"""Scene x zone edge aggregation (new analytical branch, before any class aggregation).

For each Landsat scene of the validated manifest (cloud_cover <= 50 %, as in the submitted
version) the script fetches TST/NDVI/NDMI on the Albers 30 m grid, tile by tile, and writes

  revision/data/scenes/{year}/{scene_id}.parquet
      label, nb_class, ring, n_total, n_valid, valid_fraction,
      t30_mean, t30_sd, box60_mean, box90_mean, box120_mean, psf_mean, ndvi_mean, ndmi_mean
      [+ blk60/90/120_mean when --block is set: grid-anchored block means, pilot only]
  revision/data/scenes/{year}/{scene_id}_sector.parquet
      label, sector, ring (0-2), n_total, n_valid, t30_mean

for every polygon with geometric support >= 90 m (max_ring >= 3). Rows are kept even when
n_valid == 0 (fully clouded zone) so that support can be audited.

Usage:
  python aggregate_scenes.py --year 2024 --scenes LC08_L2SP_... [--block]
  python aggregate_scenes.py --year 1985 2002 2024 --all --workers 6
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import aggregate as ag  # noqa: E402
from tedge import fetch  # noqa: E402
from tedge.paths import (MANIFEST, NATIVE_TIR_M, geometry_dir, label_raster,  # noqa: E402
                         scene_dir, DATA)

TILE = 1024  # memory-bound machine (~2 GB free RAM); 2048 px tiles exhausted memory
HALO = 8
MIN_RING_ELIGIBLE = 3
MAX_CLOUD = 50.0


class YearGrid:
    """Label grid + geometry layers + eligibility lookup for one year (thread-safe reads)."""

    def __init__(self, year: int):
        self.year = year
        self.lock = threading.Lock()
        self.lab = rasterio.open(label_raster(year))
        self.ring = rasterio.open(geometry_dir(year) / "ring.tif")
        self.nb = rasterio.open(geometry_dir(year) / "nbclass.tif")
        self.sec = rasterio.open(geometry_dir(year) / "sector.tif")
        self.crs_wkt = self.lab.crs.to_wkt()
        self.transform = self.lab.transform
        self.H, self.W = self.lab.shape
        poly = pd.read_parquet(geometry_dir(year) / "polygons.parquet", columns=["label", "max_ring"])
        self.elig = np.zeros(int(poly["label"].max()) + 1, dtype=bool)
        self.elig[poly.loc[poly["max_ring"] >= MIN_RING_ELIGIBLE, "label"].to_numpy()] = True
        self.to_grid = Transformer.from_crs("EPSG:4326", self.lab.crs, always_xy=True)

    def read(self, r0, c0, h, w):
        win = Window(c0, r0, w, h)
        with self.lock:
            return (self.lab.read(1, window=win), self.ring.read(1, window=win),
                    self.nb.read(1, window=win), self.sec.read(1, window=win))

    def tiles_for(self, footprint_geojson):
        fp = shp_transform(self.to_grid.transform, shape(footprint_geojson))
        T = self.transform
        tiles = []
        for r0 in range(0, self.H, TILE):
            for c0 in range(0, self.W, TILE):
                h, w = min(TILE, self.H - r0), min(TILE, self.W - c0)
                x0, y0 = T.c + c0 * T.a, T.f + r0 * T.e
                tb = box(x0, y0 + h * T.e, x0 + w * T.a, y0)
                if fp.intersects(tb):
                    tiles.append((r0, c0, h, w))
        return tiles


def block_mean(values, valid, k, grow0, gcol0):
    """Grid-anchored k x k block mean over valid pixels, broadcast back to 30 m pixels."""
    H, W = values.shape
    rows = (np.arange(H) + grow0) // k
    cols = (np.arange(W) + gcol0) // k
    rid = (rows - rows.min())[:, None] * (cols.max() - cols.min() + 1) + (cols - cols.min())[None, :]
    v = np.where(valid, values, 0.0).ravel()
    s = np.bincount(rid.ravel(), weights=v)
    n = np.bincount(rid.ravel(), weights=valid.ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s / n)[rid]
    out[~valid] = np.nan
    return out.astype(np.float32)


def process_tile(grid: YearGrid, arrs: dict, r0, c0, h, w, native_m, block: bool):
    tst_h, in_fp_h, valid_h = ag.decode_tst(arrs["tst"])
    variants_h = ag.support_variants(tst_h, valid_h, native_m)
    if block:
        for m in (60, 90, 120):
            variants_h[f"blk{m}"] = block_mean(tst_h, valid_h, m // 30, r0 - HALO, c0 - HALO)
    core = (slice(HALO, HALO + h), slice(HALO, HALO + w))
    variants = {k: v[core] for k, v in variants_h.items()}
    in_fp, valid = in_fp_h[core], valid_h[core]
    ndvi = ag.decode_index(arrs["ndvi"])[core]
    ndmi = ag.decode_index(arrs["ndmi"])[core]
    lab, ring, nb, sec = grid.read(r0, c0, h, w)
    lab = lab.astype(np.int64)
    ok = (lab > 0) & (ring < 255)
    ok[ok] = grid.elig[lab[ok]]
    keys = np.where(ok, ag.zone_keys(lab, nb, ring), -1)
    zone = ag.aggregate_tile(keys, in_fp, valid, variants, ndvi, ndmi)
    near = ok & (ring <= 2)
    skeys = np.where(near, (lab * 8 + sec.astype(np.int64)) * 16 + ring, -1)
    sector = ag.aggregate_tile(skeys, in_fp, valid, {"t30": variants["t30"]})
    return zone, sector, int(in_fp.sum()), int(valid.sum())


def _merge(parts):
    df = pd.concat([pd.DataFrame(p) for p in parts if p is not None], ignore_index=True)
    return df.groupby("key", as_index=False).sum()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_scenes(year, rows, workers=6, block=False, out_dir=None):
    fetch.init()
    grid = YearGrid(year)
    out_dir = Path(out_dir or scene_dir(year))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = DATA / "scene_run_log.csv"
    meta_path = DATA / f"scene_metadata_{year}.jsonl"
    todo = [r for r in rows if not (out_dir / f"{r['scene_id']}.parquet").exists()]
    print(f"{year}: {len(todo)} scenes to process ({len(rows) - len(todo)} already done)", flush=True)
    native = NATIVE_TIR_M[year]
    for chunk_start in range(0, len(todo), 8):
        chunk = todo[chunk_start:chunk_start + 8]
        metas = {m["image_id"]: m for m in fetch.scene_metadata([r["image_id"] for r in chunk])}
        for r in chunk:
            t0 = time.time()
            meta = metas[r["image_id"]]
            img = fetch.scene_image(r["image_id"], year)
            tiles = grid.tiles_for(meta["footprint"])
            zparts, sparts, n_fp, n_valid = [], [], 0, 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(fetch.fetch_tile, img, grid.crs_wkt, grid.transform,
                                    r0 - HALO, c0 - HALO, h + 2 * HALO, w + 2 * HALO): (r0, c0, h, w)
                        for (r0, c0, h, w) in tiles}
                for fut in as_completed(futs):
                    r0, c0, h, w = futs[fut]
                    z, s, a, b = process_tile(grid, fut.result(), r0, c0, h, w, native, block)
                    n_fp += a; n_valid += b
                    if z is not None:
                        zparts.append(z)
                    if s is not None:
                        sparts.append(s)
            status = "ok"
            if zparts:
                zone = ag.finalize(_merge(zparts))
                zone.to_parquet(out_dir / f"{r['scene_id']}.parquet", index=False)
                sec = ag.finalize(_merge(sparts)) if sparts else pd.DataFrame()
                if len(sec):
                    # sector keys were (label*8+sector)*16+ring: undo the zone-key split
                    k = (sec["label"].astype(np.int64) * 256 + sec["nb_class"].astype(np.int64))
                    sec = sec.assign(label=(k // 8).astype(np.int32), sector=(k % 8).astype(np.uint8))
                    sec = sec[["label", "sector", "ring", "n_total", "n_valid", "t30_mean"]]
                sec.to_parquet(out_dir / f"{r['scene_id']}_sector.parquet", index=False)
                n_rows, sha = len(zone), _sha(out_dir / f"{r['scene_id']}.parquet")
            else:
                status, n_rows, sha = "no_eligible_zone", 0, ""
                pd.DataFrame().to_parquet(out_dir / f"{r['scene_id']}.parquet")
            meta_row = {k: v for k, v in meta.items() if k != "footprint"}
            meta_row.update({"scene_id": r["scene_id"], "year": year, "n_tiles": len(tiles),
                             "n_footprint_px": n_fp, "n_valid_px": n_valid})
            with open(meta_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta_row) + "\n")
            new = not log_path.exists()
            with open(log_path, "a", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                if new:
                    wr.writerow(["year", "scene_id", "status", "tiles", "rows", "footprint_px",
                                 "valid_px", "seconds", "sha256_16", "block_variants", "finished"])
                wr.writerow([year, r["scene_id"], status, len(tiles), n_rows, n_fp, n_valid,
                             round(time.time() - t0, 1), sha, int(block),
                             time.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"  {r['scene_id']}: {status} tiles={len(tiles)} rows={n_rows} "
                  f"valid={n_valid / max(n_fp, 1):.2f} {time.time() - t0:.0f}s", flush=True)


def manifest_rows(years, scenes=None):
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rows = [r for r in rows if int(r["year"]) in years and float(r["cloud_cover"]) <= MAX_CLOUD]
    if scenes:
        rows = [r for r in rows if r["scene_id"] in set(scenes)]
    return rows


if __name__ == "__main__":
    import os
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--block", action="store_true")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if not a.all and not a.scenes:
        ap.error("--scenes or --all")
    for y in a.year:
        rows = manifest_rows([y], a.scenes)
        run_scenes(y, rows, workers=a.workers, block=a.block, out_dir=a.out_dir)
