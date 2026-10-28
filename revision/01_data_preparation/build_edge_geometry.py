"""Build the per-year edge-geometry layers and zone/contact tables (local, tiled).

Inputs (from sub-project A, unchanged): polygon_labels_{year}.tif and parana_albers_{year}.tif
(MapBiomas C9 reclassified to the Albers 30 m grid).

Outputs under revision/data/geometry/{year}/:
  ring.tif, nbclass.tif, sector.tif   uint8 rasters on the label grid (255 = no polygon)
  zones.parquet      label, nb_class, ring -> n_px (geometric pixel count of the zone)
  zones_sector.parquet  label, sector, ring (0-2) -> n_px (orientation analysis)
  contacts.parquet   label, nb_class -> faces, normal_e, normal_n (4-neighbour pixel faces)
  polygons.parquet   label, focal_class, n_px, max_ring, perimeter_faces, ...
  qc.json            own-boundary agreement, timings, counts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import geometry as g  # noqa: E402
from tedge.paths import geometry_dir, label_raster, class_raster  # noqa: E402

TILE = 2048
PAD = 16  # > ring cap + 1: deeper pixels are capped anyway


def _read(src, r0, c0, h, w):
    win = Window(c0, r0, w, h)
    return src.read(1, window=win, boundless=True, fill_value=0)


def run(year: int, tile: int = TILE) -> dict:
    """Resumable, constant-memory build: each tile row spills its parts to parts/r{r0}.*"""
    import duckdb

    out = geometry_dir(year)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with rasterio.open(label_raster(year)) as lsrc, rasterio.open(class_raster(year)) as csrc:
        H, W = lsrc.shape
        transform = lsrc.transform
        profile = lsrc.profile.copy()
        profile.update(dtype="uint8", nodata=255, compress="deflate", predictor=2,
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        resume = any(parts.glob("r*.done"))
        dsts = {k: (rasterio.open(out / f"{k}.tif", "r+") if resume and (out / f"{k}.tif").exists()
                    else rasterio.open(out / f"{k}.tif", "w", **profile))
                for k in ("ring", "nbclass", "sector")}
        for r0 in range(0, H, tile):
            if (parts / f"r{r0}.done").exists():
                continue
            zone_parts, sector_parts, contact_parts, poly_parts = [], [], [], []
            own_hits = own_tot = 0
            for c0 in range(0, W, tile):
                h, w = min(tile, H - r0), min(tile, W - c0)
                lab = _read(lsrc, r0 - PAD, c0 - PAD, h + 2 * PAD, w + 2 * PAD).astype(np.int64)
                core = (slice(PAD, PAD + h), slice(PAD, PAD + w))
                if not (lab[core] > 0).any():
                    for k, d in dsts.items():
                        d.write(np.full((h, w), 255 if k != "sector" else 0, np.uint8), 1,
                                window=Window(c0, r0, w, h))
                    continue
                cls = _read(csrc, r0 - PAD, c0 - PAD, h + 2 * PAD, w + 2 * PAD).astype(np.int16)
                geo = g.edge_geometry(lab, cls)
                L = lab[core]
                ring = geo["ring"][core]
                nb = geo["nb_class"][core]
                sec = geo["sector"][core]
                inside = L > 0
                own_hits += int(geo["own_boundary"][core][inside].sum())
                own_tot += int(inside.sum())
                del geo
                nb8 = np.where(nb < 0, 255, nb).astype(np.uint8)
                dsts["ring"].write(ring, 1, window=Window(c0, r0, w, h))
                dsts["nbclass"].write(nb8, 1, window=Window(c0, r0, w, h))
                dsts["sector"].write(sec, 1, window=Window(c0, r0, w, h))
                li, ri, ni, si = L[inside], ring[inside].astype(np.int64), \
                    nb8[inside].astype(np.int64), sec[inside].astype(np.int64)
                key = (li * 256 + ni) * 16 + ri
                uk, cnt = np.unique(key, return_counts=True)
                zone_parts.append(pd.DataFrame({"key": uk, "n_px": cnt}))
                near = ri <= 2  # orientation analysis uses the near-edge rings only
                skey = (li[near] * 8 + si[near]) * 16 + ri[near]
                uk, cnt = np.unique(skey, return_counts=True)
                sector_parts.append(pd.DataFrame({"skey": uk, "n_px": cnt}))
                # contacts counted once: faces whose focal pixel lies in the tile core
                cf = _faces_of_focal(lab, cls, _core_mask(lab.shape, core))
                contact_parts.append(pd.DataFrame(cf))
                rows, cols = np.nonzero(inside)
                fc = cls[core][inside]
                poly_parts.append(pd.DataFrame({
                    "label": li, "focal_class": fc.astype(np.int16),
                    "row": rows + r0, "col": cols + c0, "ring": ri,
                }).groupby("label").agg(
                    focal_class=("focal_class", "first"), n_px=("ring", "size"),
                    max_ring=("ring", "max"), sum_row=("row", "sum"), sum_col=("col", "sum"),
                ).reset_index())
            for name, lst in (("zones", zone_parts), ("sector", sector_parts),
                              ("contacts", contact_parts), ("poly", poly_parts)):
                if lst:
                    pd.concat(lst, ignore_index=True).to_parquet(parts / f"r{r0}.{name}.parquet", index=False)
            (parts / f"r{r0}.done").write_text(json.dumps({"own_hits": own_hits, "own_tot": own_tot}))
            del zone_parts, sector_parts, contact_parts, poly_parts
            print(f"  {year}: rows {r0}/{H} ({time.time() - t0:.0f}s)", flush=True)
        for d in dsts.values():
            d.close()
    return assemble(year, transform, t0)


def assemble(year, transform, t0):
    """Merge the spilled parts with DuckDB (bounded memory) and write the QC summary."""
    import duckdb

    out = geometry_dir(year)
    parts = out / "parts"
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'")
    tmp = out / "duck_tmp"
    tmp.mkdir(exist_ok=True)
    c.execute(f"SET temp_directory='{tmp.as_posix()}'")
    P = parts.as_posix()
    c.execute(f"""COPY (SELECT (key // 4096)::BIGINT AS label, ((key // 16) % 256)::UTINYINT AS nb_class,
                 (key % 16)::UTINYINT AS ring, sum(n_px)::BIGINT AS n_px
               FROM read_parquet('{P}/*.zones.parquet') GROUP BY key)
               TO '{(out / "zones.parquet").as_posix()}' (FORMAT parquet)""")
    c.execute(f"""COPY (SELECT (skey // 128)::BIGINT AS label, ((skey // 16) % 8)::UTINYINT AS sector,
                 (skey % 16)::UTINYINT AS ring, sum(n_px)::BIGINT AS n_px
               FROM read_parquet('{P}/*.sector.parquet') GROUP BY skey)
               TO '{(out / "zones_sector.parquet").as_posix()}' (FORMAT parquet)""")
    c.execute(f"""COPY (SELECT label, nb_class, sum(faces) AS faces, sum(normal_e) AS normal_e,
                 sum(normal_n) AS normal_n FROM read_parquet('{P}/*.contacts.parquet') GROUP BY ALL)
               TO '{(out / "contacts.parquet").as_posix()}' (FORMAT parquet)""")
    contacts = (out / "contacts.parquet").as_posix()
    c.execute(f"""COPY (
        WITH p AS (SELECT label, first(focal_class) AS focal_class, sum(n_px) AS n_px,
                          max(max_ring) AS max_ring, sum(sum_row) AS sr, sum(sum_col) AS sc
                   FROM read_parquet('{P}/*.poly.parquet') GROUP BY label),
             k AS (SELECT label, sum(faces) AS perimeter_faces,
                          sum(faces) FILTER (WHERE nb_class = 0) AS limit_faces
                   FROM read_parquet('{contacts}') GROUP BY label)
        SELECT p.label, focal_class, n_px, max_ring,
               {transform.c} + (sc / n_px + 0.5) * {transform.a} AS centroid_x,
               {transform.f} + (sr / n_px + 0.5) * {transform.e} AS centroid_y,
               perimeter_faces, coalesce(limit_faces, 0) AS limit_faces
        FROM p LEFT JOIN k USING (label))
      TO '{(out / "polygons.parquet").as_posix()}' (FORMAT parquet)""")
    own = [json.loads(f.read_text()) for f in parts.glob("r*.done")]

    def q(sql):
        return c.execute(sql).fetchone()[0]

    pq = (out / "polygons.parquet").as_posix()
    qc = {
        "year": year,
        "seconds": round(time.time() - t0, 1),
        "polygons": q(f"SELECT count(*) FROM '{pq}'"),
        "polygons_max_ring_ge3": q(f"SELECT count(*) FROM '{pq}' WHERE max_ring >= 3"),
        "zones": q(f"SELECT count(*) FROM '{(out / 'zones.parquet').as_posix()}'"),
        "zones_sector": q(f"SELECT count(*) FROM '{(out / 'zones_sector.parquet').as_posix()}'"),
        "contacts": q(f"SELECT count(*) FROM '{contacts}'"),
        "own_boundary_share": sum(o["own_hits"] for o in own) / max(sum(o["own_tot"] for o in own), 1),
        "pixels_in_polygons": sum(o["own_tot"] for o in own),
    }
    (out / "qc.json").write_text(json.dumps(qc, indent=2))
    return qc


def _core_mask(shape, core):
    m = np.zeros(shape, dtype=bool)
    m[core] = True
    return m


def _faces_of_focal(lab, cls, focal_mask):
    """contact_faces computed only for focal pixels in ``focal_mask`` (neighbours unrestricted)."""
    out_l, out_c, out_e, out_n = [], [], [], []
    cls = cls.astype(np.int16)
    for dr, dc in g.OFFSETS_4:
        nl = g._shift(lab, dr, dc, 0)
        nc = np.where(nl == 0, 0, g._shift(cls, dr, dc, 0))
        m = focal_mask & (lab > 0) & (nl != lab)
        out_l.append(lab[m]); out_c.append(nc[m])
        out_e.append(np.full(m.sum(), dc, np.float64)); out_n.append(np.full(m.sum(), -dr, np.float64))
    lab_v = np.concatenate(out_l).astype(np.int64)
    if lab_v.size == 0:
        return {"label": [], "nb_class": [], "faces": [], "normal_e": [], "normal_n": []}
    key = lab_v * 64 + np.concatenate(out_c).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    return {"label": uk // 64, "nb_class": uk % 64, "faces": np.bincount(inv),
            "normal_e": np.bincount(inv, weights=np.concatenate(out_e)),
            "normal_n": np.bincount(inv, weights=np.concatenate(out_n))}


if __name__ == "__main__":
    import os
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--tile", type=int, default=TILE)
    a = ap.parse_args()
    for y in a.year:
        print(json.dumps(run(y, a.tile), indent=2))
