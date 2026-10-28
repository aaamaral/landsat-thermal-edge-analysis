"""Rebuild a year's class raster and polygon labels from MapBiomas Collection 10.1.

  python rebuild_class_raster.py --year 2002

Why: the submitted 2002 class raster (dados/tst_edge_leakage/parana_albers_2002.tif, from the
author's original land-use raster) agrees 99.2 % with Collection 9 and only ~91 % with
Collection 10.1, while 1985 and 2024 agree 99.2-99.3 % with Collection 10.1 (decision D-11).
All three years must come from the same collection.

Same grid, Parana clip (pixels > 0 in the template raster) and labelling rule as
scripts/landscape/polygon_inventory.py (8-connected components per class, classes in ascending
order, components in scan order, global labels from 1), but labelled through disk-backed arrays so
that it fits in the available memory. Originals are not touched; outputs go to
revision/data/inputs/ and tedge.paths picks them up when present.

Template: the original raster of the study when present (dados/tst_edge_leakage/), otherwise the
reference grid built by scripts/prepare_reference_grid.py (tedge.paths.template_raster).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gee_python"))
sys.path.insert(0, str(ROOT))
from tedge import fetch  # noqa: E402
from tedge.paths import INPUTS, REFERENCE_GRID, template_raster  # noqa: E402
from edge_config import MAPBIOMAS_ASSET  # noqa: E402
from scripts.gee_python.polygon_ids import polygon_id  # noqa: E402

TILE = 1024
ROWS = 1024  # row chunk for streaming passes


def fetch_classes(year, workers=4):
    src_path = template_raster(year)
    compare = src_path != REFERENCE_GRID  # the reference grid holds 1/0, not classes
    out = INPUTS / f"parana_albers_{year}.tif"
    INPUTS.mkdir(parents=True, exist_ok=True)
    fetch.init()
    img = ee.Image(MAPBIOMAS_ASSET).select([f"classification_{year}"], ["c"]).unmask(0).toInt16()
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
        H, W, wkt, T = src.height, src.width, src.crs.to_wkt(), src.transform
        tiles = [(r, c, min(TILE, H - r), min(TILE, W - c)) for r in range(0, H, TILE)
                 for c in range(0, W, TILE)]
        diff = tot = 0
        t0 = time.time()
        with rasterio.open(out, "w", **prof) as dst, ThreadPoolExecutor(workers) as ex:
            def work(t):
                r, c, h, w = t
                return t, fetch.fetch_tile(img, wkt, T, r, c, h, w)["c"]
            for i, ((r, c, h, w), arr) in enumerate(ex.map(work, tiles)):
                old = src.read(1, window=Window(c, r, w, h))
                new = np.where(old > 0, arr, 0).astype(np.int16)  # same Parana clip
                dst.write(new, 1, window=Window(c, r, w, h))
                m = old > 0
                diff += int((new[m] != old[m]).sum())
                tot += int(m.sum())
                if i % 40 == 0:
                    print(f"  fetch {i + 1}/{len(tiles)} ({time.time() - t0:.0f}s)", flush=True)
    if compare:
        print(f"{year}: C10.1 raster written; disagreement with the original raster {diff / tot:.4f}",
              flush=True)
    else:
        print(f"{year}: C10.1 raster written on the reference grid ({tot} pixels inside Parana)",
              flush=True)
    return out


def label(year):
    cls_path = INPUTS / f"parana_albers_{year}.tif"
    lab_path = INPUTS / f"polygon_labels_{year}.tif"
    tmp = INPUTS / f"_tmp_{year}"
    tmp.mkdir(exist_ok=True)
    with rasterio.open(cls_path) as src:
        H, W, T, crs = src.height, src.width, src.transform, src.crs
        cls = np.lib.format.open_memmap(tmp / "cls.npy", "w+", np.uint8, (H, W))
        for r in range(0, H, ROWS):
            cls[r:r + ROWS] = src.read(1, window=Window(0, r, W, min(ROWS, H - r))).astype(np.uint8)
    present = sorted(int(c) for c in np.unique(np.concatenate(
        [np.unique(cls[r:r + ROWS]) for r in range(0, H, ROWS)])) if c != 0)
    glob = np.lib.format.open_memmap(tmp / "labels.npy", "w+", np.int32, (H, W))
    comp = np.lib.format.open_memmap(tmp / "comp.npy", "w+", np.int32, (H, W))
    structure = ndimage.generate_binary_structure(2, 2)
    records, next_label = [], 1
    for c in present:
        t0 = time.time()
        mask = np.lib.format.open_memmap(tmp / "mask.npy", "w+", np.bool_, (H, W))
        for r in range(0, H, ROWS):
            mask[r:r + ROWS] = cls[r:r + ROWS] == c
        count = ndimage.label(mask, structure=structure, output=comp)
        del mask
        if count == 0:
            continue
        sizes = np.zeros(count + 1, np.int64)
        sr = np.zeros(count + 1, np.float64)
        sc = np.zeros(count + 1, np.float64)
        cols = np.arange(W, dtype=np.float64)
        for r in range(0, H, ROWS):
            blk = comp[r:r + ROWS]
            nz = blk > 0
            ids = blk[nz]
            rr, cc = np.nonzero(nz)
            sizes += np.bincount(ids, minlength=count + 1)
            sr += np.bincount(ids, (rr + r).astype(np.float64), minlength=count + 1)
            sc += np.bincount(ids, cols[cc], minlength=count + 1)
            g = glob[r:r + ROWS]
            g[nz] = ids + (next_label - 1)
        for k in range(1, count + 1):
            x, y = T * (sc[k] / sizes[k] + 0.5, sr[k] / sizes[k] + 0.5)
            records.append({"polygon_id": polygon_id(c, next_label + k - 1), "year": year,
                            "focal_class": c, "pixel_count": int(sizes[k]),
                            "area_m2": int(sizes[k]) * 900.0, "centroid_x": float(x),
                            "centroid_y": float(y)})
        next_label += count
        print(f"  class {c}: {count} polygons ({time.time() - t0:.0f}s)", flush=True)
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="int32", crs=crs, transform=T,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(lab_path, "w", **prof) as dst:
        for r in range(0, H, ROWS):
            dst.write(np.asarray(glob[r:r + ROWS]), 1, window=Window(0, r, W, min(ROWS, H - r)))
    with open(INPUTS / f"polygon_inventory_{year}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    del cls, glob, comp, blk, g
    import gc
    gc.collect()
    for f in tmp.glob("*.npy"):  # Windows keeps memmaps locked until released; best effort
        try:
            f.unlink()
        except PermissionError:
            pass
    print(f"{year}: {next_label - 1} polygons labelled", flush=True)


if __name__ == "__main__":
    import os
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--step", choices=["all", "fetch", "label"], default="all")
    a = ap.parse_args()
    if a.step in ("all", "fetch"):
        fetch_classes(a.year)
    if a.step in ("all", "label"):
        label(a.year)
