"""Temporal stability of each polygon's class in the adjacent annual MapBiomas maps.

  python class_stability.py --year 1985 2002 2024

The annual map labels the whole year, while each scene is a single date. Pixels whose class differs
in the adjacent years (y-1, y+1, within 1985-2024) are the ones most likely to be in transition or
misclassified at the scene date. For each eligible polygon (max_ring >= 3) we count, separately for
the edge ring (0) and the rest, the pixels whose MapBiomas C10.1 class equals the focal class in all
available adjacent years. The same-year band is fetched as well: its agreement with the local class
raster measures grid/resampling consistency (should be ~1).

Output: revision/data/covariates/class_stability_{year}.parquet
        (label, n_edge, stable_edge, n_inner, stable_inner, n_sameyear_agree, n_px)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gee_python"))
from tedge import fetch  # noqa: E402
from tedge.paths import DATA, class_raster, geometry_dir, label_raster  # noqa: E402
from edge_config import MAPBIOMAS_ASSET  # noqa: E402

TILE = 1024
FIRST, LAST = 1985, 2024


def adjacent(year):
    return [y for y in (year - 1, year + 1) if FIRST <= y <= LAST]


def run(year, workers=4):
    fetch.init()
    adj = adjacent(year)
    img = ee.Image(MAPBIOMAS_ASSET).select(
        [f"classification_{y}" for y in [year] + adj], [f"c{y}" for y in [year] + adj]).unmask(0).toInt16()
    poly = pd.read_parquet(geometry_dir(year) / "polygons.parquet", columns=["label", "max_ring"])
    elig = np.zeros(int(poly["label"].max()) + 1, bool)
    elig[poly.loc[poly["max_ring"] >= 3, "label"].to_numpy()] = True
    lsrc, csrc = rasterio.open(label_raster(year)), rasterio.open(class_raster(year))
    rsrc = rasterio.open(geometry_dir(year) / "ring.tif")
    H, W = lsrc.shape
    wkt, T = lsrc.crs.to_wkt(), lsrc.transform
    tiles = []
    for r0 in range(0, H, TILE):
        for c0 in range(0, W, TILE):
            h, w = min(TILE, H - r0), min(TILE, W - c0)
            lab = lsrc.read(1, window=Window(c0, r0, w, h))
            if elig[lab[lab > 0]].any():
                tiles.append((r0, c0, h, w))
    print(f"{year}: {len(tiles)} tiles, adjacent years {adj}", flush=True)

    def work(t):
        r0, c0, h, w = t
        arr = fetch.fetch_tile(img, wkt, T, r0, c0, h, w)
        return t, arr

    parts, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, ((r0, c0, h, w), arr) in enumerate(ex.map(work, tiles)):
            win = Window(c0, r0, w, h)
            lab = lsrc.read(1, window=win).astype(np.int64)
            cls = csrc.read(1, window=win).astype(np.int16)
            ring = rsrc.read(1, window=win)
            m = (lab > 0) & elig[lab]
            if not m.any():
                continue
            stable = np.ones(m.sum(), bool)
            for y in adj:
                stable &= arr[f"c{y}"][m] == cls[m]
            same = arr[f"c{year}"][m] == cls[m]
            edge = ring[m] == 0
            df = pd.DataFrame({"label": lab[m], "edge": edge, "stable": stable, "same": same,
                               "edge_stable": edge & stable})
            parts.append(df.groupby("label").agg(
                n_edge=("edge", "sum"), stable_edge=("edge_stable", "sum"),
                n_px=("edge", "size"), stable_all=("stable", "sum"),
                n_sameyear_agree=("same", "sum")).reset_index())
            if i % 25 == 0:
                print(f"  {year}: tile {i + 1}/{len(tiles)} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(parts).groupby("label", as_index=False).sum()
    res["n_inner"] = res["n_px"] - res["n_edge"]
    res["stable_inner"] = res["stable_all"] - res["stable_edge"]
    out = DATA / "covariates" / f"class_stability_{year}.parquet"
    res.drop(columns="stable_all").to_parquet(out, index=False)
    print(f"{year}: {len(res)} polygons; same-year agreement "
          f"{res['n_sameyear_agree'].sum() / res['n_px'].sum():.4f}; "
          f"stable edge {res['stable_edge'].sum() / res['n_edge'].sum():.3f}, "
          f"inner {res['stable_inner'].sum() / res['n_inner'].sum():.3f}", flush=True)


if __name__ == "__main__":
    import os
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    for y in ap.parse_args().year:
        run(y)
