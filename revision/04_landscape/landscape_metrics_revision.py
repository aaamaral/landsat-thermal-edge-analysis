"""Landscape metrics for the polygons of the revision database (needed for the rebuilt 2002, D-11).

  python landscape_metrics_revision.py --year 2002

Reuses the submitted metric code unchanged (scripts/landscape/polygon_landscape.py:
_polygon_landscape_row, has_real_core; windows 90/150/300 m), but only for the eligible polygons
of the revision geometry (max_ring >= 3). Memory-bounded: bounding boxes are found in row blocks and
each polygon reads only its own window (bbox + 300 m buffer) from disk, so no full-raster array is
held (the full arrays of the original code need > 3 GB). Output has the same columns as
dados/tst_edge_leakage/landscape/landscape_{year}.csv and goes to
revision/data/inputs/landscape_{year}.csv (the original files are not overwritten).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision" / "src"))
import scripts.landscape.landscape_metrics as lm  # noqa: E402
from scripts.landscape.polygon_landscape import _polygon_landscape_row, has_real_core  # noqa: E402
from tedge.paths import INPUTS, class_raster, geometry_dir, label_raster  # noqa: E402

WINDOWS_M = (90, 150, 300)
BAND_PX = 3  # 90 m influence band, as in the submitted analysis
ROWS = 512
ADJ_ROWS = 256


def _adjacency_matrix_blocked(array, classes, nodata=0):
    """Same counts as landscape_metrics._adjacency_matrix, accumulated over row blocks.

    Horizontal pairs never cross rows; vertical pairs are taken within each block and across the
    block seam (last row of a block with the first row of the next), so every pair is counted once.
    """
    k = len(classes)
    remap = np.full(int(array.max()) + 1, -1, dtype=np.int64)
    for index, class_value in enumerate(classes):
        remap[class_value] = index
    g = np.zeros((k, k), dtype=np.float64)

    def add(first, second):
        first, second = first.ravel(), second.ravel()
        keep = (first != nodata) & (second != nodata)
        codes = remap[first[keep]] * k + remap[second[keep]]
        counts = np.bincount(codes, minlength=k * k).reshape(k, k).astype(np.float64)
        return counts + counts.T

    H = array.shape[0]
    for r in range(0, H, ADJ_ROWS):
        blk = array[r:r + ADJ_ROWS + 1]  # one extra row: vertical pairs across the seam
        g += add(blk[:ADJ_ROWS, :-1], blk[:ADJ_ROWS, 1:])
        g += add(blk[:-1, :], blk[1:, :])
    return g


lm._adjacency_matrix = _adjacency_matrix_blocked  # used by window_metrics via module lookup


def bboxes(lsrc, want):
    """(label -> r0, r1, c0, c1) for the wanted labels, stop-exclusive, in row blocks."""
    keep = np.zeros(int(want.max()) + 1, bool)
    keep[want] = True
    parts = []
    for r in range(0, lsrc.height, ROWS):
        h = min(ROWS, lsrc.height - r)
        lab = lsrc.read(1, window=Window(0, r, lsrc.width, h))
        m = (lab > 0) & (lab <= want.max())
        m[m] = keep[lab[m]]
        rr, cc = np.nonzero(m)
        if rr.size:
            df = pd.DataFrame({"label": lab[rr, cc], "r": rr + r, "c": cc})
            parts.append(df.groupby("label").agg(r0=("r", "min"), r1=("r", "max"),
                                                 c0=("c", "min"), c1=("c", "max")))
    b = pd.concat(parts).groupby(level=0).agg(r0=("r0", "min"), r1=("r1", "max"),
                                              c0=("c0", "min"), c1=("c1", "max"))
    b[["r1", "c1"]] += 1
    return b


def run(year, limit=None):
    poly = pd.read_parquet(geometry_dir(year) / "polygons.parquet", columns=["label", "max_ring"])
    want = np.sort(poly.loc[poly["max_ring"] >= 3, "label"].to_numpy())
    if limit:
        want = want[:limit]
    windows_px = {w: int(round(w / 30)) for w in WINDOWS_M}
    pad = max(windows_px.values())
    t0 = time.time()
    rows, failed = [], []
    with rasterio.open(label_raster(year)) as lsrc, rasterio.open(class_raster(year)) as csrc:
        bb = bboxes(lsrc, want)
        print(f"{year}: bounding boxes {time.time() - t0:.0f}s; {len(bb)} polygons", flush=True)
        H, W = lsrc.height, lsrc.width
        for i, (label, r) in enumerate(bb.iterrows()):
            # read bbox + buffer; local bounds are relative to the read window
            R0, R1 = max(0, r.r0 - pad), min(H, r.r1 + pad)
            C0, C1 = max(0, r.c0 - pad), min(W, r.c1 + pad)
            win = Window(C0, R0, C1 - C0, R1 - R0)
            cls = csrc.read(1, window=win)
            lab = lsrc.read(1, window=Window(r.c0, r.r0, r.c1 - r.c0, r.r1 - r.r0))
            sub = lab == label
            has_core, core_px = has_real_core(sub, BAND_PX)
            focal = int(cls[r.r0 - R0:r.r1 - R0, r.c0 - C0:r.c1 - C0][sub][0])
            bounds = (r.r0 - R0, r.r1 - R0, r.c0 - C0, r.c1 - C0)
            try:
                rows.append(_polygon_landscape_row(cls, int(label), focal, year, core_px, bounds,
                                                   windows_px))
            except MemoryError:  # giant polygon window: keep it listed with NaN metrics
                failed.append(int(label))
                rows.append({"polygon_id": f"c{focal:03d}_p{int(label):019d}", "year": year,
                             "focal_class": focal, "core_area_m2": core_px * 900.0})
                print(f"  MemoryError for label {label} ({cls.shape})", flush=True)
            if i % 10000 == 0:
                print(f"  {i}/{len(bb)} ({time.time() - t0:.0f}s)", flush=True)
    out = INPUTS / f"landscape_{year}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{year}: {len(rows)} rows -> {out} ({time.time() - t0:.0f}s); "
          f"{len(failed)} polygons without metrics (memory): {failed}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("GDAL_CACHEMAX", "256")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.year, a.limit)
