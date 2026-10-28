from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np
import rasterio
from joblib import Parallel, delayed
from scipy import ndimage

try:
    from .landscape_metrics import window_metrics
    from ..gee_python.polygon_ids import polygon_id
except ImportError:
    from scripts.landscape.landscape_metrics import window_metrics
    from scripts.gee_python.polygon_ids import polygon_id


def has_real_core(mask, influence_band_px=3):
    """A polygon has a real core if erosion by the influence band is non-empty."""
    mask = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, iterations=int(influence_band_px))
    return bool(eroded.any()), int(eroded.sum())


def polygon_window(class_array, label_array, polygon_label, buffer_px):
    """Window = the polygon's bounding box expanded by buffer_px (clipped to bounds)."""
    mask = np.asarray(label_array) == polygon_label
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    rmin = max(0, rows[0] - buffer_px)
    rmax = min(class_array.shape[0] - 1, rows[-1] + buffer_px)
    cmin = max(0, cols[0] - buffer_px)
    cmax = min(class_array.shape[1] - 1, cols[-1] + buffer_px)
    return np.asarray(class_array)[rmin:rmax + 1, cmin:cmax + 1]


def _window_from_bounds(class_array, bounds, buffer_px):
    r0, r1, c0, c1 = bounds  # row start/stop, col start/stop (stop exclusive, find_objects style)
    rmin = max(0, r0 - buffer_px)
    rmax = min(class_array.shape[0], r1 + buffer_px)
    cmin = max(0, c0 - buffer_px)
    cmax = min(class_array.shape[1], c1 + buffer_px)
    return class_array[rmin:rmax, cmin:cmax]


def _polygon_landscape_row(class_array, label, focal_class, year, core_px, bounds, windows_px):
    row = {
        "polygon_id": polygon_id(int(focal_class), int(label)),
        "year": year,
        "focal_class": int(focal_class),
        "core_area_m2": core_px * 900.0,
    }
    for window_m, buffer_px in windows_px.items():
        window = _window_from_bounds(class_array, bounds, buffer_px)
        for key, value in window_metrics(window, int(focal_class)).items():
            row[f"{key}_w{window_m}"] = value
    return row


def select_study_polygons(class_array, label_array, min_pixels=56, influence_band_m=90, res=30.0):
    """Fast study filter: polygons with >= min_pixels AND a real core (non-empty erosion
    by the influence band). Returns [(label, focal_class, core_px, bounds), ...].

    Vectorized: one pass for per-label sizes and bounding boxes; per-label work happens
    on the small bbox sub-array, never the full raster.
    """
    band_px = int(round(influence_band_m / res))
    sizes = np.bincount(np.asarray(label_array).ravel())
    objects = ndimage.find_objects(label_array)
    studied = []
    for label in range(1, len(sizes)):
        if int(sizes[label]) < min_pixels:
            continue
        slc = objects[label - 1]
        if slc is None:
            continue
        sub_mask = label_array[slc] == label
        has_core, core_px = has_real_core(sub_mask, band_px)
        if not has_core:
            continue
        focal_class = int(class_array[slc][sub_mask][0])
        bounds = (slc[0].start, slc[0].stop, slc[1].start, slc[1].stop)
        studied.append((label, focal_class, core_px, bounds))
    return studied


def landscape_rows_for_studied(
    class_array, label_array, studied, year, windows_m=(90, 150, 300), res=30.0, n_jobs=-1
):
    """Compute landscape-metric rows for an already-selected studied list, in parallel.

    Operates on in-memory arrays so callers can read rasters and select polygons ONCE
    (the heavy fixed cost) and reuse them across timing/sample/full runs.
    """
    windows_px = {int(w): int(round(w / res)) for w in windows_m}
    return Parallel(n_jobs=n_jobs)(
        delayed(_polygon_landscape_row)(
            class_array, label, focal_class, year, core_px, bounds, windows_px
        )
        for label, focal_class, core_px, bounds in studied
    )


def save_landscape_csv(rows, out_csv):
    if not rows:
        return 0
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_landscape_table(
    class_raster,
    label_raster,
    year,
    out_csv,
    windows_m=(90, 150, 300),
    min_pixels=56,
    influence_band_m=90,
    res=30.0,
    n_jobs=-1,
    limit=None,
):
    with rasterio.open(class_raster) as src:
        class_array = src.read(1)
    with rasterio.open(label_raster) as src:
        label_array = src.read(1)

    studied = select_study_polygons(class_array, label_array, min_pixels, influence_band_m, res)
    if limit is not None:
        studied = studied[:limit]

    rows = landscape_rows_for_studied(class_array, label_array, studied, year, windows_m, res, n_jobs)
    save_landscape_csv(rows, out_csv)
    return len(rows)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-polygon landscape metrics for study polygons (real core)."
    )
    parser.add_argument("--class-raster", required=True, type=Path)
    parser.add_argument("--label-raster", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--windows-m", nargs="+", type=int, default=[90, 150, 300])
    parser.add_argument("--min-pixels", type=int, default=56)
    parser.add_argument("--influence-band-m", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N study polygons (quick test).")
    return parser.parse_args()


def main():
    args = parse_args()
    count = build_landscape_table(
        args.class_raster, args.label_raster, args.year, args.out_csv,
        windows_m=tuple(args.windows_m), min_pixels=args.min_pixels,
        influence_band_m=args.influence_band_m, n_jobs=args.n_jobs, limit=args.limit,
    )
    print(f"Study polygons with landscape metrics: {count}")
    print(f"Saved landscape table: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
