from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

try:
    from ..gee_python.polygon_ids import polygon_id
except ImportError:
    from scripts.gee_python.polygon_ids import polygon_id


INVENTORY_FIELDS = [
    "polygon_id",
    "year",
    "focal_class",
    "pixel_count",
    "area_m2",
    "centroid_x",
    "centroid_y",
]


def read_class_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def write_label_raster(path, labels, transform, crs):
    # int32 supports ~2.1B polygons and halves size vs int64.
    labels = np.asarray(labels).astype("int32")
    with rasterio.open(
        path, "w", driver="GTiff", height=labels.shape[0], width=labels.shape[1],
        count=1, dtype="int32", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(labels, 1)


def write_inventory_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def label_polygons(class_array, ignore_classes=(0,), connectivity=2):
    """Label connected same-class components (8-connected by default).

    Returns (labels, records): ``labels`` is an int64 array (0 = background);
    ``records`` is a list of {polygon_id, focal_class, pixel_count}, one per
    polygon, in label order (records[i] corresponds to label i + 1).
    """
    class_array = np.asarray(class_array)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels = np.zeros(class_array.shape, dtype=np.int64)
    records = []
    next_label = 1
    present = [int(c) for c in np.unique(class_array) if int(c) not in set(ignore_classes)]
    for focal_class in present:
        component_labels, count = ndimage.label(class_array == focal_class, structure=structure)
        if count == 0:
            continue
        # Vectorized: per-component sizes in one pass (no per-component full-array scan),
        # and a single offset remap of component ids into the global label space.
        sizes = np.bincount(component_labels.ravel())  # sizes[i] = pixels in component i
        nonzero = component_labels > 0
        labels[nonzero] = component_labels[nonzero].astype(np.int64) + (next_label - 1)
        for local_index in range(1, count + 1):
            records.append(
                {
                    "polygon_id": polygon_id(focal_class, next_label + local_index - 1),
                    "focal_class": focal_class,
                    "pixel_count": int(sizes[local_index]),
                }
            )
        next_label += count
    return labels, records


def build_polygon_inventory(labels, records, year, transform, pixel_area_m2=900.0):
    """One inventory row per polygon: id, year, class, pixel_count, area, centroid (CRS units)."""
    labels = np.asarray(labels)
    centers = ndimage.center_of_mass(
        np.ones_like(labels), labels, [index + 1 for index in range(len(records))]
    )
    rows = []
    for record, (row_mean, col_mean) in zip(records, centers):
        x, y = transform * (col_mean + 0.5, row_mean + 0.5)
        rows.append(
            {
                "polygon_id": record["polygon_id"],
                "year": year,
                "focal_class": record["focal_class"],
                "pixel_count": record["pixel_count"],
                "area_m2": record["pixel_count"] * pixel_area_m2,
                "centroid_x": float(x),
                "centroid_y": float(y),
            }
        )
    return rows


def build_inventory_for_raster(parana_albers_raster, year, out_dir):
    """Read a Parana-Albers raster -> label -> inventory; write label raster + CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_array, transform, crs = read_class_raster(parana_albers_raster)
    labels, records = label_polygons(class_array)
    rows = build_polygon_inventory(labels, records, year, transform)
    label_path = out_dir / f"polygon_labels_{year}.tif"
    csv_path = out_dir / f"polygon_inventory_{year}.csv"
    write_label_raster(label_path, labels, transform, crs)
    write_inventory_csv(csv_path, rows)
    return label_path, csv_path, len(rows)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Label Parana-Albers MapBiomas polygons and build a per-year inventory."
    )
    parser.add_argument(
        "--raster", required=True, type=Path,
        help="Parana-Albers raster (from raster_prep.prepare_parana_raster).",
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "polygon_inventory")
    return parser.parse_args()


def main():
    args = parse_args()
    label_path, csv_path, count = build_inventory_for_raster(args.raster, args.year, args.out_dir)
    print(f"Polygons: {count}")
    print(f"Saved label raster: {label_path}")
    print(f"Saved inventory: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
