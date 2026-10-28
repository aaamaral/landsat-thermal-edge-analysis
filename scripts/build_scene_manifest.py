"""Rebuild the Landsat scene manifest (Earth Engine).

  python scripts/build_scene_manifest.py                 # writes to outputs/scene_manifests/
  python scripts/build_scene_manifest.py --out-dir my_dir

Lists every Collection 2 Tier 1 Level 2 scene of the collections in
scripts/gee_python/edge_config.py (YEAR_SETTINGS) that intersects the Parana boundary
(PARANA_GPKG), was acquired in the calendar year, and lies on WRS-2 paths 220-224 and rows 76-78.
Same filters, fields and sort order as the manifest used in the study
(outputs/scene_manifests/landsat_scene_manifest_y1985-2002-2024_p220-224_r76-78.csv). The
aggregation uses the scenes with cloud_cover <= 50 %. Earth Engine may reprocess or add scenes
over time, so a rebuilt manifest can differ from the one distributed here; the distributed file
is the one to use to reproduce the study.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "gee_python"))
from edge_config import YEAR_SETTINGS, load_parana_geometry  # noqa: E402

YEARS = [1985, 2002, 2024]
WRS_PATHS = [220, 221, 222, 223, 224]
WRS_ROWS = [76, 77, 78]
FIELDS = ["year", "sensor_group", "collection_id", "scene_id", "image_id", "date", "month",
          "sensor", "wrs_path", "wrs_row", "cloud_cover"]


def manifest_rows(region, years=YEARS, wrs_paths=WRS_PATHS, wrs_rows=WRS_ROWS):
    rows = []
    for year in years:
        cfg = YEAR_SETTINGS[year]
        start, end = ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1)
        for collection_id in cfg.collections:
            col = (ee.ImageCollection(collection_id).filterBounds(region).filterDate(start, end)
                   .filter(ee.Filter.inList("WRS_PATH", wrs_paths))
                   .filter(ee.Filter.inList("WRS_ROW", wrs_rows)))

            def to_feature(image, year=year, cfg=cfg, collection_id=collection_id):
                date = ee.Date(image.get("system:time_start"))
                return ee.Feature(None, {
                    "year": year, "sensor_group": cfg.label, "collection_id": collection_id,
                    "scene_id": image.get("LANDSAT_PRODUCT_ID"),
                    "image_id": ee.String(collection_id).cat("/").cat(ee.String(image.id())),
                    "date": date.format("YYYY-MM-dd"), "month": date.get("month"),
                    "sensor": image.get("SPACECRAFT_ID"), "wrs_path": image.get("WRS_PATH"),
                    "wrs_row": image.get("WRS_ROW"), "cloud_cover": image.get("CLOUD_COVER")})

            feats = ee.FeatureCollection(col.map(to_feature)).getInfo().get("features", [])
            rows += [{f: ft.get("properties", {}).get(f, "") for f in FIELDS} for ft in feats]
    return sorted(rows, key=lambda r: (int(r["year"] or 0), int(r["wrs_path"] or 0),
                                       int(r["wrs_row"] or 0), str(r["date"]), str(r["scene_id"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "scene_manifests")
    a = ap.parse_args()
    project = os.environ.get("EE_PROJECT")
    if not project:
        sys.exit("EE_PROJECT is not set. Set it to a Google Cloud project registered for Earth "
                 "Engine after running 'earthengine authenticate'.")
    ee.Initialize(project=project)
    rows = manifest_rows(load_parana_geometry())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "landsat_scene_manifest_y1985-2002-2024_p220-224_r76-78.csv"
    if out.exists():
        out = out.with_name(out.stem + "_rebuilt.csv")  # never overwrite the study manifest
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scenes -> {out}")
