"""Export the aggregated derived dataset deposited on Zenodo (revision round 2, comment R2.3).

The full derived database (scene x polygon x neighbouring class x distance-ring cells, ~6.5 GB of
scene parquet files) is NOT exported. This script writes aggregated levels from which the published
estimates can be recomputed without the Earth Engine extraction and raster processing:

Tier 1 (default; small)
  contrasts/contrasts_W150_{year}.parquet   one row per (scene, polygon): c0..c4 = TST(ring k) minus
                                            the reference band of the 150 m window (t30 support)
  contrasts/contrasts_W300_{year}.parquet   same for the 300 m window (c0..c9): profiles and DEP
  dtec/dtec_W150_ring0_{year}.parquet       one row per (scene, polygon, neighbouring class): DTEC
  polygons/polygons_{year}.parquet          attributes of the polygons present in the tables above
  results/*.csv                             class-level results behind the figures and tables
Tier 2 (--with-polygon-ring; ~1.8 GB)
  polygon_ring/polygon_ring_{year}.parquet  (scene, polygon, ring) means for all thermal supports
                                            (t30, box60, box90, box120, psf) + NDVI/NDMI: needed for
                                            other windows, edge exclusion and thermal-support sweeps.

Exactly the same engine functions and parameters as revision/08_statistics/core_estimates.py are
used (MAIN = window 150 m, t30, no exclusion, no valid-fraction threshold, >= 3 valid pixels).
A MANIFEST.csv (path, bytes, rows, sha256) is written last.

Usage:
  python revision/11_zenodo/export_derived_data.py                       # tier 1
  python revision/11_zenodo/export_derived_data.py --with-polygon-ring   # tier 1 + tier 2
  python revision/11_zenodo/export_derived_data.py --years 1985          # quick test
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import engine  # noqa: E402
from tedge.paths import DATA, OUT, geometry_dir  # noqa: E402

YEARS = (1985, 2002, 2024)
DEFAULT_OUT = DATA / "zenodo_deposit"
POLY_COLS = ["label", "focal_class", "n_px", "max_ring", "centroid_x", "centroid_y",
             "perimeter_faces", "limit_faces"]


def table(year: int) -> Path:
    """The polygon x ring x scene table used for the published estimates (2002 = rebuilt C10.1)."""
    return DATA / "tables" / f"full_polygon_ring_{year}.parquet"


def write(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return len(df)


def export_contrasts(year: int, window: int, out: Path) -> set:
    df = engine.contrasts(year, window, table=table(year))
    k = [c for c in df.columns if c.startswith("c") and c[1:].isdigit()]
    keep = ["scene_id", "label", "focal_class"] + k + ["ref"]
    df = df[keep].copy()
    df["ref"] = df["ref"].astype("float32")
    n = write(df, out / "contrasts" / f"contrasts_W{window}_{year}.parquet")
    print(f"  contrasts W{window} {year}: {n:,} rows, {df.label.nunique():,} polygons, "
          f"{df.scene_id.nunique()} scenes", flush=True)
    return set(df.label.unique())


def export_dtec(year: int, out: Path) -> set:
    df = engine.interface_contrasts(year, 150, excl=0, table=table(year))
    df["dtec"] = df["dtec"].astype("float32")
    n = write(df, out / "dtec" / f"dtec_W150_ring0_{year}.parquet")
    print(f"  dtec {year}: {n:,} rows", flush=True)
    return set(df.label.unique())


def export_polygons(year: int, labels: set, out: Path) -> None:
    src = (geometry_dir(year) / "polygons.parquet").as_posix()
    df = duckdb.sql(f"SELECT {', '.join(POLY_COLS)} FROM read_parquet('{src}')").df()
    df = df[df.label.isin(labels)]
    n = write(df, out / "polygons" / f"polygons_{year}.parquet")
    print(f"  polygons {year}: {n:,} rows", flush=True)


def copy_results(out: Path) -> None:
    dst = out / "results"
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("full", "tables"):
        for f in sorted((OUT / sub).glob("*.csv")):
            shutil.copy2(f, dst / f.name)
    print(f"  results: {len(list(dst.glob('*.csv')))} csv files", flush=True)


def manifest(out: Path) -> None:
    rows = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        nrows = (duckdb.sql(f"SELECT count(*) FROM read_parquet('{f.as_posix()}')").fetchone()[0]
                 if f.suffix == ".parquet" else "")
        rows.append({"path": f.relative_to(out).as_posix(), "bytes": f.stat().st_size,
                     "rows": nrows, "sha256": h.hexdigest()})
    m = pd.DataFrame(rows)
    m.to_csv(out / "MANIFEST.csv", index=False)
    print(f"MANIFEST: {len(m)} files, {m.bytes.sum() / 1e6:,.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--with-polygon-ring", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    shutil.copy2(here / "DATASET_README.md", a.out / "README.md")
    shutil.copy2(here / "verify_deposit.py", a.out / "verify_deposit.py")
    t0 = time.time()
    for y in a.years:
        print(f"[{y}]", flush=True)
        labels = export_contrasts(y, 150, a.out) | export_contrasts(y, 300, a.out) | export_dtec(y, a.out)
        export_polygons(y, labels, a.out)
        if a.with_polygon_ring:
            dst = a.out / "polygon_ring" / f"polygon_ring_{y}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(table(y), dst)
            print(f"  polygon_ring {y}: copied", flush=True)
    copy_results(a.out)
    manifest(a.out)
    print(f"done in {time.time() - t0:,.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
