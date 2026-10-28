"""Canonical paths for the revision pipeline (heavy data under dados/ and revision/data/)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DADOS = ROOT / "dados" / "tst_edge_leakage"
REV = ROOT / "revision"
DATA = REV / "data"
OUT = REV / "outputs"
MANIFEST = ROOT / "outputs" / "scene_manifests" / \
    "landsat_scene_manifest_y1985-2002-2024_p220-224_r76-78.csv"

YEARS = (1985, 2002, 2024)
YEAR_SENSOR = {1985: "1985_L5TM", 2002: "2002_L7ETM", 2024: "2024_L8L9TIRS"}
NATIVE_TIR_M = {1985: 120, 2002: 60, 2024: 100}


INPUTS = DATA / "inputs"  # rebuilt inputs (D-11); the originals in dados/ are never overwritten
# 30 m analysis grid (1 = inside Parana, 0 = outside) written by scripts/prepare_reference_grid.py
REFERENCE_GRID = INPUTS / "reference_grid_30m.tif"


def _missing(path: Path, hint: str) -> FileNotFoundError:
    return FileNotFoundError(f"{path} not found. {hint}")


def template_raster(year: int) -> Path:
    """Grid and Parana clip used to rasterise MapBiomas (step 1 of the workflow)."""
    original = DADOS / f"parana_albers_{year}.tif"
    if original.exists():
        return original
    if REFERENCE_GRID.exists():
        return REFERENCE_GRID
    raise _missing(REFERENCE_GRID, "Build it first: python scripts/prepare_reference_grid.py "
                                   "(needs the Parana boundary in PARANA_GPKG).")


def label_raster(year: int) -> Path:
    p = INPUTS / f"polygon_labels_{year}.tif"
    if p.exists():
        return p
    q = DADOS / "polygon_inventory" / f"polygon_labels_{year}.tif"
    if q.exists():
        return q
    raise _missing(p, f"Run: python revision/01_data_preparation/rebuild_class_raster.py --year {year}")


def class_raster(year: int) -> Path:
    p = INPUTS / f"parana_albers_{year}.tif"
    if p.exists() and (INPUTS / f"polygon_labels_{year}.tif").exists():
        return p
    q = DADOS / f"parana_albers_{year}.tif"
    if q.exists():
        return q
    raise _missing(p, f"Run: python revision/01_data_preparation/rebuild_class_raster.py --year {year}")


def geometry_dir(year: int) -> Path:
    return DATA / "geometry" / str(year)


def scene_dir(year: int) -> Path:
    return DATA / "scenes" / str(year)
