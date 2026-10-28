"""Build the 30 m reference grid of Parana used as the template of the analysis.

  python scripts/prepare_reference_grid.py                       # study grid (default)
  python scripts/prepare_reference_grid.py --extent boundary     # grid fitted to the boundary
  python scripts/prepare_reference_grid.py --boundary my.gpkg --out grid.tif

The original study used a 30 m class raster of Parana as template. That raster is not
distributed, so this script rebuilds a grid with the same definition:

  CRS         Albers Equal Area, IBGE parameters for Brazil (lat_0 -12, lon_0 -54, standard
              parallels -2 and -22, false easting 5,000,000 m, false northing 10,000,000 m,
              GRS 1980 ellipsoid, null datum shift; STUDY_CRS_WKT below)
  resolution  30 m x 30 m, north-up
  alignment   pixel corners on x = STUDY_X0 + 30 i and y = STUDY_Y0 - 30 j (upper-left corner
              of the study grid); the same lattice is kept with --extent boundary
  extent      --extent study (default): STUDY_WIDTH x STUDY_HEIGHT pixels from (STUDY_X0, STUDY_Y0)
              --extent boundary: bounding box of the boundary snapped outward to the lattice,
              plus --pad pixels on each side
  clip        pixels whose centre falls inside the Parana boundary (value 1); 0 elsewhere

The output is an int16 GeoTIFF (1 inside, 0 outside) that rebuild_class_raster.py fills with
MapBiomas Collection 10.1 classes. The grid lattice, CRS and extent are those of the study. The
clip depends on the boundary file supplied, so the set of in-state pixels along the border may
differ slightly from the original raster; the grid is spatially and analytically equivalent, not
a byte-for-byte copy.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window, transform as window_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge.paths import REFERENCE_GRID  # noqa: E402

STUDY_CRS_WKT = (
    'PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown based on GRS 1980 ellipsoid using '
    'towgs84=0,0,0,0,0,0,0",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],'
    'TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]]],PROJECTION["Albers_Conic_Equal_Area"],'
    'PARAMETER["latitude_of_center",-12],PARAMETER["longitude_of_center",-54],'
    'PARAMETER["standard_parallel_1",-2],PARAMETER["standard_parallel_2",-22],'
    'PARAMETER["false_easting",5000000],PARAMETER["false_northing",10000000],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)
PIXEL_M = 30.0
STUDY_X0 = 4936261.972889816  # upper-left corner of the study grid (m)
STUDY_Y0 = 8824729.933096489
STUDY_WIDTH = 22656  # columns
STUDY_HEIGHT = 15576  # rows
ROWS = 2048  # row chunk for rasterisation


def boundary_path(arg: str | None) -> Path:
    path = Path(arg or os.environ.get("PARANA_GPKG", "data/external/PARANA.gpkg"))
    if not path.exists():
        sys.exit(f"Parana boundary not found at {path}. Set PARANA_GPKG to a GeoPackage (or any "
                 "vector file readable by GeoPandas) with the state boundary, or pass --boundary.")
    return path


def grid_from_boundary(bounds, pad: int):
    """Bounding box snapped outward to the study lattice -> (x0, y0, width, height)."""
    left, bottom, right, top = bounds
    i0 = math.floor((left - STUDY_X0) / PIXEL_M) - pad
    i1 = math.ceil((right - STUDY_X0) / PIXEL_M) + pad
    j0 = math.floor((STUDY_Y0 - top) / PIXEL_M) - pad
    j1 = math.ceil((STUDY_Y0 - bottom) / PIXEL_M) + pad
    return STUDY_X0 + i0 * PIXEL_M, STUDY_Y0 - j0 * PIXEL_M, i1 - i0, j1 - j0


def build(boundary: Path, out: Path, extent: str = "study", pad: int = 0) -> dict:
    crs = CRS.from_wkt(STUDY_CRS_WKT)
    gdf = gpd.read_file(boundary)
    if gdf.empty:
        sys.exit(f"No geometries found in {boundary}")
    if gdf.crs is None:
        sys.exit(f"{boundary} has no CRS; define it before building the grid")
    geom = gdf.to_crs(crs).geometry.union_all()
    if extent == "study":
        x0, y0, width, height = STUDY_X0, STUDY_Y0, STUDY_WIDTH, STUDY_HEIGHT
    else:
        x0, y0, width, height = grid_from_boundary(geom.bounds, pad)
    T = Affine(PIXEL_M, 0.0, x0, 0.0, -PIXEL_M, y0)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", dtype="int16", count=1, width=width, height=height, crs=crs,
                transform=T, nodata=None, tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", BIGTIFF="IF_SAFER")
    inside = 0
    with rasterio.open(out, "w", **prof) as dst:
        for r in range(0, height, ROWS):
            win = Window(0, r, width, min(ROWS, height - r))
            m = rasterize([geom], out_shape=(win.height, width),
                          transform=window_transform(win, T), fill=0, default_value=1,
                          dtype="uint8", all_touched=False)
            inside += int(m.sum())
            dst.write(m.astype(np.int16), 1, window=win)
    return {"path": str(out), "width": width, "height": height, "x0": x0, "y0": y0,
            "pixels_inside": inside}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boundary", help="state boundary file (default: $PARANA_GPKG)")
    ap.add_argument("--out", type=Path, default=REFERENCE_GRID)
    ap.add_argument("--extent", choices=["study", "boundary"], default="study")
    ap.add_argument("--pad", type=int, default=0, help="extra pixels per side (--extent boundary)")
    a = ap.parse_args()
    info = build(boundary_path(a.boundary), a.out, a.extent, a.pad)
    print(f"reference grid written: {info['path']}\n"
          f"  {info['width']} x {info['height']} pixels of 30 m, upper-left "
          f"({info['x0']:.6f}, {info['y0']:.6f})\n"
          f"  pixels inside the boundary: {info['pixels_inside']}")
