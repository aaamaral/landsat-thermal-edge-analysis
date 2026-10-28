"""Earth Engine side: build the per-scene TST/NDVI/NDMI image and fetch grid-aligned tiles.

The thermal chain is the validated pipeline of the submitted version, imported unchanged
from scripts/gee_python/edge_config.py: Landsat Collection 2 Level 2 ST band,
ST * 0.00341802 + 149.0 - 273.15, QA_PIXEL bits 0-5 (fill, dilated cloud, cirrus, cloud,
shadow, snow) masked pixel by pixel. What changes is the output: pixels are fetched on the
Albers 30 m label grid with ee.data.computePixels and aggregated locally, instead of being
reduced on Earth Engine into four fixed distance bands.
"""
from __future__ import annotations

import os
import sys
import time

import ee
import numpy as np

from .paths import ROOT
from .aggregate import OUTSIDE, INVALID, TST_SCALE, IDX_SCALE

sys.path.insert(0, str(ROOT / "scripts" / "gee_python"))
from edge_config import YEAR_SETTINGS, add_scaled_landsat_bands, mask_landsat_c2_l2  # noqa: E402

BANDS = ("tst", "ndvi", "ndmi")


def init():
    """Initialise Earth Engine with the Google Cloud project named in EE_PROJECT."""
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError(
            "EE_PROJECT is not set. Set it to a Google Cloud project registered for Earth Engine "
            "(e.g. export EE_PROJECT=my-project) after running 'earthengine authenticate'.")
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(f"Earth Engine initialisation failed for project '{project}': {exc}. "
                           "Run 'earthengine authenticate' and check that the project is "
                           "registered for Earth Engine.") from exc


def scene_image(image_id: str, year: int) -> ee.Image:
    """int16 image: tst (centi-degC), ndvi, ndmi (x1e4) with OUTSIDE / INVALID sentinels."""
    cfg = YEAR_SETTINGS[year]
    raw = ee.Image(image_id)
    # footprint = not QA fill; a footprint pixel without a usable ST value counts as INVALID
    footprint = raw.select("QA_PIXEL").bitwiseAnd(1).eq(0).selfMask()
    scaled = add_scaled_landsat_bands(mask_landsat_c2_l2(raw), cfg)
    tst = scaled.select("tst_c").multiply(TST_SCALE)
    nir, red, swir1 = scaled.select("nir"), scaled.select("red"), scaled.select("swir1")
    ndvi = nir.subtract(red).divide(nir.add(red)).multiply(IDX_SCALE)
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).multiply(IDX_SCALE)
    out = []
    for name, img in (("tst", tst), ("ndvi", ndvi), ("ndmi", ndmi)):
        b = (img.round().clamp(-32000, 32000).toInt16()
             .unmask(INVALID).updateMask(footprint).unmask(OUTSIDE).toInt16().rename(name))
        out.append(b)
    return ee.Image.cat(out)


def fetch_tile(image: ee.Image, crs_wkt: str, transform, row0: int, col0: int,
               height: int, width: int, retries: int = 6) -> dict:
    """Fetch an int16 block on the label grid (row0/col0 may be negative: halo)."""
    grid = {
        "dimensions": {"width": int(width), "height": int(height)},
        "affineTransform": {
            "scaleX": transform.a, "shearX": 0, "translateX": transform.c + col0 * transform.a,
            "shearY": 0, "scaleY": transform.e, "translateY": transform.f + row0 * transform.e,
        },
        "crsWkt": crs_wkt,
    }
    delay = 4.0
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels(
                {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            return {b: np.asarray(arr[b]) for b in arr.dtype.names}
        except Exception as exc:  # quota / transient errors: back off and retry
            if attempt == retries - 1:
                raise
            msg = str(exc)
            time.sleep(delay * (2 if "429" in msg or "quota" in msg.lower() else 1))
            delay *= 2
    raise RuntimeError("unreachable")


def scene_metadata(image_ids: list[str]) -> list[dict]:
    """Acquisition properties needed downstream (solar geometry, time, cloud cover)."""
    keys = ["SUN_AZIMUTH", "SUN_ELEVATION", "CLOUD_COVER", "DATE_ACQUIRED",
            "SCENE_CENTER_TIME", "SPACECRAFT_ID", "WRS_PATH", "WRS_ROW"]
    fc = ee.FeatureCollection([
        ee.Feature(ee.Image(i).geometry(), ee.Image(i).toDictionary(keys)).set("image_id", i)
        for i in image_ids])
    info = fc.getInfo()["features"]
    out = []
    for f in info:
        p = dict(f["properties"])
        p["footprint"] = f["geometry"]
        out.append(p)
    return out
