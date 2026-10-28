from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Final

import ee
import geopandas as gpd


# Paraná state boundary (GeoPackage); set PARANA_GPKG or place the file at the default path
PARANA_GPKG: Final = Path(os.environ.get("PARANA_GPKG", "data/external/PARANA.gpkg"))
MAPBIOMAS_ASSET: Final = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/"
    "mapbiomas_brazil_collection10_1_coverage_v1"
)
EXPORT_FOLDER: Final = "tst_edge_leakage_parana"
DISTANCE_BINS_M: Final = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
DISTANCE_ABS_MAX_M: Final = 150
SAMPLE_SCALE_M: Final = 30
MAX_SAMPLES_PER_PAIR: Final = 5000
AGRI_CLASSES: Final = [39, 20, 40, 62, 41]

# Heuristic thresholds; require sensitivity/validation across sensors and phenology.
AGRI_THRESHOLDS: Final = {
    "ndvi_bare_max": 0.15,
    "mbi_bare_min": 0.05,
    "ndvi_residue_max": 0.20,
    "ndti_residue_min": 0.05,
    "ndvi_emerg_min": 0.20,
    "ndvi_emerg_max": 0.40,
    "ndvi_dev_min": 0.40,
    "ndvi_mat_min": 0.25,
    "ndvi_mat_max": 0.55,
    "ndti_mat_min": 0.02,
    "mbi_mat_max": 0.55,
    "ndvi_low_split": 0.20,
}


@dataclass
class YearSetting:
    label: str
    collections: list[str]
    sensors: list[str]
    mapbiomas_band: str
    native_thermal_resolution_m: int
    thermal_band: str
    blue: str
    green: str
    red: str
    nir: str
    swir1: str
    swir2: str


YEAR_SETTINGS: Final = {
    1985: YearSetting(
        label="L5_TM_1985",
        collections=["LANDSAT/LT05/C02/T1_L2"],
        sensors=["LT05"],
        mapbiomas_band="classification_1985",
        native_thermal_resolution_m=120,
        thermal_band="ST_B6",
        blue="SR_B1",
        green="SR_B2",
        red="SR_B3",
        nir="SR_B4",
        swir1="SR_B5",
        swir2="SR_B7",
    ),
    2002: YearSetting(
        label="L7_ETM_2002",
        collections=["LANDSAT/LE07/C02/T1_L2"],
        sensors=["LE07"],
        mapbiomas_band="classification_2002",
        native_thermal_resolution_m=60,
        thermal_band="ST_B6",
        blue="SR_B1",
        green="SR_B2",
        red="SR_B3",
        nir="SR_B4",
        swir1="SR_B5",
        swir2="SR_B7",
    ),
    2024: YearSetting(
        label="L8_L9_TIRS_2024",
        collections=["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"],
        sensors=["LC08", "LC09"],
        mapbiomas_band="classification_2024",
        native_thermal_resolution_m=100,
        thermal_band="ST_B10",
        blue="SR_B2",
        green="SR_B3",
        red="SR_B4",
        nir="SR_B5",
        swir1="SR_B6",
        swir2="SR_B7",
    ),
}


def initialize_ee(project: str | None = None) -> None:
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def load_parana_geometry(path: Path = PARANA_GPKG) -> ee.Geometry:
    """Load the Parana AOI as an EE geometry after initialize_ee() has run."""
    if not path.exists():
        raise FileNotFoundError(
            f"Parana boundary not found at {path}. Set PARANA_GPKG to a GeoPackage with the state "
            "boundary or place it at data/external/PARANA.gpkg.")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    geom = gdf.geometry.union_all()
    geojson = json.loads(json.dumps(geom.__geo_interface__))
    return ee.Geometry(geojson)


def add_scaled_landsat_bands(img: ee.Image, cfg: YearSetting) -> ee.Image:
    sr = (
        img.select(
            [cfg.blue, cfg.green, cfg.red, cfg.nir, cfg.swir1, cfg.swir2],
            ["blue", "green", "red", "nir", "swir1", "swir2"],
        )
        .multiply(0.0000275)
        .add(-0.2)
    )
    tst = (
        img.select(cfg.thermal_band)
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("tst_c")
    )
    return img.addBands(sr, None, True).addBands(tst, None, True)


def scale_surface_reflectance(img: ee.Image, cfg: YearSetting) -> ee.Image:
    return add_scaled_landsat_bands(img, cfg)


def mask_landsat_c2_l2(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(0b111111).eq(0)
    return img.updateMask(mask)


# QA_PIXEL bits 1-4: dilated cloud, cirrus, cloud, cloud shadow (fill=bit0 and snow=bit5
# are excluded so the swath edge / no-data does not count as cloud contamination).
CLOUD_SHADOW_BITS = 0b11110


def cloud_shadow_mask(img: ee.Image) -> ee.Image:
    """1 where a pixel is flagged as cloud, dilated cloud, cirrus, or cloud shadow."""
    qa = img.select("QA_PIXEL")
    return qa.bitwiseAnd(CLOUD_SHADOW_BITS).neq(0).rename("cloud_shadow")


def add_spectral_indices(img: ee.Image) -> ee.Image:
    eps = 0.000001
    ndvi = img.expression(
        "(nir - red) / (nir + red + eps)",
        {
            "nir": img.select("nir"),
            "red": img.select("red"),
            "eps": eps,
        },
    ).rename("ndvi")
    ndti = img.expression(
        "(swir1 - swir2) / (swir1 + swir2 + eps)",
        {
            "swir1": img.select("swir1"),
            "swir2": img.select("swir2"),
            "eps": eps,
        },
    ).rename("ndti")
    mbi = img.expression(
        "((swir1 - swir2 - nir) / (swir1 + swir2 + nir + eps)) + 0.5",
        {
            "swir1": img.select("swir1"),
            "swir2": img.select("swir2"),
            "nir": img.select("nir"),
            "eps": eps,
        },
    ).rename("mbi")
    return img.addBands(ee.Image([ndvi, ndti, mbi]), None, True)


def classify_agricultural_state(img: ee.Image, agri_mask: ee.Image) -> ee.Image:
    t = AGRI_THRESHOLDS
    ndvi = img.select("ndvi")
    ndti = img.select("ndti")
    mbi = img.select("mbi")

    bare = ndvi.lte(t["ndvi_bare_max"]).And(mbi.gte(t["mbi_bare_min"]))
    residue = (
        ndvi.lte(t["ndvi_residue_max"])
        .And(ndti.gte(t["ndti_residue_min"]))
        .And(bare.Not())
    )
    maturation = (
        ndvi.gte(t["ndvi_mat_min"])
        .And(ndvi.lte(t["ndvi_mat_max"]))
        .And(ndti.gte(t["ndti_mat_min"]))
        .And(mbi.lte(t["mbi_mat_max"]))
        .And(bare.Not())
        .And(residue.Not())
    )
    emergence = (
        ndvi.gte(t["ndvi_emerg_min"])
        .And(ndvi.lt(t["ndvi_emerg_max"]))
        .And(bare.Not())
        .And(residue.Not())
        .And(maturation.Not())
    )
    development = (
        ndvi.gte(t["ndvi_dev_min"])
        .And(bare.Not())
        .And(residue.Not())
        .And(maturation.Not())
        .And(emergence.Not())
    )

    agri_state = (
        img.select("ndvi")
        .multiply(0)
        .rename("agri_state")
        .where(bare, 1)
        .where(residue, 2)
        .where(maturation, 5)
        .where(emergence, 3)
        .where(development, 4)
        .updateMask(agri_mask)
        .rename("agri_state")
    )
    return img.addBands(agri_state, None, True)
