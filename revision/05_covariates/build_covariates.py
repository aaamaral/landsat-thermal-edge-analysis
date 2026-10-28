"""Static topographic and per-scene meteorological covariates.

dem      SRTM GL1 v3 (USGS/SRTMGL1_003, 1 arc-second, acquired Feb 2000) resampled bilinearly
         to the Albers 30 m label grid -> revision/data/covariates/srtm_albers.tif (int16, m).
demzones Per (label, nb_class, ring) zone of a year: mean elevation and the three terms whose
         zone means give the exact zone-mean illumination for any solar position:
             m1 = cos(slope); m2 = sin(slope)*cos(aspect); m3 = sin(slope)*sin(aspect)
             cos(i) = cos(z)*m1 + sin(z)*(cos(phi)*m2 + sin(phi)*m3)
         with z = solar zenith, phi = solar azimuth, aspect clockwise from north (Horn 1981).
era5     ERA5-Land hourly (ECMWF/ERA5_LAND/HOURLY, 0.1 deg) at the hour closest to each scene
         centre time: air temperature 2 m, dewpoint 2 m (-> VPD), downward solar radiation
         (hourly), 10 m wind speed, volumetric soil water layer 1, and precipitation summed over
         the 120 h before acquisition. Stored per scene on the native 0.1 deg grid; values are
         attached to polygons by their centroid cell in the analysis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import fetch  # noqa: E402
from tedge.aggregate import zone_keys, split_keys  # noqa: E402
from tedge.paths import DATA, YEARS, geometry_dir, label_raster  # noqa: E402

COV = DATA / "covariates"
TILE = 2048
NODATA = -32768
ERA5_BBOX = (-54.8, -26.9, -47.9, -22.4)  # lon_min, lat_min, lon_max, lat_max (Parana + margin)
ERA5_RES = 0.1


def fetch_dem():
    COV.mkdir(parents=True, exist_ok=True)
    fetch.init()
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").resample("bilinear").toInt16()
    with rasterio.open(label_raster(2024)) as lab:
        prof = lab.profile.copy()
        H, W, T, wkt = lab.height, lab.width, lab.transform, lab.crs.to_wkt()
        prof.update(dtype="int16", nodata=NODATA, compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(COV / "srtm_albers.tif", "w", **prof) as dst:
            for r0 in range(0, H, TILE):
                for c0 in range(0, W, TILE):
                    h, w = min(TILE, H - r0), min(TILE, W - c0)
                    if not (lab.read(1, window=Window(c0, r0, w, h)) > 0).any():
                        dst.write(np.full((h, w), NODATA, np.int16), 1, window=Window(c0, r0, w, h))
                        continue
                    arr = fetch.fetch_tile(dem.unmask(NODATA), wkt, T, r0, c0, h, w)
                    dst.write(arr["elevation"].astype(np.int16), 1, window=Window(c0, r0, w, h))
                print(f"  dem rows {r0}/{H}", flush=True)


def _slope_aspect(z, px=30.0):
    """Horn (1981) slope and aspect (radians, aspect clockwise from north) on a haloed tile."""
    z = z.astype(np.float64)
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * px)   # towards east
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * px)   # towards south (rows grow south)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    # downslope direction: -grad; north component = +dzdy (since dzdy points south)
    aspect = (np.arctan2(-dzdx, dzdy)) % (2 * np.pi)
    return slope, aspect


def dem_zone_stats(year: int):
    out = []
    with rasterio.open(COV / "srtm_albers.tif") as dsrc, \
            rasterio.open(label_raster(year)) as lsrc, \
            rasterio.open(geometry_dir(year) / "ring.tif") as rsrc, \
            rasterio.open(geometry_dir(year) / "nbclass.tif") as nsrc:
        H, W = lsrc.shape
        for r0 in range(0, H, TILE):
            for c0 in range(0, W, TILE):
                h, w = min(TILE, H - r0), min(TILE, W - c0)
                win = Window(c0, r0, w, h)
                lab = lsrc.read(1, window=win).astype(np.int64)
                ring = rsrc.read(1, window=win)
                ok = (lab > 0) & (ring < 255)
                if not ok.any():
                    continue
                nb = nsrc.read(1, window=win)
                z = dsrc.read(1, window=Window(c0 - 1, r0 - 1, w + 2, h + 2), boundless=True,
                              fill_value=NODATA).astype(np.float64)
                z[z == NODATA] = np.nan
                slope, aspect = _slope_aspect(z)
                elev = z[1:-1, 1:-1]
                good = ok & np.isfinite(slope) & np.isfinite(elev)
                k = zone_keys(lab[good], nb[good], ring[good])
                uk, inv = np.unique(k, return_inverse=True)
                s, a = slope[good], aspect[good]
                out.append(pd.DataFrame({
                    "key": uk, "n": np.bincount(inv),
                    "elev": np.bincount(inv, elev[good]),
                    "m1": np.bincount(inv, np.cos(s)),
                    "m2": np.bincount(inv, np.sin(s) * np.cos(a)),
                    "m3": np.bincount(inv, np.sin(s) * np.sin(a)),
                    "slope_deg": np.bincount(inv, np.degrees(s)),
                }))
            print(f"  {year} dem zones rows {r0}/{H}", flush=True)
    df = pd.concat(out).groupby("key", as_index=False).sum()
    lab, nb, ring = split_keys(df["key"].to_numpy())
    res = pd.DataFrame({"label": lab.astype(np.int32), "nb_class": nb.astype(np.uint8),
                        "ring": ring.astype(np.uint8), "n_dem": df["n"].astype(np.uint32)})
    for c in ("elev", "m1", "m2", "m3", "slope_deg"):
        res[c] = (df[c] / df["n"]).astype(np.float32)
    res.to_parquet(COV / f"dem_zones_{year}.parquet", index=False)
    return len(res)


def _scene_time(meta):
    t = meta["SCENE_CENTER_TIME"].split(".")[0]
    return dt.datetime.fromisoformat(f"{meta['DATE_ACQUIRED']}T{t}")


def era5_for_scene(meta) -> dict:
    t = _scene_time(meta)
    hour = (t + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    h = col.filterDate(hour.isoformat(), (hour + dt.timedelta(hours=1)).isoformat()).first()
    prec = col.filterDate((hour - dt.timedelta(hours=120)).isoformat(), hour.isoformat()) \
        .select("total_precipitation_hourly").sum().multiply(1000).rename("prec_120h_mm")
    t2 = h.select("temperature_2m").subtract(273.15).rename("t2m_c")
    td = h.select("dewpoint_temperature_2m").subtract(273.15)
    es = t2.expression("0.6108*exp(17.27*t/(t+237.3))", {"t": t2})
    ea = td.expression("0.6108*exp(17.27*t/(t+237.3))", {"t": td})
    vpd = es.subtract(ea).rename("vpd_kpa")
    rad = h.select("surface_solar_radiation_downwards_hourly").divide(3600).rename("ssrd_wm2")
    wind = h.select("u_component_of_wind_10m").hypot(h.select("v_component_of_wind_10m")).rename("wind10_ms")
    swc = h.select("volumetric_soil_water_layer_1").rename("swvl1")
    img = ee.Image.cat([t2, vpd, rad, wind, swc, prec]).toFloat()
    x0, y0, x1, y1 = ERA5_BBOX
    nx, ny = int(round((x1 - x0) / ERA5_RES)), int(round((y1 - y0) / ERA5_RES))
    grid = {"dimensions": {"width": nx, "height": ny},
            "affineTransform": {"scaleX": ERA5_RES, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -ERA5_RES, "translateY": y1},
            "crsCode": "EPSG:4326"}
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    return {"hour_utc": hour.isoformat(), **{b: np.asarray(arr[b]) for b in arr.dtype.names}}


def build_era5():
    fetch.init()
    COV.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in YEARS:
        mpath = DATA / f"scene_metadata_{year}.jsonl"
        metas = {}
        for line in open(mpath, encoding="utf-8"):
            m = json.loads(line)
            metas[m["scene_id"]] = m
        for sid, m in metas.items():
            res = era5_for_scene(m)
            ny, nx = res["t2m_c"].shape
            iy, ix = np.indices((ny, nx))
            df = pd.DataFrame({"scene_id": sid, "year": year, "hour_utc": res["hour_utc"],
                               "iy": iy.ravel(), "ix": ix.ravel()})
            for b in ("t2m_c", "vpd_kpa", "ssrd_wm2", "wind10_ms", "swvl1", "prec_120h_mm"):
                df[b] = res[b].ravel().astype(np.float32)
            rows.append(df)
            print(f"  era5 {sid} {res['hour_utc']}", flush=True)
    era = pd.concat(rows, ignore_index=True)
    era.to_parquet(COV / "era5land_scene_grid.parquet", index=False)
    (COV / "era5land_grid.json").write_text(json.dumps(
        {"bbox": ERA5_BBOX, "res_deg": ERA5_RES, "origin": "upper-left", "crs": "EPSG:4326"}))
    return len(era)


if __name__ == "__main__":
    import os
    os.environ.setdefault("GDAL_CACHEMAX", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dem", "demzones", "era5"])
    ap.add_argument("--year", type=int, nargs="*", default=list(YEARS))
    a = ap.parse_args()
    if a.step == "dem":
        fetch_dem()
    elif a.step == "demzones":
        for y in a.year:
            print(y, dem_zone_stats(y))
    else:
        print(build_era5())
