"""Shared cartographic elements for the map figures (study-area map, spatial coverage).

Projection: the analysis grid CRS, Albers equal-area with the IBGE parameters for Brazil
(lat_0 -12, lon_0 -54, standard parallels -2 and -22, GRS 1980). Boundaries:
  Brazilian states  IBGE, Malha Territorial 2022 (BR_UF_2022, SIRGAS 2000), the zip file from
                    geoftp.ibge.gov.br, at the path in IBGE_UF or data/external/ibge/BR_UF_2022.zip
                    (dados/external/ibge/ is also accepted)
  countries         Natural Earth 1:10m Admin 0 (public domain), through cartopy's downloader
Scale bars are calibrated with geodesic distances on GRS 1980, so their length is true at the
latitude where they are drawn.
"""
from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
from matplotlib.patches import Rectangle
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
IBGE_URL = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
            "malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip")
IBGE_UF = Path(os.environ["IBGE_UF"]) if os.environ.get("IBGE_UF") else next(
    (p for p in (ROOT / "data" / "external" / "ibge" / "BR_UF_2022.zip",
                 ROOT / "dados" / "external" / "ibge" / "BR_UF_2022.zip") if p.exists()),
    ROOT / "data" / "external" / "ibge" / "BR_UF_2022.zip")
ALBERS = ccrs.AlbersEqualArea(central_longitude=-54, central_latitude=-12, false_easting=5e6,
                              false_northing=1e7, standard_parallels=(-2, -22),
                              globe=ccrs.Globe(ellipse="GRS80"))
GEOD = Geod(ellps="GRS80")
INK, INK2 = "#1f1f1e", "#5f5e5a"


def states(geographic=False):
    if not IBGE_UF.exists():
        raise FileNotFoundError(
            f"{IBGE_UF} not found. Download the IBGE 2022 state boundaries from {IBGE_URL} "
            "and place the zip there, or set IBGE_UF to its path.")
    g = gpd.read_file(f"zip://{IBGE_UF.as_posix()}")
    return g if geographic else g.to_crs(ALBERS.proj4_init)


def countries():
    path = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
    return gpd.read_file(path)


def gridlines(ax, xs, ys, left=True, bottom=True, size=6.5):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, xlocs=xs, ylocs=ys,
                      linewidth=0.4, color="#9a9994", alpha=0.7, linestyle=(0, (2, 2)),
                      x_inline=False, y_inline=False, rotate_labels=False)
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left, bottom
    gl.xlabel_style = gl.ylabel_style = {"size": size, "color": INK2}
    return gl


def north_arrow(ax, x, y, size=0.07, fontsize=8):
    """North arrow in axes fraction coordinates (map is north-up at the central meridian)."""
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(facecolor=INK, edgecolor=INK, width=2.2, headwidth=7,
                                headlength=7))
    ax.text(x, y + size + 0.015, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=INK)


def scale_bar(ax, x0, y0, km, segments=4, labels=None, height_frac=0.012, fontsize=6.5):
    """Alternating scale bar starting at axes fraction (x0, y0), geodesically true length,
    on a white backing box."""
    inv = ax.transData.inverted()
    X0, Y0 = inv.transform(ax.transAxes.transform((x0, y0)))
    lon0, lat0 = ccrs.PlateCarree().transform_point(X0, Y0, ALBERS)
    lon1, lat1, _ = GEOD.fwd(lon0, lat0, 90, km * 1e3)
    X1, _ = ALBERS.transform_point(lon1, lat1, ccrs.PlateCarree())
    L = X1 - X0
    yl = ax.get_ylim()
    h = (yl[1] - yl[0]) * height_frac
    ax.add_patch(Rectangle((X0 - 0.06 * L, Y0 - 1.6 * h), 1.24 * L, 5.4 * h, transform=ax.transData,
                           facecolor="white", edgecolor="none", alpha=0.85, zorder=19))
    for i in range(segments):
        ax.add_patch(Rectangle((X0 + i * L / segments, Y0), L / segments, h, transform=ax.transData,
                               facecolor=INK if i % 2 == 0 else "white", edgecolor=INK,
                               linewidth=0.6, zorder=20))
    labels = labels if labels is not None else [km * i / segments for i in range(segments + 1)]
    for v in labels:
        ax.text(X0 + L * v / km, Y0 + 1.6 * h, f"{v:g}", transform=ax.transData, ha="center",
                va="bottom", fontsize=fontsize, color=INK, zorder=20)
    ax.text(X0 + L + 0.03 * L, Y0 + 0.5 * h, "km", transform=ax.transData, ha="left",
            va="center", fontsize=fontsize, color=INK, zorder=20)


def extent_lonlat(ax, lon_min, lon_max, lat_min, lat_max, n=50):
    """Set an Albers extent that encloses the given lon/lat box."""
    lons = np.r_[np.linspace(lon_min, lon_max, n), np.full(n, lon_max),
                 np.linspace(lon_max, lon_min, n), np.full(n, lon_min)]
    lats = np.r_[np.full(n, lat_min), np.linspace(lat_min, lat_max, n), np.full(n, lat_max),
                 np.linspace(lat_max, lat_min, n)]
    xy = ALBERS.transform_points(ccrs.PlateCarree(), lons, lats)
    ax.set_extent([xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()], crs=ALBERS)
