"""Location of the study area (State of Paraná, southern Brazil).

  python fig_study_area.py

Main panel: Paraná and the surrounding states and countries, Albers equal-area (analysis CRS),
graticule in degrees, north arrow and geodesic scale bar. Inset: South America, Brazil and
Paraná, with the extent of the main panel. Boundaries: IBGE Malha Territorial 2022 (states) and
Natural Earth 1:10m (countries); see cartography.py. Output: revision/outputs/figures/
fig02_study_area.{png,pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "revision" / "src"))
import cartography as carto  # noqa: E402
from tedge.paths import OUT  # noqa: E402

OCEAN, FOREIGN, STATE, PR_FILL, PR_EDGE = "#e7f0fa", "#e4e3de", "#f7f6f3", "#9cc3f0", "#184f95"
EXT = (-55.6, -47.2, -27.4, -21.8)  # lon_min, lon_max, lat_min, lat_max of the main panel

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "savefig.dpi": 300})


def main():
    uf = carto.states()
    pr = uf[uf["SIGLA_UF"] == "PR"]
    ufg = carto.states(geographic=True)  # SIRGAS 2000 lon/lat, for the inset
    prg = ufg[ufg["SIGLA_UF"] == "PR"]
    world = carto.countries()
    sa = world[world["CONTINENT"] == "South America"]
    foreign = sa[sa["ADM0_A3"] != "BRA"]

    fig = plt.figure(figsize=(6.6, 4.0))
    ax = fig.add_axes([0.065, 0.07, 0.66, 0.9], projection=carto.ALBERS)
    carto.extent_lonlat(ax, *EXT)
    ax.set_facecolor(OCEAN)
    ax.add_geometries(foreign.geometry, crs=ccrs.PlateCarree(), facecolor=FOREIGN,
                      edgecolor="#8f8e89", linewidth=0.5, zorder=1)
    ax.add_geometries(uf.geometry, crs=carto.ALBERS, facecolor=STATE, edgecolor="#8f8e89",
                      linewidth=0.5, zorder=2)
    ax.add_geometries(pr.geometry, crs=carto.ALBERS, facecolor=PR_FILL, edgecolor=PR_EDGE,
                      linewidth=1.1, zorder=3)
    lab = dict(transform=ccrs.PlateCarree(), ha="center", va="center", color=carto.INK2,
               fontsize=7, zorder=5)
    ax.text(-51.4, -24.6, "PARANÁ\n(study area)", fontsize=9, fontweight="bold", color=PR_EDGE,
            **{k: v for k, v in lab.items() if k not in ("fontsize", "color")})
    for lon, lat, name in [(-49.3, -22.4, "São Paulo"), (-54.6, -22.3, "Mato Grosso\ndo Sul"),
                           (-50.6, -27.0, "Santa Catarina"), (-54.95, -24.15, "PARAGUAY"),
                           (-54.85, -26.75, "ARGENTINA")]:
        ax.text(lon, lat, name, **lab)
    ax.text(-47.75, -25.9, "Atlantic\nOcean", style="italic", **{k: v for k, v in lab.items()
                                                                  if k != "color"},
            color="#4f7fb3")
    carto.gridlines(ax, np.arange(-56, -46, 1), np.arange(-28, -20, 1), size=7)
    carto.north_arrow(ax, 0.94, 0.82)
    carto.scale_bar(ax, 0.06, 0.05, 200, segments=4, labels=[0, 100, 200], fontsize=7)
    for s in ax.spines.values():
        s.set_edgecolor(carto.INK2)
        s.set_linewidth(0.8)

    # inset: South America -> Brazil -> Paraná
    proj_i = ccrs.LambertAzimuthalEqualArea(central_longitude=-60, central_latitude=-20)
    axi = fig.add_axes([0.745, 0.30, 0.25, 0.62], projection=proj_i)
    axi.set_extent([-82, -34, -56, 13], crs=ccrs.PlateCarree())
    axi.set_facecolor(OCEAN)
    axi.add_geometries(foreign.geometry, crs=ccrs.PlateCarree(), facecolor=FOREIGN,
                       edgecolor="#8f8e89", linewidth=0.25)
    axi.add_geometries(ufg.geometry, crs=ccrs.PlateCarree(), facecolor="#fbfaf8",
                       edgecolor="#b5b4ae", linewidth=0.2)
    axi.add_geometries([ufg.geometry.union_all()], crs=ccrs.PlateCarree(), facecolor="none",
                       edgecolor=carto.INK, linewidth=0.6)
    axi.add_geometries(prg.geometry, crs=ccrs.PlateCarree(), facecolor=PR_EDGE, edgecolor=PR_EDGE,
                       linewidth=0.3)
    axi.add_geometries([box(EXT[0], EXT[2], EXT[1], EXT[3])], crs=ccrs.PlateCarree(),
                       facecolor="none", edgecolor="#a32c2b", linewidth=0.8)
    axi.text(-52, -9, "BRAZIL", transform=ccrs.PlateCarree(), ha="center", fontsize=7,
             fontweight="bold", color=carto.INK)
    axi.text(-64, -46, "South\nAmerica", transform=ccrs.PlateCarree(), ha="center",
             fontsize=7, style="italic", color=carto.INK2)
    for s in axi.spines.values():
        s.set_edgecolor(carto.INK)
        s.set_linewidth(0.8)
    fig.text(0.748, 0.265, "Main map: Albers equal-area\n(IBGE parameters, GRS 1980);\n"
             "graticule in degrees.\nBoundaries: IBGE territorial\ngrid 2022; Natural Earth\n1:10m.",
             ha="left", va="top", fontsize=7, color=carto.INK2, linespacing=1.3)

    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig02_study_area.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("fig02_study_area: ok")


if __name__ == "__main__":
    main()
