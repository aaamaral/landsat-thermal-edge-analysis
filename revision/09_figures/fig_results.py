"""Result figures of the revised manuscript, drawn from revision/outputs/full/*.csv.

  python fig_results.py [--src full]

fig03_coverage      usable polygons per 10 km cell (W = 150 m), one map panel per year_sensor
fig04_profiles      TST contrast vs distance to the mapped edge (W = 300 m, 95 % CI), per class
fig05_multiscale    DET vs analysis window (90-300 m) and vs edge exclusion (0-90 m)
fig06_dtec          DTEC(A <- B) heatmap (focal x neighbour) per year_sensor, diverging, cells
                    without enough interfaces left blank and marked
fig07_asymmetry     DTEC(A <- B) - DTEC(B <- A), upper triangle, CI excluding 0 marked
Year_sensor conditions are separate panels/series, never joined as a trend.
Colours: validated categorical slots 1-3 (dataviz reference palette), each also with its own
marker and line style (the third slot is < 3:1 on the surface -> secondary encoding).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge.paths import OUT  # noqa: E402

NAMES = {3: "Forest formation", 9: "Silviculture", 11: "Wetland", 12: "Grassland",
         15: "Pasture", 20: "Sugar cane", 21: "Mosaic of uses", 24: "Urban area",
         33: "Water", 39: "Soybean", 41: "Other temp. crops", 46: "Coffee"}
SHORT = {3: "FOR", 9: "SIL", 11: "WET", 12: "GRA", 15: "PAS", 20: "SUG", 21: "MOS", 24: "URB",
         33: "WAT", 39: "SOY", 41: "OTC", 46: "COF"}
MAIN_CLASSES = [3, 9, 15, 21, 24, 39, 41, 33]
YS = {1985: ("1985 · L5 TM", "#2a78d6", "o", "-"),
      2002: ("2002 · L7 ETM+", "#eb6834", "s", "--"),
      2024: ("2024 · L8/9 TIRS", "#1baf7a", "^", "-.")}
INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e4e3de", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f29a94", "#a32c2b"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    "legend.frameon": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
})


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def year_legend(fig, ncol=3, y=1.02):
    h = [plt.Line2D([], [], color=c, marker=m, ls=ls, lw=1.6, ms=4.5, label=l)
         for l, c, m, ls in YS.values()]
    fig.legend(handles=h, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y))


def fig_coverage(src, out):
    """Usable polygons per 10 km cell, drawn as the 10 km cells themselves on the analysis CRS,
    with the Paraná boundary, a shared graticule, one north arrow and one scale bar (the three
    panels have the same extent, projection and orientation)."""
    import cartopy.crs as ccrs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cartography as carto

    g = pd.read_csv(src.parent / "audit" / "coverage_grid.csv")
    pr = carto.states().query("SIGLA_UF == 'PR'")
    vmax = np.nanpercentile(g["polygons_usable"], 98)
    gx0, gy0 = g["gx"].min(), g["gy"].min()
    nx, ny = g["gx"].max() - gx0 + 1, g["gy"].max() - gy0 + 1
    cell = 10_000.0  # cell (gx, gy) spans [gx, gx + 1) x [gy, gy + 1) x 10 km
    assert np.allclose(g["x_center"], g["gx"] * cell + cell / 2)
    assert np.allclose(g["y_center"], g["gy"] * cell + cell / 2)
    ext = [gx0 * cell, (gx0 + nx) * cell, gy0 * cell, (gy0 + ny) * cell]
    fig = plt.figure(figsize=(7.2, 2.9))
    axs = [fig.add_axes([0.035 + i * 0.295, 0.13, 0.285, 0.72], projection=carto.ALBERS)
           for i in range(3)]
    for i, (ax, (y, (lab, *_))) in enumerate(zip(axs, YS.items())):
        d = g[g["year"] == y]
        arr = np.full((ny, nx), np.nan)
        arr[(d["gy"] - gy0).to_numpy(), (d["gx"] - gx0).to_numpy()] = d["polygons_usable"].to_numpy()
        ax.set_extent([4.93e6, 5.62e6, 8.28e6, 8.83e6], crs=carto.ALBERS)
        im = ax.imshow(arr, origin="lower", extent=ext, transform=carto.ALBERS, cmap=SEQ,
                       vmin=0, vmax=vmax, interpolation="nearest", zorder=2)
        ax.add_geometries(pr.geometry, crs=carto.ALBERS, facecolor="none", edgecolor=INK,
                          linewidth=0.6, zorder=3)
        ax.set_title(f"{lab}\n(n = {int(d['polygons_usable'].sum()):,} polygons)", loc="left",
                     fontsize=8.5)
        carto.gridlines(ax, np.arange(-53, -48, 2), np.arange(-26, -21, 1), left=(i == 0),
                        size=8)
        for s in ax.spines.values():
            s.set_edgecolor(INK2)
            s.set_linewidth(0.6)
    carto.north_arrow(axs[2], 0.9, 0.70, size=0.13, fontsize=7)
    carto.scale_bar(axs[0], 0.06, 0.05, 200, segments=4, labels=[0, 100, 200], fontsize=8)
    cax = fig.add_axes([0.925, 0.22, 0.012, 0.54])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Usable polygons per\n10 km cell (W = 150 m)", color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    save(fig, "fig03_coverage", out)


def fig_profiles(src, out, window=300):
    p = pd.read_csv(src / "profiles.csv")
    p = p[p["window"] == window]
    dep = pd.read_csv(src / "dep.csv")
    dep = dep[dep["window"] == window]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True)
    for ax, cls in zip(axs.flat, MAIN_CLASSES):
        ax.axhline(0, color=INK2, lw=0.8)
        for y, (lab, col, mk, ls) in YS.items():
            d = p[(p["focal_class"] == cls) & (p["year"] == y)].sort_values("distance_m")
            if d.empty:
                continue
            ax.fill_between(d["distance_m"], d["lo"], d["hi"], color=col, alpha=0.15, lw=0)
            ax.plot(d["distance_m"], d["contrast"], color=col, ls=ls, lw=1.4, marker=mk, ms=3,
                    markevery=2)
            r = dep[(dep["focal_class"] == cls) & (dep["year"] == y)]
            if len(r) and np.isfinite(r["dep_m"].iloc[0]):
                ax.axvline(r["dep_m"].iloc[0], color=col, lw=0.7, ls=":")
        ax.set_title(NAMES[cls], loc="left")
    fig.supxlabel("Distance to mapped edge (m)", fontsize=8, color=INK)
    fig.supylabel("TST − interior reference (°C)", fontsize=8, color=INK)
    year_legend(fig)
    fig.text(0.01, -0.03, f"Ribbons: 95 % CI (crossed polygon × scene bootstrap). Dotted: breakpoint "
             f"DEP. Reference band ≥ {window} m.", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig04_profiles", out)


def fig_multiscale(src, out):
    s = pd.read_csv(src / "det_sweep.csv")
    base = (s["variant"] == "t30") & (s["vf"] == 0)
    win = s[base & (s["excl"] == 0)]
    exc = s[base & (s["window"] == 150)]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for j, cls in enumerate([3, 15, 24, 39]):
        for row, (d, x, xl) in enumerate(((win, "window", "Analysis window W (m)"),
                                          (exc, "excl", "Excluded edge width (m)"))):
            ax = axs[row, j]
            ax.axhline(0, color=INK2, lw=0.8)
            for y, (lab, col, mk, ls) in YS.items():
                q = d[(d["focal_class"] == cls) & (d["year"] == y)].sort_values(x)
                xv = q[x] * (30 if x == "excl" else 1)
                ax.fill_between(xv, q["det_lo"], q["det_hi"], color=col, alpha=0.15, lw=0)
                ax.plot(xv, q["det"], color=col, ls=ls, marker=mk, ms=3.5, lw=1.4)
            ax.set_xlabel(xl)
            if row == 0:
                ax.set_title(NAMES[cls], loc="left")
            if j == 0:
                ax.set_ylabel("DET (°C)")
    year_legend(fig)
    fig.tight_layout()
    save(fig, "fig05_multiscale", out)


def _matrix(df, value, classes):
    m = df.pivot_table(index="focal_class", columns="nb_class", values=value)
    return m.reindex(index=classes, columns=classes)


def fig_dtec(src, out):
    d = pd.read_csv(src / "dtec_pairs.csv")
    classes = [c for c in NAMES if c in set(d["focal_class"]) | set(d["nb_class"])]
    lim = np.nanpercentile(np.abs(d["dtec_unweighted"]), 97)
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (y, (lab, *_)) in zip(axs, YS.items()):
        q = d[d["year"] == y]
        m = _matrix(q, "dtec_unweighted", classes)
        im = ax.imshow(m.to_numpy(float), cmap=DIV, norm=norm)
        lo, hi = _matrix(q, "dtec_unweighted_lo", classes), _matrix(q, "dtec_unweighted_hi", classes)
        for i in range(len(classes)):
            for k in range(len(classes)):
                v = m.iat[i, k]
                if i == k:
                    continue
                if not np.isfinite(v):
                    ax.text(k, i, "·", ha="center", va="center", color=INK2, fontsize=6)
                elif not (lo.iat[i, k] > 0 or hi.iat[i, k] < 0):
                    ax.text(k, i, "×", ha="center", va="center", color=INK2, fontsize=5.5)
        ax.set_xticks(range(len(classes)), [SHORT[c] for c in classes], rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)), [SHORT[c] for c in classes], fontsize=6.5)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("Neighbour class B")
        ax.grid(False)
    axs[0].set_ylabel("Focal class A")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("DTEC(A ← B) (°C)", color=INK)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.06, "×: 95 % CI includes 0. '·': fewer than 10 interfaces (not estimated). "
             "Diagonal: no interface. Values and CIs: Table S-DTEC.", color=INK2, fontsize=7)
    save(fig, "fig06_dtec", out)


def fig_asymmetry(src, out):
    a = pd.read_csv(src / "dtec_asymmetry.csv")
    classes = [c for c in NAMES if c in set(a["class_a"]) | set(a["class_b"])]
    lim = np.nanpercentile(np.abs(a["asymmetry"]), 97)
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (y, (lab, *_)) in zip(axs, YS.items()):
        q = a[a["year"] == y]
        m = np.full((len(classes), len(classes)), np.nan)
        sig = np.zeros_like(m, bool)
        for _, r in q.iterrows():
            i, k = classes.index(r["class_a"]), classes.index(r["class_b"])
            m[i, k] = r["asymmetry"]
            sig[i, k] = r["asym_lo"] > 0 or r["asym_hi"] < 0
        im = ax.imshow(m, cmap=DIV, norm=norm)
        for i, k in zip(*np.nonzero(np.isfinite(m) & ~sig)):
            ax.text(k, i, "×", ha="center", va="center", color=INK2, fontsize=5.5)
        ax.set_xticks(range(len(classes)), [SHORT[c] for c in classes], rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)), [SHORT[c] for c in classes], fontsize=6.5)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("Class B")
        ax.grid(False)
    axs[0].set_ylabel("Class A")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("DTEC(A ← B) − DTEC(B ← A) (°C)", color=INK)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.06, "Upper triangle only (the lower one is the same value with opposite sign). "
             "×: 95 % CI includes 0. Read with both directional DTEC values.", color=INK2, fontsize=7)
    save(fig, "fig07_asymmetry", out)


def fig_asymmetry_interior(src, out):
    """Fig. S2: Asym(A,B) against the interior difference T_int,A - T_int,B (results_audit.py)."""
    p = pd.read_csv(src.parent / "audit" / "asymmetry_pairs.csv")
    s = pd.read_csv(src.parent / "audit" / "asymmetry_vs_interior.csv").set_index("year")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, sharey=True)
    for ax, (y, (lab, col, mk, _)) in zip(axs, YS.items()):
        q = p[p["year"] == y]
        lim = np.nanmax(np.abs(p[["interior_diff", "asymmetry"]].to_numpy())) * 1.05
        ax.plot([-lim, lim], [lim, -lim], color=INK2, lw=0.8, ls=":")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.scatter(q["interior_diff"], q["asymmetry"], s=14, color=col, marker=mk,
                   edgecolors="white", linewidths=0.5)
        r = s.loc[y]
        ax.text(0.04, 0.06, f"r = {r.pearson_r:.2f} [{r.r_lo:.2f}, {r.r_hi:.2f}]\n"
                f"slope = {r.ols_slope:.2f}, n = {int(r.n_pairs)}", transform=ax.transAxes,
                fontsize=6.5, color=INK)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("T_int,A − T_int,B (°C)")
    axs[0].set_ylabel("DTEC(A ← B) − DTEC(B ← A) (°C)")
    fig.text(0.01, -0.07, "Dotted line: slope −1 (asymmetry equal to minus the interior difference).",
             color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "figS2_asymmetry_vs_interior", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="full")
    a = ap.parse_args()
    src = OUT / a.src
    out = OUT / "figures" if a.src == "full" else OUT / "figures" / a.src
    for f in (fig_coverage, fig_profiles, fig_multiscale, fig_dtec, fig_asymmetry,
              fig_asymmetry_interior):
        f(src, out)
        print(f"{f.__name__}: ok", flush=True)


if __name__ == "__main__":
    main()
