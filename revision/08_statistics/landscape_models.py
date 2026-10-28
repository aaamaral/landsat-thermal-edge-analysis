"""Landscape metrics remodelled at the polygon level (replaces the submitted correlation screening).

  python landscape_models.py --year 1985 2002 2024

Response: scene-adjusted polygon DET (W = 150 m): DET minus the class x scene mean plus the class
mean, averaged over the scenes of each polygon (removes scene composition differences).
Predictors: landscape metrics in the 150 m window of the submitted analysis
(dados/tst_edge_leakage/landscape/landscape_{year}.csv), log10 polygon area, polygon-level
topography (mean d_cosi over scenes, edge slope, d_elev).
Steps per class x year_sensor:
  1. VIF screening: drop the landscape metric with the largest VIF until all VIF <= 5
  2. linear model, standardised predictors, SE clustered on 50 km spatial blocks
  3. additive model with natural cubic regression splines (df = 4) per landscape metric
     (statsmodels/patsy `cr`; R^2 compared with the linear model)
  4. variance partition (commonality): R^2 of landscape only, topography only, both
Outputs (revision/outputs/full/): landscape_effects.csv, landscape_partition.csv,
landscape_vif.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covariate_models as cm  # noqa: E402
from tedge.paths import DADOS, INPUTS, OUT, YEAR_SENSOR, geometry_dir  # noqa: E402

METRICS = ["pland_focal", "patch_density", "lpi", "ed", "area_mn", "shape_mn", "shdi", "contag",
           "cohesion", "ai"]
TOPO = ["d_cosi", "slope_edge", "d_elev"]
VIF_MAX = 5.0
BLOCK_KM = 50


def vif_screen(X):
    cols = list(X.columns)
    dropped = []
    while len(cols) > 1:
        Z = X[cols].to_numpy(float)
        Z = (Z - Z.mean(0)) / Z.std(0)
        inv = np.linalg.pinv(np.corrcoef(Z, rowvar=False))
        v = pd.Series(np.diag(inv), index=cols)
        if v.max() <= VIF_MAX:
            return cols, v, dropped
        dropped.append((v.idxmax(), float(v.max())))
        cols.remove(v.idxmax())
    return cols, pd.Series(1.0, index=cols), dropped


def polygon_table(year):
    df = cm.build(year)
    sm = df.groupby(["focal_class", "scene_id"])["det"].transform("mean")
    cm_ = df.groupby("focal_class")["det"].transform("mean")
    df["det_adj"] = df["det"] - sm + cm_
    p = df.groupby(["label", "focal_class"]).agg(
        det_adj=("det_adj", "mean"), n_scenes=("scene_id", "nunique"),
        d_cosi=("d_cosi", "mean"), slope_edge=("slope_edge", "first"),
        d_elev=("d_elev", "first"), n_px=("n_px", "first")).reset_index()
    rebuilt = INPUTS / f"landscape_{year}.csv"  # recomputed for the C10.1 rebuild (D-11)
    if (INPUTS / f"polygon_labels_{year}.tif").exists() and not rebuilt.exists():
        # the submitted metrics belong to the old polygons: joining them by label would pair
        # unrelated polygons
        raise FileNotFoundError(f"{rebuilt} missing for a rebuilt year; run landscape_metrics_revision.py")
    land = pd.read_csv(rebuilt if rebuilt.exists() else DADOS / "landscape" / f"landscape_{year}.csv",
                       usecols=["polygon_id", "focal_class"]
                       + [f"{m.replace('patch_density', 'pd')}_w150" for m in METRICS])
    land["label"] = land["polygon_id"].str.split("_p").str[1].astype(np.int64)
    land.columns = [c.replace("_w150", "") for c in land.columns]
    land = land.rename(columns={"pd": "patch_density"})  # avoid the pandas name inside formulas
    geo = pd.read_parquet(geometry_dir(year) / "polygons.parquet",
                          columns=["label", "centroid_x", "centroid_y"])
    # labels are global; matching the class as well guards against an ID mismatch
    p = p.merge(land.drop(columns="polygon_id"), on=["label", "focal_class"], how="inner").merge(geo, on="label")
    p["log_area"] = np.log10(p["n_px"] * 900.0)
    p["block"] = (np.floor(p["centroid_x"] / (BLOCK_KM * 1000)).astype(int).astype(str) + "_"
                  + np.floor(p["centroid_y"] / (BLOCK_KM * 1000)).astype(int).astype(str))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    a = ap.parse_args()
    eff, part, vifs = [], [], []
    for y in a.year:
        p = polygon_table(y)
        for cls, g in p.groupby("focal_class"):
            g = g.dropna(subset=["det_adj"] + METRICS + TOPO + ["log_area"])
            g = g.loc[:, ~g.columns.duplicated()]
            if len(g) < 50:
                continue
            base = {"year": y, "year_sensor": YEAR_SENSOR[y], "focal_class": cls, "n_polygons": len(g),
                    "n_blocks": g["block"].nunique()}
            usable = [m for m in METRICS if g[m].std() > 0]
            keep, v, dropped = vif_screen(g[usable + ["log_area"]])
            keep_land = [k for k in keep if k != "log_area"]
            for k, val in v.items():
                vifs.append({**base, "term": k, "vif": val, "kept": True})
            for k, val in dropped:
                vifs.append({**base, "term": k, "vif": val, "kept": False})
            s = g.copy()
            for c in keep + TOPO:
                s[c] = (s[c] - s[c].mean()) / s[c].std() if s[c].std() > 0 else 0.0
            w = s["n_scenes"]
            rhs_land = " + ".join(keep)
            rhs_topo = " + ".join(TOPO)
            fit = smf.wls(f"det_adj ~ {rhs_land} + {rhs_topo}", s, weights=w).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(s["block"])[0]})
            for term in fit.params.index:
                ci = fit.conf_int().loc[term]
                eff.append({**base, "model": "linear", "term": term, "estimate_per_sd": fit.params[term],
                            "lo": ci[0], "hi": ci[1], "p": fit.pvalues[term]})
            # adjusted R^2: the spline model has ~4x the parameters, which inflates raw R^2 in the
            # small classes (water, grassland, wetland: 57-192 polygons)
            r2 = lambda f: smf.wls(f"det_adj ~ {f}", s, weights=w).fit().rsquared_adj  # noqa: E731
            r_land, r_topo, r_both = r2(rhs_land), r2(rhs_topo), r2(f"{rhs_land} + {rhs_topo}")
            spl = " + ".join(f"cr({k}, df=4)" for k in keep_land) + " + log_area + " + rhs_topo
            try:
                r_gam = r2(spl)
            except Exception:  # degenerate spline basis for a near-constant metric
                r_gam = np.nan
            part.append({**base, "r2adj_landscape": r_land, "r2adj_topography": r_topo,
                         "r2adj_both": r_both, "unique_landscape": r_both - r_topo,
                         "unique_topography": r_both - r_land, "shared": r_land + r_topo - r_both,
                         "r2adj_additive_splines": r_gam, "low_support": len(g) < 200,
                         "metrics_kept": ";".join(keep)})
            print(f"  {y} class {cls}: done", flush=True)
    out = OUT / "full"
    pd.DataFrame(eff).to_csv(out / "landscape_effects.csv", index=False)
    pd.DataFrame(part).to_csv(out / "landscape_partition.csv", index=False)
    pd.DataFrame(vifs).to_csv(out / "landscape_vif.csv", index=False)


if __name__ == "__main__":
    main()
