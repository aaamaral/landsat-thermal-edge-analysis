"""Covariate models and variance decomposition of the polygon x scene edge contrast (W = 150 m).

  python covariate_models.py --year 1985 2002 2024

Response: DET of a polygon in a scene (ring 0 minus reference band, as in core_estimates).
Covariates (standardised within class x year_sensor):
  scene / meteorology  ERA5-Land at the polygon centroid cell and overpass hour: t2m, VPD, ssrd,
                       wind 10 m, swvl1, precipitation of the previous 120 h; sun elevation
  topography           edge - reference difference of elevation and of illumination cos(i)
                       (scene sun geometry on the zone-mean slope/aspect terms), edge slope
  biophysical (model B only, potential mediators, not confounders)
                       edge - reference difference of NDVI and NDMI in the same scene
Model A = meteorology + topography; Model B = A + biophysical. OLS point estimates with two-way
cluster-robust SE (polygon and scene; Cameron, Gelbach & Miller 2011), so that repeated scenes and
repeated polygons are not treated as independent observations.
Variance decomposition: crossed random intercepts (polygon, scene) by REML (statsmodels MixedLM,
variance components) on a random subsample of polygons, with and without the covariates.

Outputs (revision/outputs/full/): covariate_effects.csv, variance_components.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tedge import engine, profiles as pr  # noqa: E402
from tedge.paths import DATA, OUT, YEAR_SENSOR, geometry_dir  # noqa: E402
from core_estimates import FOCUS, MIN_UNITS  # noqa: E402

WINDOW = 150
COV = DATA / "covariates"
METEO = ["t2m_c", "vpd_kpa", "ssrd_wm2", "wind10_ms", "swvl1", "prec_120h_mm", "sun_elev"]
TOPO = ["d_elev", "d_cosi", "slope_edge"]
BIO = ["d_ndvi", "d_ndmi"]
VC_POLYGONS = 400


def scene_meta(year):
    rows = [json.loads(l) for l in open(DATA / f"scene_metadata_{year}.jsonl", encoding="utf-8")]
    m = pd.DataFrame(rows).drop_duplicates("scene_id")
    return m[["scene_id", "SUN_AZIMUTH", "SUN_ELEVATION"]]


def era5_at_polygons(year, poly):
    """ERA5-Land value of the polygon centroid cell; permanently empty cells -> nearest valid cell."""
    from pyproj import Transformer
    from scipy import ndimage

    g = json.loads((COV / "era5land_grid.json").read_text())
    x0, _, _, y1 = g["bbox"]
    res = g["res_deg"]
    era =pd.read_parquet(COV / "era5land_scene_grid.parquet", filters=[("year", "==", year)])
    for c in METEO[:-1]:
        era.loc[~np.isfinite(era[c]), c] = np.nan
    ny, nx = era["iy"].max() + 1, era["ix"].max() + 1
    valid = np.zeros((ny, nx), bool)
    ok = era.groupby(["iy", "ix"])["t2m_c"].apply(lambda s: s.notna().any())
    valid[ok.index.get_level_values(0)[ok.values], ok.index.get_level_values(1)[ok.values]] = True
    _, (iy_n, ix_n) = ndimage.distance_transform_edt(~valid, return_indices=True)
    import rasterio
    from tedge.paths import label_raster
    with rasterio.open(label_raster(year)) as src:
        crs = src.crs
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        poly["centroid_x"].to_numpy(), poly["centroid_y"].to_numpy())
    iy = np.clip(np.floor((y1 - lat) / res).astype(int), 0, ny - 1)
    ix = np.clip(np.floor((lon - x0) / res).astype(int), 0, nx - 1)
    cell = pd.DataFrame({"label": poly["label"].to_numpy(), "iy": iy_n[iy, ix], "ix": ix_n[iy, ix]})
    return cell, era[["scene_id", "iy", "ix"] + METEO[:-1]]


def zone_terms(year, labels, refs):
    """Edge (ring 0) and reference-band means of elevation, slope and the m1-m3 terms per polygon."""
    dem = pd.read_parquet(COV / f"dem_zones_{year}.parquet")
    dem = dem[dem["label"].isin(labels) & (dem["nb_class"] != 0)]
    dem["zone"] = np.where(dem["ring"] == 0, "edge", np.where(dem["ring"].isin(refs), "ref", None))
    dem = dem[dem["zone"].notna()]
    cols = ["elev", "m1", "m2", "m3", "slope_deg"]
    w = dem[cols].mul(dem["n_dem"], axis=0)
    w["n"] = dem["n_dem"]
    w[["label", "zone"]] = dem[["label", "zone"]]
    agg = w.groupby(["label", "zone"]).sum()
    agg = agg[cols].div(agg["n"], axis=0).unstack("zone")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def two_way_cluster(X, y, g1, g2):
    """OLS with two-way cluster-robust covariance (CGM 2011); g1 x g2 cells are single obs."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta

    def meat(g):
        s = pd.DataFrame(X * e[:, None]).groupby(g).sum().to_numpy()
        return s.T @ s

    m0 = (X * e[:, None]).T @ (X * e[:, None])
    V = XtX_inv @ (meat(g1) + meat(g2) - m0) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


def variance_components(df, cols, seed=0):
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(seed)
    labs = df["label"].unique()
    pick = rng.choice(labs, size=min(VC_POLYGONS, len(labs)), replace=False)
    d = df[df["label"].isin(pick)].copy()
    d["one"] = 1
    out = {}
    for name, rhs in (("no_covariates", "1"), ("modelA", " + ".join(cols))):
        m = smf.mixedlm(f"det ~ {rhs}", d, groups="one",
                        vc_formula={"polygon": "0 + C(label)", "scene": "0 + C(scene_id)"})
        r = m.fit(reml=True, method="lbfgs")
        vc = dict(zip(m.exog_vc.names, r.vcomp))
        out[name] = {"var_polygon": vc.get("polygon", np.nan), "var_scene": vc.get("scene", np.nan),
                     "var_residual": r.scale, "n_obs": len(d), "n_polygons": d["label"].nunique(),
                     "n_scenes": d["scene_id"].nunique()}
    return out


def build(year):
    kr, refs = pr.k_ref(WINDOW), pr.ref_rings(WINDOW)
    table = DATA / "tables" / f"full_polygon_ring_{year}.parquet"
    df = engine.contrasts(year, WINDOW, classes=list(FOCUS), table=table)
    df = df[np.isfinite(df["det"])]
    # biophysical edge - reference contrasts in the same scene
    q = f"""SELECT scene_id, label,
              sum(ndvi * n_valid) FILTER (WHERE ring = 0) / sum(n_valid) FILTER (WHERE ring = 0)
              - sum(ndvi * n_valid) FILTER (WHERE ring IN ({','.join(map(str, refs))}))
                / sum(n_valid) FILTER (WHERE ring IN ({','.join(map(str, refs))})) AS d_ndvi,
              sum(ndmi * n_valid) FILTER (WHERE ring = 0) / sum(n_valid) FILTER (WHERE ring = 0)
              - sum(ndmi * n_valid) FILTER (WHERE ring IN ({','.join(map(str, refs))}))
                / sum(n_valid) FILTER (WHERE ring IN ({','.join(map(str, refs))})) AS d_ndmi
            FROM read_parquet('{table.as_posix()}')
            WHERE n_valid >= 3 AND isfinite(ndvi) AND isfinite(ndmi) GROUP BY ALL"""
    df = df.merge(engine.con().execute(q).df(), on=["scene_id", "label"], how="left")
    poly = pd.read_parquet(geometry_dir(year) / "polygons.parquet",
                           columns=["label", "centroid_x", "centroid_y"])
    poly = poly[poly["label"].isin(df["label"].unique())]
    cell, era = era5_at_polygons(year, poly)
    df = df.merge(cell, on="label").merge(era, on=["scene_id", "iy", "ix"], how="left")
    df = df.merge(scene_meta(year), on="scene_id", how="left")
    zt = zone_terms(year, poly["label"], refs)
    df = df.merge(zt, on="label", how="left")
    z = np.radians(90 - df["SUN_ELEVATION"])
    phi = np.radians(df["SUN_AZIMUTH"])
    cosi = lambda s: np.cos(z) * df[f"m1_{s}"] + np.sin(z) * (np.cos(phi) * df[f"m2_{s}"]  # noqa: E731
                                                               + np.sin(phi) * df[f"m3_{s}"])
    df["d_cosi"] = cosi("edge") - cosi("ref")
    df["d_elev"] = df["elev_edge"] - df["elev_ref"]
    df["slope_edge"] = df["slope_deg_edge"]
    df["sun_elev"] = df["SUN_ELEVATION"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    a = ap.parse_args()
    eff, vcs = [], []
    for y in a.year:
        df = build(y)
        for cls, g in df.groupby("focal_class"):
            g = g.dropna(subset=["det"] + METEO + TOPO + BIO)
            if g["label"].nunique() < MIN_UNITS:
                continue
            base = {"year": y, "year_sensor": YEAR_SENSOR[y], "focal_class": cls,
                    "n_obs": len(g), "n_polygons": g["label"].nunique(), "n_scenes": g["scene_id"].nunique()}
            u1, u2 = pd.factorize(g["label"])[0], pd.factorize(g["scene_id"])[0]
            yv = g["det"].to_numpy(float)
            for model, cols in (("A", METEO + TOPO), ("B", METEO + TOPO + BIO)):
                Z = g[cols].to_numpy(float)
                sd = Z.std(0)
                keep = sd > 0
                Z = (Z[:, keep] - Z[:, keep].mean(0)) / sd[keep]
                X = np.column_stack([np.ones(len(g)), Z])
                beta, se, r2 = two_way_cluster(X, yv, u1, u2)
                names = ["intercept"] + [c for c, k in zip(cols, keep) if k]
                for n, b, s in zip(names, beta, se):
                    eff.append({**base, "model": model, "term": n, "estimate_per_sd": b, "se_2way": s,
                                "lo": b - 1.96 * s, "hi": b + 1.96 * s, "r2": r2})
            try:
                gs = g.copy()
                for c in METEO + TOPO:
                    s = gs[c].std()
                    gs[c] = (gs[c] - gs[c].mean()) / s if s > 0 else 0.0
                for name, v in variance_components(gs, METEO + TOPO).items():
                    tot = v["var_polygon"] + v["var_scene"] + v["var_residual"]
                    vcs.append({**base, "model": name, **v,
                                "share_polygon": v["var_polygon"] / tot,
                                "share_scene": v["var_scene"] / tot,
                                "share_residual": v["var_residual"] / tot})
            except Exception as exc:  # report, never drop silently
                vcs.append({**base, "model": "failed", "note": str(exc)[:200]})
            print(f"  {y} class {cls}: done", flush=True)
    out = OUT / "full"
    pd.DataFrame(eff).to_csv(out / "covariate_effects.csv", index=False)
    pd.DataFrame(vcs).to_csv(out / "variance_components.csv", index=False)


if __name__ == "__main__":
    main()
