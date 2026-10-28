"""Spatial autocorrelation of polygon-level edge contrasts and model residuals (Moran correlogram).

  python spatial_autocorrelation.py --year 1985 2002 2024

For each class x year_sensor (W = 150 m): polygon mean over scenes of (a) DET and (b) the residual
of covariate model A (covariate_models.py), at the polygon centroid. Moran's I with binary weights
in distance bands (0-2, 2-5, 5-10, 10-25, 25-50, 50-100 km), permutation p-value (199 draws), on a
random subsample of <= 3000 polygons. Also an empirical semivariogram of the residuals.
Output: revision/outputs/full/moran_correlogram.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covariate_models as cm  # noqa: E402
from tedge.paths import OUT, YEAR_SENSOR, geometry_dir  # noqa: E402

BANDS_KM = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 100)]
MAX_POLY = 3000
PERM = 199


def moran_band(z, D, lo, hi, rng):
    W = ((D > lo) & (D <= hi)).astype(float)
    np.fill_diagonal(W, 0)
    s0 = W.sum()
    if s0 == 0:
        return np.nan, np.nan, 0
    zc = z - z.mean()
    denom = zc @ zc
    i_obs = len(z) / s0 * (zc @ W @ zc) / denom
    perm = np.empty(PERM)
    for p in range(PERM):
        zp = rng.permutation(zc)
        perm[p] = len(z) / s0 * (zp @ W @ zp) / denom
    pval = (1 + np.sum(np.abs(perm) >= abs(i_obs))) / (PERM + 1)
    return i_obs, pval, int(s0 / 2)


def semivariance(z, D, lo, hi):
    m = (D > lo) & (D <= hi)
    iu = np.triu(m, 1)
    if not iu.any():
        return np.nan
    a, b = np.nonzero(iu)
    return 0.5 * np.mean((z[a] - z[b]) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    rng = np.random.default_rng(7)
    cols = cm.METEO + cm.TOPO
    for y in a.year:
        df = cm.build(y)
        poly = pd.read_parquet(geometry_dir(y) / "polygons.parquet",
                               columns=["label", "centroid_x", "centroid_y"])
        for cls, g in df.groupby("focal_class"):
            g = g.dropna(subset=["det"] + cols)
            if g["label"].nunique() < cm.MIN_UNITS:
                continue
            Z = g[cols].to_numpy(float)
            sd = Z.std(0)
            keep = sd > 0
            X = np.column_stack([np.ones(len(g)), (Z[:, keep] - Z[:, keep].mean(0)) / sd[keep]])
            beta = np.linalg.lstsq(X, g["det"].to_numpy(float), rcond=None)[0]
            g = g.assign(resid=g["det"].to_numpy(float) - X @ beta)
            pm = g.groupby("label")[["det", "resid"]].mean().reset_index().merge(poly, on="label")
            if len(pm) > MAX_POLY:
                pm = pm.sample(MAX_POLY, random_state=0)
            xy = pm[["centroid_x", "centroid_y"]].to_numpy() / 1000.0
            D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            for var in ("det", "resid"):
                z = pm[var].to_numpy(float)
                for lo, hi in BANDS_KM:
                    i, p, npairs = moran_band(z, D, lo, hi, rng)
                    rows.append({"year": y, "year_sensor": YEAR_SENSOR[y], "focal_class": cls,
                                 "variable": var, "band_lo_km": lo, "band_hi_km": hi,
                                 "moran_i": i, "p_perm": p, "n_pairs": npairs,
                                 "semivariance": semivariance(z, D, lo, hi),
                                 "variance": float(np.var(z)), "n_polygons": len(pm)})
            del D
            print(f"  {y} class {cls}: done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "full" / "moran_correlogram.csv", index=False)


if __name__ == "__main__":
    main()
