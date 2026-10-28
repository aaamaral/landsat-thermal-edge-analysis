"""Profiles, DET, DEP (breakpoint with right-censoring) and DTEC with crossed bootstrap.

Notation
--------
ring k (0..9) has centre distance d_k = 30k + 15 m from the mapped edge; ring 10 pools >= 300 m.
A *window* W (m) defines the interior reference band of a polygon: rings k_ref and k_ref+1,
k_ref = W / 30 (W = 300 -> ring 10 only). Rings are nested (a polygon that has ring m has all
rings < m), so for a given window every eligible polygon contributes to every ring 0..k_ref-1.

Within-unit contrast:  C[u, k] = T[u, k] - T[u, ref_W],  u = (scene, polygon).
DET_W(e)  = C[u, e]  where e = number of excluded edge rings (0 -> 0-30 m ring).
Profile   = mean_u C[u, k] for k < k_ref.
DTEC(A<-B) for interface i = (polygon of class A, neighbour class B), scene s:
            T[s, i, ring e] - T[s, polygon(i), ref_W].

Uncertainty: pigeonhole (two-way crossed) bootstrap - independent Poisson(1) weights for the
spatial unit (polygon or interface) and for the scene; the estimate is the weighted mean over
observed (unit, scene) cells. Records are never resampled as if independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

RING_CENTRES_M = np.array([30 * k + 15 for k in range(10)], dtype=float)
WINDOWS_M = (90, 120, 150, 180, 210, 240, 300)


def k_ref(window_m: int) -> int:
    return min(int(window_m // 30), 10)


def ref_rings(window_m: int) -> list[int]:
    k = k_ref(window_m)
    return [10] if k >= 10 else [k, min(k + 1, 10)]


# ---------------------------------------------------------------- bootstrap

def pigeonhole_means(unit_idx, scene_idx, values, n_units, n_scenes, B=1000, seed=0,
                     chunk=100):
    """Crossed bootstrap replicates of a weighted mean over observed (unit, scene) cells.

    ``values`` is (n_obs, m) - m statistics sharing the same cells (e.g. rings), NaN = missing.
    Returns (point_estimate[m], replicates[B, m]).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n_obs, m = values.shape
    ok = np.isfinite(values)
    vals = np.where(ok, values, 0.0)
    point = vals.sum(0) / np.maximum(ok.sum(0), 1)
    point[ok.sum(0) == 0] = np.nan
    rng = np.random.default_rng(seed)
    reps = np.empty((B, m))
    for j in range(m):
        X = sparse.csr_matrix((vals[:, j], (unit_idx, scene_idx)), shape=(n_units, n_scenes))
        M = sparse.csr_matrix((ok[:, j].astype(float), (unit_idx, scene_idx)), shape=(n_units, n_scenes))
        rng_j = np.random.default_rng(seed)  # same weights for every statistic
        for b0 in range(0, B, chunk):
            nb = min(chunk, B - b0)
            wu = rng_j.poisson(1.0, (nb, n_units)).astype(float)
            ws = rng_j.poisson(1.0, (n_scenes, nb)).astype(float)
            num = np.einsum("bu,ub->b", wu, X @ ws)
            den = np.einsum("bu,ub->b", wu, M @ ws)
            with np.errstate(invalid="ignore", divide="ignore"):
                reps[b0:b0 + nb, j] = num / den
    return point, reps


def ci95(reps):
    return np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)


# ---------------------------------------------------------------- DEP

def broken_stick(d, y, w=None, grid=None):
    """Fit y = c + b * (min(d, delta) - delta); returns (delta, c, b, sse, censored).

    ``delta`` is searched on a fine grid within the observed range. ``censored`` is True when
    the best breakpoint is at the last observed distance (no plateau inside the window) -> the
    depth is reported as DEP >= max(d).
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    w = np.ones_like(y) if w is None else np.asarray(w, float)[ok]
    if d.size < 3:
        return np.nan, np.nan, np.nan, np.nan, False
    grid = np.linspace(d.min(), d.max(), 200) if grid is None else grid
    best = None
    for dl in grid:
        x = np.minimum(d, dl) - dl
        A = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
        sse = float(((A @ coef - y * np.sqrt(w)) ** 2).sum())
        if best is None or sse < best[3] - 1e-12:
            best = (float(dl), float(coef[0]), float(coef[1]), sse)
    censored = best[0] >= d.max() - 1e-9
    return best + (bool(censored),)


def exp_decay(d, y, w=None):
    """y = c + A exp(-d / lam); returns (lam, c, A, sse) or NaNs (competitor model)."""
    from scipy.optimize import curve_fit

    d = np.asarray(d, float); y = np.asarray(y, float)
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    sig = None if w is None else 1 / np.sqrt(np.asarray(w, float)[ok])
    try:
        p, _ = curve_fit(lambda x, c, a, lam: c + a * np.exp(-x / lam), d, y,
                         p0=(y[-1], y[0] - y[-1], 60.0), sigma=sig,
                         bounds=([-50, -50, 1.0], [50, 50, 5000.0]), maxfev=20000)
        res = y - (p[0] + p[1] * np.exp(-d / p[2]))
        return float(p[2]), float(p[0]), float(p[1]), float((res ** 2).sum())
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def is_monotonic(y, tol=0.02):
    dy = np.diff(np.asarray(y, float))
    return bool(np.all(dy >= -tol) or np.all(dy <= tol))


def dep_from_replicates(point_profile, rep_profiles, d=RING_CENTRES_M, signal_threshold=None):
    """DEP point + crossed-bootstrap CI + share of censored replicates.

    When ``signal_threshold`` (deg C) is given, a threshold DEP is also returned: the first
    distance where |profile - plateau| falls below the threshold (practical-relevance
    sensitivity; 0.1/0.25/0.5/1.0 C), censored if never reached.
    """
    n = len(point_profile)
    dd = d[:n]
    delta, c, b, sse, cens = broken_stick(dd, point_profile)
    reps = np.array([broken_stick(dd, r)[0::4] for r in rep_profiles if np.isfinite(r).sum() >= 3])
    out = {"dep_m": delta, "dep_censored": cens, "plateau_c": c, "slope_c_per_m": b, "sse": sse}
    if reps.size:
        out["dep_lo"], out["dep_hi"] = np.percentile(reps[:, 0], [2.5, 97.5])
        out["p_censored"] = float(np.mean(reps[:, 1].astype(bool)))
    lam, ce, ae, sse_e = exp_decay(dd, point_profile)
    out.update({"monotonic": is_monotonic(point_profile), "exp_lambda_m": lam,
                "exp_sse": sse_e, "exp_dep95_m": 3 * lam if np.isfinite(lam) else np.nan})
    return out


def threshold_depth(profile, plateau, thr, d=RING_CENTRES_M):
    dev = np.abs(np.asarray(profile, float) - plateau)
    dd = d[:len(dev)]
    below = np.where(dev < thr)[0]
    if below.size == 0:
        return float(dd[-1]), True
    return float(dd[below[0]]), False
