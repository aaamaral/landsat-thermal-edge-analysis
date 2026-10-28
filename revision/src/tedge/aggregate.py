"""Pixel -> zone aggregation of one scene tile (pure numpy, unit-tested).

A *zone* is (label, nb_class, ring) on the 30 m grid. For each zone the aggregator returns
sums that can be merged across tiles: n_total (pixels of the zone inside the scene
footprint), n_valid (pixels with a QA-valid TST), sums of TST for the operational 30 m grid
and for each degraded-support variant, the sum of squares of the 30 m TST, and NDVI/NDMI sums.

Validity is pixel-wise (no all-or-nothing rule): an invalid pixel only reduces n_valid.
Degraded variants are normalised convolutions over valid pixels and are evaluated on the
same valid pixels as the 30 m grid, so differences between variants reflect spatial support
only, not a change of sample.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

OUTSIDE = -32768  # pixel outside the scene footprint (or fill)
INVALID = -32767  # inside the footprint but masked by QA_PIXEL (cloud, shadow, cirrus, snow)
TST_SCALE = 100.0  # centi-degrees Celsius
IDX_SCALE = 10000.0

BOX_SUPPORTS_M = (60, 90, 120)


def decode_tst(raw):
    """int16 centi-degC with sentinel codes -> (tst float32 NaN-masked, in_footprint, valid)."""
    raw = np.asarray(raw)
    in_fp = raw != OUTSIDE
    valid = raw > INVALID
    tst = np.where(valid, raw / TST_SCALE, np.nan).astype(np.float32)
    return tst, in_fp, valid


def decode_index(raw):
    raw = np.asarray(raw)
    return np.where(raw > INVALID, raw / IDX_SCALE, np.nan).astype(np.float32)


def normalized_filter(values, valid, kind, size_px):
    """Mean of valid neighbours under a box (size_px wide) or Gaussian (sigma=size_px) kernel."""
    v = np.where(valid, values, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    if kind == "box":
        s = ndimage.uniform_filter(v, size=size_px, mode="constant")
        n = ndimage.uniform_filter(w, size=size_px, mode="constant")
    elif kind == "gauss":
        s = ndimage.gaussian_filter(v, sigma=size_px, mode="constant", truncate=3.0)
        n = ndimage.gaussian_filter(w, sigma=size_px, mode="constant", truncate=3.0)
    else:
        raise ValueError(kind)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[~valid] = np.nan
    return out.astype(np.float32)


def psf_sigma_px(native_m, pixel_m=30.0):
    """Gaussian sigma (pixels) whose FWHM equals the native thermal resolution."""
    return native_m / 2.354820045 / pixel_m


def support_variants(tst, valid, native_m, pixel_m=30.0):
    """Dict of degraded-support TST arrays keyed by variant name (30 m grid kept as 't30')."""
    out = {"t30": tst}
    for m in BOX_SUPPORTS_M:
        out[f"box{m}"] = normalized_filter(tst, valid, "box", int(round(m / pixel_m)))
    out["psf"] = normalized_filter(tst, valid, "gauss", psf_sigma_px(native_m, pixel_m))
    return out


def zone_keys(label, nb_class, ring):
    return (label.astype(np.int64) * 256 + nb_class.astype(np.int64)) * 16 + ring.astype(np.int64)


def split_keys(keys):
    keys = np.asarray(keys, dtype=np.int64)
    return keys // (256 * 16), (keys // 16) % 256, keys % 16


def aggregate_tile(keys, in_fp, valid, variants, ndvi=None, ndmi=None):
    """Aggregate pixels (already restricted to the tile core and to eligible zones).

    ``keys`` is an int64 array of zone keys (-1 = not aggregated). Returns a dict of 1-D
    arrays: key, n_total, n_valid, sum_<variant>, sumsq_t30, sum_ndvi, n_ndvi, sum_ndmi, n_ndmi.
    """
    m = (keys >= 0) & in_fp
    if not m.any():
        return None
    k = keys[m]
    uk, inv = np.unique(k, return_inverse=True)
    out = {"key": uk, "n_total": np.bincount(inv, minlength=uk.size)}
    v = valid[m]
    out["n_valid"] = np.bincount(inv, weights=v, minlength=uk.size)
    for name, arr in variants.items():
        a = arr[m]
        ok = v & np.isfinite(a)
        out[f"sum_{name}"] = np.bincount(inv, weights=np.where(ok, a, 0.0), minlength=uk.size)
        if name == "t30":
            out["sumsq_t30"] = np.bincount(inv, weights=np.where(ok, a.astype(np.float64) ** 2, 0.0),
                                           minlength=uk.size)
    for name, arr in (("ndvi", ndvi), ("ndmi", ndmi)):
        if arr is None:
            continue
        a = arr[m]
        ok = v & np.isfinite(a)
        out[f"sum_{name}"] = np.bincount(inv, weights=np.where(ok, a, 0.0), minlength=uk.size)
        out[f"n_{name}"] = np.bincount(inv, weights=ok, minlength=uk.size)
    return out


def finalize(df):
    """Turn merged sums (pandas DataFrame) into means / SD / valid_fraction columns."""
    import pandas as pd

    label, nb, ring = split_keys(df["key"].to_numpy())
    res = pd.DataFrame({"label": label.astype(np.int32), "nb_class": nb.astype(np.uint8),
                        "ring": ring.astype(np.uint8),
                        "n_total": df["n_total"].to_numpy().astype(np.uint32),
                        "n_valid": df["n_valid"].to_numpy().astype(np.uint32)})
    nv = df["n_valid"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in [c for c in df.columns if c.startswith("sum_") and c not in ("sum_ndvi", "sum_ndmi")]:
            res[c[4:] + "_mean"] = (df[c].to_numpy() / nv).astype(np.float32)
        mean = df["sum_t30"].to_numpy() / nv
        var = df["sumsq_t30"].to_numpy() / nv - mean ** 2
        res["t30_sd"] = np.sqrt(np.clip(var, 0, None) * nv / np.maximum(nv - 1, 1)).astype(np.float32)
        for name in ("ndvi", "ndmi"):
            if f"sum_{name}" in df:
                res[f"{name}_mean"] = (df[f"sum_{name}"].to_numpy() / df[f"n_{name}"].to_numpy()).astype(np.float32)
    res["valid_fraction"] = (nv / df["n_total"].to_numpy()).astype(np.float32)
    return res
