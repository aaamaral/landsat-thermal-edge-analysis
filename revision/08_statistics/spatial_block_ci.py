"""Spatial-block sensitivity of the DET and DTEC confidence intervals.

  python spatial_block_ci.py --year 1985 2002 2024 --B 1000

The Moran correlogram (moran_correlogram.csv) shows positive autocorrelation of polygon-level
residuals up to ~25-50 km, so polygons are not independent units. Here the crossed pigeonhole
bootstrap is repeated with the polygon (or interface) unit replaced by square spatial blocks of
25 km and 50 km containing the polygon centroid (block x scene crossed weights). Point estimates
are unchanged; only the CI width responds. Output: revision/outputs/full/spatial_block_ci.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tedge import engine, profiles as pr  # noqa: E402
from tedge.paths import DATA, OUT, YEAR_SENSOR, geometry_dir  # noqa: E402
from core_estimates import FOCUS, MIN_UNITS, MAIN, _codes  # noqa: E402

BLOCKS_KM = (0, 25, 50)  # 0 = polygon / interface unit (reference)


def with_blocks(df, poly):
    df = df.merge(poly, on="label", how="left")
    for b in BLOCKS_KM[1:]:
        df[f"blk{b}"] = (np.floor(df["centroid_x"] / (b * 1000)).astype(int).astype(str) + "_"
                         + np.floor(df["centroid_y"] / (b * 1000)).astype(int).astype(str))
    return df


def ci_rows(df, value, unit0, base, B):
    rows = []
    for b in BLOCKS_KM:
        unit = unit0 if b == 0 else f"blk{b}"
        u, s = _codes(df[unit]), _codes(df["scene_id"])
        point, reps = pr.pigeonhole_means(u, s, df[[value]].to_numpy(float), u.max() + 1,
                                          s.max() + 1, B=B, seed=11)
        lo, hi = pr.ci95(reps)
        rows.append({**base, "block_km": b, "n_units": int(u.max() + 1), "estimate": point[0],
                     "lo": lo[0], "hi": hi[0], "ci_width": hi[0] - lo[0]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    rows = []
    for y in a.year:
        table = DATA / "tables" / f"full_polygon_ring_{y}.parquet"
        poly = pd.read_parquet(geometry_dir(y) / "polygons.parquet",
                               columns=["label", "centroid_x", "centroid_y"])
        det = engine.contrasts(y, MAIN["window"], classes=list(FOCUS), table=table)
        det = with_blocks(det[np.isfinite(det["det"])], poly)
        for cls, g in det.groupby("focal_class"):
            if g["label"].nunique() < MIN_UNITS:
                continue
            base = {"year": y, "year_sensor": YEAR_SENSOR[y], "estimand": "DET",
                    "focal_class": cls, "nb_class": -1}
            rows += ci_rows(g, "det", "label", base, a.B)
        dt = engine.interface_contrasts(y, MAIN["window"], table=table)
        dt = dt[dt["focal_class"].isin(FOCUS) & dt["nb_class"].isin(FOCUS)]
        dt = with_blocks(dt, poly)
        dt["iface"] = dt["label"].astype(str) + "_" + dt["nb_class"].astype(str)
        for (fc, nb), g in dt.groupby(["focal_class", "nb_class"]):
            if g["iface"].nunique() < MIN_UNITS:
                continue
            base = {"year": y, "year_sensor": YEAR_SENSOR[y], "estimand": "DTEC",
                    "focal_class": fc, "nb_class": nb}
            rows += ci_rows(g, "dtec", "iface", base, a.B)
        print(f"{y}: spatial-block CIs done", flush=True)
    res = pd.DataFrame(rows)
    ref = res[res["block_km"] == 0].set_index(["year", "estimand", "focal_class", "nb_class"])["ci_width"]
    res["width_ratio_vs_unit"] = res["ci_width"].to_numpy() / ref.reindex(
        pd.MultiIndex.from_frame(res[["year", "estimand", "focal_class", "nb_class"]])).to_numpy()
    res.to_csv(OUT / "full" / "spatial_block_ci.csv", index=False)


if __name__ == "__main__":
    main()
