"""Annual map vs scene date: DET / DTEC restricted to temporally stable polygons and by season.

  python map_date_sensitivity.py --year 1985 2002 2024 --B 1000

Scenarios per class x year_sensor (W = 150 m, ring 0):
  all          main estimate
  stable       polygons whose class is unchanged in the adjacent annual maps for >= 90 % of the
               edge-ring pixels and >= 95 % of the remaining pixels (class_stability.py)
  unstable     the complement
  wet / dry    scenes of October-March / April-September (all polygons)
Crossed polygon (interface) x scene pigeonhole bootstrap CIs.
Output: revision/outputs/full/map_date_sensitivity.csv
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
from tedge.paths import DATA, OUT, YEAR_SENSOR  # noqa: E402
from core_estimates import FOCUS, MIN_UNITS, MAIN, _codes  # noqa: E402

EDGE_MIN, INNER_MIN = 0.90, 0.95


def scenarios(df, stab):
    df = df.merge(stab, on="label", how="left")
    month = pd.to_numeric(df["scene_id"].str.slice(21, 23))  # LC08_L2SP_PPPRRR_YYYYMMDD_...
    wet = month.isin([10, 11, 12, 1, 2, 3])
    yield "all", df
    yield "stable", df[df["stable"] == True]  # noqa: E712
    yield "unstable", df[df["stable"] == False]  # noqa: E712
    yield "wet", df[wet]
    yield "dry", df[~wet]


def estimate(g, value, unit, B):
    u, s = _codes(g[unit]), _codes(g["scene_id"])
    point, reps = pr.pigeonhole_means(u, s, g[[value]].to_numpy(float), u.max() + 1, s.max() + 1,
                                      B=B, seed=21)
    lo, hi = pr.ci95(reps)
    return point[0], lo[0], hi[0], int(u.max() + 1), int(s.max() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    rows = []
    for y in a.year:
        table = DATA / "tables" / f"full_polygon_ring_{y}.parquet"
        st = pd.read_parquet(DATA / "covariates" / f"class_stability_{y}.parquet")
        st["edge_stable_share"] = st["stable_edge"] / st["n_edge"].replace(0, np.nan)
        st["inner_stable_share"] = st["stable_inner"] / st["n_inner"].replace(0, np.nan)
        st["stable"] = (st["edge_stable_share"] >= EDGE_MIN) & (st["inner_stable_share"] >= INNER_MIN)
        st = st[["label", "stable", "edge_stable_share", "inner_stable_share"]]
        det = engine.contrasts(y, MAIN["window"], classes=list(FOCUS), table=table)
        det = det[np.isfinite(det["det"])]
        dt = engine.interface_contrasts(y, MAIN["window"], table=table)
        dt = dt[dt["focal_class"].isin(FOCUS) & dt["nb_class"].isin(FOCUS)]
        dt["iface"] = dt["label"].astype(str) + "_" + dt["nb_class"].astype(str)
        for est, df, value, unit, keys in (("DET", det, "det", "label", ["focal_class"]),
                                           ("DTEC", dt, "dtec", "iface", ["focal_class", "nb_class"])):
            for scen, d in scenarios(df, st):
                for k, g in d.groupby(keys):
                    k = k if isinstance(k, tuple) else (k,)
                    base = {"year": y, "year_sensor": YEAR_SENSOR[y], "estimand": est,
                            "scenario": scen, "focal_class": k[0],
                            "nb_class": k[1] if len(k) > 1 else -1}
                    if g[unit].nunique() < MIN_UNITS:
                        rows.append({**base, "n_units": g[unit].nunique(), "note": "insufficient units"})
                        continue
                    e, lo, hi, nu, ns = estimate(g, value, unit, a.B)
                    rows.append({**base, "estimate": e, "lo": lo, "hi": hi, "n_units": nu,
                                 "n_scenes": ns, "note": ""})
        share = st["stable"].mean()
        print(f"{y}: done (stable polygons {share:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "full" / "map_date_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
