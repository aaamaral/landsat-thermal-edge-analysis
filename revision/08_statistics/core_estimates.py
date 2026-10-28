"""Class-level DET / profile / DEP and class-pair DTEC with crossed-bootstrap CIs.

Runs the same estimators on the pilot or on the full database:
  python core_estimates.py --source pilot --year 2024 --B 200
  python core_estimates.py --source full  --year 1985 2002 2024 --B 1000

Outputs (revision/outputs/{source}/):
  det_sweep.csv      class x year_sensor x window x variant x exclusion x valid_fraction scenario
                     DET estimate, 95 % CI, n polygons, n scenes, n obs
  profiles.csv       class x year_sensor x window: contrast per ring (estimate + CI)
  dep.csv            class x year_sensor x window: breakpoint DEP, CI, censoring, exp competitor,
                     threshold depths (0.1/0.25/0.5/1.0 C)
  dtec_pairs.csv     focal x neighbour x year_sensor (window 150 m, ring 0) DTEC + CI + support
                     under three contact weights; asymmetry for each unordered pair
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import engine, profiles as pr  # noqa: E402
from tedge.paths import DATA, OUT, YEAR_SENSOR  # noqa: E402

FOCUS = {3: "Forest formation", 9: "Silviculture", 11: "Wetland", 12: "Grassland",
         15: "Pasture", 20: "Sugar cane", 21: "Mosaic of uses", 24: "Urban area",
         33: "River, lake and ocean", 39: "Soybean", 41: "Other temporary crops", 46: "Coffee"}
MAIN = dict(window=150, variant="t30", excl=0, vf=0.0)
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
MIN_UNITS = 10


def _codes(s):
    return pd.factorize(s)[0]


def class_stats(df, cols, B, seed=0, unit="label"):
    u = _codes(df[unit] if isinstance(unit, str) else df[unit].astype(str).agg("|".join, axis=1))
    s = _codes(df["scene_id"])
    point, reps = pr.pigeonhole_means(u, s, df[cols].to_numpy(float), u.max() + 1, s.max() + 1,
                                      B=B, seed=seed)
    return point, reps, int(u.max() + 1), int(s.max() + 1), len(df)


def sweep_rows(year, table, B):
    """DET for the main specification and one-factor-at-a-time sensitivities."""
    specs = [dict(MAIN, window=w) for w in pr.WINDOWS_M]
    specs += [dict(MAIN, variant=v) for v in ("box60", "box90", "box120", "psf", "blk60", "blk90", "blk120")]
    specs += [dict(MAIN, excl=e) for e in (1, 2, 3)]
    specs += [dict(MAIN, vf=f) for f in (0.5, 0.75, 0.9)]
    specs += [dict(MAIN, window=300, variant=v) for v in ("box90", "box120", "psf")]
    rows, cache = [], {}
    for sp in specs:
        key = tuple(sorted(sp.items()))
        if key in cache:
            continue
        cache[key] = 1
        try:
            df = engine.contrasts(year, sp["window"], excl=sp["excl"], variant=sp["variant"],
                                  vf_min=sp["vf"], classes=list(FOCUS), table=table)
        except Exception as exc:  # variant absent (block variants only in pilot)
            if "not found" in str(exc).lower() or "binder" in str(exc).lower():
                continue
            raise
        df = df[np.isfinite(df["det"])]
        for cls, g in df.groupby("focal_class"):
            if g["label"].nunique() < MIN_UNITS:
                rows.append({"year": year, "year_sensor": YEAR_SENSOR[year], "focal_class": cls,
                             **sp, "det": np.nan, "n_polygons": g["label"].nunique(),
                             "n_scenes": g["scene_id"].nunique(), "n_obs": len(g),
                             "note": "insufficient units"})
                continue
            point, reps, nu, ns, no = class_stats(g, ["det"], B)
            lo, hi = pr.ci95(reps)
            rows.append({"year": year, "year_sensor": YEAR_SENSOR[year], "focal_class": cls, **sp,
                         "det": point[0], "det_lo": lo[0], "det_hi": hi[0], "n_polygons": nu,
                         "n_scenes": ns, "n_obs": no,
                         "valid_fraction_mean": float(np.nan), "note": ""})
    return rows


def profile_dep_rows(year, table, B, windows=(150, 300)):
    prof_rows, dep_rows = [], []
    for w in windows:
        kr = pr.k_ref(w)
        df = engine.contrasts(year, w, classes=list(FOCUS), table=table)
        cols = [f"c{k}" for k in range(kr)]
        for cls, g in df.groupby("focal_class"):
            if g["label"].nunique() < MIN_UNITS:
                continue
            point, reps, nu, ns, no = class_stats(g, cols, B, seed=1)
            lo, hi = pr.ci95(reps)
            for k in range(kr):
                prof_rows.append({"year": year, "year_sensor": YEAR_SENSOR[year], "focal_class": cls,
                                  "window": w, "ring": k, "distance_m": pr.RING_CENTRES_M[k],
                                  "contrast": point[k], "lo": lo[k], "hi": hi[k],
                                  "n_obs_ring": int(np.isfinite(g[f"c{k}"]).sum())})
            dep = pr.dep_from_replicates(point, reps)
            thr = {}
            for t in THRESHOLDS:
                d, c = pr.threshold_depth(point, 0.0, t)
                thr[f"dep_thr_{t}"] = d
                thr[f"dep_thr_{t}_censored"] = c
            dep_rows.append({"year": year, "year_sensor": YEAR_SENSOR[year], "focal_class": cls,
                             "window": w, "max_distance_m": pr.RING_CENTRES_M[kr - 1],
                             "n_polygons": nu, "n_scenes": ns, "n_obs": no, **dep, **thr})
    return prof_rows, dep_rows


def dtec_rows(year, table, B, root=None, window=150, excl=0):
    df = engine.interface_contrasts(year, window, excl=excl, root=root, table=table)
    df = df[df["focal_class"].isin(FOCUS) & df["nb_class"].isin(FOCUS)]
    df["iface"] = df["label"].astype(str) + "_" + df["nb_class"].astype(str)
    rows = []
    for (a, b), g in df.groupby(["focal_class", "nb_class"]):
        n_if = g["iface"].nunique()
        base = {"year": year, "year_sensor": YEAR_SENSOR[year], "focal_class": a, "nb_class": b,
                "window": window, "excl": excl, "n_interfaces": n_if,
                "n_scenes": g["scene_id"].nunique(), "n_obs": len(g),
                "contact_m_median": float(np.median(g.drop_duplicates("iface")["contact_faces"]) * 30)}
        if n_if < MIN_UNITS:
            rows.append({**base, "note": "insufficient interfaces"})
            continue
        u = _codes(g["iface"]); s = _codes(g["scene_id"])
        out = dict(base)
        for wname, wt in (("unweighted", None), ("sqrt_contact", np.sqrt(g["contact_faces"].to_numpy(float))),
                          ("valid_contact", g["nv_edge"].to_numpy(float))):
            vals = g["dtec"].to_numpy(float)
            if wt is None:
                point, reps = pr.pigeonhole_means(u, s, vals, u.max() + 1, s.max() + 1, B=B, seed=2)
            else:  # weighted mean = ratio of crossed-bootstrapped weighted sums
                pn, rn = pr.pigeonhole_means(u, s, vals * wt, u.max() + 1, s.max() + 1, B=B, seed=2)
                pd_, rd = pr.pigeonhole_means(u, s, wt, u.max() + 1, s.max() + 1, B=B, seed=2)
                point, reps = pn / pd_, rn / rd
            lo, hi = pr.ci95(reps)
            out[f"dtec_{wname}"] = point[0]
            out[f"dtec_{wname}_lo"] = lo[0]
            out[f"dtec_{wname}_hi"] = hi[0]
            if wname == "unweighted":
                out["_reps"] = reps[:, 0]
        rows.append(out)
    res = pd.DataFrame(rows)
    # asymmetry A,B = DTEC(A<-B) - DTEC(B<-A); CI from the difference of independent replicate sets
    asym = []
    for _, r in res.iterrows():
        a, b = r["focal_class"], r["nb_class"]
        if a >= b:
            continue
        o = res[(res["focal_class"] == b) & (res["nb_class"] == a)]
        if o.empty or not isinstance(r.get("_reps"), np.ndarray) or not isinstance(o.iloc[0].get("_reps"), np.ndarray):
            continue
        d = r["_reps"] - o.iloc[0]["_reps"]
        asym.append({"year": year, "class_a": a, "class_b": b,
                     "asymmetry": r["dtec_unweighted"] - o.iloc[0]["dtec_unweighted"],
                     "asym_lo": np.nanpercentile(d, 2.5), "asym_hi": np.nanpercentile(d, 97.5),
                     "n_if_ab": r["n_interfaces"], "n_if_ba": o.iloc[0]["n_interfaces"]})
    return res.drop(columns=["_reps"], errors="ignore"), pd.DataFrame(asym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pilot", "full"], required=True)
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    out = OUT / a.source
    out.mkdir(parents=True, exist_ok=True)
    sweep, prof, dep, dt, asy = [], [], [], [], []
    for y in a.year:
        t0 = time.time()
        root = DATA / "pilot" / str(y) if a.source == "pilot" else None
        table = DATA / "tables" / f"{a.source}_polygon_ring_{y}.parquet"
        if a.rebuild or not table.exists():
            engine.build_polygon_ring(y, root=root, out=table)
        sweep += sweep_rows(y, table, a.B)
        p, d = profile_dep_rows(y, table, a.B)
        prof += p; dep += d
        r, s = dtec_rows(y, table, a.B, root=root)
        dt.append(r); asy.append(s)
        print(f"{y}: done in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(sweep).to_csv(out / "det_sweep.csv", index=False)
    pd.DataFrame(prof).to_csv(out / "profiles.csv", index=False)
    pd.DataFrame(dep).to_csv(out / "dep.csv", index=False)
    pd.concat(dt).to_csv(out / "dtec_pairs.csv", index=False)
    pd.concat(asy).to_csv(out / "dtec_asymmetry.csv", index=False)


if __name__ == "__main__":
    main()
