"""Check that the deposited aggregated tables reproduce the published point estimates.

Self-contained (pandas + pyarrow). Run from the dataset root or pass its path:
  python verify_deposit.py [DATASET_DIR]

Point estimates are plain means over the observed (scene, polygon) or (scene, interface) cells, as
in the crossed bootstrap of the article (the bootstrap only adds the intervals):
  DET (W = 150 and 300 m)  mean of c0               vs results/S_main_results.csv  (det)
  profile (W = 300 m)      mean of c0..c9 per ring  vs results/profiles.csv        (contrast)
  DTEC (W = 150 m, ring 0) mean of dtec per pair    vs results/dtec_pairs.csv      (dtec_unweighted)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
TOL = 1e-4
YEARS = sorted(int(p.stem.split("_")[-1]) for p in (D / "contrasts").glob("contrasts_W150_*.parquet"))


def compare(name, mine, ref, keys, col_m, col_r):
    m = mine.merge(ref[keys + [col_r]], on=keys, how="inner")
    m = m[m[col_r].notna()]
    diff = (m[col_m] - m[col_r]).abs()
    bad = int((diff > TOL).sum())
    print(f"{name:<24} {len(m):>5} values compared, max |diff| = {diff.max():.2e}, "
          f"{'OK' if bad == 0 else f'{bad} MISMATCHES'}")
    return bad


def main():
    res = pd.read_csv(D / "results" / "S_main_results.csv")
    prof = pd.read_csv(D / "results" / "profiles.csv")
    pairs = pd.read_csv(D / "results" / "dtec_pairs.csv")
    bad = 0
    for y in YEARS:
        print(f"[{y}]")
        for w in (150, 300):
            c = pd.read_parquet(D / "contrasts" / f"contrasts_W{w}_{y}.parquet")
            det = c.groupby("focal_class", as_index=False)["c0"].mean().assign(year=y, window=w)
            bad += compare(f"DET W{w}", det, res, ["year", "focal_class", "window"], "c0", "det")
            if w == 300:
                ks = [k for k in c.columns if k.startswith("c") and k[1:].isdigit()]
                p = (c.groupby("focal_class")[ks].mean().reset_index()
                     .melt(id_vars="focal_class", var_name="ring", value_name="mean"))
                p["ring"] = p["ring"].str[1:].astype(int)
                p = p.assign(year=y, window=w)
                bad += compare("profile W300 (rings)", p, prof, ["year", "focal_class", "window", "ring"],
                               "mean", "contrast")
        t = pd.read_parquet(D / "dtec" / f"dtec_W150_ring0_{y}.parquet")
        g = t.groupby(["focal_class", "nb_class"], as_index=False)["dtec"].mean().assign(year=y)
        bad += compare("DTEC W150 ring 0", g, pairs, ["year", "focal_class", "nb_class"], "dtec",
                       "dtec_unweighted")
    print("ALL OK" if bad == 0 else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
