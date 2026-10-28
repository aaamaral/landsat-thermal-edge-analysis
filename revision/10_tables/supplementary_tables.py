"""Supplementary tables of the revised manuscript (CSV, one file per table) + scene-season figure.

  python supplementary_tables.py

S_scenes.csv          every scene: id, year_sensor, date, time, sun elevation/azimuth, cloud cover,
                      footprint and valid pixels, valid fraction
S_support.csv         polygons / area eligible and usable by class x window x year_sensor
S_valid_fraction.csv  DET (W = 150 m) for minimum zone valid fractions 0, 0.5, 0.75, 0.9
S_dtec.csv            DTEC(A <- B) for every pair and condition, 3 contact weightings, CIs, support
S_main_results.csv    DET, DEP (breakpoint, CI, censoring) per class x condition, W = 150 / 300 m
figS_scene_seasons    number of scenes per month and year_sensor
Reads only revision/outputs/full, revision/outputs/audit and revision/data metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(ROOT / "revision" / "09_figures"))
from tedge.paths import DATA, OUT, YEAR_SENSOR, YEARS  # noqa: E402
from fig_results import NAMES, YS, save  # noqa: E402

TAB = OUT / "tables"


def scenes():
    meta = []
    for y in YEARS:
        for line in open(DATA / f"scene_metadata_{y}.jsonl", encoding="utf-8"):
            meta.append(json.loads(line))
    m = pd.DataFrame(meta).drop_duplicates("scene_id", keep="last")
    log = pd.read_csv(DATA / "scene_run_log.csv").drop_duplicates("scene_id", keep="last")
    s = m.merge(log[["scene_id", "footprint_px", "valid_px"]], on="scene_id", how="left")
    s["year_sensor"] = s["year"].map(YEAR_SENSOR)
    s["valid_fraction"] = s["valid_px"] / s["footprint_px"]
    s = s.rename(columns={"DATE_ACQUIRED": "date", "SCENE_CENTER_TIME": "time_utc",
                          "SUN_ELEVATION": "sun_elevation", "SUN_AZIMUTH": "sun_azimuth",
                          "CLOUD_COVER": "cloud_cover", "SPACECRAFT_ID": "spacecraft",
                          "WRS_PATH": "path", "WRS_ROW": "row"})
    cols = ["scene_id", "year_sensor", "spacecraft", "path", "row", "date", "time_utc",
            "sun_elevation", "sun_azimuth", "cloud_cover", "footprint_px", "valid_px", "valid_fraction"]
    s = s[cols].sort_values(["year_sensor", "date", "path", "row"])
    s.to_csv(TAB / "S_scenes.csv", index=False)
    # seasonal distribution figure
    s["month"] = pd.to_datetime(s["date"]).dt.month
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    w = 0.27
    for j, (y, (lab, col, *_)) in enumerate(YS.items()):
        c = s[s["year_sensor"] == YEAR_SENSOR[y]].groupby("month").size().reindex(range(1, 13), fill_value=0)
        ax.bar(np.arange(1, 13) + (j - 1) * w, c.values, width=w - 0.02, color=col, label=lab)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Scenes")
    ax.axvspan(3.5, 9.5, color="#f0efec", zorder=0, lw=0)
    ax.text(6.5, ax.get_ylim()[1] * 0.92, "dry season (Apr–Sep)", ha="center", fontsize=7, color="#5f5e5a")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "figS_scene_seasons", OUT / "figures")
    return len(s)


def support():
    r = pd.read_csv(OUT / "audit" / "retention_window.csv")
    r["class_name"] = r["focal_class"].map(NAMES)
    r.sort_values(["year", "window_m", "focal_class"]).to_csv(TAB / "S_support.csv", index=False)


def valid_fraction():
    d = pd.read_csv(OUT / "full" / "det_sweep.csv")
    d = d[(d["window"] == 150) & (d["variant"] == "t30") & (d["excl"] == 0)]
    d = d[["year_sensor", "focal_class", "vf", "det", "det_lo", "det_hi", "n_polygons", "n_scenes", "n_obs"]]
    d.assign(class_name=d["focal_class"].map(NAMES)).sort_values(
        ["year_sensor", "focal_class", "vf"]).to_csv(TAB / "S_valid_fraction.csv", index=False)


def dtec():
    d = pd.read_csv(OUT / "full" / "dtec_pairs.csv")
    d["focal_name"], d["neighbour_name"] = d["focal_class"].map(NAMES), d["nb_class"].map(NAMES)
    d.sort_values(["year", "focal_class", "nb_class"]).to_csv(TAB / "S_dtec.csv", index=False)


def main_results():
    s = pd.read_csv(OUT / "full" / "det_sweep.csv")
    s = s[(s["variant"] == "t30") & (s["excl"] == 0) & (s["vf"] == 0) & s["window"].isin([150, 300])]
    dep = pd.read_csv(OUT / "full" / "dep.csv")
    k = ["year", "focal_class", "window"]
    t = s[k + ["year_sensor", "det", "det_lo", "det_hi", "n_polygons", "n_scenes", "n_obs"]].merge(
        dep[k + ["dep_m", "dep_lo", "dep_hi", "dep_censored", "p_censored", "max_distance_m"]],
        on=k, how="left")
    t["class_name"] = t["focal_class"].map(NAMES)
    t.sort_values(["window", "focal_class", "year"]).to_csv(TAB / "S_main_results.csv", index=False)


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    print("scenes:", scenes())
    support()
    valid_fraction()
    dtec()
    main_results()
    print("tables written to", TAB)
