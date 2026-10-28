"""Edge orientation: does the near-edge contrast depend on the direction the edge faces?

  python orientation.py --year 1985 2002 2024 --B 1000

Sector = quadrant (1 N, 2 E, 3 S, 4 W) of the outward normal of the polygon boundary, from the
per-scene ``*_sector.parquet`` files (rings 0-2). Two estimands per class x year_sensor x sector:
  edge_contrast    T(ring 0, sector) - polygon reference band (W = 150 m), as DET but by sector
  sector_anomaly   T(ring 0, sector) - T(ring 0, all sectors) of the same polygon and scene:
                   removes polygon/scene level, isolates orientation
CIs: crossed (polygon x scene) pigeonhole bootstrap. Output: revision/outputs/full/orientation.csv
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
from tedge.paths import DATA, OUT, YEAR_SENSOR, geometry_dir, scene_dir  # noqa: E402
from core_estimates import FOCUS, MIN_UNITS, class_stats  # noqa: E402

SECTORS = {1: "N", 2: "E", 3: "S", 4: "W"}
WINDOW = 150
N_MIN = 3


def sector_table(year, table):
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    sec = (scene_dir(year) / "*_sector.parquet").as_posix()
    refs = pr.ref_rings(WINDOW)
    sql = f"""
    WITH s AS (
      SELECT regexp_replace(regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1), '_sector$', '') AS scene_id,
             label, sector, t30_mean AS t, n_valid
      FROM read_parquet('{sec}', filename=true)
      WHERE ring = 0 AND sector BETWEEN 1 AND 4 AND n_valid >= {N_MIN} AND isfinite(t30_mean)),
    e AS (SELECT scene_id, label, sum(t * n_valid) / sum(n_valid) AS t_edge FROM s GROUP BY ALL),
    r AS (SELECT scene_id, label, sum(t30 * n_valid) / sum(n_valid) AS ref
          FROM read_parquet('{table}')
          WHERE ring IN ({','.join(map(str, refs))}) AND n_valid >= {N_MIN} AND isfinite(t30)
          GROUP BY ALL)
    SELECT s.scene_id, s.label, s.sector, p.focal_class,
           s.t - r.ref AS edge_contrast, s.t - e.t_edge AS sector_anomaly
    FROM s JOIN e USING (scene_id, label) JOIN r USING (scene_id, label)
    JOIN read_parquet('{poly}') p USING (label)
    WHERE p.max_ring >= {pr.k_ref(WINDOW)}
      AND p.focal_class IN ({','.join(map(str, FOCUS))})"""
    return engine.con().execute(sql).df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    rows = []
    for y in a.year:
        table = DATA / "tables" / f"full_polygon_ring_{y}.parquet"
        df = sector_table(y, table.as_posix())
        for (cls, sec), g in df.groupby(["focal_class", "sector"]):
            base = {"year": y, "year_sensor": YEAR_SENSOR[y], "focal_class": cls,
                    "sector": SECTORS[sec], "n_polygons": g["label"].nunique(),
                    "n_scenes": g["scene_id"].nunique(), "n_obs": len(g)}
            if base["n_polygons"] < MIN_UNITS:
                rows.append({**base, "note": "insufficient units"})
                continue
            point, reps, *_ = class_stats(g, ["edge_contrast", "sector_anomaly"], a.B, seed=3)
            lo, hi = pr.ci95(reps)
            rows.append({**base, "edge_contrast": point[0], "edge_lo": lo[0], "edge_hi": hi[0],
                         "sector_anomaly": point[1], "anom_lo": lo[1], "anom_hi": hi[1], "note": ""})
        print(f"{y}: orientation done", flush=True)
    out = OUT / "full"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "orientation.csv", index=False)


if __name__ == "__main__":
    main()
