"""Database audit and support/retention tables for the full scene x zone database.

  python audit_database.py --year 1985 2002 2024

Requires the polygon_ring tables built by 08_statistics/core_estimates.py
(revision/data/tables/full_polygon_ring_{year}.parquet).

Outputs (revision/outputs/audit/):
  integrity.csv          per year: duplicate keys, out-of-range values, labels absent from the
                         geometry, n_valid > n_total, scenes, rows
  retention_window.csv   per year x window x class: polygons / area eligible by geometry and with
                         >= 1 usable scene (valid edge ring + reference band), share of class area
  validity_by_ring.csv   per year x class x ring: pixel-wise valid fraction (cloud losses by depth)
  coverage_grid.csv      per year x 10 km cell: usable polygons and mean usable scenes per polygon
                         (W = 150 m) -> spatial coverage map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
from tedge import engine, profiles as pr  # noqa: E402
from tedge.paths import DATA, OUT, YEAR_SENSOR, geometry_dir  # noqa: E402

N_MIN = 3
GRID_M = 10_000


def integrity(year, c):
    g = engine.scene_glob(year)
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    q = lambda sql: c.execute(sql).fetchone()  # noqa: E731
    rows, keys = q(f"""SELECT count(*), count(DISTINCT (filename, label, nb_class, ring))
                       FROM read_parquet('{g}', filename=true)""")
    bad = q(f"""SELECT count(*) FILTER (WHERE n_valid > n_total),
                       count(*) FILTER (WHERE ring > 10),
                       count(*) FILTER (WHERE valid_fraction < 0 OR valid_fraction > 1),
                       count(*) FILTER (WHERE isfinite(t30_mean) AND (t30_mean < -20 OR t30_mean > 80)),
                       count(DISTINCT filename)
                FROM read_parquet('{g}', filename=true)""")
    orphan = q(f"""SELECT count(DISTINCT s.label) FROM read_parquet('{g}') s
                   ANTI JOIN read_parquet('{poly}') p USING (label)""")[0]
    return {"year": year, "year_sensor": YEAR_SENSOR[year], "rows": rows,
            "duplicate_keys": rows - keys, "n_valid_gt_total": bad[0], "ring_out_of_range": bad[1],
            "valid_fraction_out_of_range": bad[2], "tst_implausible_rows": bad[3],
            "scenes": bad[4], "labels_not_in_geometry": orphan}


def retention(year, c, table):
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    tot = c.execute(f"""SELECT focal_class, count(*) AS polygons_all, sum(n_px) * 0.09 AS area_ha_all
                        FROM read_parquet('{poly}') GROUP BY 1""").df()
    out = []
    for w in pr.WINDOWS_M:
        kr, refs = pr.k_ref(w), pr.ref_rings(w)
        ok = f"n_valid >= {N_MIN} AND isfinite(t30)"
        df = c.execute(f"""
          WITH u AS (
            SELECT label, scene_id,
                   bool_or(ring = 0 AND {ok}) AS edge_ok,
                   bool_or(ring IN ({','.join(map(str, refs))}) AND {ok}) AS ref_ok
            FROM read_parquet('{table}') GROUP BY ALL),
          s AS (SELECT label, count(*) FILTER (WHERE edge_ok AND ref_ok) AS usable_scenes
                FROM u GROUP BY label)
          SELECT p.focal_class,
                 count(*) AS polygons_geom, sum(p.n_px) * 0.09 AS area_ha_geom,
                 count(*) FILTER (WHERE coalesce(s.usable_scenes, 0) > 0) AS polygons_usable,
                 sum(p.n_px) FILTER (WHERE coalesce(s.usable_scenes, 0) > 0) * 0.09 AS area_ha_usable,
                 median(s.usable_scenes) AS median_usable_scenes,
                 avg(p.n_px) * 0.09 AS mean_polygon_ha
          FROM read_parquet('{poly}') p LEFT JOIN s USING (label)
          WHERE p.max_ring >= {kr} GROUP BY 1""").df()
        df["year"], df["year_sensor"], df["window_m"] = year, YEAR_SENSOR[year], w
        out.append(df)
    res = pd.concat(out).merge(tot, on="focal_class", how="left")
    res["share_polygons_usable"] = res["polygons_usable"] / res["polygons_all"]
    res["share_area_usable"] = res["area_ha_usable"] / res["area_ha_all"]
    return res


def validity_by_ring(year, c, table):
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    df = c.execute(f"""SELECT p.focal_class, t.ring, sum(t.n_valid) / sum(t.n_total) AS valid_fraction,
                              sum(t.n_total) AS px_scene_total
                       FROM read_parquet('{table}') t JOIN read_parquet('{poly}') p USING (label)
                       GROUP BY ALL ORDER BY 1, 2""").df()
    df["year"], df["year_sensor"] = year, YEAR_SENSOR[year]
    return df


def coverage(year, c, table, w=150):
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    refs = pr.ref_rings(w)
    ok = f"n_valid >= {N_MIN} AND isfinite(t30)"
    df = c.execute(f"""
      WITH u AS (SELECT label, scene_id, bool_or(ring = 0 AND {ok}) AND
                        bool_or(ring IN ({','.join(map(str, refs))}) AND {ok}) AS usable
                 FROM read_parquet('{table}') GROUP BY ALL),
      s AS (SELECT label, count(*) FILTER (WHERE usable) AS n FROM u GROUP BY label)
      SELECT floor(centroid_x / {GRID_M})::INT AS gx, floor(centroid_y / {GRID_M})::INT AS gy,
             count(*) AS polygons_geom, count(*) FILTER (WHERE s.n > 0) AS polygons_usable,
             avg(s.n) FILTER (WHERE s.n > 0) AS mean_usable_scenes
      FROM read_parquet('{poly}') p LEFT JOIN s USING (label)
      WHERE p.max_ring >= {pr.k_ref(w)} GROUP BY 1, 2""").df()
    df["x_center"] = (df["gx"] + 0.5) * GRID_M
    df["y_center"] = (df["gy"] + 0.5) * GRID_M
    df["year"], df["year_sensor"] = year, YEAR_SENSOR[year]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    a = ap.parse_args()
    out = OUT / "audit"
    out.mkdir(parents=True, exist_ok=True)
    res = {k: [] for k in ("integrity", "retention_window", "validity_by_ring", "coverage_grid")}
    for y in a.year:
        c = engine.con()
        table = (DATA / "tables" / f"full_polygon_ring_{y}.parquet").as_posix()
        res["integrity"].append(pd.DataFrame([integrity(y, c)]))
        res["retention_window"].append(retention(y, c, table))
        res["validity_by_ring"].append(validity_by_ring(y, c, table))
        res["coverage_grid"].append(coverage(y, c, table))
        print(f"{y}: audit done", flush=True)
    for k, v in res.items():
        pd.concat(v).to_csv(out / f"{k}.csv", index=False)


if __name__ == "__main__":
    main()
