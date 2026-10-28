"""DuckDB-backed construction of analysis tables from the scene x zone parquet files.

polygon_ring   (scene_id, label, ring) with neighbour classes pooled, excluding zones whose
               nearest boundary is the study-area limit (nb_class 0, artificial clipping edge).
contrasts      wide (scene_id, label): c0..c{k_ref-1} = T(ring k) - T(reference band of window W)
interfaces     (scene_id, label, nb_class): near-edge TST of the interface (ring e) minus the
               polygon's reference band -> DTEC(A <- B).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .paths import DATA, geometry_dir, scene_dir
from .profiles import ref_rings, k_ref

VARIANTS = ("t30", "box60", "box90", "box120", "psf")


def con(memory_gb: float = 1.5):
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{memory_gb}GB'")
    c.execute("SET threads=3")
    tmp = DATA / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    c.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return c


def scene_glob(year: int, root: Path | None = None) -> str:
    root = Path(root or scene_dir(year))
    if not any(root.glob("*_T1.parquet")):
        raise FileNotFoundError(f"No scene files (*_T1.parquet) in {root}. Run the scene "
                                "aggregation first (revision/02_edge_aggregation/run_full.sh).")
    return (root / "*_T1.parquet").as_posix()


def build_polygon_ring(year: int, root=None, out=None, extra=("blk60", "blk90", "blk120")):
    """Materialise the polygon x ring x scene table for a year (streaming COPY)."""
    out = Path(out or DATA / "tables" / f"polygon_ring_{year}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    c = con()
    cols = c.execute(f"DESCRIBE SELECT * FROM read_parquet('{scene_glob(year, root)}')").df()["column_name"].tolist()
    vs = [v for v in VARIANTS + tuple(extra) if f"{v}_mean" in cols]
    means = ",\n".join(
        f"sum({v}_mean * n_valid) FILTER (WHERE isfinite({v}_mean)) / "
        f"nullif(sum(n_valid) FILTER (WHERE isfinite({v}_mean)), 0) AS {v}" for v in vs)
    idx = ",\n".join(
        f"sum({v}_mean * n_valid) FILTER (WHERE isfinite({v}_mean)) / "
        f"nullif(sum(n_valid) FILTER (WHERE isfinite({v}_mean)), 0) AS {v}" for v in ("ndvi", "ndmi"))
    sql = f"""
    COPY (
      SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS scene_id,
             label, ring,
             sum(n_total)::UINTEGER AS n_total, sum(n_valid)::UINTEGER AS n_valid,
             {means},
             {idx}
      FROM read_parquet('{scene_glob(year, root)}', filename=true)
      WHERE nb_class <> 0
      GROUP BY ALL
    ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
    c.execute(sql)
    return out


def contrasts(year, window_m, excl=0, variant="t30", vf_min=0.0, n_min=3, classes=None,
              table=None):
    """Wide contrast table for a window: one row per (scene, polygon)."""
    table = Path(table or DATA / "tables" / f"polygon_ring_{year}.parquet").as_posix()
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    kr = k_ref(window_m)
    refs = ref_rings(window_m)
    ok = f"(n_valid >= {n_min} AND n_valid::DOUBLE / n_total >= {vf_min} AND isfinite({variant}))"
    ring_cols = ",\n".join(
        f"max(CASE WHEN ring = {k} AND {ok} THEN {variant} END) AS r{k}" for k in range(kr))
    ref = (f"sum(CASE WHEN ring IN ({','.join(map(str, refs))}) AND {ok} THEN {variant} * n_valid END) / "
           f"nullif(sum(CASE WHEN ring IN ({','.join(map(str, refs))}) AND {ok} THEN n_valid END), 0)")
    cls = "" if classes is None else f"AND p.focal_class IN ({','.join(map(str, classes))})"
    sql = f"""
    WITH w AS (
      SELECT t.scene_id, t.label, {ring_cols}, {ref} AS ref
      FROM read_parquet('{table}') t
      JOIN read_parquet('{poly}') p USING (label)
      WHERE p.max_ring >= {kr} {cls}
      GROUP BY t.scene_id, t.label
    )
    SELECT w.*, p.focal_class, p.n_px, p.max_ring FROM w JOIN read_parquet('{poly}') p USING (label)
    WHERE ref IS NOT NULL"""
    df = con().execute(sql).df()
    for k in range(kr):
        df[f"c{k}"] = (df[f"r{k}"] - df["ref"]).astype("float32")
    df["det"] = df[f"c{excl}"] if excl < kr else np.nan
    return df


def interface_contrasts(year, window_m, excl=0, variant="t30", vf_min=0.0, n_min=3, root=None,
                        table=None):
    """DTEC per (scene, polygon, neighbour class): ring ``excl`` of the interface - reference."""
    table = Path(table or DATA / "tables" / f"polygon_ring_{year}.parquet").as_posix()
    poly = (geometry_dir(year) / "polygons.parquet").as_posix()
    contacts = (geometry_dir(year) / "contacts.parquet").as_posix()
    kr = k_ref(window_m)
    refs = ref_rings(window_m)
    ok = f"(n_valid >= {n_min} AND n_valid::DOUBLE / n_total >= {vf_min} AND isfinite({variant}_mean))"
    okr = ok.replace(f"{variant}_mean", variant)
    sql = f"""
    WITH ref AS (
      SELECT scene_id, label,
             sum({variant} * n_valid) / sum(n_valid) AS ref
      FROM read_parquet('{table}')
      WHERE ring IN ({','.join(map(str, refs))}) AND {okr}
      GROUP BY ALL
    ), edge AS (
      SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS scene_id,
             label, nb_class, {variant}_mean AS t_edge, n_valid AS nv_edge, n_total AS nt_edge
      FROM read_parquet('{scene_glob(year, root)}', filename=true)
      WHERE ring = {excl} AND nb_class <> 0 AND {ok}
    )
    SELECT e.scene_id, e.label, e.nb_class, p.focal_class, e.t_edge - r.ref AS dtec,
           e.nv_edge, e.nt_edge, c.faces AS contact_faces, c.normal_e, c.normal_n
    FROM edge e JOIN ref r USING (scene_id, label)
    JOIN read_parquet('{poly}') p USING (label)
    LEFT JOIN read_parquet('{contacts}') c USING (label, nb_class)
    WHERE p.max_ring >= {kr}"""
    return con().execute(sql).df()


def support_table(df, unit="label"):
    return pd.Series({"n_units": df[unit].nunique(), "n_scenes": df["scene_id"].nunique(),
                      "n_obs": len(df)})
