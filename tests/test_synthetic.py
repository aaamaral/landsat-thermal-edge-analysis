"""Minimal tests on synthetic data (no Earth Engine, no downloads, a few seconds).

  python tests/test_synthetic.py          # or: python -m pytest tests

A synthetic landscape of forest squares (class 3) bordered by pasture (class 15) above and by
mosaic (class 21) below is given a known edge signal: inside forest, TST = 25 C + A(neighbour)
* max(0, 1 - d / 90 m), with A = 3.0 C next to pasture and 1.0 C next to mosaic, plus scene
offsets, pixel noise and QA-invalid pixels. The pipeline (edge geometry -> per-scene zone
aggregation -> DuckDB tables -> contrasts) must recover DET, a DEP breakpoint at 90 m, the two
directional contrasts DTEC(3 <- 15) and DTEC(3 <- 21), and finite crossed-bootstrap intervals.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gee_python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from tedge import aggregate, engine, geometry, paths, profiles  # noqa: E402

AMP = {15: 3.0, 21: 1.0}
BREAK_M = 90.0
N_SCENES = 8
YEAR = 2024


def _block(offset):
    """200 x 200 block: forest square (label 1), pasture above (2), mosaic below (3)."""
    lab = np.full((200, 200), 2, np.int32)
    lab[100:] = 3
    lab[40:160, 40:160] = 1
    cls = np.select([lab == 1, lab == 2], [3, 15], 21).astype(np.int16)
    return lab + offset, cls


def synthetic_landscape():
    rows = []
    for i in range(2):
        cols = [_block(3 * (2 * i + j)) for j in range(2)]
        rows.append((np.hstack([c[0] for c in cols]), np.hstack([c[1] for c in cols])))
    return np.vstack([r[0] for r in rows]), np.vstack([r[1] for r in rows])


def edge_signal(g, cls):
    d = 30.0 * g["ring"].astype(float) + 15.0
    amp = np.vectorize(lambda c: AMP.get(int(c), 0.0))(g["nb_class"])
    forest = cls == 3
    return np.where(forest, amp * np.clip(1.0 - d / BREAK_M, 0, None), 0.0)


def build_synthetic_database(tmp: Path):
    labels, classes = synthetic_landscape()
    g = geometry.edge_geometry(labels, classes)
    signal = edge_signal(g, classes)
    base = np.select([classes == 3, classes == 15], [25.0, 32.0], 30.0)
    nb = g["nb_class"].astype(np.int64)
    keys = np.where((labels > 0) & (nb >= 0),
                    aggregate.zone_keys(labels, np.clip(nb, 0, None), g["ring"]), -1)
    scenes = tmp / "scenes"
    scenes.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for s in range(N_SCENES):
        t = base + signal + rng.normal(0, 3) + rng.normal(0, 0.3, labels.shape)
        raw = np.round(t * aggregate.TST_SCALE).astype(np.int16)
        raw[rng.random(labels.shape) < 0.05] = aggregate.INVALID  # QA-masked pixels
        tst, in_fp, valid = aggregate.decode_tst(raw)
        variants = aggregate.support_variants(tst, valid, native_m=100)
        ndvi = np.where(classes == 3, 0.8, 0.4) + rng.normal(0, 0.02, labels.shape)
        ndmi = ndvi - 0.3
        agg = aggregate.aggregate_tile(keys, in_fp, valid, variants, ndvi=ndvi, ndmi=ndmi)
        aggregate.finalize(pd.DataFrame(agg)).to_parquet(scenes / f"S{s:02d}_T1.parquet", index=False)
    gdir = tmp / "geometry"
    gdir.mkdir()
    inside = labels > 0
    poly = pd.DataFrame({"label": labels[inside], "focal_class": classes[inside],
                         "ring": g["ring"][inside]}).groupby("label").agg(
        focal_class=("focal_class", "first"), n_px=("ring", "size"), max_ring=("ring", "max"))
    poly.reset_index().to_parquet(gdir / "polygons.parquet", index=False)
    pd.DataFrame(geometry.contact_faces(labels, classes)).to_parquet(gdir / "contacts.parquet",
                                                                    index=False)
    forest0 = inside & (classes == 3) & (g["ring"] == 0)
    truth = {"det": float(signal[forest0].mean())}
    for c in AMP:
        truth[c] = float(signal[forest0 & (g["nb_class"] == c)].mean())
    return scenes, gdir, truth


def _with_synthetic_engine(fn):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        scenes, gdir, truth = build_synthetic_database(tmp)
        saved = engine.DATA, engine.geometry_dir
        engine.DATA, engine.geometry_dir = tmp, (lambda year: gdir)
        try:
            table = engine.build_polygon_ring(YEAR, root=scenes, out=tmp / "polygon_ring.parquet")
            return fn(scenes, table, truth)
        finally:
            engine.DATA, engine.geometry_dir = saved


def _crossed(df, value_cols, unit="label", B=200):
    u = df[unit].astype("category").cat.codes.to_numpy()
    s = df["scene_id"].astype("category").cat.codes.to_numpy()
    return profiles.pigeonhole_means(u, s, df[value_cols].to_numpy(), u.max() + 1, s.max() + 1,
                                     B=B, seed=0)


# ------------------------------------------------------------------ DET, DEP, DTEC, bootstrap

def test_det_dep_dtec_on_synthetic_landscape():
    def check(scenes, table, truth):
        df = engine.contrasts(YEAR, 300, classes=[3], table=table)
        assert df["label"].nunique() == 4 and df["scene_id"].nunique() == N_SCENES
        point, reps = _crossed(df, ["det"])
        lo, hi = profiles.ci95(reps)
        assert abs(point[0] - truth["det"]) < 0.05, (point, truth)
        assert lo[0] <= point[0] <= hi[0]

        ring_cols = [f"c{k}" for k in range(10)]
        prof, prof_reps = _crossed(df, ring_cols)
        dep = profiles.dep_from_replicates(prof, prof_reps)
        assert 75 <= dep["dep_m"] <= 105 and not dep["dep_censored"], dep
        assert dep["dep_lo"] <= dep["dep_m"] <= dep["dep_hi"]

        dt = engine.interface_contrasts(YEAR, 300, root=scenes, table=table)
        dt = dt[dt["focal_class"] == 3]
        dt["iface"] = dt["label"].astype(str) + "_" + dt["nb_class"].astype(str)
        est = {}
        for c in AMP:
            sub = dt[dt["nb_class"] == c]
            p, r = _crossed(sub, ["dtec"], unit="iface")
            est[c] = p[0]
            assert abs(p[0] - truth[c]) < 0.05, (c, p, truth)
            assert np.isfinite(profiles.ci95(r)).all()
        assert est[15] > est[21]  # directional asymmetry is recovered
    _with_synthetic_engine(check)


def test_crossed_bootstrap_properties():
    rng = np.random.default_rng(0)
    nu, ns = 30, 12
    u, s = np.meshgrid(np.arange(nu), np.arange(ns), indexing="ij")
    keep = rng.random(u.shape) < 0.7
    u, s = u[keep], s[keep]
    v = rng.normal(0, 1, nu)[u] + rng.normal(0, 1, ns)[s] + rng.normal(0, 0.1, u.size)
    p1, r1 = profiles.pigeonhole_means(u, s, v, nu, ns, B=300, seed=5)
    p2, r2 = profiles.pigeonhole_means(u, s, v, nu, ns, B=300, seed=5)
    assert np.isclose(p1[0], v.mean())  # point estimate = mean over observed cells
    assert np.array_equal(r1, r2, equal_nan=True)  # fixed seed -> reproducible intervals
    lo, hi = profiles.ci95(r1)
    naive_se = v.std(ddof=1) / np.sqrt(v.size)
    # scene and polygon clustering widen the interval well beyond an i.i.d. record bootstrap
    assert (hi[0] - lo[0]) > 3 * 2 * 1.96 * naive_se


def test_broken_stick_recovers_breakpoint_and_censoring():
    d = profiles.RING_CENTRES_M
    y = 2.0 * np.clip(1 - d / 120.0, 0, None)
    delta, c, b, sse, cens = profiles.broken_stick(d, y)
    assert abs(delta - 120) < 3 and not cens
    *_, cens2 = profiles.broken_stick(d, 2.0 - d / 400.0)  # no plateau inside the window
    assert cens2


# ------------------------------------------------------------------ reference grid

def test_reference_grid_alignment():
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    import prepare_reference_grid as prg

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        x, y = prg.STUDY_X0 + 300_000.0 + 7.0, prg.STUDY_Y0 - 200_000.0 - 11.0
        gdf = gpd.GeoDataFrame(geometry=[box(x, y - 2000, x + 3000, y)], crs=prg.STUDY_CRS_WKT)
        gpkg = tmp / "boundary.gpkg"
        gdf.to_crs("EPSG:4674").to_file(gpkg)  # supplied in geographic SIRGAS 2000
        info = prg.build(gpkg, tmp / "grid.tif", extent="boundary", pad=2)
        with rasterio.open(tmp / "grid.tif") as src:
            T = src.transform
            assert (T.a, T.e) == (30.0, -30.0)
            assert abs(((T.c - prg.STUDY_X0) / 30) - round((T.c - prg.STUDY_X0) / 30)) < 1e-6
            assert abs(((prg.STUDY_Y0 - T.f) / 30) - round((prg.STUDY_Y0 - T.f) / 30)) < 1e-6
            assert src.crs == rasterio.crs.CRS.from_wkt(prg.STUDY_CRS_WKT)
            a = src.read(1)
        assert a.max() == 1 and a[0].sum() == 0  # padding outside the boundary
        assert abs(info["pixels_inside"] - 3000 * 2000 / 900) <= 3000 / 30 + 2000 / 30 + 4


def test_missing_boundary_message():
    env = {k: v for k, v in os.environ.items() if k != "PARANA_GPKG"}
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_reference_grid.py")],
                           cwd=d, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "PARANA_GPKG" in r.stderr, r.stderr


def test_missing_ee_project_message():
    from tedge import fetch
    saved = os.environ.pop("EE_PROJECT", None)
    try:
        fetch.init()
    except RuntimeError as exc:
        assert "EE_PROJECT is not set" in str(exc)
    else:
        raise AssertionError("fetch.init() must fail without EE_PROJECT")
    finally:
        if saved is not None:
            os.environ["EE_PROJECT"] = saved


def test_missing_inputs_message():
    if (paths.DADOS / f"parana_albers_{YEAR}.tif").exists() or paths.REFERENCE_GRID.exists():
        return  # inputs present in this working copy
    for fn in (paths.template_raster, paths.class_raster, paths.label_raster):
        try:
            fn(YEAR)
        except FileNotFoundError as exc:
            assert "Run" in str(exc) or "Build" in str(exc)
        else:
            raise AssertionError(fn.__name__)


# ------------------------------------------------------------------ repository contents

def test_scene_manifest():
    with open(paths.MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 922
    used = [r for r in rows if float(r["cloud_cover"]) <= 50]
    by_year = {y: sum(r["year"] == str(y) for r in used) for y in paths.YEARS}
    assert by_year == {1985: 109, 2002: 162, 2024: 401}, by_year


def test_no_local_paths_or_secrets():
    pat = re.compile(r"(?i)([a-z]:[\\/](users|doutorado|tese)|/home/\w|/Users/\w|"
                     r"private_key|client_secret|api[_-]?key\s*=|refresh_token|password\s*=)")
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".sh", ".md", ".txt", ".cfg", ".toml", ".yml"} \
                and ".venv" not in p.parts and p.name != Path(__file__).name:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, hits


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
