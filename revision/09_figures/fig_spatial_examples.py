"""Supplementary figure: spatial examples of the edge geometry and TST (Reviewer 2, Q4.f).

  python fig_spatial_examples.py

For one forest, one soybean and one urban polygon (2024, max_ring >= 8, away from the state limit),
a 4.8 km window around the polygon shows: (a) MapBiomas C10.1 classes, (b) distance rings of the
focal polygon (0-30 m ... >= 300 m inside its boundary), (c) neighbouring class assigned to the
focal polygon's pixels, (d) Landsat TST of a cloud-free scene, with the focal polygon outlined.
The scene is the cloud-free scene (valid fraction > 0.999 over the polygon) closest to mid-winter.
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
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from rasterio.windows import Window  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tedge import engine, fetch  # noqa: E402
from tedge.aggregate import decode_tst  # noqa: E402
from tedge.paths import DATA, OUT, class_raster, geometry_dir, label_raster  # noqa: E402
from fig_results import INK, INK2, NAMES, SEQ, save  # noqa: E402

YEAR = 2024
EXAMPLES = [(3, 120312), (39, 2112613), (24, 2024857)]
HALF = 80  # pixels -> 4.8 km window
# MapBiomas legend colours (official palette) for the classes shown
MB = {3: "#1f8d49", 9: "#7a5900", 11: "#519799", 12: "#d6bc74", 15: "#edde8e", 20: "#db7093",
      21: "#ffefc3", 24: "#d4271e", 33: "#2532e4", 39: "#f5b3c8", 41: "#f54ca9", 46: "#d68fe2"}


def clear_scene(label):
    sql = f"""SELECT filename, sum(n_valid)::DOUBLE / sum(n_total) AS vf
              FROM read_parquet('{engine.scene_glob(YEAR)}', filename=true)
              WHERE label = {label} GROUP BY filename HAVING vf > 0.999"""
    d = engine.con(1.0).execute(sql).df()
    d["scene_id"] = d["filename"].map(lambda f: Path(f).stem)
    d["doy"] = pd.to_datetime(d["scene_id"].str.slice(17, 25)).dt.dayofyear
    d = d.iloc[(d["doy"] - 196).abs().argsort()]  # closest to mid-July (dry season)
    return d["scene_id"].iloc[0]


def window_for(label):
    p = pd.read_parquet(geometry_dir(YEAR) / "polygons.parquet")
    r = p[p["label"] == label].iloc[0]
    with rasterio.open(label_raster(YEAR)) as s:
        row, col = s.index(r["centroid_x"], r["centroid_y"])
    return Window(col - HALF, row - HALF, 2 * HALF, 2 * HALF)


def main():
    meta = {}
    for line in open(DATA / f"scene_metadata_{YEAR}.jsonl", encoding="utf-8"):
        m = json.loads(line)
        meta[m["scene_id"]] = m
    fetch.init()
    fig, axs = plt.subplots(3, 4, figsize=(7.4, 6.4))
    shown = set()
    for i, (cls, label) in enumerate(EXAMPLES):
        win = window_for(label)
        with rasterio.open(label_raster(YEAR)) as s:
            lab = s.read(1, window=win)
            T, wkt = s.transform, s.crs.to_wkt()
        with rasterio.open(class_raster(YEAR)) as s:
            cl = s.read(1, window=win)
        with rasterio.open(geometry_dir(YEAR) / "ring.tif") as s:
            ring = s.read(1, window=win)
        with rasterio.open(geometry_dir(YEAR) / "nbclass.tif") as s:
            nb = s.read(1, window=win)
        sid = clear_scene(label)
        img = fetch.scene_image(meta[sid]["image_id"], YEAR)
        arr = fetch.fetch_tile(img, wkt, T, int(win.row_off), int(win.col_off), int(win.height),
                               int(win.width))
        tst = decode_tst(arr["tst"])[0]
        focal = lab == label
        classes = sorted(c for c in np.unique(cl) if c in MB)
        cmap = ListedColormap([MB[c] for c in classes])
        idx = np.searchsorted(classes, cl).astype(float)
        idx[~np.isin(cl, classes)] = np.nan
        ax = axs[i, 0]
        ax.imshow(idx, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, interpolation="nearest")
        ax.set_ylabel(f"{NAMES[cls]}\n{sid[17:25]}", fontsize=7.5, color=INK)
        rr = np.where(focal, ring.astype(float), np.nan)
        rr[rr == 255] = np.nan
        im1 = axs[i, 1].imshow(rr, cmap=SEQ.reversed(), vmin=0, vmax=10, interpolation="nearest")
        nn = np.where(focal & (ring <= 2), nb.astype(float), np.nan)
        nn[~np.isin(nn, classes)] = np.nan
        axs[i, 2].imshow(np.where(np.isfinite(nn), np.searchsorted(classes, np.nan_to_num(nn)), np.nan),
                         cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, interpolation="nearest")
        lo, hi = np.nanpercentile(tst, [2, 98])
        im3 = axs[i, 3].imshow(tst, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        for ax in axs[i]:
            ax.contour(focal, levels=[0.5], colors="white" if ax is axs[i, 3] else INK, linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        cb = fig.colorbar(im3, ax=axs[i, 3], fraction=0.046, pad=0.02)
        cb.set_label("TST (°C)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
        shown.update(classes)
    for j, t in enumerate(["(a) Land cover", "(b) Ring inside polygon", "(c) Neighbour class",
                           "(d) Landsat 8/9 TST"]):
        axs[0, j].set_title(t, fontsize=7.5, loc="left")
    cl_sorted = sorted(shown)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=MB[c]) for c in cl_sorted],
               [NAMES[c] for c in cl_sorted], loc="lower center", ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.07), frameon=False, handlelength=0.9, columnspacing=1.0)
    cb = fig.colorbar(im1, ax=axs[:, 1], fraction=0.03, pad=0.01, location="bottom")
    cb.set_label("Ring (0 = 0–30 m, 10 = ≥ 300 m)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.text(0.01, 0.0, "Windows of 4.8 × 4.8 km; outline: focal polygon. Scene: cloud-free over the "
             "polygon, closest to mid-July.", fontsize=6.5, color=INK2)
    save(fig, "figS3_spatial_examples", OUT / "figures")


if __name__ == "__main__":
    main()
