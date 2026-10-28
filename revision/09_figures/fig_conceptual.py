"""Conceptual figure: what is measured (DET, DEP, DTEC) and what is not inferred.

Synthetic, illustrative curves only (no data). Output: revision/outputs/figures/fig01_conceptual.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
CA, CB = "#2e7d4f", "#c0832a"  # class A (e.g. forest) / class B (e.g. cropland)

x = np.linspace(-300, 300, 601)  # m; negative = inside A, positive = inside B
tA = 24.0 + 1.4 * np.exp(-(-x) / 70.0)          # A interior 24 C, warmer near the edge
tB = 31.0 - 1.1 * np.exp(-(x) / 50.0)           # B interior 31 C, cooler near the edge
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 6.4), height_ratios=[1.0, 2.2],
                             gridspec_kw={"hspace": 0.08}, sharex=True)

# --- top: map strip -----------------------------------------------------------------------
ax.axvspan(-300, 0, color=CA, alpha=0.18, lw=0)
ax.axvspan(0, 300, color=CB, alpha=0.18, lw=0)
for k in range(10):
    ax.axvline(-30 * (k + 1), color=GRID, lw=0.6)
    ax.axvline(30 * (k + 1), color=GRID, lw=0.6)
ax.axvspan(-50, 50, color="#8250df", alpha=0.12, lw=0, hatch="///")
ax.axvline(0, color=INK, lw=1.6)
ax.text(-150, 0.55, "Class A (focal)", ha="center", color=CA, fontsize=10, weight="bold")
ax.text(150, 0.55, "Class B (neighbour)", ha="center", color=CB, fontsize=10, weight="bold")
ax.text(0, 0.12, "mapped (cartographic) edge", ha="center", fontsize=8, color=INK,
        bbox=dict(fc="white", ec="none", pad=1))
ax.text(0, 0.88, "zone exposed to PSF / mixed pixels\n(≈ native thermal footprint)", ha="center",
        fontsize=7.5, color="#6639ba", va="top")
ax.text(-285, 0.12, "30 m distance rings", fontsize=7.5, color=MUTED)
ax.set_ylim(0, 1); ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# --- bottom: observed LST profiles ---------------------------------------------------------
bx.plot(x[x <= 0], tA[x <= 0], color=CA, lw=2.2, label="LST profile in A (A←B)")
bx.plot(x[x >= 0], tB[x >= 0], color=CB, lw=2.2, label="LST profile in B (B←A)")
bx.axvline(0, color=INK, lw=1.2)
bx.axvspan(-50, 50, color="#8250df", alpha=0.08, lw=0)
# interior reference bands (window W = 150 m -> rings 5-6)
for sgn, c, t in ((-1, CA, 24.0), (1, CB, 31.0)):
    lo, hi = sorted((sgn * 150, sgn * 210))
    bx.axvspan(lo, hi, color=c, alpha=0.10, lw=0)
    bx.hlines(t, lo, hi, color=c, lw=3)
bx.text(-180, 23.3, "interior reference\n(window W)", ha="center", fontsize=7.5, color=CA)
bx.text(180, 31.35, "interior reference\n(window W)", ha="center", fontsize=7.5, color=CB)
# DET / DTEC A<-B
bx.annotate("", xy=(-15, 24.0), xytext=(-15, tA[285]), arrowprops=dict(arrowstyle="<->", color=CA, lw=1.4))
bx.text(-290, 25.9, "DET$_A$ / DTEC(A←B)\n= T(edge ring) − T(reference)", ha="left", fontsize=8, color=CA)
bx.hlines(24.0, -150, -15, color=CA, lw=0.8, ls=":")
# DTEC B<-A
bx.annotate("", xy=(15, 31.0), xytext=(15, tB[315]), arrowprops=dict(arrowstyle="<->", color=CB, lw=1.4))
bx.text(22, 30.0, "DTEC(B←A)", ha="left", fontsize=8, color=CB)
bx.hlines(31.0, 15, 150, color=CB, lw=0.8, ls=":")
# DEP
dep = -105
bx.axvline(dep, color=CA, lw=1, ls="--")
bx.text(dep - 4, 27.8, "DEP$_A$: breakpoint\nof the profile", ha="right", fontsize=8, color=CA)
bx.annotate("", xy=(dep, 27.4), xytext=(0, 27.4), arrowprops=dict(arrowstyle="<->", color=CA, lw=1))
bx.text(-255, 28.9, "If no plateau inside the window:\nDEP ≥ maximum distance (right-censored)",
        fontsize=7.5, color=MUTED)
bx.text(110, 26.0, "Asymmetry(A,B) = DTEC(A←B) − DTEC(B←A)\n"
        "Measured: radiometric LST contrasts.\nNot inferred: heat transfer between classes.",
        fontsize=7.5, color=INK, va="top")
bx.set_xlim(-300, 300)
bx.set_ylim(22.5, 32)
bx.set_xlabel("Distance from the mapped edge (m)   ← inside A  |  inside B →")
bx.set_ylabel("Land surface temperature (°C)")
bx.set_xticks([-300, -210, -150, -90, -30, 0, 30, 90, 150, 210, 300])
bx.set_xticklabels([300, 210, 150, 90, 30, 0, 30, 90, 150, 210, 300])
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.legend(loc="lower right", frameon=False, fontsize=8)
fig.savefig(OUT / "fig01_conceptual.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig01_conceptual.pdf", bbox_inches="tight")
print("ok")
