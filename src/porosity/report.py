"""Figures and file outputs for porosity results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation

from .analysis import PorosityResult

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ACCENT, ACCENT2 = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def overlay(result: PorosityResult) -> np.ndarray:
    """Greyscale image with pore outlines (blue) and border-excluded pores (orange)."""
    rgb = np.dstack([result.processed] * 3)
    border_ids = set(result.pores.loc[result.pores["touches_border"], "pore_id"]) if len(result.pores) else set()
    border_lab = np.isin(result.labels, list(border_ids)) * result.labels
    inner_lab = result.labels * ~np.isin(result.labels, list(border_ids))
    rgb = segmentation.mark_boundaries(rgb, inner_lab, color=(0.165, 0.47, 0.84), mode="thick")
    rgb = segmentation.mark_boundaries(rgb, border_lab, color=(0.92, 0.41, 0.20), mode="thick")
    return rgb


def figure(result: PorosityResult, raw: np.ndarray, path: Path) -> None:
    fig = plt.figure(figsize=(16, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, 1.0], wspace=0.22)

    ax0 = fig.add_subplot(gs[0]); ax0.imshow(raw, cmap="gray"); ax0.set_title("Input micrograph", loc="left")
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(overlay(result)); ax1.set_title("Segmented pores", loc="left")
    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    _scale_bar(ax0, raw.shape, result.config["um_per_px"])

    ax2 = fig.add_subplot(gs[2])
    d = result.pores.loc[~result.pores["touches_border"], "equiv_diameter_um"] if len(result.pores) else []
    if len(d):
        ax2.hist(d, bins=min(30, max(8, len(d) // 8)), color=ACCENT, edgecolor=SURFACE, linewidth=1.5)
        ax2.axvline(result.mean_equiv_diameter_um, color=TEXT, lw=1.5, ls="--")
        ax2.text(result.mean_equiv_diameter_um, ax2.get_ylim()[1] * 0.97,
                 f"  mean {result.mean_equiv_diameter_um:.1f} µm", va="top", color=TEXT, fontsize=9)
    ax2.set_xlabel("Equivalent circular diameter (µm)")
    ax2.set_ylabel("Number of pores")
    ax2.set_title("Pore size distribution", loc="left")
    _clean(ax2)

    ax3 = fig.add_subplot(gs[3]); ax3.axis("off")
    rows = [
        ("2D void fraction", f"{100 * result.void_fraction:.1f} %"),
        ("Pores measured", f"{result.n_pores_measured} of {result.n_pores}"),
        ("Mean diameter", f"{result.mean_equiv_diameter_um:.1f} µm"),
        ("Median diameter", f"{result.median_equiv_diameter_um:.1f} µm"),
        ("D10 – D90", f"{result.d10_um:.0f} – {result.d90_um:.0f} µm"),
        ("Mean pore area", f"{result.mean_pore_area_um2:,.0f} µm²"),
        ("Mean circularity", f"{result.mean_circularity:.2f}"),
        ("Mean aspect ratio", f"{result.mean_aspect_ratio:.2f}"),
        ("Wall thickness", f"{result.mean_wall_thickness_um:.1f} µm"),
        ("Pore density", f"{result.pore_density_per_mm2:,.0f} /mm²"),
    ]
    ax3.text(0.08, 1.03, "Summary", transform=ax3.transAxes, weight="bold", fontsize=11)
    for i, (k, v) in enumerate(rows):
        y = 0.93 - i * 0.092
        ax3.text(0.08, y, k, color=MUTED, fontsize=9.5, transform=ax3.transAxes)
        ax3.text(1.0, y, v, color=TEXT, fontsize=9.5, ha="right", weight="bold", transform=ax3.transAxes)
    ax3.text(0.08, -0.04, "Blue outline: measured pore.  Orange: cut by image\nborder, counted in void fraction only.",
             color=MUTED, fontsize=8, transform=ax3.transAxes, va="top")

    fig.suptitle(result.image, x=0.01, ha="left", fontsize=12, weight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _scale_bar(ax, shape, um_per_px):
    h, w = shape[:2]
    target = w * um_per_px * 0.2
    nice = min([10, 20, 50, 100, 200, 250, 500, 1000, 2000], key=lambda x: abs(x - target))
    px = nice / um_per_px
    x0, y0 = w * 0.05, h * 0.93
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=4, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - h * 0.025, f"{nice} µm", color="white", ha="center", va="bottom",
            fontsize=9, weight="bold")


def save_outputs(result: PorosityResult, raw: np.ndarray, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image).stem
    paths = {
        "pores_csv": out_dir / f"{stem}_pores.csv",
        "summary_json": out_dir / f"{stem}_summary.json",
        "figure_png": out_dir / f"{stem}_analysis.png",
        "mask_png": out_dir / f"{stem}_pore_mask.png",
    }
    result.pores.round(3).to_csv(paths["pores_csv"], index=False)
    with open(paths["summary_json"], "w") as f:
        json.dump({**result.summary(), "config": result.config}, f, indent=2)
    plt.imsave(paths["mask_png"], result.pore_mask, cmap="gray")
    figure(result, raw, paths["figure_png"])
    return paths
