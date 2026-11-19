"""
Validate the analyser against ground truth for the synthetic scaffold images.

For every image with a matching <name>.json and <name>_mask.png, this script
compares the measured void fraction, pore count and mean pore diameter with
the known values, and scores pixel-level segmentation with the Dice
coefficient and IoU. It also compares threshold methods.

Usage:
    python validate.py --images data/sample_images --out results/validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from skimage import io

sys.path.insert(0, str(Path(__file__).parent / "src"))
from porosity.analysis import AnalysisConfig, analyse_image  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ARCH_COLORS = {"salt_leached": "#2a78d6", "freeze_dried": "#eb6834", "printed_grid": "#1baf7a"}


def dice_iou(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return 2 * inter / (pred.sum() + truth.sum()), inter / union


def run(images: Path, method: str, sigma: float) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(images.glob("*.json")):
        gt = json.loads(meta_path.read_text())
        img = images / gt["image"]
        truth = io.imread(images / f"{img.stem}_mask.png") > 127
        cfg = AnalysisConfig(um_per_px=gt["um_per_px"], threshold=method, denoise_sigma=sigma)
        res = analyse_image(img, cfg)
        dice, iou = dice_iou(res.pore_mask, truth)
        rows.append({
            "image": gt["image"], "architecture": gt["architecture"], "threshold": method, "sigma": sigma,
            "void_fraction_true": gt["void_fraction"], "void_fraction_measured": res.void_fraction,
            "void_fraction_abs_error_pct_pts": 100 * (res.void_fraction - gt["void_fraction"]),
            "n_pores_true": gt["n_pores_interior"], "n_pores_measured": res.n_pores_measured,
            "mean_d_true_um": gt["mean_equiv_diameter_um"], "mean_d_measured_um": res.mean_equiv_diameter_um,
            "mean_d_rel_error_pct": 100 * (res.mean_equiv_diameter_um / gt["mean_equiv_diameter_um"] - 1),
            "dice": dice, "iou": iou,
        })
    return pd.DataFrame(rows)


def plot_agreement(df: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                         "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    specs = [("void_fraction", 100, "2D void fraction (%)"), ("mean_d", 1, "Mean equivalent diameter (µm)")]
    for ax, (key, k, label) in zip(axes, specs):
        t, m = df[f"{key}_true" if key == "void_fraction" else f"{key}_true_um"] * k, \
               df[f"{key}_measured" if key == "void_fraction" else f"{key}_measured_um"] * k
        lo, hi = min(t.min(), m.min()) * 0.8, max(t.max(), m.max()) * 1.15
        ax.plot([lo, hi], [lo, hi], ls="--", color="#8a8985", lw=1, label="Perfect agreement")
        for arch, c in ARCH_COLORS.items():
            s = df["architecture"] == arch
            ax.scatter(t[s], m[s], s=70, color=c, edgecolor=SURFACE, linewidth=1.5, label=arch.replace("_", " "))
        if key == "mean_d":
            ax.set_xscale("log"); ax.set_yscale("log")
            ticks = [20, 50, 100, 200, 300]
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
                axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Ground truth: {label}"); ax.set_ylabel(f"Measured: {label}")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_title("Void fraction", loc="left"); axes[1].set_title("Pore size", loc="left")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/sample_images"))
    p.add_argument("--out", type=Path, default=Path("results/validation"))
    p.add_argument("--sigma", type=float, default=1.0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    all_runs = pd.concat([run(a.images, m, a.sigma) for m in ["otsu", "li", "triangle", "yen"]])
    all_runs.round(4).to_csv(a.out / "validation_all_methods.csv", index=False)
    method_summary = all_runs.groupby("threshold").agg(
        mean_dice=("dice", "mean"),
        mean_abs_void_error_pct_pts=("void_fraction_abs_error_pct_pts", lambda x: x.abs().mean()),
        mean_abs_size_error_pct=("mean_d_rel_error_pct", lambda x: x.abs().mean()),
    ).sort_values("mean_dice", ascending=False)
    method_summary.round(3).to_csv(a.out / "threshold_method_comparison.csv")
    print("Threshold method comparison:\n", method_summary.round(3), "\n")

    default = all_runs[all_runs["threshold"] == "otsu"]
    plot_agreement(default, a.out / "validation_agreement.png")
    cols = ["image", "void_fraction_true", "void_fraction_measured", "n_pores_true", "n_pores_measured",
            "mean_d_true_um", "mean_d_measured_um", "dice"]
    print("Default settings (Otsu):\n", default[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
