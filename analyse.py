"""
Tissue Scaffold Porosity Analyser - command-line interface.

Analyse one image or a whole folder of scaffold micrographs (SEM, optical,
confocal or micro-CT slices) and report 2D void fraction, pore size
distribution, pore shape and wall thickness.

Examples:
    # single image, 1.5 um per pixel
    python analyse.py data/sample_images/salt_leached_01.png --um-per-px 1.5

    # whole folder; calibration read from matching <name>.json if present
    python analyse.py data/sample_images --out results

    # real SEM image with a 60 px information bar, split touching pores
    python analyse.py my_scaffold.tif --um-per-px 0.82 --crop-bottom 60 --split-touching
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from porosity.analysis import AnalysisConfig, analyse_image, load_greyscale  # noqa: E402
from porosity.report import save_outputs  # noqa: E402

IMAGE_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.suffix.lower() in IMAGE_EXT and not p.stem.endswith(("_mask", "_pore_mask", "_analysis")))


def calibration_for(img: Path, default: float | None) -> float:
    if default is not None:
        return default
    meta = img.with_suffix(".json")
    if meta.exists():
        return float(json.loads(meta.read_text()).get("um_per_px", 1.0))
    print(f"  ! no calibration for {img.name}; reporting in pixels (use --um-per-px)")
    return 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="image file or folder")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--um-per-px", type=float, default=None, help="pixel size in micrometres")
    p.add_argument("--threshold", default="otsu", help="otsu | li | triangle | yen | value in 0-1")
    p.add_argument("--bright-pores", action="store_true", help="pores are brighter than walls")
    p.add_argument("--denoise", default="gaussian", choices=["gaussian", "median", "none"])
    p.add_argument("--sigma", type=float, default=1.0, help="denoising strength (px)")
    p.add_argument("--no-flat-field", action="store_true")
    p.add_argument("--min-pore-area", type=int, default=30, help="px; smaller objects are noise")
    p.add_argument("--split-touching", action="store_true", help="watershed-split merged pores")
    p.add_argument("--crop-bottom", type=int, default=0, help="px to remove (SEM data bar)")
    p.add_argument("--include-border-pores", action="store_true")
    a = p.parse_args()

    try:
        thr: str | float = float(a.threshold)
    except ValueError:
        thr = a.threshold

    images = find_images(a.input)
    if not images:
        sys.exit(f"No images found in {a.input}")

    rows = []
    for img in images:
        cfg = AnalysisConfig(
            um_per_px=calibration_for(img, a.um_per_px), threshold=thr, bright_pores=a.bright_pores,
            denoise=a.denoise, denoise_sigma=a.sigma, flat_field=not a.no_flat_field,
            min_pore_area_px=a.min_pore_area, split_touching=a.split_touching,
            crop_bottom_px=a.crop_bottom, exclude_border_pores=not a.include_border_pores,
        )
        res = analyse_image(img, cfg)
        save_outputs(res, load_greyscale(img, cfg.crop_bottom_px), a.out)
        rows.append(res.summary())
        print(f"{res.image:<24} void fraction {100 * res.void_fraction:5.1f} %   "
              f"pores {res.n_pores_measured:4d}   mean d {res.mean_equiv_diameter_um:7.1f} um   "
              f"wall {res.mean_wall_thickness_um:6.1f} um")

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(a.out / "batch_summary.csv", index=False)
    print(f"\nResults written to {a.out.resolve()}")


if __name__ == "__main__":
    main()
