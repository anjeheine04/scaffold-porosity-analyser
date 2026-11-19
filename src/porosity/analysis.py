"""
Core image-analysis pipeline for 2D porosity of scaffold micrographs.

Pipeline
    1. Load image -> greyscale float in [0, 1]; optionally crop off an SEM
       information bar.
    2. Flat-field correction (divide by a heavily blurred background) to
       remove uneven illumination.
    3. Denoise (Gaussian or median).
    4. Global threshold (Otsu by default; Li, triangle, yen or a manual value
       are also available). Pores are assumed darker than the walls unless
       `bright_pores=True`.
    5. Morphological clean-up: remove specks below a minimum pore area and
       fill small holes in the walls.
    6. Optional watershed on the distance transform to split touching pores.
    7. Measure:
         * 2D void fraction = pore pixels / analysed pixels
         * per-pore area, equivalent circular diameter, Feret diameter,
           major/minor axes, aspect ratio, orientation, circularity, solidity
         * wall (strut) thickness from the medial axis of the solid phase
    Pores cut by the image border are counted for void fraction but excluded
    from size statistics, because their true size is unknown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, exposure, filters, io, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.util import img_as_float

THRESHOLDS = {
    "otsu": filters.threshold_otsu,
    "li": filters.threshold_li,
    "triangle": filters.threshold_triangle,
    "yen": filters.threshold_yen,
}


@dataclass
class AnalysisConfig:
    um_per_px: float = 1.0               # spatial calibration (from the scale bar)
    threshold: str | float = "otsu"      # method name or a manual value in [0, 1]
    bright_pores: bool = False           # True if pores appear brighter than the walls
    denoise: str = "gaussian"            # "gaussian", "median" or "none"
    denoise_sigma: float = 1.0
    flat_field: bool = True
    min_pore_area_px: int = 30
    fill_wall_holes_px: int = 30
    split_touching: bool = False         # watershed separation of merged pores
    watershed_min_distance_px: int = 7
    crop_bottom_px: int = 0              # strip SEM data bar
    exclude_border_pores: bool = True


@dataclass
class PorosityResult:
    image: str
    config: dict
    void_fraction: float
    threshold_value: float
    n_pores: int
    n_pores_measured: int
    mean_equiv_diameter_um: float
    median_equiv_diameter_um: float
    std_equiv_diameter_um: float
    d10_um: float
    d90_um: float
    mean_pore_area_um2: float
    mean_circularity: float
    mean_aspect_ratio: float
    mean_wall_thickness_um: float
    pore_density_per_mm2: float
    pores: pd.DataFrame = field(repr=False)
    pore_mask: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    processed: np.ndarray = field(repr=False)

    def summary(self) -> dict:
        skip = {"pores", "pore_mask", "labels", "processed", "config"}
        return {k: v for k, v in asdict(self).items() if k not in skip}


# ---------------------------------------------------------------- stages --

def load_greyscale(path: str | Path, crop_bottom_px: int = 0) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    img = img_as_float(img)
    if crop_bottom_px:
        img = img[:-crop_bottom_px]
    return img


def preprocess(img: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    out = img.astype(float)
    if cfg.flat_field:
        bg = ndi.gaussian_filter(out, sigma=max(out.shape) / 8)
        out = out / (bg + 1e-6)
    if cfg.denoise == "gaussian":
        out = filters.gaussian(out, sigma=cfg.denoise_sigma)
    elif cfg.denoise == "median":
        out = ndi.median_filter(out, size=int(2 * cfg.denoise_sigma + 1))
    return exposure.rescale_intensity(out, out_range=(0.0, 1.0))


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than `min_area` pixels (version-independent)."""
    if min_area <= 0:
        return mask
    lab, _ = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def segment_pores(processed: np.ndarray, cfg: AnalysisConfig) -> tuple[np.ndarray, float]:
    if isinstance(cfg.threshold, str):
        t = float(THRESHOLDS[cfg.threshold](processed))
    else:
        t = float(cfg.threshold)
    pores = processed > t if cfg.bright_pores else processed < t
    pores = _drop_small(pores, cfg.min_pore_area_px)
    pores = ~_drop_small(~pores, cfg.fill_wall_holes_px)  # fill specks in walls
    return pores, t


def label_pores(pores: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    if not cfg.split_touching:
        return measure.label(pores, connectivity=1)
    dist = ndi.distance_transform_edt(pores)
    peaks = peak_local_max(dist, min_distance=cfg.watershed_min_distance_px,
                           labels=measure.label(pores), exclude_border=False)
    markers = np.zeros_like(dist, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=pores)


def wall_thickness_px(pores: np.ndarray) -> float:
    """Mean strut thickness: 2 x distance-to-pore sampled on the solid phase's medial axis."""
    solid = ~pores
    if not solid.any() or not pores.any():
        return float("nan")
    skel, dist = morphology.medial_axis(solid, return_distance=True)
    # Ignore skeleton points dominated by the image edge.
    h, w = solid.shape
    yy, xx = np.nonzero(skel)
    edge = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx])
    d = dist[yy, xx]
    keep = edge > d
    return float(2 * d[keep].mean()) if keep.any() else float("nan")


def measure_pores(labels: np.ndarray, um: float) -> pd.DataFrame:
    h, w = labels.shape
    rows = []
    for p in measure.regionprops(labels):
        r0, c0, r1, c1 = p.bbox
        perim = p.perimeter if p.perimeter > 0 else np.nan
        rows.append({
            "pore_id": p.label,
            "area_um2": p.area * um ** 2,
            "equiv_diameter_um": p.equivalent_diameter_area * um,
            "feret_max_um": p.feret_diameter_max * um,
            "major_axis_um": p.axis_major_length * um,
            "minor_axis_um": p.axis_minor_length * um,
            "aspect_ratio": p.axis_major_length / max(p.axis_minor_length, 1e-6),
            "orientation_deg": float(np.degrees(p.orientation)),
            "circularity": float(min(4 * np.pi * p.area / perim ** 2, 1.0)) if perim == perim else np.nan,
            "solidity": p.solidity,
            "centroid_x_um": p.centroid[1] * um,
            "centroid_y_um": p.centroid[0] * um,
            "touches_border": bool(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main --

def analyse_image(path: str | Path, cfg: AnalysisConfig | None = None) -> PorosityResult:
    cfg = cfg or AnalysisConfig()
    raw = load_greyscale(path, cfg.crop_bottom_px)
    processed = preprocess(raw, cfg)
    pore_mask, t = segment_pores(processed, cfg)
    labels = label_pores(pore_mask, cfg)
    um = cfg.um_per_px
    pores = measure_pores(labels, um)

    measured = pores[~pores["touches_border"]] if (cfg.exclude_border_pores and len(pores)) else pores
    d = measured["equiv_diameter_um"].to_numpy() if len(measured) else np.array([np.nan])
    area_mm2 = pore_mask.size * (um / 1000) ** 2

    return PorosityResult(
        image=str(Path(path).name),
        config=asdict(cfg),
        void_fraction=float(pore_mask.mean()),
        threshold_value=t,
        n_pores=int(len(pores)),
        n_pores_measured=int(len(measured)),
        mean_equiv_diameter_um=float(np.nanmean(d)),
        median_equiv_diameter_um=float(np.nanmedian(d)),
        std_equiv_diameter_um=float(np.nanstd(d)),
        d10_um=float(np.nanpercentile(d, 10)),
        d90_um=float(np.nanpercentile(d, 90)),
        mean_pore_area_um2=float(measured["area_um2"].mean()) if len(measured) else float("nan"),
        mean_circularity=float(measured["circularity"].mean()) if len(measured) else float("nan"),
        mean_aspect_ratio=float(measured["aspect_ratio"].mean()) if len(measured) else float("nan"),
        mean_wall_thickness_um=wall_thickness_px(pore_mask) * um,
        pore_density_per_mm2=float(len(pores) / area_mm2),
        pores=pores,
        pore_mask=pore_mask,
        labels=labels,
        processed=processed,
    )
