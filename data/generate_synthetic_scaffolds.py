"""
Generate synthetic SEM-style micrographs of porous tissue scaffolds with exact
ground truth, for validating the porosity analyser.

Why synthetic?
    Validating an image-analysis pipeline needs images where the "right
    answer" is known exactly. Here every pore is drawn from a known geometry,
    so the true void fraction and pore-size distribution can be compared
    with what the analyser measures.

Three scaffold architectures are simulated:
    salt_leached   - porogen-leached sponge: polydisperse, roughly round,
                     irregular pores (log-normal sizes)
    freeze_dried   - ice-templated / freeze-cast: elongated pores aligned
                     along one direction
    printed_grid   - extrusion 3D-bioprinted lattice: regular square pores
                     between orthogonal struts

Imaging artefacts are added to mimic SEM: textured polymer walls, bright
edge (charging / edge-effect) rims, darker pore interiors with depth
shading, uneven illumination, blur and noise.

Outputs, per image, in --out:
    <name>.png          8-bit greyscale micrograph
    <name>_mask.png     ground-truth binary pore mask (255 = pore)
    <name>.json         ground truth: pixel size, void fraction, pore list

Usage:
    python data/generate_synthetic_scaffolds.py --out data/sample_images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import io, measure
from skimage.draw import ellipse, polygon

SIZE = 768


def _place_ellipses(rng, shape, radii_fn, orient_fn, target_vf, min_wall_px, max_tries=40000):
    """Random sequential addition of non-overlapping ellipses with a minimum wall gap."""
    mask = np.zeros(shape, bool)
    forbidden = np.zeros(shape, bool)   # existing pores dilated by the wall thickness
    pores = []
    tries = 0
    while mask.mean() < target_vf and tries < max_tries:
        tries += 1
        ra, rb = radii_fn()
        theta = orient_fn()
        r, c = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        rr, cc = ellipse(r, c, ra, rb, shape=shape, rotation=theta)
        if rr.size < 20 or forbidden[rr, cc].any():
            continue
        mask[rr, cc] = True
        rr2, cc2 = ellipse(r, c, ra + min_wall_px, rb + min_wall_px, shape=shape, rotation=theta)
        forbidden[rr2, cc2] = True
    return mask


def _roughen(mask, rng, amount=2.5, smooth=3.0):
    """Make pore outlines irregular by thresholding a noisy distance field."""
    sd = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    noise = ndi.gaussian_filter(rng.normal(0, 1, mask.shape), smooth)
    noise /= noise.std() + 1e-9
    rough = (sd + amount * noise) > 0
    # Keep pores separate: never let roughening bridge two original pores.
    lab, _ = ndi.label(mask)
    nearest = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    owner = lab[nearest[0], nearest[1]]
    boundary = ndi.morphological_gradient(owner, size=3) > 0
    return rough & ~boundary


def salt_leached(rng):
    def radii():
        r = rng.lognormal(np.log(18), 0.4)
        return r * rng.uniform(0.8, 1.0), r * rng.uniform(0.8, 1.0)
    m = _place_ellipses(rng, (SIZE, SIZE), radii, lambda: rng.uniform(0, np.pi), 0.52, 6)
    return _roughen(m, rng, 2.5)


def freeze_dried(rng):
    def radii():
        a = rng.lognormal(np.log(10), 0.3)
        return a, a * rng.uniform(2.5, 4.5)
    m = _place_ellipses(rng, (SIZE, SIZE), radii, lambda: rng.normal(0, 0.12), 0.45, 5)
    return _roughen(m, rng, 1.8)


def printed_grid(rng):
    m = np.zeros((SIZE, SIZE), bool)
    pitch, strut = 96, 34
    for i in range(-1, SIZE // pitch + 2):
        for j in range(-1, SIZE // pitch + 2):
            r0 = i * pitch + strut / 2 + rng.normal(0, 2)
            c0 = j * pitch + strut / 2 + rng.normal(0, 2)
            w = pitch - strut + rng.normal(0, 3)
            h = pitch - strut + rng.normal(0, 3)
            rr, cc = polygon([r0, r0, r0 + h, r0 + h], [c0, c0 + w, c0 + w, c0], shape=m.shape)
            m[rr, cc] = True
    return _roughen(m, rng, 1.2, 4)


ARCHITECTURES = {
    "salt_leached": (salt_leached, 1.5),   # um per pixel
    "freeze_dried": (freeze_dried, 1.0),
    "printed_grid": (printed_grid, 4.0),
}


def render_sem(mask: np.ndarray, rng) -> np.ndarray:
    """Turn a binary pore mask into an SEM-like greyscale micrograph."""
    h, w = mask.shape
    texture = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 1.5) * 18
    strut = 175 + texture
    depth = ndi.distance_transform_edt(mask)
    pore = 70 - 35 * np.tanh(depth / 8) + ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 2) * 8
    img = np.where(mask, pore, strut)
    rim = ndi.distance_transform_edt(~mask)
    img = img + 55 * np.exp(-rim / 1.8) * (~mask)              # bright SEM edge effect
    yy, xx = np.mgrid[0:h, 0:w]
    img *= 0.85 + 0.25 * (xx / w) * (0.6 + 0.4 * yy / h)       # uneven illumination
    img = ndi.gaussian_filter(img, 1.1)
    img += rng.normal(0, 9, img.shape)                         # detector noise
    return np.clip(img, 0, 255).astype(np.uint8)


def ground_truth(mask: np.ndarray, um_per_px: float) -> dict:
    lab = measure.label(mask, connectivity=1)
    props = measure.regionprops(lab)
    interior = [p for p in props
                if p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < mask.shape[0] and p.bbox[3] < mask.shape[1]]
    d = np.array([p.equivalent_diameter_area * um_per_px for p in interior])
    return {
        "um_per_px": um_per_px,
        "void_fraction": float(mask.mean()),
        "n_pores_total": len(props),
        "n_pores_interior": len(interior),
        "mean_equiv_diameter_um": float(d.mean()),
        "median_equiv_diameter_um": float(np.median(d)),
        "interior_pore_equiv_diameters_um": [round(float(x), 2) for x in d],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "sample_images")
    p.add_argument("--per-type", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for name, (fn, um) in ARCHITECTURES.items():
        for k in range(1, a.per_type + 1):
            mask = fn(rng)
            img = render_sem(mask, rng)
            stem = f"{name}_{k:02d}"
            io.imsave(a.out / f"{stem}.png", img, check_contrast=False)
            io.imsave(a.out / f"{stem}_mask.png", (mask * 255).astype(np.uint8), check_contrast=False)
            gt = ground_truth(mask, um)
            with open(a.out / f"{stem}.json", "w") as f:
                json.dump({"image": f"{stem}.png", "architecture": name, **gt}, f, indent=2)
            print(f"{stem}: void fraction {gt['void_fraction']:.3f}, "
                  f"{gt['n_pores_interior']} interior pores, mean d = {gt['mean_equiv_diameter_um']:.1f} um")


if __name__ == "__main__":
    main()
