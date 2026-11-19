import sys
from pathlib import Path

import numpy as np
import pytest
from skimage import io
from skimage.draw import disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porosity.analysis import AnalysisConfig, analyse_image  # noqa: E402


@pytest.fixture
def disks_image(tmp_path):
    """Bright background with 9 dark disks of radius 20 px, fully inside the frame."""
    img = np.full((300, 300), 200, np.uint8)
    truth = np.zeros_like(img, bool)
    for r in (60, 150, 240):
        for c in (60, 150, 240):
            rr, cc = disk((r, c), 20)
            img[rr, cc] = 40
            truth[rr, cc] = True
    path = tmp_path / "disks.png"
    io.imsave(path, img, check_contrast=False)
    return path, truth


def test_void_fraction_and_count(disks_image):
    path, truth = disks_image
    res = analyse_image(path, AnalysisConfig(um_per_px=1.0, flat_field=False))
    assert res.n_pores == 9
    assert res.void_fraction == pytest.approx(truth.mean(), abs=0.01)


def test_calibration_scales_diameter(disks_image):
    path, _ = disks_image
    d1 = analyse_image(path, AnalysisConfig(um_per_px=1.0, flat_field=False)).mean_equiv_diameter_um
    d2 = analyse_image(path, AnalysisConfig(um_per_px=2.5, flat_field=False)).mean_equiv_diameter_um
    assert d1 == pytest.approx(40, rel=0.05)
    assert d2 == pytest.approx(2.5 * d1, rel=1e-6)


def test_circular_pores_have_high_circularity(disks_image):
    path, _ = disks_image
    res = analyse_image(path, AnalysisConfig(flat_field=False))
    assert res.mean_circularity > 0.85
    assert res.mean_aspect_ratio < 1.1


def test_bright_pores_option(tmp_path, disks_image):
    path, truth = disks_image
    inv = 255 - io.imread(path)
    p2 = tmp_path / "inv.png"
    io.imsave(p2, inv, check_contrast=False)
    res = analyse_image(p2, AnalysisConfig(bright_pores=True, flat_field=False))
    assert res.void_fraction == pytest.approx(truth.mean(), abs=0.01)


def test_border_pores_excluded_from_size_stats(tmp_path):
    img = np.full((200, 200), 200, np.uint8)
    rr, cc = disk((100, 100), 20); img[rr, cc] = 40
    rr, cc = disk((0, 100), 30, shape=img.shape); img[rr, cc] = 40   # cut by the top edge
    p = tmp_path / "border.png"
    io.imsave(p, img, check_contrast=False)
    res = analyse_image(p, AnalysisConfig(flat_field=False))
    assert res.n_pores == 2 and res.n_pores_measured == 1


def test_synthetic_sample_matches_ground_truth():
    import json
    meta = ROOT / "data" / "sample_images" / "printed_grid_01.json"
    if not meta.exists():
        pytest.skip("sample images not generated")
    gt = json.loads(meta.read_text())
    res = analyse_image(meta.with_suffix(".png"), AnalysisConfig(um_per_px=gt["um_per_px"]))
    assert abs(res.void_fraction - gt["void_fraction"]) < 0.04
    assert res.mean_equiv_diameter_um == pytest.approx(gt["mean_equiv_diameter_um"], rel=0.05)
