import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.level1_interpolation import run_all, upscale  # noqa: E402
from pipeline.metrics import psnr  # noqa: E402


def _image(h=16, w=24):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", ["bicubic", "bilinear", "lanczos", "nearest"])
def test_upscale_shape(method):
    out = upscale(_image(), 4, method)
    assert out.shape == (64, 96, 3)
    assert out.dtype == np.uint8


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        upscale(_image(), 2, "magic")


def test_run_all_returns_every_method():
    assert set(run_all(_image(), 2)) == {"bicubic", "bilinear", "lanczos", "nearest"}


def test_psnr_identical_images_is_infinite_and_noise_lowers_it():
    img = _image()
    noisy = np.clip(img.astype(int) + 10, 0, 255).astype(np.uint8)
    assert psnr(img, img) == float("inf")
    assert 20 < psnr(img, noisy) < 40
