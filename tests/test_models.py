from pathlib import Path

import numpy as np
import pytest

from methods.level6_cnn import _infer, load_model
from workflow.pipeline import SadFsrcnnPipeline

ROOT = Path(__file__).resolve().parents[1]


def _image(h=48, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("ckpt", ["fsrcnn_div2k.ckpt", "fsrcnn_vimeo_best.ckpt"])
def test_committed_checkpoints_upscale_x4(ckpt):
    model = load_model(ROOT / "train" / ckpt, device="cpu")
    out = _infer(_image(), model, 4, "cpu")
    assert out.shape == (192, 256, 3)
    assert out.dtype == np.uint8


def test_sad_pipeline_recycles_static_blocks():
    model = load_model(ROOT / "train" / "fsrcnn_div2k.ckpt", device="cpu")
    pipeline = SadFsrcnnPipeline(model, block_size=16, sad_threshold=1500.0)
    # Smooth frame: its SR output, downscaled again, matches the LR input closely
    ramp = np.linspace(40, 200, 64, dtype=np.float32)
    frame = np.repeat(np.tile(ramp, (48, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    _, hr_y, first = pipeline.process(frame)
    _, _, second = pipeline.process(frame, prev_hr_y=hr_y)
    assert first["recycled_pct"] == 0.0  # no previous frame: every block goes through FSRCNN
    assert second["recycled_pct"] > first["recycled_pct"]  # same frame again: blocks are reused
