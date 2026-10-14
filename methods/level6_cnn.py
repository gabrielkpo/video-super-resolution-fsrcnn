"""Level 6 — CNN super-resolution: FSRCNN (Dong et al., ECCV 2016).

Reference pre-trained checkpoint: train/fsrcnn_div2k.ckpt
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import DEVICE

DEFAULT_CKPT = Path(__file__).parent.parent / "train" / "fsrcnn_div2k.ckpt"


# ── Architecture ─────────────────────────────────────────────────────────────


class FSRCNN(nn.Module):
    """Fast Super-Resolution CNN — Dong et al., ECCV 2016.

    FSRCNN(d, s, m):
        d = LR feature dimension (extraction)
        s = shrunk dimension (shrinking/expanding)
        m = number of non-linear mapping layers
    """

    def __init__(self, scale: int = 4, d: int = 56, s: int = 12, m: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.feature = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(num_parameters=s),
        )
        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(num_parameters=s)]
        self.map = nn.Sequential(*mapping)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(num_parameters=d),
        )
        self.deconv = nn.ConvTranspose2d(
            d, 1, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        return self.deconv(x)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_model(ckpt_path: str | Path = DEFAULT_CKPT, device: str = DEVICE) -> FSRCNN:
    """Load a checkpoint, inferring d, s and m from the weights."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    d = state_dict["feature.0.weight"].shape[0]
    s = state_dict["shrink.0.weight"].shape[0]
    m = sum(1 for k in state_dict if k.startswith("map.") and k.endswith(".bias"))
    model = FSRCNN(scale=4, d=d, s=s, m=m)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ── Inference ────────────────────────────────────────────────────────────────


def upscale(
    img: np.ndarray, scale: int, ckpt_path: str | Path = DEFAULT_CKPT, device: str = DEVICE
) -> np.ndarray:
    """Level 6 standard interface: full-image colour SR.

    FSRCNN on the Y channel, bicubic on Cr/Cb.
    Same signature as the other levels (img, scale).
    """
    model = load_model(ckpt_path, device)
    return _infer(img, model, scale, device)


def _infer(
    img: np.ndarray, model: FSRCNN, scale: int | None = None, device: str = DEVICE
) -> np.ndarray:
    """Internal inference: FSRCNN on Y, bicubic on Cr/Cb."""
    model.eval().to(device)
    sc = scale or model.scale
    h, w = img.shape[:2]

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    t = torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_sr = np.clip(model(t).squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)

    cr_up = cv2.resize(cr, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)

    return cv2.cvtColor(cv2.merge([y_sr, cr_up, cb_up]), cv2.COLOR_YCrCb2BGR)
