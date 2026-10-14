"""SAD + FSRCNN pipeline: batched, non-overlapping 16×16 blocks."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import FSRCNN
from .sad import compute_sad_mask

BLOCK_SIZE = 16
SCALE = 4


class SadFsrcnnPipeline:
    def __init__(
        self,
        model: FSRCNN,
        block_size: int = BLOCK_SIZE,
        sad_threshold: float = 10000.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.block_size = block_size
        self.threshold = sad_threshold
        self.device = device
        self.scale = model.scale

    def process(
        self, lr_bgr: np.ndarray, prev_hr_y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        bs = self.block_size
        sc = self.scale
        bhr = bs * sc

        # ── 1. YCrCb split ──────────────────────────────────────────────────
        ycrcb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
        lr_y, lr_cr, lr_cb = cv2.split(ycrcb)
        H, W = lr_y.shape
        H_hr, W_hr = H * sc, W * sc
        n_bh, n_bw = H // bs, W // bs

        # ── 2. Global bicubic (baseline for static blocks) ───────────────────
        bic_y = cv2.resize(lr_y.astype(np.float32), (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)

        # ── 3. SAD mask ──────────────────────────────────────────────────────
        if prev_hr_y is None:
            mask = np.ones((n_bh, n_bw), dtype=bool)
        else:
            prev_lr_y = cv2.resize(prev_hr_y, (W, H), interpolation=cv2.INTER_AREA)
            mask = compute_sad_mask(lr_y, prev_lr_y, bs, self.threshold)

        positions = [(by, bx) for by in range(n_bh) for bx in range(n_bw)]
        active_pos = [(by, bx) for by, bx in positions if mask[by, bx]]
        n_active = len(active_pos)
        n_total = n_bh * n_bw

        # ── 4. HR Y mosaic: bicubic by default, FSRCNN on active blocks ──────
        hr_y = bic_y[: n_bh * bhr, : n_bw * bhr].copy()

        if n_active > 0:
            # Bilateral pre-filter: softens compression artefacts before FSRCNN
            lr_y_clean = cv2.bilateralFilter(lr_y, d=5, sigmaColor=15, sigmaSpace=15)
            patches = np.stack(
                [
                    lr_y_clean[by * bs : (by + 1) * bs, bx * bs : (bx + 1) * bs].astype(np.float32)
                    / 255.0
                    for by, bx in active_pos
                ]
            )
            tensor = torch.from_numpy(patches).unsqueeze(1).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
            out_np = out.squeeze(1).cpu().numpy() * 255.0
            for i, (by, bx) in enumerate(active_pos):
                y_hr, x_hr = by * bhr, bx * bhr
                hr_y[y_hr : y_hr + bhr, x_hr : x_hr + bhr] = out_np[i]

        # ── 5. Leftover rows when H is not a multiple of bs ──────────────────
        if n_bh * bhr < H_hr:
            bic_y[: n_bh * bhr, : n_bw * bhr] = hr_y
            hr_y = bic_y

        hr_y_u8 = np.clip(hr_y, 0, 255).astype(np.uint8)

        # ── 6. Bicubic Cr/Cb, merge back to BGR ──────────────────────────────
        hr_cr = cv2.resize(lr_cr, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(lr_cb, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)
        hr_bgr = cv2.cvtColor(cv2.merge([hr_y_u8, hr_cr, hr_cb]), cv2.COLOR_YCrCb2BGR)

        stats = {
            "n_active": n_active,
            "n_static": n_total - n_active,
            "n_total": n_total,
            "recycled_pct": 100.0 * (n_total - n_active) / n_total,
        }
        return hr_bgr, hr_y_u8, stats
