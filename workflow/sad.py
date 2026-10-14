"""Temporal filtering by SAD on a fixed block grid."""

import numpy as np


def compute_sad_mask(
    y_curr: np.ndarray,
    y_prev: np.ndarray,
    block_size: int = 16,
    threshold: float = 1000.0,
) -> np.ndarray:
    """Return a boolean (n_bh, n_bw) mask, True for an active block (SAD >= threshold)."""
    h, w = y_curr.shape
    n_bh = h // block_size
    n_bw = w // block_size
    mask = np.zeros((n_bh, n_bw), dtype=bool)
    for by in range(n_bh):
        for bx in range(n_bw):
            y0, x0 = by * block_size, bx * block_size
            sad = np.sum(
                np.abs(
                    y_curr[y0 : y0 + block_size, x0 : x0 + block_size].astype(np.float32)
                    - y_prev[y0 : y0 + block_size, x0 : x0 + block_size].astype(np.float32)
                )
            )
            mask[by, bx] = sad >= threshold
    return mask
