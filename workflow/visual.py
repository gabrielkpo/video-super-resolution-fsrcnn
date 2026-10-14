"""Side-by-side figure HR / bicubic / SAD+FSRCNN, shared by the workflow scripts."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_comparison(
    hr: np.ndarray,
    bic: np.ndarray,
    sr: np.ndarray,
    p_bic: float,
    p_sr: float,
    name: str,
    scale: int,
    out: Path,
    figsize: tuple[float, float] = (18, 6),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"SAD+FSRCNN (×{scale}) — {name}", fontsize=11, fontweight="bold")
    panels = [
        (axes[0], hr, "HR reference", "#2ecc71"),
        (axes[1], bic, f"Bicubic\nPSNR={p_bic:.2f} dB", "#95a5a6"),
        (axes[2], sr, f"SAD+FSRCNN\nPSNR={p_sr:.2f} dB", "#3498db"),
    ]
    for ax, img, title, color in panels:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
