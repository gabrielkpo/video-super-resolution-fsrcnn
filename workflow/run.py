"""SAD + FSRCNN workflow on consecutive frames of an extracted clip.

Usage (after pipeline/extract_clip.py):
    python -m workflow.run
    python -m workflow.run --threshold 1500 --block_size 16
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import RESULTS_VISUALS_DIR, RUNS_DIR, VIDEO_SCALE
from pipeline.metrics import psnr, ssim
from workflow.model import load_model
from workflow.pipeline import SadFsrcnnPipeline
from workflow.visual import save_comparison

CKPT = Path(__file__).parent.parent / "train" / "fsrcnn_div2k.ckpt"
VIDEO_ID = "33mqqm4QlJ8"
N_FRAMES = 30
HEADER = (
    f"{'PSNR bic':>9} {'PSNR SR':>9} {'Δ PSNR':>7} {'SSIM bic':>9} {'SSIM SR':>8}"
    f" {'Recycled':>9} {'T inf':>7}"
)


def main(sad_threshold: float = 1500.0, block_size: int = 16, ckpt_path: Path = CKPT):
    if not ckpt_path.exists():
        print(f"[run] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[run] Loading the model: {ckpt_path.name}")
    print(f"[run] Blocks {block_size}×{block_size}, SAD threshold {sad_threshold}")
    model = load_model(ckpt_path)
    pipeline = SadFsrcnnPipeline(model, block_size=block_size, sad_threshold=sad_threshold)

    scale = VIDEO_SCALE[VIDEO_ID]
    lr_root = RUNS_DIR / VIDEO_ID / "frames_lr"
    hr_root = RUNS_DIR / VIDEO_ID / "frames_hr"
    frames = sorted(lr_root.glob("*.png"))[:N_FRAMES]
    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Frame':<16} {HEADER}")
    print("-" * 86)

    prev_hr_y = None
    rows = []
    for lr_f in frames:
        lr = cv2.imread(str(lr_f))
        hr = cv2.imread(str(hr_root / lr_f.name))
        bic = cv2.resize(
            lr, (lr.shape[1] * scale, lr.shape[0] * scale), interpolation=cv2.INTER_CUBIC
        )

        t0 = time.perf_counter()
        sr, hr_y, stats = pipeline.process(lr, prev_hr_y)
        t_ms = (time.perf_counter() - t0) * 1000

        prev_hr_y = hr_y

        p_bic = psnr(hr, bic)
        p_sr = psnr(hr, sr)
        s_bic = ssim(hr, bic)
        s_sr = ssim(hr, sr)
        rows.append((p_bic, p_sr, s_bic, s_sr, stats["recycled_pct"], t_ms))
        print(
            f"{lr_f.name:<16} {p_bic:>9.2f} {p_sr:>9.2f} {p_sr - p_bic:>+7.2f}"
            f" {s_bic:>9.4f} {s_sr:>8.4f} {stats['recycled_pct']:>8.1f}% {t_ms:>6.1f}ms"
        )

        out = RESULTS_VISUALS_DIR / f"workflow_{VIDEO_ID}_{lr_f.stem}.png"
        save_comparison(hr, bic, sr, p_bic, p_sr, lr_f.name, scale, out, figsize=(24, 9))

    print("-" * 86)
    pb, ps, sb, ss, rec, tm = [np.mean([r[i] for r in rows]) for i in range(6)]
    print(
        f"{'Mean':<16} {pb:>9.2f} {ps:>9.2f} {ps - pb:>+7.2f}"
        f" {sb:>9.4f} {ss:>8.4f} {rec:>8.1f}% {tm:>6.1f}ms"
    )
    print(f"\nVisuals → {RESULTS_VISUALS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=1500.0, help="SAD threshold (default 1500)"
    )
    parser.add_argument("--block_size", type=int, default=16, help="block size (default 16)")
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    args = parser.parse_args()
    main(args.threshold, args.block_size, args.ckpt)
