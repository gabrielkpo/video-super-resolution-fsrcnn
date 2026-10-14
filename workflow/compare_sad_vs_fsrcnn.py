"""SAD + FSRCNN vs FSRCNN alone on Vimeo clips (7 consecutive frames).

For each clip of the Vimeo test split, the 7 frames are processed in sequence:
  - FSRCNN alone: FSRCNN applied to EVERY block of every frame
  - SAD + FSRCNN: FSRCNN applied only to the blocks that changed (SAD mask)

Compared metrics:
  - PSNR Y (quality)
  - SSIM Y (quality)
  - % of blocks processed by FSRCNN (efficiency)
  - mean inference time per frame

Usage:
    python -m workflow.compare_sad_vs_fsrcnn
    python -m workflow.compare_sad_vs_fsrcnn --max-clips 50 --threshold 1500
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import DATA_DISK, DEVICE
from methods.level6_cnn import load_model
from workflow.pipeline import SadFsrcnnPipeline

LR_ROOT = DATA_DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DATA_DISK / "vimeo_septuplet" / "sequences"
SPLITS = DATA_DISK / "vimeo_sr" / "splits"

CKPT_VIMEO = Path(__file__).parent.parent / "train" / "fsrcnn_vimeo_best.ckpt"
RESULTS = Path(__file__).parent.parent / "results"

SCALE = 4
BLOCK_SIZE = 16
N_FRAMES = 7
HEADER = f"{'Method':<24}  {'PSNR Y':>8}  {'SSIM Y':>7}  {'CNN blocks':>10}  {'T/frame':>8}"


# ── Y-channel metrics ────────────────────────────────────────────────────────


def psnr_y(hr: np.ndarray, sr: np.ndarray) -> float:
    h = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    s = cv2.cvtColor(sr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((h - s) ** 2)
    return 100.0 if mse < 1e-10 else 10.0 * np.log10(255.0**2 / mse)


def ssim_y(hr: np.ndarray, sr: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    h = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    s = cv2.cvtColor(sr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return structural_similarity(h, s, data_range=255)


def _row(name: str, psnr: float, ssim: float, blocks: float, ms: float) -> str:
    return f"{name:<24}  {psnr:>8.4f}  {ssim:>7.4f}  {blocks:>9.1f}%  {ms:>7.1f}ms"


# ── Main ─────────────────────────────────────────────────────────────────────


def main(max_clips: int, threshold: float) -> None:
    if not DATA_DISK.exists():
        print(f"Data disk not mounted: {DATA_DISK} (set VSR_DATA_DISK)")
        sys.exit(1)

    lines = (SPLITS / "test.txt").read_text().splitlines()
    clips = [line.strip() for line in lines if line.strip()]
    valid = [
        c
        for c in clips
        if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists()
    ]
    if max_clips > 0:
        valid = valid[:max_clips]

    print(f"Device        : {DEVICE}")
    print(f"Checkpoint    : {CKPT_VIMEO.name}")
    print(f"SAD threshold : {threshold}")
    print(f"Test clips    : {len(valid)}  ×  {N_FRAMES} frames = {len(valid) * N_FRAMES} images\n")

    model = load_model(CKPT_VIMEO, device=DEVICE)
    pipeline = SadFsrcnnPipeline(
        model, block_size=BLOCK_SIZE, sad_threshold=threshold, device=DEVICE
    )

    # Per-method accumulators: [psnr_sum, ssim_sum, blocks_pct_sum, time_sum, n]
    acc_fsrcnn = [0.0, 0.0, 0.0, 0.0, 0]
    acc_sad = [0.0, 0.0, 0.0, 0.0, 0]

    rows = []

    bar = tqdm(valid, desc="Comparing", unit="clip", ncols=90)
    for clip in bar:
        prev_hr_y_sad = None  # temporal state for SAD

        for fi in range(1, N_FRAMES + 1):
            lr_bgr = cv2.imread(str(LR_ROOT / clip / f"im{fi}.png"))
            hr_bgr = cv2.imread(str(HR_ROOT / clip / f"im{fi}.png"))
            if lr_bgr is None or hr_bgr is None:
                continue

            # ── FSRCNN alone (prev=None → every block active on every frame) ──
            t0 = time.perf_counter()
            sr_f, _, _ = pipeline.process(lr_bgr, prev_hr_y=None)
            t_fsrcnn = (time.perf_counter() - t0) * 1000

            # ── SAD + FSRCNN (prev = previous HR frame) ────────────────────────
            t0 = time.perf_counter()
            sr_s, hr_y_sad, stats_s = pipeline.process(lr_bgr, prev_hr_y=prev_hr_y_sad)
            t_sad = (time.perf_counter() - t0) * 1000
            prev_hr_y_sad = hr_y_sad

            # ── Metrics ────────────────────────────────────────────────────────
            pf = psnr_y(hr_bgr, sr_f)
            sf = ssim_y(hr_bgr, sr_f)
            ps = psnr_y(hr_bgr, sr_s)
            ss = ssim_y(hr_bgr, sr_s)

            blocks_fsrcnn = 100.0  # always 100 % processed
            blocks_sad = 100.0 - stats_s["recycled_pct"]

            for acc, values in (
                (acc_fsrcnn, (pf, sf, blocks_fsrcnn, t_fsrcnn)),
                (acc_sad, (ps, ss, blocks_sad, t_sad)),
            ):
                for k, v in enumerate(values):
                    acc[k] += v
                acc[4] += 1

            rows.append(
                {
                    "clip": clip,
                    "frame": fi,
                    "psnr_fsrcnn": round(pf, 4),
                    "ssim_fsrcnn": round(sf, 4),
                    "blocks_fsrcnn_pct": 100.0,
                    "time_fsrcnn_ms": round(t_fsrcnn, 2),
                    "psnr_sad": round(ps, 4),
                    "ssim_sad": round(ss, 4),
                    "blocks_sad_pct": round(blocks_sad, 1),
                    "time_sad_ms": round(t_sad, 2),
                }
            )

        n = acc_sad[4]
        if n > 0:
            bar.set_postfix_str(
                f"PSNR fsrcnn={acc_fsrcnn[0] / n:.2f} "
                f"sad={acc_sad[0] / n:.2f} "
                f"sad_blocks={acc_sad[2] / n:.0f}%"
            )

    bar.close()

    n_f = acc_fsrcnn[4]
    n_s = acc_sad[4]
    if n_f == 0:
        print("No image evaluated.")
        sys.exit(1)

    mp_f, ms_f, mb_f, mt_f = (v / n_f for v in acc_fsrcnn[:4])
    mp_s, ms_s, mb_s, mt_s = (v / n_s for v in acc_sad[:4])
    line_f = _row("FSRCNN alone", mp_f, ms_f, mb_f, mt_f)
    line_s = _row("SAD + FSRCNN", mp_s, ms_s, mb_s, mt_s)

    # ── Summary table ─────────────────────────────────────────────────────────
    width = 72
    print(f"\n{'─' * width}\n{HEADER}\n{'─' * width}\n{line_f}\n{line_s}\n{'─' * width}")
    print(f"  ΔPSNR SAD vs FSRCNN alone : {mp_s - mp_f:+.4f} dB")
    print(f"  ΔSSIM SAD vs FSRCNN alone : {ms_s - ms_f:+.4f}")
    print(f"  Blocks saved by SAD       : {mb_f - mb_s:.1f}%")
    print(f"  Time saved by SAD         : {(mt_f - mt_s) / mt_f * 100:.1f}%")
    print(f"  Images evaluated          : {n_f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "compare_sad_vs_fsrcnn.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS / "compare_sad_vs_fsrcnn_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"SAD+FSRCNN vs FSRCNN alone — {n_f} images\n")
        f.write(f"Checkpoint    : {CKPT_VIMEO.name}\n")
        f.write(f"SAD threshold : {threshold}\n")
        f.write(f"Device        : {DEVICE}\n\n")
        f.write(f"{HEADER}\n{line_f}\n{line_s}\n")
        f.write(f"\nDelta PSNR  : {mp_s - mp_f:+.4f} dB\n")
        f.write(f"Delta SSIM  : {ms_s - ms_f:+.4f}\n")
        f.write(f"Blocks saved : {mb_f - mb_s:.1f}%\n")
        f.write(f"Time saved   : {(mt_f - mt_s) / mt_f * 100:.1f}%\n")

    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0, help="number of clips (0 = all)")
    ap.add_argument(
        "--threshold", type=float, default=1500.0, help="SAD motion threshold (default: 1500)"
    )
    args = ap.parse_args()
    main(args.max_clips, args.threshold)
