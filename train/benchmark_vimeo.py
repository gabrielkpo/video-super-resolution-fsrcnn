"""FSRCNN benchmark on the Vimeo-90K test split.

Compares bicubic / FSRCNN DIV2K / FSRCNN fine-tuned on Vimeo, on the Y channel.
Writes a summary table and a CSV in results/.

Usage:
    python -m train.benchmark_vimeo
    python -m train.benchmark_vimeo --max-clips 100   # quick run
    python -m train.benchmark_vimeo --frame im4       # one frame per clip only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import DATA_DISK, DEVICE
from methods.level6_cnn import load_model

LR_ROOT = DATA_DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DATA_DISK / "vimeo_septuplet" / "sequences"
SPLITS = DATA_DISK / "vimeo_sr" / "splits"

CKPT_DIV2K = Path(__file__).parent / "fsrcnn_div2k.ckpt"
CKPT_VIMEO = Path(__file__).parent / "fsrcnn_vimeo_best.ckpt"
RESULTS = Path(__file__).parent.parent / "results"

SCALE = 4


# ── Y-channel metrics ────────────────────────────────────────────────────────


def psnr_y(hr_bgr: np.ndarray, sr_bgr: np.ndarray) -> float:
    hr_y = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    sr_y = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((hr_y - sr_y) ** 2)
    return 100.0 if mse < 1e-10 else 10.0 * np.log10(255.0**2 / mse)


def ssim_y(hr_bgr: np.ndarray, sr_bgr: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    hr_y = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sr_y = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return structural_similarity(hr_y, sr_y, data_range=255)


# ── Bicubic SR ───────────────────────────────────────────────────────────────


def bicubic_sr(lr_bgr: np.ndarray) -> np.ndarray:
    h, w = lr_bgr.shape[:2]
    return cv2.resize(lr_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)


# ── FSRCNN SR ────────────────────────────────────────────────────────────────


def fsrcnn_sr(lr_bgr: np.ndarray, model: torch.nn.Module, device: str) -> np.ndarray:
    h, w = lr_bgr.shape[:2]
    ycrcb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    t = torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_sr = model(t).squeeze().cpu().numpy()
    y_sr = np.clip(y_sr * 255, 0, 255).astype(np.uint8)

    cr_up = cv2.resize(cr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    sr_ycrcb = cv2.merge([y_sr, cr_up, cb_up])
    return cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(max_clips: int, frames: list[str]) -> None:
    if not DATA_DISK.exists():
        print(f"[benchmark] Data disk not mounted: {DATA_DISK} (set VSR_DATA_DISK)")
        sys.exit(1)

    # Test clips
    lines = (SPLITS / "test.txt").read_text().splitlines()
    clips = [line.strip() for line in lines if line.strip()]

    # Drop broken clips (missing im4)
    valid = [
        c
        for c in clips
        if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists()
    ]
    n_bad = len(clips) - len(valid)
    if n_bad:
        print(f"  {n_bad} clips excluded (missing im4)")

    if max_clips > 0:
        valid = valid[:max_clips]

    n_images = len(valid) * len(frames)
    print(f"Device       : {DEVICE}")
    print(f"Test clips   : {len(valid)}  ×  {len(frames)} frame(s) = {n_images} images")
    print(f"Checkpoints  : {CKPT_DIV2K.name}  /  {CKPT_VIMEO.name}")

    print("\nLoading the models…")
    model_div2k = load_model(CKPT_DIV2K, device=DEVICE)
    model_vimeo = load_model(CKPT_VIMEO, device=DEVICE)
    print("  OK")

    # Benchmark loop: per method, [psnr_sum, ssim_sum]
    sums = {"bic": [0.0, 0.0], "div2k": [0.0, 0.0], "vimeo": [0.0, 0.0]}
    n_evaluated = 0
    rows = []

    bar = tqdm(valid, desc="Benchmark", unit="clip", ncols=90)
    for clip in bar:
        for fr in frames:
            lr_bgr = cv2.imread(str(LR_ROOT / clip / f"{fr}.png"))
            hr_bgr = cv2.imread(str(HR_ROOT / clip / f"{fr}.png"))
            if lr_bgr is None or hr_bgr is None:
                continue

            outputs = {
                "bic": bicubic_sr(lr_bgr),
                "div2k": fsrcnn_sr(lr_bgr, model_div2k, DEVICE),
                "vimeo": fsrcnn_sr(lr_bgr, model_vimeo, DEVICE),
            }
            row = {"clip": clip, "frame": fr}
            for name, sr in outputs.items():
                p, s = psnr_y(hr_bgr, sr), ssim_y(hr_bgr, sr)
                sums[name][0] += p
                sums[name][1] += s
                row[f"psnr_{name}"] = round(p, 4)
                row[f"ssim_{name}"] = round(s, 4)
            rows.append(row)
            n_evaluated += 1

        bar.set_postfix_str(
            f"PSNR bic={sums['bic'][0] / max(n_evaluated, 1):.2f} "
            f"vimeo={sums['vimeo'][0] / max(n_evaluated, 1):.2f}"
        )

    bar.close()

    if n_evaluated == 0:
        print("No image evaluated.")
        sys.exit(1)

    mb_p, mb_s = (v / n_evaluated for v in sums["bic"])
    md_p, md_s = (v / n_evaluated for v in sums["div2k"])
    mv_p, mv_s = (v / n_evaluated for v in sums["vimeo"])

    print(f"\n{'─' * 60}")
    print(f"{'Method':<22}  {'PSNR Y (dB)':>12}  {'SSIM Y':>8}  {'ΔPSNR':>7}")
    print(f"{'─' * 60}")
    print(f"{'Bicubic (baseline)':<22}  {mb_p:>12.4f}  {mb_s:>8.4f}  {'—':>7}")
    print(f"{'FSRCNN DIV2K':<22}  {md_p:>12.4f}  {md_s:>8.4f}  {md_p - mb_p:>+7.4f}")
    print(f"{'FSRCNN Vimeo (ft)':<22}  {mv_p:>12.4f}  {mv_s:>8.4f}  {mv_p - mb_p:>+7.4f}")
    print(f"{'─' * 60}")
    print(f"  Images evaluated: {n_evaluated}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "benchmark_vimeo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS / "benchmark_vimeo_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Vimeo test split benchmark — {n_evaluated} images\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"{'Method':<22}  {'PSNR Y':>10}  {'SSIM Y':>8}  {'ΔPSNR':>7}\n")
        f.write(f"{'Bicubic':<22}  {mb_p:>10.4f}  {mb_s:>8.4f}  {'—':>7}\n")
        f.write(f"{'FSRCNN DIV2K':<22}  {md_p:>10.4f}  {md_s:>8.4f}  {md_p - mb_p:>+7.4f}\n")
        f.write(f"{'FSRCNN Vimeo (ft)':<22}  {mv_p:>10.4f}  {mv_s:>8.4f}  {mv_p - mb_p:>+7.4f}\n")

    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FSRCNN vs bicubic on the Vimeo test split.")
    ap.add_argument("--max-clips", type=int, default=0, help="clip limit (0 = all, the default)")
    ap.add_argument(
        "--frame",
        type=str,
        default="all",
        help="frame(s) to evaluate: 'all' (default), 'im4', or 'im1,im4,im7'",
    )
    args = ap.parse_args()

    if args.frame == "all":
        frames = [f"im{i}" for i in range(1, 8)]
    else:
        frames = [f.strip() for f in args.frame.split(",")]

    main(args.max_clips, frames)
