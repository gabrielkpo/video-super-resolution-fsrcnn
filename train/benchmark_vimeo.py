"""Benchmark FSRCNN sur le test set Vimeo-90K.

Compare bicubique / FSRCNN DIV2K / FSRCNN Vimeo fine-tuné sur le canal Y.
Produit un tableau résumé et un CSV dans green_sr/results/.

Usage :
    cd green_sr
    python train/benchmark_vimeo.py
    python train/benchmark_vimeo.py --max-clips 100   # test rapide
    python train/benchmark_vimeo.py --frame im4       # 1 frame/clip seulement
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

from config import DEVICE
from methods.level6_cnn import load_model

DISK = Path("/Volumes/DisqueGab")
LR_ROOT = DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DISK / "vimeo_septuplet" / "sequences"
SPLITS = DISK / "vimeo_sr" / "splits"

CKPT_DIV2K = Path(__file__).parent / "fsrcnn_div2k.ckpt"
CKPT_VIMEO = Path(__file__).parent / "fsrcnn_vimeo_best.ckpt"
RESULTS = Path(__file__).parent.parent / "results"

SCALE = 4


# ── Métriques sur canal Y ────────────────────────────────────────────────────


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


# ── SR bicubique ─────────────────────────────────────────────────────────────


def bicubic_sr(lr_bgr: np.ndarray) -> np.ndarray:
    h, w = lr_bgr.shape[:2]
    return cv2.resize(lr_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)


# ── SR FSRCNN ────────────────────────────────────────────────────────────────


def fsrcnn_sr(lr_bgr: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    h, w = lr_bgr.shape[:2]
    ycrcb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    t = torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y_sr = model(t).squeeze().cpu().numpy()
    y_sr = np.clip(y_sr * 255, 0, 255).astype(np.uint8)

    cr_up = cv2.resize(cr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    sr_ycrcb = cv2.merge([y_sr, cr_up, cb_up])
    return cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(max_clips: int, frames: list[str]) -> None:

    if not DISK.exists():
        print(f"[benchmark] DisqueGab non monté : {DISK}")
        sys.exit(1)

    # Chargement des clips de test
    test_file = SPLITS / "test.txt"
    clips = [l.strip() for l in test_file.read_text().splitlines() if l.strip()]

    # Filtre clips cassés (im4 absent)
    valid = [
        c
        for c in clips
        if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists()
    ]
    n_bad = len(clips) - len(valid)
    if n_bad:
        print(f"  {n_bad} clips exclus (im4 manquant)")

    if max_clips > 0:
        valid = valid[:max_clips]

    print(f"Device       : {DEVICE}")
    print(
        f"Test clips   : {len(valid)}  ×  {len(frames)} frame(s) = {len(valid) * len(frames)} images"
    )
    print(f"Checkpoints  : {CKPT_DIV2K.name}  /  {CKPT_VIMEO.name}")

    # Chargement des modèles
    print("\nChargement des modèles…")
    model_div2k = load_model(CKPT_DIV2K, device=DEVICE)
    model_vimeo = load_model(CKPT_VIMEO, device=DEVICE)
    print("  OK")

    # Boucle de benchmark
    sums = {"bic": [0.0, 0.0], "div2k": [0.0, 0.0], "vimeo": [0.0, 0.0]}
    n_evaluated = 0
    rows = []

    bar = tqdm(valid, desc="Benchmark", unit="clip", ncols=90)
    for clip in bar:
        for fr in frames:
            lr_path = LR_ROOT / clip / f"{fr}.png"
            hr_path = HR_ROOT / clip / f"{fr}.png"

            lr_bgr = cv2.imread(str(lr_path))
            hr_bgr = cv2.imread(str(hr_path))
            if lr_bgr is None or hr_bgr is None:
                continue

            bic = bicubic_sr(lr_bgr)
            sr_d = fsrcnn_sr(lr_bgr, model_div2k)
            sr_v = fsrcnn_sr(lr_bgr, model_vimeo)

            pb, sb = psnr_y(hr_bgr, bic), ssim_y(hr_bgr, bic)
            pd, sd = psnr_y(hr_bgr, sr_d), ssim_y(hr_bgr, sr_d)
            pv, sv = psnr_y(hr_bgr, sr_v), ssim_y(hr_bgr, sr_v)

            sums["bic"][0] += pb
            sums["bic"][1] += sb
            sums["div2k"][0] += pd
            sums["div2k"][1] += sd
            sums["vimeo"][0] += pv
            sums["vimeo"][1] += sv
            n_evaluated += 1

            rows.append(
                {
                    "clip": clip,
                    "frame": fr,
                    "psnr_bic": round(pb, 4),
                    "ssim_bic": round(sb, 4),
                    "psnr_div2k": round(pd, 4),
                    "ssim_div2k": round(sd, 4),
                    "psnr_vimeo": round(pv, 4),
                    "ssim_vimeo": round(sv, 4),
                }
            )

        bar.set_postfix_str(
            f"PSNR bic={sums['bic'][0] / max(n_evaluated, 1):.2f} "
            f"vimeo={sums['vimeo'][0] / max(n_evaluated, 1):.2f}"
        )

    bar.close()

    if n_evaluated == 0:
        print("Aucune image évaluée.")
        sys.exit(1)

    # Moyennes
    mb_p = sums["bic"][0] / n_evaluated
    mb_s = sums["bic"][1] / n_evaluated
    md_p = sums["div2k"][0] / n_evaluated
    md_s = sums["div2k"][1] / n_evaluated
    mv_p = sums["vimeo"][0] / n_evaluated
    mv_s = sums["vimeo"][1] / n_evaluated

    # Affichage
    print(f"\n{'─' * 60}")
    print(f"{'Méthode':<22}  {'PSNR Y (dB)':>12}  {'SSIM Y':>8}  {'ΔPSNR':>7}")
    print(f"{'─' * 60}")
    print(f"{'Bicubique (baseline)':<22}  {mb_p:>12.4f}  {mb_s:>8.4f}  {'—':>7}")
    print(f"{'FSRCNN DIV2K':<22}  {md_p:>12.4f}  {md_s:>8.4f}  {md_p - mb_p:>+7.4f}")
    print(f"{'FSRCNN Vimeo (ft)':<22}  {mv_p:>12.4f}  {mv_s:>8.4f}  {mv_p - mb_p:>+7.4f}")
    print(f"{'─' * 60}")
    print(f"  Images évaluées : {n_evaluated}")

    # Sauvegarde CSV
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "benchmark_vimeo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Sauvegarde résumé
    summary_path = RESULTS / "benchmark_vimeo_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Benchmark Vimeo test set — {n_evaluated} images\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"{'Méthode':<22}  {'PSNR Y':>10}  {'SSIM Y':>8}  {'ΔPSNR':>7}\n")
        f.write(f"{'Bicubique':<22}  {mb_p:>10.4f}  {mb_s:>8.4f}  {'—':>7}\n")
        f.write(f"{'FSRCNN DIV2K':<22}  {md_p:>10.4f}  {md_s:>8.4f}  {md_p - mb_p:>+7.4f}\n")
        f.write(f"{'FSRCNN Vimeo (ft)':<22}  {mv_p:>10.4f}  {mv_s:>8.4f}  {mv_p - mb_p:>+7.4f}\n")

    print(f"\nCSV     → {csv_path}")
    print(f"Résumé  → {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark FSRCNN vs bicubique sur test Vimeo.")
    ap.add_argument(
        "--max-clips", type=int, default=0, help="Limite de clips (0 = tous, défaut: tous)"
    )
    ap.add_argument(
        "--frame",
        type=str,
        default="all",
        help="Frame(s) à évaluer : 'all' (défaut), 'im4', ou 'im1,im4,im7'",
    )
    args = ap.parse_args()

    if args.frame == "all":
        frames = [f"im{i}" for i in range(1, 8)]
    else:
        frames = [f.strip() for f in args.frame.split(",")]

    main(args.max_clips, frames)
