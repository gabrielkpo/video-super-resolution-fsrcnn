"""Comparaison SAD+FSRCNN vs FSRCNN seul sur clips Vimeo (7 frames consécutives).

Pour chaque clip du test set Vimeo, on traite les 7 frames en séquence :
  - FSRCNN seul   : FSRCNN appliqué à TOUS les blocs de chaque frame
  - SAD + FSRCNN  : FSRCNN appliqué seulement aux blocs qui ont changé (SAD mask)

Métriques comparées :
  - PSNR Y  (qualité)
  - SSIM Y  (qualité)
  - % blocs traités par FSRCNN  (efficacité)
  - Temps d'inférence moyen par frame

Usage :
    cd green_sr
    python workflow/compare_sad_vs_fsrcnn.py
    python workflow/compare_sad_vs_fsrcnn.py --max-clips 50 --threshold 1500
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE
from methods.level6_cnn import load_model
from workflow.pipeline import SadFsrcnnPipeline

DISK    = Path("/Volumes/DisqueGab")
LR_ROOT = DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DISK / "vimeo_septuplet" / "sequences"
SPLITS  = DISK / "vimeo_sr" / "splits"

CKPT_VIMEO = Path(__file__).parent.parent / "train" / "fsrcnn_vimeo_best.ckpt"
RESULTS    = Path(__file__).parent.parent / "results"

SCALE      = 4
BLOCK_SIZE = 16
N_FRAMES   = 7


# ── Métriques Y ──────────────────────────────────────────────────────────────

def psnr_y(hr: np.ndarray, sr: np.ndarray) -> float:
    h = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    s = cv2.cvtColor(sr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((h - s) ** 2)
    return 100.0 if mse < 1e-10 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_y(hr: np.ndarray, sr: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    h = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    s = cv2.cvtColor(sr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return structural_similarity(h, s, data_range=255)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(max_clips: int, threshold: float) -> None:

    if not DISK.exists():
        print(f"DisqueGab non monté : {DISK}")
        sys.exit(1)

    clips = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    valid = [c for c in clips
             if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists()]
    if max_clips > 0:
        valid = valid[:max_clips]

    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CKPT_VIMEO.name}")
    print(f"Seuil SAD    : {threshold}")
    print(f"Clips test   : {len(valid)}  ×  {N_FRAMES} frames = {len(valid)*N_FRAMES} images\n")

    model    = load_model(CKPT_VIMEO, device=DEVICE)
    pipeline = SadFsrcnnPipeline(model, block_size=BLOCK_SIZE,
                                 sad_threshold=threshold, device=DEVICE)

    # Accumulateurs par méthode : [psnr_sum, ssim_sum, blocks_pct_sum, time_sum, n]
    acc_fsrcnn = [0.0, 0.0, 0.0, 0.0, 0]
    acc_sad    = [0.0, 0.0, 0.0, 0.0, 0]

    rows = []

    bar = tqdm(valid, desc="Comparaison", unit="clip", ncols=90)
    for clip in bar:
        prev_hr_y_sad = None   # état temporel pour SAD

        for fi in range(1, N_FRAMES + 1):
            lr_bgr = cv2.imread(str(LR_ROOT / clip / f"im{fi}.png"))
            hr_bgr = cv2.imread(str(HR_ROOT / clip / f"im{fi}.png"))
            if lr_bgr is None or hr_bgr is None:
                continue

            # ── FSRCNN seul (prev=None → tous les blocs actifs à chaque frame) ─
            t0 = time.perf_counter()
            sr_f, _, stats_f = pipeline.process(lr_bgr, prev_hr_y=None)
            t_fsrcnn = (time.perf_counter() - t0) * 1000

            # ── SAD + FSRCNN (prev = frame HR précédente) ──────────────────────
            t0 = time.perf_counter()
            sr_s, hr_y_sad, stats_s = pipeline.process(lr_bgr, prev_hr_y=prev_hr_y_sad)
            t_sad = (time.perf_counter() - t0) * 1000
            prev_hr_y_sad = hr_y_sad

            # ── Métriques ──────────────────────────────────────────────────────
            pf = psnr_y(hr_bgr, sr_f);  sf = ssim_y(hr_bgr, sr_f)
            ps = psnr_y(hr_bgr, sr_s);  ss = ssim_y(hr_bgr, sr_s)

            blocks_fsrcnn = 100.0          # toujours 100 % traités
            blocks_sad    = 100.0 - stats_s["recycled_pct"]

            acc_fsrcnn[0] += pf;  acc_fsrcnn[1] += sf
            acc_fsrcnn[2] += blocks_fsrcnn;  acc_fsrcnn[3] += t_fsrcnn
            acc_fsrcnn[4] += 1

            acc_sad[0] += ps;  acc_sad[1] += ss
            acc_sad[2] += blocks_sad;  acc_sad[3] += t_sad
            acc_sad[4] += 1

            rows.append({
                "clip": clip, "frame": fi,
                "psnr_fsrcnn": round(pf, 4), "ssim_fsrcnn": round(sf, 4),
                "blocks_fsrcnn_pct": 100.0,  "time_fsrcnn_ms": round(t_fsrcnn, 2),
                "psnr_sad": round(ps, 4),    "ssim_sad": round(ss, 4),
                "blocks_sad_pct": round(blocks_sad, 1), "time_sad_ms": round(t_sad, 2),
            })

        n = acc_sad[4]
        if n > 0:
            bar.set_postfix_str(
                f"PSNR fsrcnn={acc_fsrcnn[0]/n:.2f} "
                f"sad={acc_sad[0]/n:.2f} "
                f"blocs_sad={acc_sad[2]/n:.0f}%"
            )

    bar.close()

    n_f = acc_fsrcnn[4]
    n_s = acc_sad[4]
    if n_f == 0:
        print("Aucune image évaluée.")
        sys.exit(1)

    mp_f = acc_fsrcnn[0] / n_f;  ms_f = acc_fsrcnn[1] / n_f
    mb_f = acc_fsrcnn[2] / n_f;  mt_f = acc_fsrcnn[3] / n_f
    mp_s = acc_sad[0]    / n_s;  ms_s = acc_sad[1]    / n_s
    mb_s = acc_sad[2]    / n_s;  mt_s = acc_sad[3]    / n_s

    # ── Tableau résumé ────────────────────────────────────────────────────────
    W = 70
    print(f"\n{'─'*W}")
    print(f"{'Méthode':<24}  {'PSNR Y':>8}  {'SSIM Y':>7}  {'Blocs CNN':>9}  {'T/frame':>8}")
    print(f"{'─'*W}")
    print(f"{'FSRCNN seul':<24}  {mp_f:>8.4f}  {ms_f:>7.4f}  {mb_f:>8.0f}%  {mt_f:>7.1f}ms")
    print(f"{'SAD + FSRCNN':<24}  {mp_s:>8.4f}  {ms_s:>7.4f}  {mb_s:>8.1f}%  {mt_s:>7.1f}ms")
    print(f"{'─'*W}")
    print(f"  ΔPSNR  SAD vs FSRCNN seul : {mp_s - mp_f:+.4f} dB")
    print(f"  ΔSSIM  SAD vs FSRCNN seul : {ms_s - ms_f:+.4f}")
    print(f"  Blocs économisés par SAD  : {mb_f - mb_s:.1f}%")
    print(f"  Gain temps SAD            : {(mt_f - mt_s)/mt_f*100:.1f}%")
    print(f"  Images évaluées           : {n_f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "compare_sad_vs_fsrcnn.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS / "compare_sad_vs_fsrcnn_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Comparaison SAD+FSRCNN vs FSRCNN seul — {n_f} images\n")
        f.write(f"Checkpoint : {CKPT_VIMEO.name}\n")
        f.write(f"Seuil SAD  : {threshold}\n")
        f.write(f"Device     : {DEVICE}\n\n")
        f.write(f"{'Méthode':<24}  {'PSNR Y':>8}  {'SSIM Y':>7}  {'Blocs CNN':>9}  {'T/frame':>8}\n")
        f.write(f"{'FSRCNN seul':<24}  {mp_f:>8.4f}  {ms_f:>7.4f}  {mb_f:>8.0f}%  {mt_f:>7.1f}ms\n")
        f.write(f"{'SAD + FSRCNN':<24}  {mp_s:>8.4f}  {ms_s:>7.4f}  {mb_s:>8.1f}%  {mt_s:>7.1f}ms\n")
        f.write(f"\nDelta PSNR  : {mp_s - mp_f:+.4f} dB\n")
        f.write(f"Delta SSIM  : {ms_s - ms_f:+.4f}\n")
        f.write(f"Blocs économisés : {mb_f - mb_s:.1f}%\n")
        f.write(f"Gain temps       : {(mt_f - mt_s)/mt_f*100:.1f}%\n")

    print(f"\nCSV    → {csv_path}")
    print(f"Résumé → {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0,
                    help="Nombre de clips à traiter (0 = tous)")
    ap.add_argument("--threshold", type=float, default=1500.0,
                    help="Seuil SAD pour détection de mouvement (défaut: 1500)")
    args = ap.parse_args()
    main(args.max_clips, args.threshold)
