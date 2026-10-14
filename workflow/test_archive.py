"""Test du workflow SAD+FSRCNN sur les vidéos UCF-101 (dataset_UFC101/test).

Usage :
    python -m workflow.test_archive                          # vidéo aléatoire
    python -m workflow.test_archive --category Basketball    # catégorie choisie
    python -m workflow.test_archive --video dataset_UFC101/test/Basketball/v_Basketball_g01_c01.avi
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_VISUALS_DIR
from pipeline.metrics import psnr, ssim
from workflow.model import load_model
from workflow.pipeline import SadFsrcnnPipeline

CKPT = Path(__file__).parent.parent / "train" / "fsrcnn_div2k.ckpt"
ARCHIVE = Path(__file__).parent.parent / "dataset_UFC101" / "test"
SCALE = 4
N_FRAMES = 30
SAD_THRESH = 1500.0
BLOCK_SIZE = 16


def pick_video(category: str | None, video_path: str | None, seed: int | None = None) -> Path:
    if video_path:
        p = Path(video_path)
        if not p.exists():
            print(f"[test_archive] Fichier introuvable : {p}")
            sys.exit(1)
        return p
    if seed is not None:
        random.seed(seed)
    cats = [d for d in ARCHIVE.iterdir() if d.is_dir()]
    if category:
        cats = [c for c in cats if c.name == category]
        if not cats:
            print(f"[test_archive] Catégorie introuvable : {category}")
            print(
                f"Disponibles : {[c.name for c in sorted(ARCHIVE.iterdir()) if c.is_dir()][:10]}..."
            )
            sys.exit(1)
    cat = random.choice(cats)
    vids = list(cat.glob("*.avi"))
    return random.choice(vids)


def extract_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def degrade(frame: np.ndarray, scale: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def main(category=None, video_path=None, sad_threshold=SAD_THRESH, ckpt_path=CKPT, seed=None):
    if not ckpt_path.exists():
        print(f"[test_archive] Checkpoint introuvable : {ckpt_path}")
        sys.exit(1)

    video = pick_video(category, video_path, seed=seed)
    print(f"[test_archive] Vidéo : {video.parent.name}/{video.name}")

    frames_hr = extract_frames(video, N_FRAMES)
    if not frames_hr:
        print("[test_archive] Impossible de lire la vidéo.")
        sys.exit(1)
    print(
        f"[test_archive] {len(frames_hr)} frames extraites — résolution HR : {frames_hr[0].shape[1]}×{frames_hr[0].shape[0]}"
    )
    print(f"[test_archive] Checkpoint : {ckpt_path.name}")

    model = load_model(ckpt_path)
    pipeline = SadFsrcnnPipeline(model, block_size=BLOCK_SIZE, sad_threshold=sad_threshold)

    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'Frame':<10} {'PSNR bic':>9} {'PSNR SR':>9} {'Δ PSNR':>7} {'SSIM bic':>9} {'SSIM SR':>8} {'Recyclé':>9} {'T inf':>7}"
    )
    print("-" * 78)

    prev_hr_y = None
    rows = []

    for i, hr in enumerate(frames_hr):
        lr = degrade(hr, SCALE)
        h, w = lr.shape[:2]
        bic = cv2.resize(lr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

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
            f"frame_{i + 1:04d}  {p_bic:>9.2f} {p_sr:>9.2f} {p_sr - p_bic:>+7.2f}"
            f" {s_bic:>9.4f} {s_sr:>8.4f} {stats['recycled_pct']:>8.1f}% {t_ms:>6.1f}ms"
        )

    print("-" * 78)
    pb, ps, sb, ss, rec, tm = [np.mean([r[i] for r in rows]) for i in range(6)]
    print(
        f"{'Moyenne':<10} {pb:>9.2f} {ps:>9.2f} {ps - pb:>+7.2f}"
        f" {sb:>9.4f} {ss:>8.4f} {rec:>8.1f}% {tm:>6.1f}ms"
    )

    # Visuel sur la dernière frame
    out = RESULTS_VISUALS_DIR / f"ucf101_{video.parent.name}_{video.stem}.png"
    _save_visual(frames_hr[-1], bic, sr, p_bic, p_sr, video.name, SCALE, out)
    print(f"\nVisuel → {out}")


def _save_visual(hr, bic, sr, p_bic, p_sr, name, scale, out):
    def crop(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAD+FSRCNN (×{scale}) — {name}", fontsize=11, fontweight="bold")
    for ax, img, title, color in [
        (axes[0], hr, "HR référence", "#2ecc71"),
        (axes[1], bic, f"Bicubique\nPSNR={p_bic:.2f} dB", "#95a5a6"),
        (axes[2], sr, f"SAD+FSRCNN\nPSNR={p_sr:.2f} dB", "#3498db"),
    ]:
        ax.imshow(crop(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(color)
            s.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=SAD_THRESH)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=CKPT,
        help="FSRCNN checkpoint (default: train/fsrcnn_div2k.ckpt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la sélection aléatoire de la vidéo (défaut : 42)",
    )
    args = parser.parse_args()
    main(args.category, args.video, args.threshold, ckpt_path=args.ckpt, seed=args.seed)
