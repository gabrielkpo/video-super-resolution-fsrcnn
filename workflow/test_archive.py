"""SAD + FSRCNN workflow on UCF-101 videos (dataset_UFC101/test).

Usage:
    python -m workflow.test_archive                          # random video
    python -m workflow.test_archive --category Basketball    # chosen category
    python -m workflow.test_archive --video dataset_UFC101/test/Basketball/v_Basketball_g01_c01.avi
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import RESULTS_VISUALS_DIR
from pipeline.metrics import psnr, ssim
from workflow.model import load_model
from workflow.pipeline import SadFsrcnnPipeline
from workflow.run import HEADER
from workflow.visual import save_comparison

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
            print(f"[test_archive] File not found: {p}")
            sys.exit(1)
        return p
    if seed is not None:
        random.seed(seed)
    cats = [d for d in ARCHIVE.iterdir() if d.is_dir()]
    if category:
        cats = [c for c in cats if c.name == category]
        if not cats:
            available = [c.name for c in sorted(ARCHIVE.iterdir()) if c.is_dir()][:10]
            print(f"[test_archive] Category not found: {category}")
            print(f"Available: {available}...")
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
        print(f"[test_archive] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    video = pick_video(category, video_path, seed=seed)
    print(f"[test_archive] Video: {video.parent.name}/{video.name}")

    frames_hr = extract_frames(video, N_FRAMES)
    if not frames_hr:
        print("[test_archive] Cannot read the video.")
        sys.exit(1)
    h, w = frames_hr[0].shape[:2]
    print(f"[test_archive] {len(frames_hr)} frames extracted, HR resolution {w}×{h}")
    print(f"[test_archive] Checkpoint: {ckpt_path.name}")

    model = load_model(ckpt_path)
    pipeline = SadFsrcnnPipeline(model, block_size=BLOCK_SIZE, sad_threshold=sad_threshold)

    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Frame':<10} {HEADER}")
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
        f"{'Mean':<10} {pb:>9.2f} {ps:>9.2f} {ps - pb:>+7.2f}"
        f" {sb:>9.4f} {ss:>8.4f} {rec:>8.1f}% {tm:>6.1f}ms"
    )

    # Figure for the last frame
    out = RESULTS_VISUALS_DIR / f"ucf101_{video.parent.name}_{video.stem}.png"
    save_comparison(frames_hr[-1], bic, sr, p_bic, p_sr, video.name, SCALE, out)
    print(f"\nFigure → {out}")


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
        "--seed", type=int, default=42, help="seed for the random video choice (default: 42)"
    )
    args = parser.parse_args()
    main(args.category, args.video, args.threshold, ckpt_path=args.ckpt, seed=args.seed)
