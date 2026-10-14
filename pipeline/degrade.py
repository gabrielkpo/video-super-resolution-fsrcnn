"""Generate two sets of LR frames from the HR frames.

  Phase 1 — plain bicubic                     → frames_lr/phase1_bicubic/<video_id>/
  Phase 2 — Gaussian blur + bicubic + AWGN σ=25 → frames_lr/phase2_noisy/<video_id>/

Usage:
    python -m pipeline.degrade
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    BLUR_KERNEL_SIZE,
    FRAMES_HR_DIR,
    FRAMES_LR_PHASE1_DIR,
    FRAMES_LR_PHASE2_DIR,
    NOISE_STD_PHASE2,
    SCALE_FACTOR,
    VIDEO_SCALE,
)


def degrade_phase1(img: np.ndarray, scale: int = SCALE_FACTOR) -> np.ndarray:
    """Plain bicubic degradation, no noise."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def degrade_phase2(
    img: np.ndarray,
    scale: int = SCALE_FACTOR,
    blur_k: int = BLUR_KERNEL_SIZE,
    noise_std: float = NOISE_STD_PHASE2,
) -> np.ndarray:
    """Realistic degradation: Gaussian blur → bicubic → AWGN σ=25."""
    h, w = img.shape[:2]
    if blur_k > 1:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    noise = np.random.normal(0, noise_std, lr.shape).astype(np.float32)
    return np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_directory(hr_dir: Path, phase1_root: Path, phase2_root: Path) -> None:
    video_id = hr_dir.name
    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)
    out1 = phase1_root / video_id
    out2 = phase2_root / video_id
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    frames = sorted(hr_dir.glob("*.png"))
    if not frames:
        print(f"[degrade] No frame in {hr_dir}, skipped.")
        return

    print(f"[degrade] {video_id} — factor ×{scale}")
    for f in frames:
        img = cv2.imread(str(f))
        cv2.imwrite(str(out1 / f.name), degrade_phase1(img, scale))
        cv2.imwrite(str(out2 / f.name), degrade_phase2(img, scale))

    print(f"[degrade] {len(frames)} frames  →  {out1}")
    print(f"[degrade] {len(frames)} frames  →  {out2}")


if __name__ == "__main__":
    hr_subdirs = [d for d in FRAMES_HR_DIR.iterdir() if d.is_dir()]
    if not hr_subdirs:
        print(f"[degrade] No sub-folder in {FRAMES_HR_DIR}.")
        print("          Run first: python -m pipeline.extract_frames")
    for hr_subdir in hr_subdirs:
        degrade_directory(hr_subdir, FRAMES_LR_PHASE1_DIR, FRAMES_LR_PHASE2_DIR)
