"""Level 1 — classical interpolation: bicubic, bilinear, Lanczos.

Usage (visual comparison and metrics):
    python -m methods.level1_interpolation
    python -m methods.level1_interpolation --video_id 33mqqm4QlJ8 --frame 5
    python -m methods.level1_interpolation --patch 200 300 128 128
"""

import cv2
import numpy as np

_METHODS = {
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
}


def upscale(img: np.ndarray, scale: int, method: str = "bicubic") -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method}. Choices: {list(_METHODS)}")
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=_METHODS[method])


def run_all(img: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    return {name: upscale(img, scale, name) for name in _METHODS}


if __name__ == "__main__":
    import argparse
    import sys
    import time

    import matplotlib.pyplot as plt

    from config import (
        FRAMES_HR_DIR,
        FRAMES_LR_PHASE1_DIR,
        RESULTS_VISUALS_DIR,
        SCALE_FACTOR,
        VIDEO_SCALE,
    )
    from pipeline.metrics import evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument(
        "--patch",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="patch of the HR image (default: centre)",
    )
    args = parser.parse_args()

    # Pick the video_id automatically
    if args.video_id:
        video_id = args.video_id
    else:
        dirs = [d for d in FRAMES_LR_PHASE1_DIR.iterdir() if d.is_dir()]
        if not dirs:
            print("No LR data found. Run pipeline.extract_frames and pipeline.degrade first.")
            sys.exit(1)
        video_id = dirs[0].name

    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)
    frame_name = f"frame_{args.frame:05d}.png"
    lr = cv2.imread(str(FRAMES_LR_PHASE1_DIR / video_id / frame_name))
    hr = cv2.imread(str(FRAMES_HR_DIR / video_id / frame_name))
    if lr is None or hr is None:
        print(f"Frame {frame_name} not found.")
        sys.exit(1)

    # Upscaling and metrics
    print(f"{'Method':<12} {'PSNR':>8} {'SSIM':>8} {'Tinf (ms)':>10}")
    print("-" * 42)
    results = {}
    for name in ("bilinear", "bicubic", "lanczos"):
        t0 = time.perf_counter()
        sr = upscale(lr, scale, name)
        tinf = (time.perf_counter() - t0) * 1000
        m = evaluate(hr, sr)
        results[name] = sr
        print(f"{name:<12} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {tinf:>10.2f}")

    # Figure 1: visual comparison on a patch
    h_hr, w_hr = hr.shape[:2]
    if args.patch:
        px, py, pw, ph = args.patch
    else:
        pw, ph = min(128, w_hr // 3), min(128, h_hr // 3)
        px, py = (w_hr - pw) // 2, (h_hr - ph) // 2

    def patch(img):
        return cv2.cvtColor(img[py : py + ph, px : px + pw], cv2.COLOR_BGR2RGB)

    panels = {
        "HR reference": patch(hr),
        "Bilinear": patch(results["bilinear"]),
        "Bicubic": patch(results["bicubic"]),
        "Lanczos": patch(results["lanczos"]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img) in zip(axes, panels.items(), strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"Level 1 — interpolation ×{scale} | {video_id} / {frame_name}", fontsize=10)
    plt.tight_layout()

    RESULTS_VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_VISUALS_DIR / f"figure1_interpolation_{video_id}.png"
    plt.savefig(out, dpi=150)
    print(f"\nFigure 1 → {out}")
    plt.show()
