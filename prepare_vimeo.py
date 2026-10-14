"""
prepare_vimeo.py — prepare the Vimeo-90K dataset for FSRCNN.

This script (run once):
  1. Scans every clip in <VSR_DATA_DISK>/vimeo_septuplet/sequences/
  2. Generates the LR frames (bicubic ×SCALE) of every clip
  3. Saves them in <VSR_DATA_DISK>/vimeo_sr/sequences_lr/ (same tree)
  4. Writes the train/val/test split files in <VSR_DATA_DISK>/vimeo_sr/splits/
  5. Everything stays on the data disk, nothing is written to the internal drive

Resumable: clips already processed are detected and skipped.

Usage:
    python prepare_vimeo.py
    python prepare_vimeo.py --scale 4 --workers 4 --val 0.1 --test 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import cv2

from config import DATA_DISK

# ── Paths (all on the data disk) ──────────────────────────────────────────────
SRC_SEQ_DIR = DATA_DISK / "vimeo_septuplet" / "sequences"
DST_ROOT = DATA_DISK / "vimeo_sr"
DST_LR_DIR = DST_ROOT / "sequences_lr"
DST_SPLIT_DIR = DST_ROOT / "splits"
STATS_FILE = DST_ROOT / "stats.json"

N_FRAMES = 7  # im1.png … im7.png per clip
MIN_SIZE = (32, 32)  # smallest acceptable HR size


# ── One clip ──────────────────────────────────────────────────────────────────


def process_clip(clip_rel: str, scale: int) -> tuple[str, str]:
    """Degrade the 7 frames of a clip.

    Returns (clip_rel, status) with status in {"ok", "skip", "error:<msg>"}.
    """
    src_clip = SRC_SEQ_DIR / clip_rel
    dst_clip = DST_LR_DIR / clip_rel

    # Resume: clip already processed?
    existing = list(dst_clip.glob("im*.png")) if dst_clip.exists() else []
    if len(existing) == N_FRAMES:
        return clip_rel, "skip"

    dst_clip.mkdir(parents=True, exist_ok=True)

    for i in range(1, N_FRAMES + 1):
        src_path = src_clip / f"im{i}.png"
        dst_path = dst_clip / f"im{i}.png"

        if dst_path.exists():
            continue

        hr = cv2.imread(str(src_path))
        if hr is None:
            return clip_rel, f"error:cannot read im{i}.png"

        h, w = hr.shape[:2]
        if h < MIN_SIZE[0] or w < MIN_SIZE[1]:
            return clip_rel, f"error:too small ({w}×{h})"

        lr = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        if not cv2.imwrite(str(dst_path), lr):
            return clip_rel, f"error:write failed im{i}.png"

    return clip_rel, "ok"


# ── Split ─────────────────────────────────────────────────────────────────────


def make_splits(
    clips: list[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_splits(splits: dict[str, list[str]]) -> None:
    DST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, clips in splits.items():
        out = DST_SPLIT_DIR / f"{name}.txt"
        out.write_text("\n".join(sorted(clips)) + "\n")
        print(f"  {name:5s}: {len(clips):6d} clips → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def _progress(done: int, total: int, t0: float, n_ok: int, n_skip: int, n_error: int) -> None:
    rate = done / (time.time() - t0)
    eta = (total - done) / rate
    print(
        f"  {100.0 * done / total:5.1f}%  {done}/{total} clips"
        f"  ok={n_ok} skip={n_skip} err={n_error}"
        f"  {rate:.1f} clips/s  ETA {eta / 60:.0f}min",
        flush=True,
    )


def main(scale: int, workers: int, val_ratio: float, test_ratio: float, seed: int) -> None:
    if not SRC_SEQ_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SRC_SEQ_DIR}\n"
            "Check that the data disk is mounted, or set VSR_DATA_DISK."
        )

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    DST_LR_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Scan the clips ─────────────────────────────────────────────────────
    print("Scanning clips…", flush=True)
    clips: list[str] = []
    for group_dir in sorted(SRC_SEQ_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for clip_dir in sorted(group_dir.iterdir()):
            # Skip empty folders: a clip needs at least im1.png
            if clip_dir.is_dir() and (clip_dir / "im1.png").exists():
                clips.append(f"{group_dir.name}/{clip_dir.name}")

    print(f"  {len(clips)} clips found in {SRC_SEQ_DIR}")
    print(f"  → {len(clips) * N_FRAMES} frames to process (scale ×{scale})")

    # ── 2. Degrade and save ───────────────────────────────────────────────────
    print(f"\nDegrading (×{scale}) with {workers} worker(s)…", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_error = 0
    done = 0
    report_every = max(1, len(clips) // 100)  # log every 1 %

    def record(clip_rel: str, status: str) -> None:
        nonlocal n_ok, n_skip, n_error, done
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_error += 1
            print(f"  ERROR {clip_rel}: {status}")
        done += 1
        if done % report_every == 0:
            _progress(done, len(clips), t0, n_ok, n_skip, n_error)

    if workers == 1:
        # Sequential mode (simpler to debug)
        for clip_rel in clips:
            record(*process_clip(clip_rel, scale))
    else:
        worker_fn = partial(process_clip, scale=scale)
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(worker_fn, c) for c in clips]
            for fut in as_completed(futures):
                record(*fut.result())

    elapsed_total = time.time() - t0
    print(f"\nDegradation done in {elapsed_total / 60:.1f} min")
    print(f"  Processed: {n_ok}  Skipped: {n_skip}  Errors: {n_error}")

    # ── 3. Splits ─────────────────────────────────────────────────────────────
    print("\nWriting the splits…")
    good_clips = [c for c in clips if len(list((DST_LR_DIR / c).glob("im*.png"))) == N_FRAMES]
    print(f"  Clips eligible for the split: {len(good_clips)}")

    splits = make_splits(good_clips, val_ratio, test_ratio, seed)
    write_splits(splits)

    # ── 4. Stats ──────────────────────────────────────────────────────────────
    # One sample to record the frame sizes
    sample_clip = good_clips[0] if good_clips else None
    hr_shape = lr_shape = None
    if sample_clip:
        hr = cv2.imread(str(SRC_SEQ_DIR / sample_clip / "im4.png"))
        lr = cv2.imread(str(DST_LR_DIR / sample_clip / "im4.png"))
        if hr is not None and lr is not None:
            hr_shape = list(hr.shape)
            lr_shape = list(lr.shape)

    stats = {
        "scale": scale,
        "n_clips_total": len(clips),
        "n_clips_valid": len(good_clips),
        "n_frames_valid": len(good_clips) * N_FRAMES,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "hr_shape_hwc": hr_shape,
        "lr_shape_hwc": lr_shape,
        "elapsed_s": round(elapsed_total, 1),
        "errors": n_error,
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(f"\nStats → {STATS_FILE}")
    print(json.dumps(stats, indent=2))

    print("\nDataset ready.")
    print(f"  HR  : {SRC_SEQ_DIR}  (originals, untouched)")
    print(f"  LR  : {DST_LR_DIR}")
    print(f"  Splits : {DST_SPLIT_DIR}/{{train,val,test}}.txt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vimeo-90K for FSRCNN training.")
    parser.add_argument("--scale", type=int, default=4, help="degradation factor (default: 4)")
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel processes (default: 4; 1 = sequential)"
    )
    parser.add_argument("--val", type=float, default=0.1, help="validation fraction (0.10)")
    parser.add_argument("--test", type=float, default=0.1, help="test fraction (0.10)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()
    main(args.scale, args.workers, args.val, args.test, args.seed)
