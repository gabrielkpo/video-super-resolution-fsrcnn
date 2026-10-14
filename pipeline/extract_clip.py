"""Extract CLIP_DURATION seconds of a video at its native FPS.

Output layout:
  data_ytb_24fps/<video_id>/frames_hr/frame_00001.png ...
  data_ytb_24fps/<video_id>/frames_hr/fps.txt           ← native FPS
  data_ytb_24fps/<video_id>/frames_hr/start.txt         ← start point
  data_ytb_24fps/<video_id>/frames_lr/                  ← bicubic degradation

Usage:
  python -m pipeline.extract_clip
  python -m pipeline.extract_clip --video_id DGbwtVtthu8
  python -m pipeline.extract_clip --video_id DGbwtVtthu8 --start 30
"""

import argparse
import subprocess
from pathlib import Path

import cv2

from config import CLIP_DURATION, RAW_DIR, RUNS_DIR, SCALE_FACTOR, VIDEO_SCALE
from pipeline.degrade import degrade_phase1


def get_native_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    num, den = result.stdout.strip().split("/")
    return round(float(num) / float(den), 6)


def extract_clip(video_id: str, start: float = 0.0) -> float:
    """Extract CLIP_DURATION seconds at native FPS from `start` seconds.

    Returns the native FPS. An extraction that already exists with the same `start`
    is skipped; if `start` changed, the frames are overwritten.
    """
    video_path = RAW_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video missing: {video_path}\n"
            "Run python -m pipeline.extract_frames first to download it."
        )

    hr_dir = RUNS_DIR / video_id / "frames_hr"
    fps_file = hr_dir / "fps.txt"
    start_file = hr_dir / "start.txt"
    hr_dir.mkdir(parents=True, exist_ok=True)

    existing = list(hr_dir.glob("frame_*.png"))
    if existing and fps_file.exists() and start_file.exists():
        if float(start_file.read_text().strip()) == start:
            fps = float(fps_file.read_text().strip())
            print(
                f"[extract_clip] {video_id} — {len(existing)} frames already there "
                f"(start={start}s, {fps} fps), skipped."
            )
            return fps
        print(f"[extract_clip] {video_id} — different start, extracting again.")

    fps = get_native_fps(video_path)
    print(f"[extract_clip] {video_id} — FPS {fps} | start {start}s | duration {CLIP_DURATION}s")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(CLIP_DURATION),
            "-pix_fmt",
            "rgb24",
            "-q:v",
            "1",
            str(hr_dir / "frame_%05d.png"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    fps_file.write_text(str(fps))
    start_file.write_text(str(start))
    n = len(list(hr_dir.glob("frame_*.png")))
    print(f"[extract_clip] {n} frames extracted → {hr_dir}")
    return fps


def degrade_clip(video_id: str) -> None:
    """Generate the LR frames (bicubic degradation) of the clip."""
    hr_dir = RUNS_DIR / video_id / "frames_hr"
    lr_dir = RUNS_DIR / video_id / "frames_lr"
    lr_dir.mkdir(parents=True, exist_ok=True)

    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)
    frames = sorted(hr_dir.glob("frame_*.png"))

    if len(list(lr_dir.glob("frame_*.png"))) == len(frames):
        print(f"[degrade_clip] {video_id} — LR frames already there, skipped.")
        return

    print(f"[degrade_clip] {video_id} — degrading ×{scale}  ({len(frames)} frames)")
    for f in frames:
        img = cv2.imread(str(f))
        cv2.imwrite(str(lr_dir / f.name), degrade_phase1(img, scale))
    print(f"[degrade_clip] OK → {lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", default="33mqqm4QlJ8")
    parser.add_argument(
        "--start", type=float, default=0.0, help="clip start in seconds (default: 0)"
    )
    args = parser.parse_args()

    extract_clip(args.video_id, start=args.start)
    degrade_clip(args.video_id)
