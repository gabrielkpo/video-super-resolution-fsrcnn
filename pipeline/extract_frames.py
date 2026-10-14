"""Download a video with yt-dlp and extract its frames with ffmpeg.

Usage:
    python -m pipeline.extract_frames
    python -m pipeline.extract_frames --url <YouTube URL> --fps 1
"""

import argparse
import re
import subprocess
from pathlib import Path

from config import EXTRACT_FPS, FRAMES_HR_DIR, RAW_DIR, VIDEO_URL2


def download_video(url: str) -> Path:
    video_id = _parse_video_id(url)
    output_path = RAW_DIR / f"{video_id}.mp4"

    if output_path.exists():
        print(f"[download] Already there: {output_path}, skipped.")
        return output_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading {url} ...")
    cmd = [
        "yt-dlp",
        "--format",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "--output",
        str(RAW_DIR / f"{video_id}.%(ext)s"),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for: {url}")

    candidates = list(RAW_DIR.glob(f"{video_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"No file found after the download ({video_id})")
    downloaded = candidates[0]
    if downloaded.suffix != ".mp4":
        downloaded = downloaded.rename(output_path)
    print(f"[download] OK — {downloaded}  ({downloaded.stat().st_size / 1e6:.1f} MB)")
    return downloaded


def _parse_video_id(url: str) -> str:
    for pattern in (r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return re.sub(r"[^\w]", "_", url)[-20:]


def extract_frames(
    video_path: Path, output_dir: Path, fps: int = EXTRACT_FPS, max_frames: int | None = None
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("frame_*.png"))
    if existing:
        print(f"[extract_frames] {len(existing)} frames already there, skipped.")
        return len(existing)

    pattern = output_dir / "frame_%05d.png"
    cmd = ["ffmpeg", "-i", str(video_path)]
    if fps and fps > 0:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-pix_fmt", "rgb24", "-q:v", "1"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [str(pattern), "-y"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr}")

    frames = list(output_dir.glob("frame_*.png"))
    print(f"[extract_frames] {len(frames)} frames extracted → {output_dir}")
    return len(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=VIDEO_URL2, help="YouTube URL (default: VIDEO_URL2 in config.py)"
    )
    parser.add_argument("--fps", type=int, default=EXTRACT_FPS)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    video = download_video(args.url)
    extract_frames(video, FRAMES_HR_DIR / video.stem, fps=args.fps, max_frames=args.max_frames)
