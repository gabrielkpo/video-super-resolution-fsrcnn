"""Extrait CLIP_DURATION secondes d'une vidéo à son FPS natif.

Structure de sortie :
  data_ytb_24fps/<video_id>/frames_hr/frame_00001.png ...
  data_ytb_24fps/<video_id>/frames_hr/fps.txt           ← FPS natif
  data_ytb_24fps/<video_id>/frames_hr/start.txt         ← point de départ
  data_ytb_24fps/<video_id>/frames_lr/                  ← dégradation bicubique

Usage :
  python pipeline/extract_clip.py
  python pipeline/extract_clip.py --video_id DGbwtVtthu8
  python pipeline/extract_clip.py --video_id DGbwtVtthu8 --start 30
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
    """Extrait CLIP_DURATION secondes à FPS natif depuis `start` secondes.

    Retourne le FPS natif. Si l'extraction existe déjà avec le même `start`,
    elle est ignorée. Si `start` a changé, les frames sont réécrasées.
    """
    video_path = RAW_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(
            f"Vidéo source absente : {video_path}\n"
            "Lancez d'abord pipeline/extract_frames.py pour la télécharger."
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
                f"[extract_clip] {video_id} — {len(existing)} frames déjà présentes "
                f"(start={start}s, {fps} fps) — ignoré."
            )
            return fps
        print(f"[extract_clip] {video_id} — start différent, réextraction.")

    fps = get_native_fps(video_path)
    print(
        f"[extract_clip] {video_id} — FPS : {fps}  |  start : {start}s  |  durée : {CLIP_DURATION}s"
    )

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
    print(f"[extract_clip] {n} frames extraites → {hr_dir}")
    return fps


def degrade_clip(video_id: str) -> None:
    """Génère les frames LR (dégradation bicubique) pour le clip."""
    hr_dir = RUNS_DIR / video_id / "frames_hr"
    lr_dir = RUNS_DIR / video_id / "frames_lr"
    lr_dir.mkdir(parents=True, exist_ok=True)

    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)
    frames = sorted(hr_dir.glob("frame_*.png"))

    if len(list(lr_dir.glob("frame_*.png"))) == len(frames):
        print(f"[degrade_clip] {video_id} — frames LR déjà présentes — ignoré.")
        return

    print(f"[degrade_clip] {video_id} — dégradation ×{scale}  ({len(frames)} frames)")
    for f in frames:
        img = cv2.imread(str(f))
        cv2.imwrite(str(lr_dir / f.name), degrade_phase1(img, scale))
    print(f"[degrade_clip] OK → {lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", default="33mqqm4QlJ8")
    parser.add_argument(
        "--start", type=float, default=0.0, help="Début du clip en secondes (défaut : 0)"
    )
    args = parser.parse_args()

    extract_clip(args.video_id, start=args.start)
    degrade_clip(args.video_id)
