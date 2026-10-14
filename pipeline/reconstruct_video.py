"""Reconstruit une vidéo fluide à partir des frames SR/HR/LR d'un run.

Usage :
  python pipeline/reconstruct_video.py --method hr
  python pipeline/reconstruct_video.py --method fft_zeropad
  python pipeline/reconstruct_video.py --method dwt_haar --video_id DGbwtVtthu8
"""

import argparse
import subprocess
from pathlib import Path

import cv2

import methods.level1_interpolation as l1
from config import RUNS_DIR, SCALE_FACTOR, VIDEO_SCALE


def get_fps(video_id: str) -> float:
    fps_file = RUNS_DIR / video_id / "frames_hr" / "fps.txt"
    if not fps_file.exists():
        raise FileNotFoundError(
            f"fps.txt introuvable dans {RUNS_DIR / video_id / 'frames_hr'}.\n"
            "Lancez d'abord : python pipeline/extract_clip.py"
        )
    return float(fps_file.read_text().strip())


def apply_sr(video_id: str, method_name: str, fn) -> Path:
    """Applique fn(lr, scale) sur toutes les frames LR du run. Retourne le dossier SR."""
    lr_dir = RUNS_DIR / video_id / "frames_lr"
    sr_dir = RUNS_DIR / video_id / "frames_sr" / method_name
    sr_dir.mkdir(parents=True, exist_ok=True)

    scale = VIDEO_SCALE.get(video_id, SCALE_FACTOR)
    frames = sorted(lr_dir.glob("frame_*.png"))

    if len(list(sr_dir.glob("frame_*.png"))) == len(frames):
        print(f"[apply_sr] {method_name}/{video_id} — frames déjà présentes — ignoré.")
        return sr_dir

    print(f"[apply_sr] {method_name}  {len(frames)} frames ×{scale} …")
    for i, f in enumerate(frames, 1):
        sr = fn(cv2.imread(str(f)), scale)
        cv2.imwrite(str(sr_dir / f.name), sr)
        if i % 20 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}", end="\r")
    print()
    print(f"[apply_sr] OK → {sr_dir}")
    return sr_dir


def frames_to_video(frames_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"[reconstruct] → {output_path}")


def reconstruct(video_id: str, method_name: str, fn=None) -> None:
    fps = get_fps(video_id)
    out_path = RUNS_DIR / video_id / "videos" / f"{video_id}_{method_name}.mp4"

    if method_name == "hr":
        frames_dir = RUNS_DIR / video_id / "frames_hr"
    elif method_name == "lr":
        frames_dir = apply_sr(video_id, "lr_bicubic", lambda img, s: l1.upscale(img, s, "bicubic"))
    else:
        if fn is None:
            raise ValueError("fn requis pour une méthode SR custom.")
        frames_dir = apply_sr(video_id, method_name, fn)

    frames_to_video(frames_dir, out_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", default="33mqqm4QlJ8")
    parser.add_argument(
        "--method", default="fft_zeropad", help="Nom de la méthode SR, ou 'hr' / 'lr'"
    )
    args = parser.parse_args()

    import methods.level2_frequency as l2

    METHOD_MAP = {
        "bicubic": lambda img, s: l1.upscale(img, s, "bicubic"),
        "fft_zeropad": lambda img, s: l2.fft_zeropad(img, s),
        "dwt_haar": lambda img, s: l2.dwt_upscale(img, s, wavelet="haar"),
        "dwt_db2": lambda img, s: l2.dwt_upscale(img, s, wavelet="db2"),
    }

    fn = METHOD_MAP.get(args.method)
    reconstruct(args.video_id, args.method, fn)
