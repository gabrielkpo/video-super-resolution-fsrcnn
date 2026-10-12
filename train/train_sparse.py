"""Entraînement du Sparse Coding SR.

Sources disponibles :
    --source div2k    (défaut) : 100 images DIV2K haute qualité
    --source frames   : toutes les frames benchmark (1 FPS, très diversifiées)

Usage :
    python train/train_sparse.py --source frames
    python train/train_sparse.py --source div2k --n_imgs 100
"""
import sys
import warnings
import argparse
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIV2K_DIR, SCALE_FACTOR, FRAMES_HR_DIR, FRAMES_LR_PHASE1_DIR
from methods.level4_ml_classic import SparseSR

MODEL_PATH = Path(__file__).parent / "sparse_sr.joblib"


def load_div2k_pairs(n_imgs: int, scale: int) -> list:
    hr_dir = DIV2K_DIR / "DIV2K_train_HR"
    lr_dir = DIV2K_DIR / f"DIV2K_train_LR_bicubic_X{scale}" / f"X{scale}"

    if not hr_dir.exists():
        print(f"[train_sparse] DIV2K introuvable : {hr_dir}")
        print("  → Télécharger depuis https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        sys.exit(1)

    pairs = []
    for hr_path in sorted(hr_dir.glob("*.png"))[:n_imgs]:
        hr_img = cv2.imread(str(hr_path))
        if hr_img is None:
            continue
        lr_path = lr_dir / f"{hr_path.stem}x{scale}.png"
        lr_img  = cv2.imread(str(lr_path)) if lr_path.exists() else _degrade(hr_img, scale)
        if lr_img is not None:
            pairs.append((lr_img, hr_img))
    return pairs


def load_frames_pairs() -> list:
    """Charge toutes les frames benchmark (1 FPS) — diversité maximale disponible localement."""
    pairs = []
    for lr_video_dir in sorted(FRAMES_LR_PHASE1_DIR.iterdir()):
        if not lr_video_dir.is_dir():
            continue
        hr_video_dir = FRAMES_HR_DIR / lr_video_dir.name
        for lr_f in sorted(lr_video_dir.glob("*.png")):
            hr_f = hr_video_dir / lr_f.name
            if not hr_f.exists():
                continue
            lr = cv2.imread(str(lr_f))
            hr = cv2.imread(str(hr_f))
            if lr is not None and hr is not None:
                pairs.append((lr, hr))
    return pairs


def _degrade(hr_img, scale: int):
    h, w = hr_img.shape[:2]
    return cv2.resize(hr_img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     choices=["div2k", "frames", "combined"], default="div2k")
    parser.add_argument("--n_imgs",     type=int, default=100, help="Nb images DIV2K (ignoré si --source frames)")
    parser.add_argument("--n_atoms",    type=int, default=256)
    parser.add_argument("--n_nonzero",  type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=8)
    args = parser.parse_args()

    if args.source == "frames":
        print("[train_sparse] Source : frames benchmark (1 FPS)…")
        pairs = load_frames_pairs()
    elif args.source == "combined":
        scale = SCALE_FACTOR
        print(f"[train_sparse] Source : DIV2K ({args.n_imgs} images) + frames benchmark…")
        pairs = load_div2k_pairs(args.n_imgs, scale) + load_frames_pairs()
        print(f"[train_sparse] {len(pairs)} paires au total.")
    else:
        scale = SCALE_FACTOR
        print(f"[train_sparse] Source : DIV2K ({args.n_imgs} images, ×{scale})…")
        pairs = load_div2k_pairs(args.n_imgs, scale)

    if not pairs:
        print("[train_sparse] Aucune paire chargée.")
        sys.exit(1)

    print(f"[train_sparse] {len(pairs)} paires chargées.")

    model = SparseSR(
        n_atoms=args.n_atoms,
        patch_size=args.patch_size,
        stride_train=6,
        stride_infer=2,
        n_nonzero=args.n_nonzero,
        scale=SCALE_FACTOR,
        max_patches=60_000,
    )
    model.fit([p[0] for p in pairs], [p[1] for p in pairs])
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
