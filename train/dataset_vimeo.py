"""Dataset Vimeo-90K pour le fine-tuning FSRCNN.

Structure attendue sur DisqueGab (produite par prepare_vimeo.py) :
    /Volumes/DisqueGab/vimeo_sr/sequences_lr/XXXXX/YYYY/im{1..7}.png  (LR 64×112)
    /Volumes/DisqueGab/vimeo_septuplet/sequences/XXXXX/YYYY/im{1..7}.png  (HR 256×448)
    /Volumes/DisqueGab/vimeo_sr/splits/{train,val,test}.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DISK = Path("/Volumes/DisqueGab")
LR_ROOT = DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DISK / "vimeo_septuplet" / "sequences"
SPLITS = DISK / "vimeo_sr" / "splits"


class VimeoSRDataset(Dataset):
    """Paires (LR patch Y, HR patch Y) tirées du dataset Vimeo-90K.

    Canal Y uniquement — cohérent avec l'inférence FSRCNN (Cr/Cb en bicubique).
    Patches aléatoires + augmentation à chaque accès pour maximiser la diversité.
    """

    def __init__(
        self, split: str = "train", patch_size: int = 32, scale: int = 4, augment: bool = True
    ) -> None:
        split_file = SPLITS / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Fichier split introuvable : {split_file}\n"
                "Lancez d'abord : python prepare_vimeo.py"
            )
        clips = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        if not clips:
            raise ValueError(f"Split '{split}' est vide.")

        # Filtre les clips dont im4.png est absent côté LR ou HR (clips tronqués)
        valid_clips, n_bad = [], 0
        for c in clips:
            if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists():
                valid_clips.append(c)
            else:
                n_bad += 1
        if n_bad:
            print(f"  [dataset] {n_bad} clip(s) exclus (im4 manquant) sur {len(clips)}")

        # Un item = (clip_rel, indice de frame 1–7)
        self.items = [(c, i) for c in valid_clips for i in range(1, 8)]
        self.ps = patch_size
        self.scale = scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_rel, frame_idx = self.items[idx]

        lr_bgr = cv2.imread(str(LR_ROOT / clip_rel / f"im{frame_idx}.png"))
        hr_bgr = cv2.imread(str(HR_ROOT / clip_rel / f"im{frame_idx}.png"))

        # Fallback sur im4 si frame individuelle corrompue
        if lr_bgr is None:
            lr_bgr = cv2.imread(str(LR_ROOT / clip_rel / "im4.png"))
        if hr_bgr is None:
            hr_bgr = cv2.imread(str(HR_ROOT / clip_rel / "im4.png"))
        if lr_bgr is None or hr_bgr is None:
            # Disque inaccessible ou clip entièrement corrompu — fail fast
            if not DISK.exists():
                raise OSError(f"DisqueGab non monté : {DISK}")
            raise OSError(f"Clip illisible (LR et im4 absents) : {clip_rel}")

        # Canal Y uniquement — float32 normalisé [0, 1]
        lr_y = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
        hr_y = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        # Crop aléatoire aligné LR/HR
        lh, lw = lr_y.shape
        ps, sc = self.ps, self.scale
        y0 = random.randint(0, lh - ps)
        x0 = random.randint(0, lw - ps)
        lr_patch = lr_y[y0 : y0 + ps, x0 : x0 + ps]
        hr_patch = hr_y[y0 * sc : (y0 + ps) * sc, x0 * sc : (x0 + ps) * sc]

        # Augmentation (flip H/V + rotation 90°)
        if self.augment:
            if random.random() > 0.5:
                lr_patch = np.fliplr(lr_patch).copy()
                hr_patch = np.fliplr(hr_patch).copy()
            if random.random() > 0.5:
                lr_patch = np.flipud(lr_patch).copy()
                hr_patch = np.flipud(hr_patch).copy()
            k = random.randint(0, 3)
            if k:
                lr_patch = np.rot90(lr_patch, k).copy()
                hr_patch = np.rot90(hr_patch, k).copy()

        return (
            torch.from_numpy(lr_patch).unsqueeze(0),  # (1, ps, ps)
            torch.from_numpy(hr_patch).unsqueeze(0),
        )  # (1, ps*sc, ps*sc)
