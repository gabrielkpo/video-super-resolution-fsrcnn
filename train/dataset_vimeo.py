"""Vimeo-90K dataset for FSRCNN fine-tuning.

Expected layout on the data disk (produced by prepare_vimeo.py):
    <VSR_DATA_DISK>/vimeo_sr/sequences_lr/XXXXX/YYYY/im{1..7}.png     (LR 64×112)
    <VSR_DATA_DISK>/vimeo_septuplet/sequences/XXXXX/YYYY/im{1..7}.png  (HR 256×448)
    <VSR_DATA_DISK>/vimeo_sr/splits/{train,val,test}.txt
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_DISK

LR_ROOT = DATA_DISK / "vimeo_sr" / "sequences_lr"
HR_ROOT = DATA_DISK / "vimeo_septuplet" / "sequences"
SPLITS = DATA_DISK / "vimeo_sr" / "splits"


class VimeoSRDataset(Dataset):
    """(LR Y patch, HR Y patch) pairs drawn from Vimeo-90K.

    Y channel only, as at FSRCNN inference (Cr/Cb are upscaled bicubically).
    Random patches and augmentation on every access, to maximise diversity.
    """

    def __init__(
        self, split: str = "train", patch_size: int = 32, scale: int = 4, augment: bool = True
    ) -> None:
        split_file = SPLITS / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\nRun first: python prepare_vimeo.py"
            )
        lines = split_file.read_text().splitlines()
        clips = [line.strip() for line in lines if line.strip()]
        if not clips:
            raise ValueError(f"Split '{split}' is empty.")

        # Drop clips whose im4.png is missing on the LR or HR side (truncated clips)
        valid_clips, n_bad = [], 0
        for c in clips:
            if (LR_ROOT / c / "im4.png").exists() and (HR_ROOT / c / "im4.png").exists():
                valid_clips.append(c)
            else:
                n_bad += 1
        if n_bad:
            print(f"  [dataset] {n_bad} clip(s) excluded (missing im4) out of {len(clips)}")

        # One item = (clip_rel, frame index 1–7)
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

        # Fall back to im4 if a single frame is corrupted
        if lr_bgr is None:
            lr_bgr = cv2.imread(str(LR_ROOT / clip_rel / "im4.png"))
        if hr_bgr is None:
            hr_bgr = cv2.imread(str(HR_ROOT / clip_rel / "im4.png"))
        if lr_bgr is None or hr_bgr is None:
            # Disk unreachable or clip fully corrupted: fail fast
            if not DATA_DISK.exists():
                raise OSError(f"Data disk not mounted: {DATA_DISK}")
            raise OSError(f"Unreadable clip (frame and im4 missing): {clip_rel}")

        # Y channel only, float32 normalised to [0, 1]
        lr_y = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
        hr_y = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        # Random crop, aligned between LR and HR
        lh, lw = lr_y.shape
        ps, sc = self.ps, self.scale
        y0 = random.randint(0, lh - ps)
        x0 = random.randint(0, lw - ps)
        lr_patch = lr_y[y0 : y0 + ps, x0 : x0 + ps]
        hr_patch = hr_y[y0 * sc : (y0 + ps) * sc, x0 * sc : (x0 + ps) * sc]

        # Augmentation (horizontal/vertical flip, 90° rotation)
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
            torch.from_numpy(hr_patch).unsqueeze(0),  # (1, ps*sc, ps*sc)
        )
