"""Fine-tune FSRCNN from DIV2K to Vimeo-90K.

Loads the checkpoint pre-trained on DIV2K and fine-tunes it on the Vimeo frames.
The dataset stays on the data disk, nothing is copied to the internal drive.

Usage:
    python -m train.train_finetune
    python -m train.train_finetune --epochs 10 --lr 1e-5 --batch 32 --patience 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_DISK, DEVICE
from methods.level6_cnn import FSRCNN, load_model

CKPT_IN = Path(__file__).parent / "fsrcnn_div2k.ckpt"  # pre-trained on DIV2K
CKPT_OUT = Path(__file__).parent / "fsrcnn_vimeo_best.ckpt"  # best fine-tuned


# ── Metrics ───────────────────────────────────────────────────────────────────


def mse_to_psnr(mse: float) -> float:
    return 100.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


# ── Train / val epoch ─────────────────────────────────────────────────────────


def run_epoch(
    model: FSRCNN,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    epoch: int,
    n_epochs: int,
) -> tuple[float, float, float]:
    """Return (l1_loss, psnr, elapsed_s)."""
    training = optimizer is not None
    model.train(training)

    total_l1 = total_mse = n_samples = 0
    t0 = time.time()

    phase = "Train" if training else "Val  "
    bar = tqdm(
        loader,
        desc=f"Ep {epoch + 1}/{n_epochs} {phase}",
        unit="batch",
        ncols=90,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] L1={postfix}",
    )

    for lr_t, hr_t in bar:
        lr_t = lr_t.to(device)
        hr_t = hr_t.to(device)

        with torch.set_grad_enabled(training):
            out = model(lr_t)
            loss = loss_fn(out, hr_t)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = lr_t.size(0)
        total_l1 += loss.item() * bs
        total_mse += nn.functional.mse_loss(out.detach(), hr_t).item() * bs
        n_samples += bs

        bar.set_postfix_str(f"{total_l1 / n_samples:.5f}")

    bar.close()
    mean_l1 = total_l1 / n_samples
    mean_mse = total_mse / n_samples
    return mean_l1, mse_to_psnr(mean_mse), time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    epochs: int, lr: float, batch: int, patch: int, patience: int, workers: int, resume: bool
) -> None:
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not (DATA_DISK / "vimeo_sr" / "splits" / "train.txt").exists():
        print("[fine-tune] Dataset not ready: run prepare_vimeo.py first")
        sys.exit(1)
    if not CKPT_IN.exists():
        print(f"[fine-tune] DIV2K checkpoint not found: {CKPT_IN}")
        sys.exit(1)

    # Imported here, not at module level, to keep the DataLoader workers light
    from train.dataset_vimeo import VimeoSRDataset

    print(f"Device        : {DEVICE}")
    print(f"Checkpoint in : {CKPT_IN.name}")
    print(f"Checkpoint out: {CKPT_OUT.name}")
    print(f"LR={lr}  batch={batch}  patch={patch}×{patch}  epochs={epochs}  patience={patience}")

    # ── Datasets and loaders ──────────────────────────────────────────────────
    print("\nLoading the Vimeo datasets…")
    train_ds = VimeoSRDataset("train", patch_size=patch, augment=True)
    val_ds = VimeoSRDataset("val", patch_size=patch, augment=False)
    print(f"  Train : {len(train_ds):,} items  ({len(train_ds) // 7:,} clips × 7 frames)")
    print(f"  Val   : {len(val_ds):,} items  ({len(val_ds) // 7:,} clips × 7 frames)")

    loader_args = {
        "batch_size": batch,
        "num_workers": workers,
        "pin_memory": False,
        "persistent_workers": workers > 0,
    }
    train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_args)

    # ── Model ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_l1 = float("inf")
    patience_ctr = 0

    if resume and CKPT_OUT.exists():
        print("\nResuming from the Vimeo checkpoint…")
        resume_meta = torch.load(str(CKPT_OUT), map_location="cpu")
        model = load_model(CKPT_OUT, device=DEVICE)
        start_epoch = resume_meta["epoch"] + 1
        best_val_l1 = resume_meta["val_l1"]
        ra = resume_meta.get("args", {})
        d, s, m = ra.get("d", "?"), ra.get("s", "?"), ra.get("m", "?")
        print(f"  Resuming at epoch {start_epoch}  (best val_l1={best_val_l1:.5f})")
    else:
        print("\nLoading the DIV2K model…")
        model = load_model(CKPT_IN, device=DEVICE)
        meta = torch.load(str(CKPT_IN), map_location="cpu")
        sd = meta["model"] if "model" in meta else meta
        d = sd["feature.0.weight"].shape[0]
        s = sd["shrink.0.weight"].shape[0]
        m = sum(1 for k in sd if k.startswith("map.") and k.endswith(".bias"))
        print(f"  DIV2K epoch : {meta.get('epoch', '?')}")

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FSRCNN({d}, {s}, {m})  — {n_params:,} parameters")

    # ── Optimizer and loss ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Ep':>3}  {'Train L1':>10}  {'Val L1':>10}  {'Val PSNR':>9}  {'Time':>7}  Status")
    print("─" * 62)

    for epoch in range(start_epoch, epochs):
        tr_l1, _, t_tr = run_epoch(model, train_dl, loss_fn, optimizer, DEVICE, epoch, epochs)
        va_l1, va_psnr, t_va = run_epoch(model, val_dl, loss_fn, None, DEVICE, epoch, epochs)

        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            patience_ctr = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_l1": va_l1,
                    "val_psnr": va_psnr,
                    "args": {"lr": lr, "batch": batch, "patch": patch, "d": d, "s": s, "m": m},
                },
                str(CKPT_OUT),
            )
            status = "✓ saved"
        else:
            patience_ctr += 1
            status = f"patience {patience_ctr}/{patience}"

        print(
            f"{epoch:>3}  {tr_l1:>10.5f}  {va_l1:>10.5f}  "
            f"{va_psnr:>8.2f}dB  {t_tr + t_va:>6.0f}s  {status}"
        )

        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\n{'─' * 62}")
    print(f"Best val L1      : {best_val_l1:.5f}")
    print(f"Final checkpoint : {CKPT_OUT}")
    print("\nTo use the fine-tuned model:")
    print("  from methods.level6_cnn import load_model")
    print("  model = load_model('train/fsrcnn_vimeo_best.ckpt')")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune FSRCNN (DIV2K) on Vimeo-90K.")
    ap.add_argument("--epochs", type=int, default=10, help="max epochs (default: 10)")
    ap.add_argument(
        "--lr", type=float, default=1e-5, help="learning rate (default: 1e-5, 10× below DIV2K)"
    )
    ap.add_argument("--batch", type=int, default=32, help="batch size (default: 32)")
    ap.add_argument(
        "--patch", type=int, default=32, help="LR patch size in pixels (default: 32 → HR 128×128)"
    )
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience (default: 3)")
    ap.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers (default: 4; 0 = no processes)"
    )
    ap.add_argument(
        "--resume", action="store_true", help="resume from fsrcnn_vimeo_best.ckpt if present"
    )
    args = ap.parse_args()
    main(args.epochs, args.lr, args.batch, args.patch, args.patience, args.workers, args.resume)
