"""
prepare_vimeo.py — Préparation du dataset Vimeo-90K pour FSRCNN.

Ce script (1 seul lancement) :
  1. Scanne tous les clips dans /Volumes/DisqueGab/vimeo_septuplet/sequences/
  2. Génère les LR (bicubique ×SCALE) pour chaque frame de chaque clip
  3. Sauvegarde tout dans /Volumes/DisqueGab/vimeo_sr/sequences_lr/ (même arborescence)
  4. Crée les fichiers de split train/val/test dans /Volumes/DisqueGab/vimeo_sr/splits/
  5. Tout reste sur DisqueGab — rien n'est écrit sur le SSD interne

Résumable : les clips déjà traités sont détectés et sautés.

Usage :
    python prepare_vimeo.py
    python prepare_vimeo.py --scale 4 --workers 4 --val 0.1 --test 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

# ── Chemins (tout sur DisqueGab) ──────────────────────────────────────────────
DISK          = Path("/Volumes/DisqueGab")
SRC_SEQ_DIR   = DISK / "vimeo_septuplet" / "sequences"
DST_ROOT      = DISK / "vimeo_sr"
DST_LR_DIR    = DST_ROOT / "sequences_lr"
DST_SPLIT_DIR = DST_ROOT / "splits"
STATS_FILE    = DST_ROOT / "stats.json"

N_FRAMES = 7          # im1.png … im7.png par clip
MIN_SIZE = (32, 32)   # taille minimale HR acceptable


# ── Traitement d'un clip ──────────────────────────────────────────────────────

def process_clip(clip_rel: str, scale: int) -> tuple[str, str]:
    """Dégrade les 7 frames d'un clip.

    Retourne (clip_rel, status) où status ∈ {"ok", "skip", "error:<msg>"}.
    Tout I/O reste sur DisqueGab.
    """
    src_clip = SRC_SEQ_DIR / clip_rel
    dst_clip = DST_LR_DIR  / clip_rel

    # Resume : clip déjà traité ?
    existing = [f for f in dst_clip.glob("im*.png")] if dst_clip.exists() else []
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

        lr = cv2.resize(hr, (w // scale, h // scale),
                        interpolation=cv2.INTER_CUBIC)
        if not cv2.imwrite(str(dst_path), lr):
            return clip_rel, f"error:write failed im{i}.png"

    return clip_rel, "ok"


# ── Split ─────────────────────────────────────────────────────────────────────

def make_splits(clips: list[str], val_ratio: float, test_ratio: float,
                seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)

    n      = len(shuffled)
    n_val  = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def write_splits(splits: dict[str, list[str]]) -> None:
    DST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, clips in splits.items():
        out = DST_SPLIT_DIR / f"{name}.txt"
        out.write_text("\n".join(sorted(clips)) + "\n")
        print(f"  {name:5s}: {len(clips):6d} clips → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(scale: int, workers: int, val_ratio: float, test_ratio: float,
         seed: int) -> None:

    if not SRC_SEQ_DIR.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {SRC_SEQ_DIR}\n"
            "Vérifiez que DisqueGab est monté et que le chemin est correct."
        )

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    DST_LR_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Scan des clips ─────────────────────────────────────────────────────
    print("Scan des clips…", flush=True)
    clips: list[str] = []
    for group_dir in sorted(SRC_SEQ_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for clip_dir in sorted(group_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            # Vérifie qu'il y a au moins im1.png (ignore dossiers vides)
            if (clip_dir / "im1.png").exists():
                clips.append(f"{group_dir.name}/{clip_dir.name}")

    print(f"  {len(clips)} clips trouvés dans {SRC_SEQ_DIR}")
    print(f"  → {len(clips) * N_FRAMES} frames à traiter (scale ×{scale})")

    # ── 2. Dégradation + sauvegarde ───────────────────────────────────────────
    print(f"\nDégradation (×{scale}) avec {workers} worker(s)…", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_error = 0
    done = 0
    report_every = max(1, len(clips) // 100)   # log tous les 1 %

    if workers == 1:
        # Mode séquentiel (plus simple, debug)
        for clip_rel in clips:
            _, status = process_clip(clip_rel, scale)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_error += 1
                print(f"  ERREUR {clip_rel}: {status}")
            done += 1
            if done % report_every == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(clips) - done) / rate
                pct     = 100.0 * done / len(clips)
                print(f"  {pct:5.1f}%  {done}/{len(clips)} clips"
                      f"  ok={n_ok} skip={n_skip} err={n_error}"
                      f"  {rate:.1f} clips/s  ETA {eta/60:.0f}min",
                      flush=True)
    else:
        from functools import partial
        worker_fn = partial(process_clip, scale=scale)
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(worker_fn, c): c for c in clips}
            for fut in as_completed(futures):
                _, status = fut.result()
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_error += 1
                    print(f"  ERREUR {futures[fut]}: {status}")
                done += 1
                if done % report_every == 0:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (len(clips) - done) / rate
                    pct     = 100.0 * done / len(clips)
                    print(f"  {pct:5.1f}%  {done}/{len(clips)} clips"
                          f"  ok={n_ok} skip={n_skip} err={n_error}"
                          f"  {rate:.1f} clips/s  ETA {eta/60:.0f}min",
                          flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDégradation terminée en {elapsed_total/60:.1f} min")
    print(f"  Traités : {n_ok}  Sautés : {n_skip}  Erreurs : {n_error}")

    # ── 3. Splits ─────────────────────────────────────────────────────────────
    print("\nGénération des splits…")
    good_clips = [c for c in clips
                  if len(list((DST_LR_DIR / c).glob("im*.png"))) == N_FRAMES]
    print(f"  Clips valides pour le split : {len(good_clips)}")

    splits = make_splits(good_clips, val_ratio, test_ratio, seed)
    write_splits(splits)

    # ── 4. Stats ──────────────────────────────────────────────────────────────
    # Échantillon pour vérification des dimensions
    sample_clip = good_clips[0] if good_clips else None
    hr_shape = lr_shape = None
    if sample_clip:
        hr = cv2.imread(str(SRC_SEQ_DIR / sample_clip / "im4.png"))
        lr = cv2.imread(str(DST_LR_DIR  / sample_clip / "im4.png"))
        if hr is not None and lr is not None:
            hr_shape = list(hr.shape)
            lr_shape = list(lr.shape)

    stats = {
        "scale":        scale,
        "n_clips_total":  len(clips),
        "n_clips_valid":  len(good_clips),
        "n_frames_valid": len(good_clips) * N_FRAMES,
        "n_train":      len(splits["train"]),
        "n_val":        len(splits["val"]),
        "n_test":       len(splits["test"]),
        "hr_shape_hwc": hr_shape,
        "lr_shape_hwc": lr_shape,
        "elapsed_s":    round(elapsed_total, 1),
        "errors":       n_error,
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(f"\nStats → {STATS_FILE}")
    print(json.dumps(stats, indent=2))

    print("\nDataset prêt.")
    print(f"  HR  : {SRC_SEQ_DIR}  (originaux, intacts)")
    print(f"  LR  : {DST_LR_DIR}")
    print(f"  Splits : {DST_SPLIT_DIR}/{{train,val,test}}.txt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prépare le dataset Vimeo-90K pour l'entraînement FSRCNN."
    )
    parser.add_argument("--scale",   type=int,   default=4,
                        help="Facteur de dégradation (défaut: 4)")
    parser.add_argument("--workers", type=int,   default=4,
                        help="Nombre de processus parallèles (défaut: 4 ; 1 = séquentiel)")
    parser.add_argument("--val",     type=float, default=0.1,
                        help="Fraction val   (défaut: 0.10)")
    parser.add_argument("--test",    type=float, default=0.1,
                        help="Fraction test  (défaut: 0.10)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Seed aléatoire (défaut: 42)")
    args = parser.parse_args()
    main(args.scale, args.workers, args.val, args.test, args.seed)
