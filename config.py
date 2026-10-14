from pathlib import Path

ROOT = Path(__file__).parent

# ── Données brutes (vidéos source) ───────────────────────────────────────────
DATA_DIR = ROOT / "data_ytb_1fps"
RAW_DIR = DATA_DIR / "raw"
DIV2K_DIR = ROOT / "div2k"
DIV2K_TRAIN_HR = DIV2K_DIR / "DIV2K_train_HR"
DIV2K_TRAIN_LR = DIV2K_DIR / "DIV2K_train_LR_bicubic_X4" / "X4"
DIV2K_VALID_HR = DIV2K_DIR / "DIV2K_valid_HR"
DIV2K_VALID_LR = DIV2K_DIR / "DIV2K_valid_LR_bicubic_X4" / "X4"

# ── Benchmark (1 fps, toutes durées)
FRAMES_HR_DIR = DATA_DIR / "frames_hr"
FRAMES_LR_DIR = DATA_DIR / "frames_lr"
FRAMES_LR_PHASE1_DIR = FRAMES_LR_DIR / "phase1_bicubic"
FRAMES_LR_PHASE2_DIR = FRAMES_LR_DIR / "phase2_noisy"

# ── Clips FPS natif ── structure par vidéo
#    data_ytb_24fps/<video_id>/frames_hr/
#    data_ytb_24fps/<video_id>/frames_lr/
#    data_ytb_24fps/<video_id>/frames_sr/<method>/
#    data_ytb_24fps/<video_id>/videos/
RUNS_DIR = ROOT / "data_ytb_24fps"

# ── Résultats (benchmark + visuels) ─────────────────────────────────────────
RESULTS_DIR = ROOT / "results"
RESULTS_VISUALS_DIR = RESULTS_DIR / "visuals"

# ── Facteurs d'upscaling ─────────────────────────────────────────────────────
SCALE_FACTOR = 4
VIDEO_SCALE = {
    "DGbwtVtthu8": 8,  # source 4K → LR ×8
    "33mqqm4QlJ8": 4,  # source 1080p → LR ×4
}

# ── Patches pour l'entraînement ──────────────────────────────────────────────
PATCH_SIZE = 33
STRIDE = 14

# ── Dégradation ──────────────────────────────────────────────────────────────
BLUR_KERNEL_SIZE = 3
NOISE_STD_PHASE2 = 25  # AWGN σ=25 (phase 2 uniquement)

# ── Entraînement CNN ─────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50


def _detect_device() -> str:
    import torch  # imported lazily: the classical methods do not need torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def __getattr__(name: str):
    """`from config import DEVICE` detects the device on first use only."""
    if name == "DEVICE":
        return _detect_device()
    raise AttributeError(name)


# ── CodeCarbon ───────────────────────────────────────────────────────────────
COUNTRY_ISO = "FRA"

# ── Vidéos source ────────────────────────────────────────────────────────────
VIDEO_URL1 = "https://www.youtube.com/watch?v=DGbwtVtthu8"
VIDEO_URL2 = "https://www.youtube.com/watch?v=33mqqm4QlJ8"
EXTRACT_FPS = 1  # 1 frame/seconde pour le benchmark sparse
CLIP_DURATION = 5  # secondes extraites à FPS natif pour les runs
