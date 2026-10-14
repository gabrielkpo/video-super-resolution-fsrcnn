"""PSNR, SSIM, et métriques énergétiques via CodeCarbon."""

from contextlib import contextmanager

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import COUNTRY_ISO

try:
    from codecarbon import EmissionsTracker

    _CODECARBON = True
except ImportError:
    _CODECARBON = False


def psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    return peak_signal_noise_ratio(hr, sr, data_range=255)


def ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    channel_axis = 2 if hr.ndim == 3 else None
    return structural_similarity(hr, sr, data_range=255, channel_axis=channel_axis)


@contextmanager
def energy_tracker(name: str = "run"):
    """Context manager retournant (emissions_kg, energy_kwh) à la sortie."""
    if not _CODECARBON:
        yield lambda: (None, None)
        return

    tracker = EmissionsTracker(
        project_name=name,
        country_iso_code=COUNTRY_ISO,
        log_level="error",
        save_to_file=False,
    )
    tracker.start()
    result = {}

    class _Result:
        def __call__(self):
            return result.get("emissions"), result.get("energy")

    r = _Result()
    try:
        yield r
    finally:
        emissions = tracker.stop()
        result["emissions"] = emissions
        result["energy"] = getattr(tracker, "_total_energy", None)


def evaluate(hr: np.ndarray, sr: np.ndarray) -> dict:
    return {"psnr": psnr(hr, sr), "ssim": ssim(hr, sr)}
