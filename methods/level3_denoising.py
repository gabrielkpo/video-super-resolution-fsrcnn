"""Level 3 — denoising then upscaling: bilateral, NLM, BM3D."""

import cv2
import numpy as np

try:
    import bm3d as _bm3d_lib

    _BM3D_AVAILABLE = True
except ImportError:
    _BM3D_AVAILABLE = False


def bilateral_sr(
    img: np.ndarray, scale: int, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    h, w = denoised.shape[:2]
    return cv2.resize(denoised, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def nlm_sr(
    img: np.ndarray, scale: int, h_param: float = 10, template_size: int = 7, search_size: int = 21
) -> np.ndarray:
    if img.ndim == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h_param, h_param, template_size, search_size
        )
    else:
        denoised = cv2.fastNlMeansDenoising(img, None, h_param, template_size, search_size)
    h, w = denoised.shape[:2]
    return cv2.resize(denoised, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def bm3d_sr(img: np.ndarray, scale: int, sigma_psd: float = 25) -> np.ndarray:
    if not _BM3D_AVAILABLE:
        raise ImportError("bm3d is not installed: pip install bm3d")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    denoised = _bm3d_lib.bm3d(gray.astype(float) / 255.0, sigma_psd / 255.0)
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    h, w = denoised.shape[:2]
    up = cv2.resize(denoised, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    if img.ndim == 3:
        up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    return up


def run_all(img: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    results = {
        "bilateral_sr": bilateral_sr(img, scale),
        "nlm_sr": nlm_sr(img, scale),
    }
    if _BM3D_AVAILABLE:
        results["bm3d_sr"] = bm3d_sr(img, scale)
    return results
