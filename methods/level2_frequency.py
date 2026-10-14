"""Level 2 — frequency-domain methods: DWT and FFT zero-padding."""

import cv2
import numpy as np
import pywt


def fft_zeropad(img: np.ndarray, scale: int) -> np.ndarray:
    """Zero-padding in the frequency domain (FFT shift)."""

    def _upscale_channel(channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        F = np.fft.fftshift(np.fft.fft2(channel))
        ph, pw = h * scale, w * scale
        pad_h, pad_w = (ph - h) // 2, (pw - w) // 2
        F_pad = np.zeros((ph, pw), dtype=complex)
        F_pad[pad_h : pad_h + h, pad_w : pad_w + w] = F * (scale**2)
        return np.abs(np.fft.ifft2(np.fft.ifftshift(F_pad)))

    if img.ndim == 2:
        out = _upscale_channel(img)
    else:
        out = np.stack([_upscale_channel(img[:, :, c]) for c in range(img.shape[2])], axis=2)

    return np.clip(out, 0, 255).astype(np.uint8)


def dwt_upscale(img: np.ndarray, scale: int, wavelet: str = "haar") -> np.ndarray:
    """Upscaling in the wavelet domain.

    Strategy: one-level DWT → upscale each sub-band (LL, LH, HL, HH)
    by `scale` → IDWT → output at ×scale resolution.
    LL is upscaled bicubically (low-frequency content);
    the detail sub-bands are upscaled bilinearly.
    """

    def _upscale_channel(channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        LL, (LH, HL, HH) = pywt.dwt2(channel.astype(float), wavelet)

        # Upscale each sub-band by `scale` from its actual size
        # (long-filter wavelets produce sub-bands slightly larger than h//2,
        # so the target size must not be fixed in advance)
        sh, sw = LL.shape[0] * scale, LL.shape[1] * scale
        LL_up = cv2.resize(LL, (sw, sh), interpolation=cv2.INTER_CUBIC)
        LH_up = cv2.resize(LH, (sw, sh), interpolation=cv2.INTER_LINEAR)
        HL_up = cv2.resize(HL, (sw, sh), interpolation=cv2.INTER_LINEAR)
        HH_up = cv2.resize(HH, (sw, sh), interpolation=cv2.INTER_LINEAR)

        rec = pywt.idwt2((LL_up, (LH_up, HL_up, HH_up)), wavelet)
        return rec[: h * scale, : w * scale]

    if img.ndim == 2:
        out = _upscale_channel(img)
    else:
        out = np.stack([_upscale_channel(img[:, :, c]) for c in range(img.shape[2])], axis=2)

    return np.clip(out, 0, 255).astype(np.uint8)


def run_all(img: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    return {
        "fft_zeropad": fft_zeropad(img, scale),
        "dwt_haar": dwt_upscale(img, scale, wavelet="haar"),
        "dwt_db2": dwt_upscale(img, scale, wavelet="db2"),
    }
