"""Level 4 — classical machine learning: sparse coding on LR/HR patches."""

import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Ridge, orthogonal_mp

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════


def _to_y(img: np.ndarray) -> np.ndarray:
    """Y channel (YCrCb luma, BT.601), same formula as BGR2GRAY."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0] if img.ndim == 3 else img


def _lr_features(patch: np.ndarray) -> np.ndarray:
    """
    Yang et al. (2010) descriptor: first and second order gradients (Sobel).
    Invariant to the DC component, consistent with the high-frequency residual to predict.
    """
    p = patch.astype(np.float32) / 255.0
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(p, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(p, cv2.CV_32F, 0, 2, ksize=3)
    return np.concatenate([gx.flatten(), gy.flatten(), gxx.flatten(), gyy.flatten()])


def _extract_pairs(
    lr_gray: np.ndarray, hr_gray: np.ndarray, patch_size: int, stride: int, scale: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (LR features, HR residual) pairs.

    A p×p LR patch covers a (p·scale)×(p·scale) HR region.
    Residual = HR_patch - bic_global_patch, with the bicubic computed on the WHOLE image,
    exactly as at inference (same OpenCV centring convention).
    Guarantee: smooth patch → features ≈ 0 → α ≈ 0 → residual ≈ 0 → output = bicubic.
    """
    h, w = lr_gray.shape
    H, W = h * scale, w * scale
    ph_hr = patch_size * scale

    # Global bicubic, computed as at inference to avoid a sub-pixel shift
    bic_global = cv2.resize(lr_gray.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

    lr_feats, hr_residuals = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            lr_patch = lr_gray[y : y + patch_size, x : x + patch_size]
            y_hr, x_hr = y * scale, x * scale
            hr_patch = hr_gray[y_hr : y_hr + ph_hr, x_hr : x_hr + ph_hr]
            bic_patch = bic_global[y_hr : y_hr + ph_hr, x_hr : x_hr + ph_hr]

            if hr_patch.shape != (ph_hr, ph_hr):
                continue  # incomplete border patch

            residual = (hr_patch.astype(np.float32) - bic_patch) / 128.0

            lr_feats.append(_lr_features(lr_patch))
            hr_residuals.append(residual.flatten())

    return np.array(lr_feats, dtype=np.float32), np.array(hr_residuals, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Sparse coding SR — coupled DLR / DHR dictionaries
# ══════════════════════════════════════════════════════════════════════════════


class SparseSR:
    """
    Sparse-coding super-resolution with coupled DLR and DHR dictionaries.

    Residual formulation (Yang et al., 2010):
        DLR learns the gradient (Sobel) descriptors of the LR patches
        DHR learns the high-frequency residuals HR - bicubic(LR)

    A p×p LR patch maps to a (p·scale)×(p·scale) HR region.
    Graceful degradation: smooth areas → residual ≈ 0 → output = bicubic.

    Training:
        1. Extract LR gradient descriptors and full HR residuals
        2. Learn DLR with MiniBatchDictionaryLearning (ODL)
        3. Compute the α codes with OMP on DLR
        4. Learn DHR by ridge regression: DHR = ridge.fit(A, X_res)

    Inference:
        α* = OMP(DLR, grad(LR_patch))
        SR = bicubic(LR) + DHR · α*
    """

    def __init__(
        self,
        n_atoms: int = 256,
        patch_size: int = 8,
        stride_train: int = 4,
        stride_infer: int = 4,
        n_nonzero: int = 3,
        scale: int = 4,
        max_patches: int = 60_000,
    ):
        self.n_atoms = n_atoms
        self.patch_size = patch_size
        self.stride_train = stride_train
        self.stride_infer = stride_infer
        self.n_nonzero = n_nonzero
        self.scale = scale
        self.max_patches = max_patches
        self.D_lr = None  # (n_atoms, d_lr)
        self.D_hr = None  # (d_hr,   n_atoms)   d_hr = (patch_size * scale)²

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, lr_imgs: list, hr_imgs: list) -> "SparseSR":
        """Learn DLR and DHR from (LR, HR) image pairs."""

        # Step 1: extraction, stopped early to avoid running out of memory
        print(f"[SparseSR] Extracting patches ({len(lr_imgs)} images)…")
        all_lr, all_hr = [], []
        total = 0
        for lr, hr in zip(lr_imgs, hr_imgs, strict=False):
            lf, hr_res = _extract_pairs(
                _to_y(lr),
                _to_y(hr),
                self.patch_size,
                self.stride_train,
                self.scale,
            )
            all_lr.append(lf)
            all_hr.append(hr_res)
            total += len(lf)
            if total >= self.max_patches:
                break

        X_lr = np.vstack(all_lr)[: self.max_patches]
        X_res = np.vstack(all_hr)[: self.max_patches]
        print(f"[SparseSR] {len(X_lr):,} patches extracted.")

        # Step 2: ODL, learn the DLR dictionary
        print("[SparseSR] Learning DLR (ODL)…")
        odl = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            alpha=1.0,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=self.n_nonzero,
            max_iter=1000,
            batch_size=64,
            random_state=42,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            odl.fit(X_lr)
        self.D_lr = odl.components_  # (n_atoms, d_lr)

        # Step 3: sparse α codes with OMP
        print("[SparseSR] Computing the α codes (OMP)…")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = odl.transform(X_lr)  # (N, n_atoms)

        # Step 4: DHR by ridge regression (robust to the sparse conditioning)
        print("[SparseSR] Learning DHR (ridge)…")
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(A, X_res)
        self.D_hr = ridge.coef_  # (d_hr, n_atoms), sklearn multi-output convention

        expected_d_hr = (self.patch_size * self.scale) ** 2
        assert self.D_hr.shape == (expected_d_hr, self.n_atoms), (
            f"D_hr shape {self.D_hr.shape} ≠ ({expected_d_hr}, {self.n_atoms})"
        )

        print("[SparseSR] Training done.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, lr_img: np.ndarray) -> np.ndarray:
        """SR on the Y (luma) channel, bicubic upscale of the chroma channels."""
        assert self.D_lr is not None, "Call fit() or load() before predict()."

        is_color = lr_img.ndim == 3
        if is_color:
            ycrcb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)
            lr_y, lr_cr, lr_cb = cv2.split(ycrcb)
        else:
            lr_y = lr_img

        lr_gray = lr_y.astype(np.float32)
        h, w = lr_gray.shape
        ph = self.patch_size
        ph_hr = ph * self.scale
        H, W = h * self.scale, w * self.scale

        bicubic = cv2.resize(lr_gray, (W, H), interpolation=cv2.INTER_CUBIC)
        residual = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)

        positions, feats = [], []
        for y in range(0, h - ph + 1, self.stride_infer):
            for x in range(0, w - ph + 1, self.stride_infer):
                feats.append(_lr_features(lr_gray[y : y + ph, x : x + ph]))
                positions.append((y, x))

        X = np.array(feats, dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = orthogonal_mp(self.D_lr.T, X.T, n_nonzero_coefs=self.n_nonzero)

        RES_patches = self.D_hr @ A

        # Gating: zero the residual on patches without structure.
        # OMP always uses n_nonzero atoms, even on a flat patch, which would add fake detail.
        patch_energy = np.linalg.norm(X, axis=1)
        RES_patches[:, patch_energy < np.median(patch_energy) / 4] = 0.0

        for i, (y, x) in enumerate(positions):
            y_hr, x_hr = y * self.scale, x * self.scale
            if y_hr + ph_hr <= H and x_hr + ph_hr <= W:
                res_patch = RES_patches[:, i].reshape(ph_hr, ph_hr)
                residual[y_hr : y_hr + ph_hr, x_hr : x_hr + ph_hr] += res_patch
                weights[y_hr : y_hr + ph_hr, x_hr : x_hr + ph_hr] += 1.0

        weights = np.maximum(weights, 1)
        hr_y = np.clip(bicubic + (residual / weights) * 128.0, 0, 255).astype(np.uint8)

        if not is_color:
            return hr_y

        hr_cr = cv2.resize(lr_cr, (W, H), interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(lr_cb, (W, H), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(cv2.merge([hr_y, hr_cr, hr_cb]), cv2.COLOR_YCrCb2BGR)

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                k: getattr(self, k)
                for k in (
                    "D_lr",
                    "D_hr",
                    "n_atoms",
                    "patch_size",
                    "n_nonzero",
                    "scale",
                    "stride_train",
                    "stride_infer",
                )
            },
            path,
        )
        print(f"[SparseSR] Model saved → {path}")

    def load(self, path: Path | str) -> "SparseSR":
        for k, v in joblib.load(path).items():
            setattr(self, k, v)
        return self


def sparse_sr(lr_img: np.ndarray, scale: int, model_path: Path | str | None = None) -> np.ndarray:
    """Functional interface: load the pre-trained model and apply SR."""
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            f"Sparse model not found: {model_path}\nRun first: python -m train.train_sparse"
        )
    return SparseSR(scale=scale).load(model_path).predict(lr_img)
