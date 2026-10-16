"""Classical single-image deconvolution baselines.

Wiener (FFT, scalar NSR) and Richardson-Lucy (iterative, spatial). Both take
a grayscale blurred image in [0,1] and a non-negative PSF normalised to sum
1, and return a deblurred image in [0,1]. These are the floor the learned
restorer has to clear before any "learned deblurring" claim is defensible.
"""
from __future__ import annotations

import cv2
import numpy as np


def _pad_kernel(kernel: np.ndarray, shape) -> np.ndarray:
    """Pad `kernel` into an array of `shape`, centred then shifted so the
    kernel's centre lands at (0, 0) in Fourier terms."""
    H, W = shape
    kh, kw = kernel.shape
    out = np.zeros((H, W), dtype=np.float32)
    out[:kh, :kw] = kernel
    out = np.roll(out, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return out


def wiener_deconvolve(
    blurred_gray: np.ndarray,
    kernel: np.ndarray,
    nsr: float = 0.01,
) -> np.ndarray:
    """Scalar-NSR Wiener filter. `blurred_gray` and the output are float32
    in [0, 1]. `nsr` is the inverse SNR; higher = smoother result."""
    assert blurred_gray.ndim == 2
    img = blurred_gray.astype(np.float32)
    k_pad = _pad_kernel(kernel.astype(np.float32), img.shape)
    K = np.fft.fft2(k_pad)
    B = np.fft.fft2(img)
    W = np.conj(K) / (np.abs(K) ** 2 + nsr)
    out = np.real(np.fft.ifft2(W * B))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def richardson_lucy(
    blurred_gray: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Classical Richardson-Lucy on grayscale in [0, 1]. The kernel is
    assumed non-negative; small/negative entries get clipped before use."""
    assert blurred_gray.ndim == 2
    k = np.clip(kernel, 0, None).astype(np.float32)
    s = k.sum()
    if s < 1e-9:
        return blurred_gray.astype(np.float32)
    k = k / s
    k_mirror = k[::-1, ::-1]

    estimate = blurred_gray.astype(np.float32).clip(eps, 1.0)
    for _ in range(iterations):
        reblurred = cv2.filter2D(estimate, -1, k, borderType=cv2.BORDER_REPLICATE)
        reblurred = np.clip(reblurred, eps, None)
        ratio = blurred_gray / reblurred
        correction = cv2.filter2D(ratio, -1, k_mirror, borderType=cv2.BORDER_REPLICATE)
        estimate = estimate * correction
    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def deblur_rgb(
    img_rgb_u8: np.ndarray,
    kernel: np.ndarray,
    method: str = "wiener",
    nsr: float = 0.01,
    iterations: int = 20,
) -> np.ndarray:
    """Deblur each colour channel independently and return uint8 RGB."""
    assert img_rgb_u8.dtype == np.uint8 and img_rgb_u8.ndim == 3
    out = np.empty_like(img_rgb_u8)
    for c in range(3):
        chan = img_rgb_u8[..., c].astype(np.float32) / 255.0
        if method == "wiener":
            d = wiener_deconvolve(chan, kernel, nsr=nsr)
        elif method == "rl":
            d = richardson_lucy(chan, kernel, iterations=iterations)
        else:
            raise ValueError(f"unknown method {method}")
        out[..., c] = (d * 255.0).round().clip(0, 255).astype(np.uint8)
    return out
