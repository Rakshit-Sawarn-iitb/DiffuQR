"""Blur kernel estimation and summary statistics.

Given a pair of aligned grayscale patches (blurred frame and clean reference,
both in [0,1]), solve for the small PSF `k` that minimises ||B - k * R||^2
via a Tikhonov-regularised Fourier division. Only the central support is
kept and the kernel is normalised to sum 1 (so convolving the reference by
it produces the frame's brightness).
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def estimate_kernel_fft(
    blurred_gray: np.ndarray,
    reference_gray: np.ndarray,
    kernel_size: int = 25,
    eps: float = 1e-2,
) -> np.ndarray:
    """Tikhonov-regularised Fourier division to recover a `kernel_size`-sided
    PSF. Inputs are float32 grayscale images in [0, 1] of the same shape."""
    assert blurred_gray.shape == reference_gray.shape, "shape mismatch"
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    H, W = blurred_gray.shape
    B = np.fft.fft2(blurred_gray.astype(np.float32))
    R = np.fft.fft2(reference_gray.astype(np.float32))
    K = (np.conj(R) * B) / (np.abs(R) ** 2 + eps)
    k_full = np.real(np.fft.ifft2(K))
    k_full = np.fft.fftshift(k_full)
    cy, cx = H // 2, W // 2
    half = kernel_size // 2
    k = k_full[cy - half : cy + half + 1, cx - half : cx + half + 1].copy()
    s = k.sum()
    if s > 1e-6:
        k = k / s
    return k


def kernel_moments(k: np.ndarray, peak_frac: float = 0.20) -> Dict[str, float]:
    """PSF spatial statistics via second-order image moments.

    FFT-divided kernels have ringing lobes scattered across the whole
    support; clipping to non-negative keeps those lobes, which pulls the
    second-moment ellipse out to the kernel size regardless of the actual
    PSF. Fix: build a mask from pixels >= `peak_frac * peak`, keep only
    the connected component touching the peak, and compute moments on
    that support.

    Returns: length (2 * sqrt of major eigenvalue), thickness (minor),
    aspect ratio, orientation in [0, 180) degrees, and `pos_energy` =
    fraction of total |k| captured by the kept support.
    """
    k = np.asarray(k, dtype=np.float32)
    peak = float(k.max())
    if peak < 1e-9:
        return {"length": 0.0, "thickness": 0.0, "aspect": 0.0, "angle_deg": 0.0, "pos_energy": 0.0}

    bin_mask = (k >= peak_frac * peak).astype(np.uint8)
    py, px = np.unravel_index(int(np.argmax(k)), k.shape)
    # Keep only the connected component containing the peak.
    n_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    peak_label = int(labels[py, px])
    if peak_label == 0:
        peak_label = int(np.argmax(np.bincount(labels.ravel())[1:]) + 1) if n_labels > 1 else 0
    cc_mask = (labels == peak_label).astype(np.float32) if peak_label > 0 else bin_mask.astype(np.float32)

    k_pos = np.clip(k, 0, None) * cc_mask
    s = k_pos.sum()
    if s < 1e-9:
        return {"length": 0.0, "thickness": 0.0, "aspect": 0.0, "angle_deg": 0.0, "pos_energy": 0.0}
    p = k_pos / s
    H, W = p.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (xx * p).sum()
    cy = (yy * p).sum()
    dx = xx - cx
    dy = yy - cy
    mxx = (dx * dx * p).sum()
    myy = (dy * dy * p).sum()
    mxy = (dx * dy * p).sum()
    C = np.array([[mxx, mxy], [mxy, myy]], dtype=np.float64)
    w, v = np.linalg.eigh(C)
    w = np.clip(w, 0, None)
    major = float(np.sqrt(w[1]))
    minor = float(np.sqrt(w[0]))
    length = 2.0 * major
    thickness = 2.0 * minor
    aspect = length / thickness if thickness > 1e-6 else float("inf")
    vmaj = v[:, 1]
    angle = float(np.degrees(np.arctan2(vmaj[0], vmaj[1]))) % 180.0
    total_pos = float(np.clip(k, 0, None).sum())
    pos_energy = float(s / total_pos) if total_pos > 1e-9 else 0.0
    return {
        "length": length,
        "thickness": thickness,
        "aspect": aspect if np.isfinite(aspect) else 999.0,
        "angle_deg": angle,
        "pos_energy": pos_energy,
    }


def kernel_grid_image(kernels: np.ndarray, cols: int = 8, upscale: int = 6) -> np.ndarray:
    """Pack N kernels into a tiled grid image for visual inspection.
    `kernels` has shape (N, k, k); each tile is normalised to [0, 255]."""
    N, h, w = kernels.shape
    rows = int(np.ceil(N / cols))
    out = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i, k in enumerate(kernels):
        r, c = divmod(i, cols)
        km = k - k.min()
        if km.max() > 1e-9:
            km = km / km.max()
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = km
    out = (out * 255).astype(np.uint8)
    out = cv2.resize(out, (out.shape[1] * upscale, out.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)
    return out
