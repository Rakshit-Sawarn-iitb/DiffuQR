"""Calibrated degradation pipeline for synthetic QR crops.

Pipeline:
  perspective warp -> motion blur (sampled from the Phase 0.3 kernel bank)
  -> defocus (Gaussian) -> illumination (gamma + contrast + vignette)
  -> JPEG compression -> sensor noise.

Only motion blur is calibrated to real buggy frames; the rest are
distributionally plausible and parameterised so a sampling policy can
be swept in Phase 2. Everything is deterministic given a seed.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# -- Kernel bank -----------------------------------------------------------

def load_kernel_bank(path: Path, target_support: int = 25) -> np.ndarray:
    """Load kernels.npz, clip to non-negative, renormalise, and rescale
    the spatial support to `target_support` by bilinear interpolation.

    Rescaling preserves the kernel-to-image size RATIO used during
    estimation, so if callers generate crops at a different pixel size
    than 256 they pass `target_support = round(25 * crop_size / 256)`.
    """
    d = np.load(path, allow_pickle=True)
    kernels = d["kernels"].astype(np.float32)
    kernels = np.clip(kernels, 0, None)
    s = kernels.reshape(len(kernels), -1).sum(axis=1).reshape(-1, 1, 1)
    s = np.where(s < 1e-9, 1.0, s)
    kernels = kernels / s

    if kernels.shape[1] != target_support:
        rescaled = np.empty((len(kernels), target_support, target_support), dtype=np.float32)
        for i, k in enumerate(kernels):
            k_up = cv2.resize(k, (target_support, target_support), interpolation=cv2.INTER_LINEAR)
            ss = k_up.sum()
            if ss > 1e-9:
                k_up = k_up / ss
            rescaled[i] = k_up
        kernels = rescaled
    return kernels


# -- Component transforms --------------------------------------------------

def perspective_warp(img: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """Mild random perspective warp. `strength` is max corner displacement
    as a fraction of the shorter image side."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    delta = strength * min(h, w)
    dst = src + np.random.uniform(-delta, delta, size=src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def motion_blur_from_bank(img: np.ndarray, kernel_bank: np.ndarray) -> np.ndarray:
    k = kernel_bank[random.randrange(len(kernel_bank))]
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def defocus_blur(img: np.ndarray, sigma_range=(0.3, 2.0)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def illumination(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    gamma = random.uniform(0.75, 1.35)
    contrast = random.uniform(0.75, 1.2)
    brightness = random.uniform(-0.08, 0.08)
    x = np.clip(x, 0, 1) ** gamma
    x = np.clip((x - 0.5) * contrast + 0.5 + brightness, 0, 1)
    if random.random() < 0.5:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * random.uniform(0.3, 0.7), h * random.uniform(0.3, 0.7)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(h, w)
        amt = random.uniform(0.1, 0.35)
        v = 1 - amt * np.clip(r * 2, 0, 1) ** 2
        x = x * v[..., None]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality_range=(30, 85)) -> np.ndarray:
    q = random.randint(*quality_range)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sensor_noise(img: np.ndarray, sigma_range=(0.005, 0.025)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    x = img.astype(np.float32) / 255.0 + noise
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# -- Pipeline --------------------------------------------------------------

def calibrated_degrade(
    clean_rgb: np.ndarray,
    kernel_bank: np.ndarray,
    seed: Optional[int] = None,
    *,
    perspective: bool = True,
    motion: bool = True,
    defocus_prob: float = 0.5,
    illum: bool = True,
    jpeg: bool = True,
    noise: bool = True,
) -> np.ndarray:
    """Apply the full degradation pipeline. Returns uint8 RGB."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img = clean_rgb.copy()
    if perspective:
        img = perspective_warp(img)
    if motion:
        img = motion_blur_from_bank(img, kernel_bank)
    if defocus_prob > 0 and random.random() < defocus_prob:
        img = defocus_blur(img)
    if illum:
        img = illumination(img)
    if jpeg:
        img = jpeg_compress(img)
    if noise:
        img = sensor_noise(img)
    return img
