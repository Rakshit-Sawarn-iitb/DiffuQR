"""Small, reusable image ops.

Kept intentionally tiny: the training notebook already shows that subtle
choices here (INTER_AREA vs LANCZOS, resizing cached images twice) corrupt
dense QR codes, so these helpers are the one place that behaviour lives.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


Box = Sequence[float]  # (x1, y1, x2, y2)


def crop_square(image: Image.Image, box: Box, pad_frac: float = 0.15) -> Image.Image:
    """Square crop around `box` with padding. The window is SHIFTED to stay
    inside the image rather than clipped, so the aspect ratio is preserved
    (clipping + a square resize would stretch the QR).
    """
    W, H = image.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = min(max(x2 - x1, y2 - y1) * (1.0 + pad_frac), W, H)
    left = min(max(cx - side / 2.0, 0.0), W - side)
    top = min(max(cy - side / 2.0, 0.0), H - side)
    s, l, t = int(side), int(left), int(top)
    return image.crop((l, t, l + s, t + s))


def laplacian_variance(image_rgb: np.ndarray) -> float:
    """Variance of the Laplacian: a classical blur score. Higher = sharper."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def upscale_lanczos(image_rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Lanczos upscale to a square `target_size`. No-op if already larger."""
    h, w = image_rgb.shape[:2]
    if max(h, w) >= target_size:
        return image_rgb
    return cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def unsharp(image_rgb: np.ndarray, radius: float = 1.5, amount: float = 1.5) -> np.ndarray:
    """Classic unsharp mask: sharpened = img + amount * (img - blur(img))."""
    blur = cv2.GaussianBlur(image_rgb, (0, 0), radius)
    out = cv2.addWeighted(image_rgb, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """PIL -> uint8 RGB ndarray."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image)
