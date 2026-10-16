"""Procedural QR generator for training-time synthetic data.

Builds on `segno` because it supports every version/EC combo and exposes
the module matrix directly. The matrix is both rendered to an RGB image
(so the training pipeline can treat it like a real crop) and returned as
a 2D bit grid (so the Phase 2 module-space model can target it directly).

Payload sampler is deliberately coarse: EMV-like digits-heavy strings
(like UPI merchant QRs), URLs, and short alphanumeric text. The goal is
character-distribution variety and version-range coverage, not spec
compliance.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import segno


@dataclass
class SynthQR:
    image_rgb: np.ndarray      # uint8 (H, W, 3), white bg, black modules, >= 4-module quiet zone
    modules: np.ndarray        # uint8 (N, N), 1 = dark module
    version: int
    error: str                 # 'L' / 'M' / 'Q' / 'H'
    payload: str
    px_per_module: int         # integer scale factor (so the rendering is exact)


def _render(qr: segno.QRCode, img_size: int, quiet_zone: int = 4) -> Tuple[np.ndarray, np.ndarray, int]:
    """Render a segno QR to a square RGB image of side `img_size`.
    Returns (image_rgb, modules_2d, integer px/module scale).

    The scale is chosen so at least `quiet_zone` modules of white border
    remain on every side (the spec asks for 4). Sizing from `img_size // n`
    alone left 0-2 modules for some versions, and the perspective warp in
    `degrade.py` then pushed the code off-canvas (v11/v13/v14 failed to
    decode even without blur)."""
    # segno's `matrix` is the inner module grid without the quiet zone.
    matrix = np.array(qr.matrix, dtype=np.uint8)
    n = matrix.shape[0]
    scale = max(1, img_size // (n + 2 * quiet_zone))
    inner = (1 - np.repeat(np.repeat(matrix, scale, axis=0), scale, axis=1)) * 255
    inner_px = n * scale
    pad = (img_size - inner_px) // 2
    img = np.full((img_size, img_size), 255, dtype=np.uint8)
    img[pad : pad + inner_px, pad : pad + inner_px] = inner
    rgb = np.repeat(img[..., None], 3, axis=-1)
    return rgb, matrix, int(scale)


def generate_qr(
    payload: str,
    version: Optional[int] = None,
    error: str = "L",
    img_size: int = 512,
) -> SynthQR:
    """Encode `payload` as a QR at `version` (or the smallest version that
    fits) and render to `img_size` pixels square."""
    qr = segno.make(payload, version=version, error=error)
    rgb, modules, pxm = _render(qr, img_size)
    return SynthQR(
        image_rgb=rgb,
        modules=modules,
        version=int(qr.version) if isinstance(qr.version, int) else int(str(qr.version)),
        error=error.upper(),
        payload=payload,
        px_per_module=pxm,
    )


# -- Payload samplers ------------------------------------------------------

_UPI_FIELD_CHARS = string.digits + string.ascii_uppercase


def _emv_like(min_len: int, max_len: int) -> str:
    """Approximates the EMV merchant-QR TLV structure in character
    distribution and density. Not spec-valid; just dense digits + caps."""
    out = []
    while sum(len(p) for p in out) < min_len:
        tag = f"{random.randint(0, 99):02d}"
        data = "".join(random.choices(_UPI_FIELD_CHARS, k=random.randint(2, 30)))
        out.append(f"{tag}{len(data):02d}{data}")
    s = "".join(out)
    return s[:max_len]


def _url(min_len: int, max_len: int) -> str:
    host = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    tld = random.choice([".com", ".in", ".io", ".org", ".ai"])
    path = "".join(random.choices(string.ascii_lowercase + string.digits + "/-_", k=random.randint(5, max_len)))
    s = f"https://{host}{tld}/{path}"
    if len(s) < min_len:
        s = s + "/" + "".join(random.choices(string.ascii_lowercase + string.digits, k=min_len - len(s)))
    return s[:max_len]


def _text(min_len: int, max_len: int) -> str:
    pool = string.ascii_letters + string.digits + " !?.,-"
    k = random.randint(min_len, max_len)
    return "".join(random.choices(pool, k=k))


def random_payload(min_len: int = 60, max_len: int = 280, kinds: Tuple[str, ...] = ("emv", "url", "text")) -> str:
    """Random payload from one of the configured kinds."""
    kind = random.choice(kinds)
    if kind == "emv":
        return _emv_like(min_len, max_len)
    if kind == "url":
        return _url(min_len, max_len)
    return _text(min_len, max_len)


# Approximate char budgets per version at EC-L (numeric / alphanumeric mode;
# segno's byte mode uses less). Rough values; segno bumps the version if
# the payload doesn't fit, which is fine.
_CHAR_TARGET = {10: 110, 11: 150, 12: 190, 13: 230, 14: 270, 15: 310,
                16: 350, 17: 390, 18: 430, 19: 480, 20: 530}


def random_qr(
    min_version: int = 10,
    max_version: int = 18,
    errors: Tuple[str, ...] = ("L", "M"),
    img_size: int = 512,
) -> SynthQR:
    """Sample a QR whose version lands in [min_version, max_version] by
    picking an EC level and tuning payload length to a per-version budget.
    segno may still bump the version up a step for byte-mode payloads;
    we accept that so long as the result is inside the requested range."""
    error = random.choice(errors)
    target_version = random.randint(min_version, max_version)
    pl_len = _CHAR_TARGET.get(target_version, 300)
    for _ in range(5):
        payload = random_payload(min_len=max(60, pl_len - 40), max_len=pl_len + 10)
        try:
            qr = generate_qr(payload, version=None, error=error, img_size=img_size)
        except Exception:
            continue
        if min_version <= qr.version <= max_version:
            return qr
    # Fallback: force the version.
    try:
        return generate_qr(payload, version=target_version, error=error, img_size=img_size)
    except Exception:
        return generate_qr(_text(80, 120), version=min_version, error=error, img_size=img_size)
