"""Module-space training pairs (Phase 2 data engine).

A sample is what the module-space restorer sees at inference:

    input   rectified grayscale crop, exactly `px_per_module` px per module,
            covering the symbol plus `margin` modules on every side
            -> side (N + 2*margin) * px_per_module
    target  N x N module grid (1 = dark), plus (version, EC, mask)

Synthetic generation mirrors the real pipeline in CAMERA space, then
rectifies, so the model sees the same artefacts real frames will have:

    segno QR (random payload/version/EC/mask)
      -> render on paper at s px/module (s ~ real 7-16), quiet zone >= 4
      -> perspective warp into a camera image (calibrated to Phase 0.4)
      -> REAL motion kernel from the bank, rescaled in module units
      -> optional defocus, illumination/vignette, sensor noise, JPEG
      -> rectify from the TRUE corners + random corner error
      -> resample to px_per_module

Kernel calibration: the bank was fit on 256-px crops of (SAM3 box * 1.15)
around a 61-module symbol, i.e. 256 / (1.15 * 61) = 3.65 px per module.
Kernels are rescaled by s / 3.65 so blur extent is preserved IN MODULES,
which is what decides whether modules smear into each other.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import segno

from diffuqr.synth import random_payload, _CHAR_TARGET

BANK_PX_PER_MODULE = 256 / (1.15 * 61)


@dataclass
class ModuleSampleConfig:
    px_per_module: int = 8                 # model input resolution
    margin: int = 2                        # modules of context around the symbol
    min_version: int = 10
    max_version: int = 18
    errors: Tuple[str, ...] = ("L", "M", "Q", "H")
    camera_ppm: Tuple[float, float] = (7.0, 16.0)   # measured real range 7.0-15.9
    # Per-sample corner-jitter amplitude (fraction of symbol side). (0.05, 0.17)
    # gives side ratio p50/p95 1.19/1.47 and angle dev 9.2/20 deg vs real
    # 1.18/1.48 and 7.6/18 (Phase 0.4).
    persp_jitter: Tuple[float, float] = (0.05, 0.17)
    corner_err_modules: float = 0.4        # max rectification corner error (modules, uniform per axis)
    hard_frac: float = 0.5                 # fraction of kernels drawn from the longest third
    defocus_prob: float = 0.3
    defocus_modules: Tuple[float, float] = (0.05, 0.25)  # Gaussian sigma in modules
    noise_sigma: Tuple[float, float] = (0.005, 0.03)
    jpeg_quality: Tuple[int, int] = (40, 90)


@dataclass
class KernelBank:
    kernels: np.ndarray        # (K, k, k) float32, non-negative, sum 1, in BANK px
    lengths: np.ndarray        # (K,) kernel length in bank px

    @classmethod
    def load(cls, path: Path) -> "KernelBank":
        d = np.load(path, allow_pickle=True)
        k = np.clip(d["kernels"].astype(np.float32), 0, None)
        s = k.reshape(len(k), -1).sum(1).reshape(-1, 1, 1)
        k = k / np.where(s < 1e-9, 1.0, s)
        lengths = np.array([json.loads(m)["k_length"] for m in d["meta"]], dtype=np.float32)
        return cls(k, lengths)

    def sample(self, rng: random.Random, hard_frac: float) -> np.ndarray:
        if rng.random() < hard_frac:
            cut = np.percentile(self.lengths, 100 * 2 / 3)
            pool = np.flatnonzero(self.lengths >= cut)
        else:
            pool = np.arange(len(self.kernels))
        return self.kernels[int(pool[rng.randrange(len(pool))])]


def rescale_kernel(k: np.ndarray, factor: float) -> np.ndarray:
    """Spatially rescale a PSF by `factor`, keep it odd-sized and sum 1."""
    side = max(1, int(round(k.shape[0] * factor)))
    side += 1 - side % 2
    out = cv2.resize(k, (side, side), interpolation=cv2.INTER_LINEAR if factor >= 1 else cv2.INTER_AREA)
    out = np.clip(out, 0, None)
    s = out.sum()
    return out / s if s > 1e-9 else out


def rectify_quad(gray: np.ndarray, quad: np.ndarray, n: int, px_per_module: int, margin: int) -> np.ndarray:
    """Warp so `quad` (symbol corners TL, TR, BR, BL) lands on the symbol
    region of an (n + 2*margin) * px_per_module square."""
    side = (n + 2 * margin) * px_per_module
    a, b = margin * px_per_module, (margin + n) * px_per_module
    dst = np.float32([[a, a], [b, a], [b, b], [a, b]])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(gray, M, (side, side), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _random_code(rng: random.Random, cfg: ModuleSampleConfig):
    error = rng.choice(cfg.errors)
    mask = rng.randrange(8)
    target_v = rng.randint(cfg.min_version, cfg.max_version)
    state = random.getstate()
    random.seed(rng.random())  # synth payload samplers use the global RNG
    try:
        budget = _CHAR_TARGET.get(target_v, 300)
        scale = {"L": 1.0, "M": 0.78, "Q": 0.56, "H": 0.43}[error]
        payload = random_payload(min_len=20, max_len=max(25, int(budget * scale)))
    finally:
        random.setstate(state)
    for v in range(target_v, 41):
        try:
            qr = segno.make(payload, version=v, error=error, mask=mask, boost_error=False, micro=False)
            break
        except Exception:
            continue
    return qr, payload, error, mask


def make_synthetic_sample(rng: random.Random, bank: KernelBank, cfg: ModuleSampleConfig) -> dict:
    qr, payload, error, mask = _random_code(rng, cfg)
    grid = np.array(qr.matrix, dtype=np.uint8)
    n = grid.shape[0]
    s = rng.uniform(*cfg.camera_ppm)

    # Clean render on paper at >= camera resolution (integer px/module), then map into camera space.
    ss = int(np.ceil(s))
    qz = rng.randint(4, 8)
    paper = rng.uniform(0.75, 0.95)
    ink = rng.uniform(0.02, 0.18)
    canvas_modules = n + 2 * qz
    padded = np.zeros((canvas_modules, canvas_modules), np.float32)
    padded[qz:qz + n, qz:qz + n] = grid
    clean = cv2.resize(paper + (ink - paper) * padded, (canvas_modules * ss, canvas_modules * ss),
                       interpolation=cv2.INTER_NEAREST)

    sym_px = n * s
    cam_side = int(np.ceil((n + 2 * qz) * s * 1.25))
    c = cam_side / 2
    sq = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (sym_px / 2) + c
    amp = rng.uniform(*cfg.persp_jitter)
    jit = np.float32([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * amp * sym_px
    cam_quad = sq + jit
    src_sym = np.float32([[qz, qz], [qz + n, qz], [qz + n, qz + n], [qz, qz + n]]) * ss
    H = cv2.getPerspectiveTransform(src_sym, cam_quad)
    bg = rng.uniform(0.2, 0.9)
    cam = cv2.warpPerspective(clean, H, (cam_side, cam_side), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    k = rescale_kernel(bank.sample(rng, cfg.hard_frac), s / BANK_PX_PER_MODULE)
    cam = cv2.filter2D(cam, -1, k, borderType=cv2.BORDER_REPLICATE)
    if rng.random() < cfg.defocus_prob:
        cam = cv2.GaussianBlur(cam, (0, 0), rng.uniform(*cfg.defocus_modules) * s)

    # Illumination: gain/offset/gamma + soft linear gradient (shadow across the sticker).
    ramp = np.linspace(-0.5, 0.5, cam_side, dtype=np.float32)
    grad = 1 + rng.uniform(-0.25, 0.25) * ramp[None, :] + rng.uniform(-0.25, 0.25) * ramp[:, None]
    cam *= grad * np.float32(rng.uniform(0.8, 1.15))
    cam += np.float32(rng.uniform(-0.05, 0.05))
    np.clip(cam, 0, 1, out=cam)
    cv2.pow(cam, rng.uniform(0.8, 1.25), dst=cam)
    nrng = np.random.default_rng(rng.randrange(2 ** 31))
    cam += nrng.standard_normal(cam.shape, dtype=np.float32) * np.float32(rng.uniform(*cfg.noise_sigma))
    u8 = cv2.convertScaleAbs(np.clip(cam, 0, 1), alpha=255.0)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*cfg.jpeg_quality)])
    if ok:
        u8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    err = np.float32([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * cfg.corner_err_modules * s
    inp = rectify_quad(u8, cam_quad + err, n, cfg.px_per_module, cfg.margin)
    return {
        "input": inp, "target": grid, "version": int(qr.version), "error": error, "mask": mask,
        "payload": payload, "camera_ppm": float(s), "kernel_len_modules": float(
            _kernel_extent(k) / s),
    }


def baseline_read(inp: np.ndarray, n: int, px_per_module: int, margin: int) -> Tuple[np.ndarray, np.ndarray]:
    """Learning-free reader on a module-space input: mean of the central
    half of each module cell, local (Gaussian-smoothed) threshold.
    Returns (bits, confidence). The floor every learned model must beat."""
    p = px_per_module
    a = margin * p
    sym = inp[a:a + n * p, a:a + n * p].astype(np.float32)
    lo, hi = p // 4, p - p // 4
    cells = sym.reshape(n, p, n, p)[:, lo:hi, :, lo:hi].mean(axis=(1, 3))
    local = cv2.GaussianBlur(cells, (0, 0), sigmaX=4.0, borderType=cv2.BORDER_REFLECT)
    return (cells <= local).astype(np.uint8), np.abs(cells - local)


def _kernel_extent(k: np.ndarray) -> float:
    """Second-moment length (2 * sqrt of the major eigenvalue), in px."""
    H, W = k.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    p = k / max(k.sum(), 1e-12)
    cx, cy = (xx * p).sum(), (yy * p).sum()
    C = np.array([[((xx - cx) ** 2 * p).sum(), ((xx - cx) * (yy - cy) * p).sum()],
                  [((xx - cx) * (yy - cy) * p).sum(), ((yy - cy) ** 2 * p).sum()]])
    return float(2 * np.sqrt(max(np.linalg.eigvalsh(C)[-1], 0)))
