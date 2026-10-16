"""Module-grid reading for real QR crops.

Rectify a QR from its 4-corner quad, sample the N x N module grid, and
identify (version, EC level, mask) by agreement with the modules that are
FIXED for that combination: finder/separator/timing/alignment patterns plus
format info and version info. Those depend only on (version, EC, mask), not
on the payload or on which encoder produced the symbol, so the match is
encoder-independent.

Fixed modules come from segno's spec helpers (see `fixed_template`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import segno
from segno import consts as segno_consts
from segno import encoder as segno_encoder  # internal API; verified on segno 1.6.6


def modules_for_version(version: int) -> int:
    return 17 + 4 * version


# -- Geometry ----------------------------------------------------------------

def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 points clockwise starting from the one nearest the image
    top-left. Orientation of the symbol itself is resolved later."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords (y down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def rectify(image_rgb: np.ndarray, quad: Sequence[Sequence[float]], out_px: int,
            reorder: bool = True) -> np.ndarray:
    """Warp the quad (symbol outer corners, no quiet zone) to an
    out_px x out_px grayscale square. With reorder=False the quad is used
    as given (TL, TR, BR, BL of the output), which keeps point
    correspondence when the quad was mapped from another image."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    src = order_quad(quad) if reorder else np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.float32([[0, 0], [out_px, 0], [out_px, out_px], [0, out_px]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (out_px, out_px), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REPLICATE)


def sample_grid(square_gray: np.ndarray, n: int, center_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an n x n grid from a rectified square. Returns (bits, means)
    where bits is uint8 (1 = dark) via Otsu over the per-module means, and
    means are the central-window intensities in [0, 255]."""
    S = square_gray.shape[0]
    cell = S / n
    half = max(1, int(cell * center_frac / 2))
    means = np.empty((n, n), dtype=np.float32)
    for r in range(n):
        cy = int((r + 0.5) * cell)
        for c in range(n):
            cx = int((c + 0.5) * cell)
            means[r, c] = square_gray[cy - half: cy + half + 1, cx - half: cx + half + 1].mean()
    u8 = np.clip(means, 0, 255).astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (means <= thr).astype(np.uint8)  # Otsu's low class is <= thr (thr is 0 on pure 0/255 input)
    return bits, means


# -- Fixed-module templates --------------------------------------------------

_SEGNO_ERROR = {"L": segno_consts.ERROR_LEVEL_L, "M": segno_consts.ERROR_LEVEL_M,
                "Q": segno_consts.ERROR_LEVEL_Q, "H": segno_consts.ERROR_LEVEL_H}


@lru_cache(maxsize=None)
def fixed_template(version: int, error: str, mask: int) -> Tuple[np.ndarray, np.ndarray]:
    """(fixed_mask, fixed_values) for (version, EC, mask). fixed_mask is
    bool (n, n): True for function patterns, format info, dark module and
    version info, i.e. every module that does not depend on the payload.

    Built with segno's own spec helpers: `make_matrix` marks data cells
    0x2 and the add_* calls fill everything else. (Comparing encodings of
    random short payloads does NOT work: their identical padding codewords
    make large data regions look fixed.)"""
    n = modules_for_version(version)
    m = segno_encoder.make_matrix(n, n)
    segno_encoder.add_finder_patterns(m, n, n)
    segno_encoder.add_alignment_patterns(m, n, n)
    segno_encoder.add_format_info(m, version, _SEGNO_ERROR[error], mask)
    segno_encoder.add_version_info(m, version)
    arr = np.array([list(r) for r in m], dtype=np.uint8)
    fixed = arr != 0x2
    return fixed, np.where(fixed, arr, 0).astype(np.uint8)


@lru_cache(maxsize=None)
def version_fixed(version: int) -> Tuple[np.ndarray, np.ndarray]:
    """(mask, values) of modules fixed by the VERSION alone: finders,
    separators, timing, alignment, dark module, version info. Excludes the
    format-info modules, which depend on (EC, mask) and must be read."""
    combos = [fixed_template(version, e, m) for e in "LMQH" for m in range(8)]
    fixed0, vals0 = combos[0]
    same = np.ones_like(fixed0)
    for f, v in combos[1:]:
        same &= (f == fixed0) & (v == vals0)
    mask = fixed0 & same
    return mask, np.where(mask, vals0, 0).astype(np.uint8)


def read_format(bits: np.ndarray, version: int) -> Tuple[str, int, int]:
    """Most consistent (EC, mask) for a grid: compares the 30 format-info
    modules (two copies of the BCH-protected 15-bit word) with every
    combination. Returns (error, mask, n_disagreeing_modules)."""
    vmask, _ = version_fixed(version)
    best = None
    for e in "LMQH":
        for m in range(8):
            f, vals = fixed_template(version, e, m)
            region = f & ~vmask
            d = int((bits[region] != vals[region]).sum())
            if best is None or d < best[2]:
                best = (e, m, d)
    return best


# -- Identification ----------------------------------------------------------

@dataclass
class GridMatch:
    version: int
    error: str
    mask: int
    rotation: int           # np.rot90 k applied to the sampled grid
    fixed_agreement: float  # fraction of fixed modules that match
    n_fixed: int
    bits: np.ndarray        # oriented sampled grid (1 = dark)


def identify(
    image_rgb: np.ndarray,
    quad: Sequence[Sequence[float]],
    versions: Iterable[int] = range(1, 41),
    errors: Iterable[str] = ("L", "M", "Q", "H"),
    px_per_module: int = 12,
) -> Optional[GridMatch]:
    """Best (version, EC, mask, rotation) for the QR inside `quad`."""
    best: Optional[GridMatch] = None
    for v in versions:
        n = modules_for_version(v)
        sq = rectify(image_rgb, quad, n * px_per_module)
        bits0, _ = sample_grid(sq, n)
        for rot in range(4):
            bits = np.rot90(bits0, rot)
            for e in errors:
                for m in range(8):
                    fixed, vals = fixed_template(v, e, m)
                    agree = float((bits[fixed] == vals[fixed]).mean())
                    if best is None or agree > best.fixed_agreement:
                        best = GridMatch(v, e, m, rot, agree, int(fixed.sum()), bits.copy())
    return best


def payload_agreement(match: GridMatch, payload: str) -> Tuple[float, Optional[np.ndarray]]:
    """Re-encode `payload` with segno at the identified (version, EC, mask)
    and return (fraction of all modules that agree, re-encoded matrix).
    ~1.0 means the sampled grid is an exact, verified ground-truth bit grid.
    Lower values can just mean the original encoder segmented the payload
    differently (modes), so this is a bonus check, not the identification."""
    try:
        qr = segno.make(payload, version=match.version, error=match.error, mask=match.mask,
                        boost_error=False, micro=False)
    except Exception:
        return 0.0, None
    ref = np.array(qr.matrix, dtype=np.uint8)
    if ref.shape != match.bits.shape:
        return 0.0, None
    return float((ref == match.bits).mean()), ref
