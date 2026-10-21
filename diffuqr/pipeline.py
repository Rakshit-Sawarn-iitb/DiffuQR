"""Reference-free end-to-end reader (Phase 3).

    frame -> SAM3 mask -> quad fit -> version + orientation + corner refinement
          -> module-space input (diffuqr.modspace format) -> diffusion -> RS decode

Nothing here uses the reference photo or the ground-truth grid: geometry comes
from the segmentation mask, and the version, orientation and exact corners
come from the QR's own version-fixed modules (finders, timing, alignment,
version info) plus the light quiet zone around the symbol.

Corner refinement: the mask outline follows the blurred edge, so its corners
can be off by a module or more, while the model was trained with <= 0.4
modules of corner error. `locate` therefore moves each corner (coordinate
descent, step halving from 2 modules down to 1/8) to maximise agreement
between the rectified cell means and the version's fixed-module template.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from diffuqr.qrgrid import modules_for_version, order_quad, version_fixed

MARGIN = 2           # quiet-zone modules scored as light (= diffuqr.diffusion.MARGIN)
SEARCH_PPM = 4       # px per module while searching (the model input uses 8)
MAX_VERSION = 18     # largest version on the model's 96-cell canvas


# -- Mask -> quad ------------------------------------------------------------

def pick_mask(masks: np.ndarray, scores: np.ndarray) -> Optional[int]:
    """Index of the highest-score mask whose bounding box is roughly square
    (same rule as diffuqr.annotations.pick_square_box), or None."""
    best = None
    for i, (m, s) in enumerate(zip(masks, scores)):
        ys, xs = np.nonzero(m)
        if len(xs) < 64:
            continue
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if w > 8 and h > 8 and 0.5 <= w / h <= 2.0 and (best is None or s > scores[best]):
            best = i
    return best


def _intersect(l1, l2) -> Optional[np.ndarray]:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, [x2 - x1, y2 - y1])[0]
    return np.array([x1 + t * vx1, y1 + t * vy1])


def quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Four corners (TL, TR, BR, BL in image order) of the largest blob.

    Polygon approximation of the convex hull gives rough corners; each side
    is then re-fit as a line through the contour points near its middle and
    adjacent lines are intersected (corners of a blurred blob are rounded,
    its straight sides are not)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 64:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in np.linspace(0.01, 0.12, 23):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a.reshape(4, 2).astype(np.float64)
            break
        if len(a) < 4:
            break
    if approx is None:
        approx = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
    approx = order_quad(approx).astype(np.float64)

    pts = c.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p, q = approx[i], approx[(i + 1) % 4]
        d = q - p
        L = np.linalg.norm(d)
        u = d / max(L, 1e-9)
        rel = pts - p
        t = rel @ u / max(L, 1e-9)
        dist = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.04 * peri)]
        if len(sel) < 5:
            return approx.astype(np.float32)
        lines.append(cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4))
    corners = [_intersect(lines[i - 1], lines[i]) for i in range(4)]
    if any(x is None for x in corners):
        return approx.astype(np.float32)
    out = np.array(corners)
    # A wild intersection (near-parallel sides) means the line fit failed: keep the polygon.
    if np.abs(out - approx).max() > 0.25 * peri / 4:
        return approx.astype(np.float32)
    return out.astype(np.float32)


# -- Structure score -----------------------------------------------------------

@lru_cache(maxsize=None)
def structure_template(version: int) -> Tuple[np.ndarray, np.ndarray]:
    """(known, sign) over the (n + 2*MARGIN)^2 cells: version-fixed modules
    and the quiet-zone margin; sign +1 = light, -1 = dark."""
    n = modules_for_version(version)
    N = n + 2 * MARGIN
    known = np.ones((N, N), bool)
    sign = np.ones((N, N), np.float32)
    vmask, vvals = version_fixed(version)
    s = slice(MARGIN, MARGIN + n)
    known[s, s] = vmask
    sign[s, s] = np.where(vvals > 0, -1.0, 1.0)
    return known, sign


def _warp(gray: np.ndarray, quad: np.ndarray, n: int, ppm: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    side = (n + 2 * MARGIN) * ppm
    a, b = MARGIN * ppm, (MARGIN + n) * ppm
    dst = np.float32([[a, a], [b, a], [b, b], [a, b]])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(gray, M, (side, side), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def warp_pixels(gray: np.ndarray, quad: np.ndarray, n: int, ppm: int = SEARCH_PPM) -> np.ndarray:
    return _warp(gray, quad, n, ppm).astype(np.float32)


@lru_cache(maxsize=None)
def _pixel_template(version: int, ppm: int) -> Tuple[np.ndarray, np.ndarray]:
    known, sign = structure_template(version)
    k = np.ones((ppm, ppm), np.float32)
    return np.kron(known, k) > 0, np.kron(sign, k)


def structure_score(warped: np.ndarray, version: int, rotation: int, ppm: int = SEARCH_PPM) -> float:
    """Pearson correlation between the warped pixels (rotated upright) and
    the +-1 template over the pixels of known cells. Brightness/contrast
    invariant; pixel-level, so it peaks sharply at the true alignment."""
    known, sign = _pixel_template(version, ppm)
    x = np.rot90(warped, rotation)[known]
    t = sign[known]
    x = x - x.mean()
    t = t - t.mean()
    return float((x * t).sum() / (np.sqrt((x * x).sum() * (t * t).sum()) + 1e-6))


def _best_rotation(warped: np.ndarray, version: int) -> Tuple[int, float]:
    s = [structure_score(warped, version, r) for r in range(4)]
    r = int(np.argmax(s))
    return r, s[r]


# -- Geometry search -------------------------------------------------------------

@dataclass
class Geometry:
    version: int
    rotation: int           # np.rot90 k that brings the rectified crop upright
    quad: np.ndarray        # (4, 2) refined symbol corners, image order (TL, TR, BR, BL)
    score: float            # structure score after refinement
    init_score: float       # structure score at the mask quad (same version/rotation)


def refine_quad(gray: np.ndarray, quad: np.ndarray, version: int, rotation: int,
                steps_modules: Sequence[float] = (2.0, 1.0, 0.5, 0.25, 0.125),
                max_iter: int = 4, ppm: int = SEARCH_PPM) -> Tuple[np.ndarray, float]:
    """Coordinate descent on the 8 corner coordinates, maximising the
    structure score at a fixed version and rotation."""
    n = modules_for_version(version)
    score = lambda c: structure_score(warp_pixels(gray, c, n, ppm), version, rotation, ppm)
    q = np.asarray(quad, np.float64).copy()
    side = np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])
    module_px = side / n
    best = score(q)
    dirs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for step in steps_modules:
        h = step * module_px
        for _ in range(max_iter):
            improved = False
            # Whole-quad moves first (shift, scale, rotate by about one step at the corners):
            # single-corner moves alone get stuck when the whole symbol is offset.
            c0 = q.mean(axis=0)
            rad = max(np.linalg.norm(q - c0, axis=1).mean(), 1e-6)
            moves = [q + (dx * h, dy * h) for dx, dy in dirs]
            moves += [c0 + (q - c0) * (1 + sg * h / rad) for sg in (-1, 1)]
            moves += [c0 + (q - c0) @ R(sg * h / rad).T for sg in (-1, 1)]
            scores = [score(c) for c in moves]
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-4:
                q, best, improved = moves[j], scores[j], True
            for i in range(4):
                cand_best, cand_q = best, None
                for dx, dy in dirs:
                    c = q.copy()
                    c[i] += (dx * h, dy * h)
                    s = score(c)
                    if s > cand_best + 1e-4:
                        cand_best, cand_q = s, c
                if cand_q is not None:
                    q, best, improved = cand_q, cand_best, True
            if not improved:
                break
    return q.astype(np.float32), best


def locate(gray: np.ndarray, quad: np.ndarray, versions: Iterable[int] = range(1, MAX_VERSION + 1),
           top: int = 3, min_module_px: float = 1.5, coarse_ppm: int = 2) -> Optional[Geometry]:
    """Version, orientation and refined corners for the QR roughly at `quad`.

    Far from the true corners neither the version nor the rotation can be
    read off reliably, so every (version, rotation) pair whose modules would
    be at least `min_module_px` wide gets its own coarse refinement (2 and 1
    module steps, at `coarse_ppm` px/module to keep it cheap); the `top`
    best pairs are then refined finely, and the highest score wins."""
    quad = order_quad(quad)
    side = np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
    # Work on a crop around the quad (cheaper blurs); coordinates go back to the frame at the end.
    pad = 0.5 * side
    x0, y0 = (max(int(v - pad), 0) for v in quad.min(axis=0))
    x1, y1 = (int(v + pad) for v in quad.max(axis=0))
    crop = gray[y0:y1, x0:x1]
    off = np.float32([x0, y0])
    quad = quad - off
    coarse = []
    for v in versions:
        n = modules_for_version(v)
        if side / n < min_module_px:
            continue
        w0 = warp_pixels(crop, quad, n)
        # Anti-alias for the low-resolution coarse warp.
        sigma = 0.5 * side / n / coarse_ppm
        g = cv2.GaussianBlur(crop, (0, 0), sigma) if sigma > 0.7 else crop
        for r in range(4):
            s0 = structure_score(w0, v, r)
            q, s = refine_quad(g, quad, v, r, steps_modules=(2.0, 1.0), max_iter=2, ppm=coarse_ppm)
            coarse.append((s, v, r, q, s0))
    if not coarse:
        return None
    best = None
    for _, v, r, q, s0 in sorted(coarse, key=lambda c: -c[0])[:top]:
        q, s = refine_quad(crop, q, v, r, steps_modules=(0.5, 0.25, 0.125))
        if best is None or s > best.score:
            best = Geometry(v, r, q + off, s, s0)
    return best


# -- Model input -----------------------------------------------------------------

def contrast_stretch(img: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    """Map the lo..hi percentile range to 0..255 (dark or washed-out crops ->
    the full range the model was trained on)."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)


def module_input(gray: np.ndarray, geom: Geometry, ppm: int = 8, stretch: bool = True) -> np.ndarray:
    """Upright module-space crop for diffuqr.diffusion.make_example:
    (n + 2*MARGIN) * ppm square, symbol at the canonical position."""
    n = modules_for_version(geom.version)
    inp = np.ascontiguousarray(np.rot90(_warp(gray, geom.quad, n, ppm, interp=cv2.INTER_AREA), geom.rotation))
    return contrast_stretch(inp) if stretch else inp


def frame_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb


def mask_touches_border(mask: np.ndarray, pad: int = 2) -> bool:
    """True when the QR blob reaches the frame edge (QR likely cut off)."""
    return bool(mask[:pad].any() or mask[-pad:].any() or mask[:, :pad].any() or mask[:, -pad:].any())


def quad_error_modules(q: np.ndarray, ref: np.ndarray, n: int) -> float:
    """Worst-corner distance between two quads, in modules of `ref`."""
    q, ref = order_quad(q), order_quad(ref)
    mod = np.mean([np.linalg.norm(ref[i] - ref[(i + 1) % 4]) for i in range(4)]) / n
    return float(np.linalg.norm(q - ref, axis=1).max() / mod)
