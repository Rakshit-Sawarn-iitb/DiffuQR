"""Alignment between a blurry frame crop and its clean reference.

ECC on band-passed grayscale (illumination-invariant), optionally initialised
with an ORB homography, gated by normalised cross-correlation. Ported from
the training notebook so Phase 0+ code can reuse exactly the same alignment
the training pipeline uses.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def bandpass(gray_u8: np.ndarray) -> np.ndarray:
    """Illumination- and blur-tolerant representation for alignment/scoring."""
    g = gray_u8.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 12)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


def orb_homography(frame_gray: np.ndarray, ref_gray: np.ndarray, min_matches: int = 10) -> Optional[np.ndarray]:
    """Homography mapping REF coords -> FRAME coords from ORB matches, or None."""
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(frame_gray, None)
    k2, d2 = orb.detectAndCompute(ref_gray, None)
    if d1 is None or d2 is None or len(k1) < min_matches or len(k2) < min_matches:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [p[0] for p in matches if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_matches:
        return None
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < min_matches:
        return None
    return H


def ecc_refine(frame_bp: np.ndarray, ref_bp: np.ndarray, init_ref_to_frame: np.ndarray) -> np.ndarray:
    """ECC refinement. Returns W mapping FRAME coords -> REF coords, which is
    what cv2.warpPerspective(..., flags=WARP_INVERSE_MAP) needs to warp the
    reference onto the frame."""
    try:
        W = np.linalg.inv(init_ref_to_frame).astype(np.float32)
        W = W / W[2, 2]
        if not np.all(np.isfinite(W)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        W = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, W_ref = cv2.findTransformECC(frame_bp, ref_bp, W.copy(), cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
        if np.all(np.isfinite(W_ref)):
            W = W_ref
    except cv2.error:
        pass
    return W


def align_reference_to_frame(
    frame_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    ncc_thresh: float = 0.5,
) -> Tuple[Optional[np.ndarray], float]:
    """Warp the clean reference onto the degraded frame's geometry.
    Tries ORB-initialised and identity-initialised ECC, keeps the better one.
    Returns (warped_reference_or_None, alignment_score).
    """
    h, w = frame_bgr.shape[:2]
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    frame_bp, ref_bp = bandpass(frame_gray), bandpass(ref_gray)

    inits = [np.eye(3, dtype=np.float64)]
    H0 = orb_homography(frame_gray, ref_gray)
    if H0 is not None:
        inits.insert(0, H0)

    best_img, best_score = None, -1.0
    for init in inits:
        W = ecc_refine(frame_bp, ref_bp, init)
        warped = cv2.warpPerspective(
            ref_bgr, W, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        score = ncc(frame_bp, bandpass(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)))
        if score > best_score:
            best_img, best_score = warped, score
    return (best_img if best_score >= ncc_thresh else None), best_score
