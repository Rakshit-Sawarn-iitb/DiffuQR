"""QR decoder wrappers.

Each decoder returns a UTF-8 string on success or None. Decoders whose
dependencies are missing are reported by `available_decoders()` and skipped
cleanly by `decode_with` / `decode_any`.

- pyzbar: zbar bindings; robust on dense QR codes. Needs libzbar (bundled in
  the Windows wheel; Linux usually needs `apt install libzbar0`).
- opencv: `cv2.QRCodeDetector`, ships with every opencv build. Weakest on
  dense codes but always available.
- wechat: `cv2.wechat_qrcode_WeChatQRCode`, ships only with
  `opencv-contrib-python`. Also needs four model files whose location is
  found either via WECHAT_QR_MODELS_DIR or next to cv2.data.haarcascades.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np


def _decode_pyzbar(image_rgb: np.ndarray) -> Optional[str]:
    from pyzbar.pyzbar import decode as zbar_decode

    for r in zbar_decode(image_rgb):
        if r.type == "QRCODE":
            return r.data.decode("utf-8", errors="ignore")
    return None


def _decode_opencv(image_rgb: np.ndarray) -> Optional[str]:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    return data if data else None


_wechat_detector = None


def _find_wechat_models() -> Optional[Path]:
    env = os.environ.get("WECHAT_QR_MODELS_DIR")
    if env and Path(env).exists():
        return Path(env)
    try:
        import cv2.data as cv2data

        candidate = Path(cv2data.haarcascades).parent / "wechat_qrcode"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return None


def _decode_wechat(image_rgb: np.ndarray) -> Optional[str]:
    global _wechat_detector
    if _wechat_detector is None:
        d = _find_wechat_models()
        _wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            str(d / "detect.prototxt"),
            str(d / "detect.caffemodel"),
            str(d / "sr.prototxt"),
            str(d / "sr.caffemodel"),
        )
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results, _ = _wechat_detector.detectAndDecode(bgr)
    for s in results:
        if s:
            return str(s)
    return None


def _probe_pyzbar() -> Optional[Callable]:
    try:
        from pyzbar.pyzbar import decode as _  # noqa: F401
        return _decode_pyzbar
    except Exception:
        return None


def _probe_opencv() -> Optional[Callable]:
    # QRCodeDetector is in every opencv build.
    return _decode_opencv


def _probe_wechat() -> Optional[Callable]:
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return None
    if _find_wechat_models() is None:
        return None
    return _decode_wechat


_PROBES = {"pyzbar": _probe_pyzbar, "opencv": _probe_opencv, "wechat": _probe_wechat}


def available_decoders() -> Dict[str, Callable]:
    """{name: fn} for every decoder that can actually run on this machine."""
    out: Dict[str, Callable] = {}
    for name, probe in _PROBES.items():
        fn = probe()
        if fn is not None:
            out[name] = fn
    return out


def decode_with(name: str, image_rgb: np.ndarray) -> Optional[str]:
    """Decode with a specific decoder; returns None on any failure."""
    fn = _PROBES[name]() if name in _PROBES else None
    if fn is None:
        return None
    try:
        return fn(image_rgb)
    except Exception:
        return None


def decode_any(image_rgb: np.ndarray, order=("pyzbar", "wechat", "opencv")) -> Optional[str]:
    """First decoder in `order` that succeeds."""
    for name in order:
        s = decode_with(name, image_rgb)
        if s:
            return s
    return None
