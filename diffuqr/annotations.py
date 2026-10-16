"""SAM3 annotation helpers (annotations/all_boxes.jsonl).

Boxes are in RAW pixel coordinates (PIL `Image.open`, no EXIF transpose).
Never pair them with `cv2.imread`, which auto-rotates EXIF-tagged photos.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from diffuqr import config


def load_sam3_boxes() -> Dict[str, dict]:
    """{file_name: record}; file_name like `frames/<session>/<file>.jpg`."""
    path = config.ANNOTATIONS_DIR / "all_boxes.jsonl"
    out: Dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                out[r["file_name"]] = r
    return out


def pick_square_box(boxes) -> Optional[List[float]]:
    """Highest-score SAM3 box that is at least roughly square (QR codes are)."""
    ok = []
    for b in boxes:
        x1, y1, x2, y2 = b["box_xyxy"]
        w, h = x2 - x1, y2 - y1
        if w > 8 and h > 8 and 0.5 <= w / h <= 2.0:
            ok.append(b)
    if not ok:
        return None
    return max(ok, key=lambda b: b["score"])["box_xyxy"]
