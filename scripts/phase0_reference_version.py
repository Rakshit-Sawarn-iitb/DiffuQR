"""phase0_reference_version.py

Measure each physical QR's actual version / EC level / mask from its clean
reference photo, instead of assuming version 11 (Phase 0.2 did).

Per reference photo:
  1. Crop around the SAM3 box (PIL coordinates: the annotation was made on
     the raw pixel array, ignoring EXIF orientation - cv2.imread would
     auto-rotate and misalign the box).
  2. pyzbar -> payload + 4-corner quad.
  3. `diffuqr.qrgrid.identify`: rectify, sample the module grid per
     candidate version, match against the payload-independent fixed
     modules (function patterns + format + version info).
  4. Re-encode the payload with segno at the identified settings; if all
     modules agree, the sampled grid is a verified ground-truth bit grid.
  5. Re-express Phase 0.2 frame px/module with the measured module count.

Outputs:
  artifacts/phase0/reference_version.md
  artifacts/phase0/reference_grid_<qr_id>.png     rectified crop | sampled grid
  data/calibration/<qr_id>_modules.npy            oriented GT module grid (1 = dark)
  data/calibration/<qr_id>_qrmeta.json            version / EC / mask / GT payload

Usage:
  python -m scripts.phase0_reference_version
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from diffuqr import config
from diffuqr.qrcodec import decode_grid, write_codewords
from diffuqr.qrgrid import identify, modules_for_version, payload_agreement, rectify


def load_reference_boxes() -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    path = config.ANNOTATIONS_DIR / "all_boxes.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") == "reference" and r["boxes"]:
                out[r["qr_id"]] = max(r["boxes"], key=lambda b: b["score"])["box_xyxy"]
    return out


def zbar_with_quad(rgb: np.ndarray):
    """Try a few downscales; return (payload, quad in `rgb` coords) or None."""
    for f in (1.0, 0.5, 0.3):
        im = rgb if f == 1.0 else cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        for r in zbar_decode(im):
            if r.type == "QRCODE" and len(r.polygon) == 4:
                quad = [(p.x / f, p.y / f) for p in r.polygon]
                return r.data.decode("utf-8", errors="ignore"), quad
    return None


def frame_px_per_module(n_modules: int) -> Optional[dict]:
    path = config.PHASE0_DIR / "pixel_module_stats.csv"
    if not path.exists():
        return None
    sides = np.array([float(r["qr_side_px"]) for r in csv.DictReader(open(path, encoding="utf-8"))
                      if r["qr_side_px"]])
    ppm = sides / n_modules
    return {
        "n": len(ppm),
        "p5": float(np.percentile(ppm, 5)), "p50": float(np.median(ppm)),
        "p95": float(np.percentile(ppm, 95)), "min": float(ppm.min()),
    }


def grid_decodes_to(bits: np.ndarray, payload: str, scale: int = 10) -> bool:
    """Render the sampled grid as a clean code (4-module quiet zone) and
    check it decodes to `payload`. RS may silently fix a few module errors,
    so True means "valid code for this payload", not "bit-exact"."""
    n = bits.shape[0]
    img = np.full(((n + 8) * scale,) * 2, 255, np.uint8)
    img[4 * scale:(4 + n) * scale, 4 * scale:(4 + n) * scale] = (1 - np.kron(bits, np.ones((scale, scale), np.uint8))) * 255
    return any(r.data.decode("utf-8", errors="ignore") == payload for r in zbar_decode(np.dstack([img] * 3)))


def save_panel(rgb: np.ndarray, quad, match, out_path: Path) -> None:
    n = match.bits.shape[0]
    sq = np.rot90(rectify(rgb, quad, n * 12), match.rotation)
    grid = cv2.resize((1 - match.bits) * 255, (n * 12, n * 12), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    panel = np.concatenate([sq, np.full((n * 12, 16), 128, np.uint8), grid], axis=1)
    cv2.imwrite(str(out_path), cv2.resize(panel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def main() -> None:
    boxes = load_reference_boxes()
    lines = ["# Phase 0.2b - Measured QR version per reference photo", ""]
    lines += ["Replaces the version-11 assumption in `pixel_module_stats_summary.md`.", ""]
    for qr_id, box in sorted(boxes.items()):
        path = config.reference_photo(qr_id)
        pil = Image.open(path).convert("RGB")  # raw orientation, matches SAM3 box
        x1, y1, x2, y2 = box
        pad = 0.08 * max(x2 - x1, y2 - y1)
        l, t = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        crop = np.asarray(pil.crop((l, t, int(min(pil.width, x2 + pad)), int(min(pil.height, y2 + pad)))))

        z = zbar_with_quad(crop)
        if z is None:
            print(f"[{qr_id}] pyzbar could not decode the reference crop")
            lines += [f"## {qr_id}", "", "pyzbar could not decode the reference crop; not measured.", ""]
            continue
        payload, quad = z
        match = identify(crop, quad)
        n = modules_for_version(match.version)
        full_agree, _ = payload_agreement(match, payload)
        decodes = grid_decodes_to(match.bits, payload)
        runner_up = max(
            (s for v in range(1, 41) if v != match.version
             for s in [identify(crop, quad, versions=[v]).fixed_agreement]),
        )
        side = float(np.mean([np.linalg.norm(np.subtract(quad[i], quad[(i + 1) % 4])) for i in range(4)]))

        print(f"[{qr_id}] v{match.version} ({n}x{n}) EC-{match.error} mask {match.mask} rot {match.rotation} "
              f"fixed-agree {match.fixed_agreement:.4f} over {match.n_fixed}  payload-agree {full_agree:.4f}")

        # Own RS decode of the sampled grid; if it succeeds, rebuild the grid
        # from corrected codewords -> bit-exact GT (sampling noise removed).
        rs = decode_grid(match.bits, match.version, match.error, match.mask)
        rs_ok = rs.ok and rs.payload == payload
        gt_grid = write_codewords(rs.corrected_codewords, match.version, match.error, match.mask) if rs_ok else match.bits
        n_fixed_cw = sum(b.n_corrected for b in rs.blocks) if rs_ok else None
        n_mod_diff = int((gt_grid != match.bits).sum())

        config.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.CALIBRATION_DIR / f"{qr_id}_modules.npy", gt_grid)
        (config.CALIBRATION_DIR / f"{qr_id}_qrmeta.json").write_text(json.dumps({
            "qr_id": qr_id, "version": match.version, "error": match.error, "mask": match.mask,
            "payload": payload, "gt_grid_exact": bool(rs_ok),
        }, indent=2), encoding="utf-8")
        save_panel(crop, quad, match, config.PHASE0_DIR / f"reference_grid_{qr_id}.png")

        lines += [f"## {qr_id}", "",
                  "| field | value |", "|---|---|",
                  f"| payload length | {len(payload)} chars (prefix `{payload[:12]}...`) |",
                  f"| version | **{match.version}** ({n} x {n} modules) |",
                  f"| EC level | **{match.error}** |",
                  f"| mask | {match.mask} |",
                  f"| fixed-module agreement | {match.fixed_agreement:.2%} of {match.n_fixed} payload-independent modules |",
                  f"| best agreement at any other version | {runner_up:.2%} |",
                  f"| sampled grid, re-rendered clean, decodes to payload | {'yes' if decodes else 'NO'} |",
                  f"| own RS decode of sampled grid (`diffuqr.qrcodec`) | "
                  f"{'payload match, ' + str(n_fixed_cw) + ' codewords corrected' if rs_ok else 'FAILED'} |",
                  f"| saved GT grid | {'bit-exact (rebuilt from RS-corrected codewords)' if rs_ok else 'sampled (not exact)'}; "
                  f"differs from sampled grid in {n_mod_diff} modules |",
                  f"| full-grid agreement with segno re-encode | {full_agree:.2%} |",
                  f"| reference symbol side | {side:.0f} px -> {side / n:.1f} px/module |",
                  ""]
        fp = frame_px_per_module(n)
        if fp:
            lines += [f"Frame px/module with the measured {n} modules (SAM3 box side / {n}, n={fp['n']}): "
                      f"p5 {fp['p5']:.1f}, median {fp['p50']:.1f}, p95 {fp['p95']:.1f}, min {fp['min']:.1f}.",
                      ""]
    lines += ["## Notes",
              "- Identification uses only payload-independent modules, so it does not depend on the",
              "  original encoder's mode segmentation.",
              "- Full-grid agreement with segno < 100% is expected when the original encoder segmented",
              "  the payload into different modes (segno uses one mode for the whole string). It is",
              "  not an identification error.",
              "- `data/calibration/<qr_id>_modules.npy` is rebuilt from the RS-corrected codewords when",
              "  our own decode reproduces the payload, so it is bit-exact ground truth.",
              "- Reference JPEGs carry EXIF orientation; everything here uses raw (PIL) pixel coordinates,",
              "  matching the SAM3 annotations.", ""]
    out = config.PHASE0_DIR / "reference_version.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
