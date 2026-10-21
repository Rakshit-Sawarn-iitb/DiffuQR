"""phase0_grid_errors.py

How far is each real frame from decodable, in the code's own units?

For every frame: align the clean reference onto the frame (ECC, as in
Phase 0.3), map the reference's QR corners into the native-resolution frame
crop, rectify, sample the 61 x 61 module grid, and compare it with the
bit-exact GT grid from `phase0_reference_version`. Then decode the grid
with our own Reed-Solomon layer (`diffuqr.qrcodec`):

  hard        plain RS (per block: errors <= nsym/2)
  soft        erase the least-confident codewords per block (confidence =
              |module intensity - threshold|), no learning   [C3, learning-free]
  oracle-er   erase exactly the wrong codewords (needs GT): the ceiling of
              ANY erasure strategy (per block: wrong codewords <= nsym)

Two thresholds are compared: one global Otsu threshold, and a local one
(Gaussian-smoothed module means) that tolerates illumination gradients.

ORACLE CAVEATS (state in the report): geometry comes from aligning the
reference photo, and version/EC/mask are taken from the reference. This
isolates the module-reading + decoding question from detection.

Also reports perspective strength of each frame's QR quad.

Outputs:
  artifacts/phase0/grid_errors.csv
  artifacts/phase0/grid_errors_summary.md

Usage:
  python -m scripts.phase0_grid_errors
  python -m scripts.phase0_grid_errors --sample 40
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from diffuqr import config
from diffuqr.align import align_reference_to_frame, bandpass, ecc_refine, ncc
from diffuqr.annotations import load_sam3_boxes, pick_square_box
from diffuqr.image import crop_square
from diffuqr.qrcodec import (block_layout, decode_grid, decode_soft, interleave_map,
                             read_codewords)
from diffuqr.qrgrid import fixed_template, identify, modules_for_version, order_quad, rectify, sample_grid

FRAME_RE = re.compile(r"^(?P<session>[a-z0-9]+_\d{8}_\d{2})_f(?P<idx>\d+)\.jpg$")
ALIGN_PX = 512
PAD = 0.15


def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_reference(qr_id: str, sam3) -> dict:
    meta = json.loads((config.CALIBRATION_DIR / f"{qr_id}_qrmeta.json").read_text(encoding="utf-8"))
    gt = np.load(config.CALIBRATION_DIR / f"{qr_id}_modules.npy")
    photo = config.reference_photo(qr_id)
    box = pick_square_box(sam3[f"raw/reference_photos/{photo.name}"]["boxes"])
    ref = np.asarray(crop_square(Image.open(photo).convert("RGB"),
                                 box, pad_frac=PAD))
    quad = None
    for f in (1.0, 0.5, 0.3):
        im = ref if f == 1.0 else cv2.resize(ref, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        hits = [r for r in zbar_decode(im) if r.type == "QRCODE" and len(r.polygon) == 4]
        if hits:
            quad = [(p.x / f, p.y / f) for p in hits[0].polygon]
            break
    if quad is None:
        raise SystemExit(f"[{qr_id}] reference crop does not decode; cannot get its quad")
    m = identify(ref, quad, versions=[meta["version"]], errors=[meta["error"]])
    assert m.mask == meta["mask"], "reference re-identification disagrees with qrmeta.json"
    s = ref.shape[0]
    return {
        "meta": meta, "gt": gt, "rotation": m.rotation,
        "ref512": cv2.resize(ref, (ALIGN_PX, ALIGN_PX), interpolation=cv2.INTER_AREA),
        "quad512": order_quad(quad) * (ALIGN_PX / s),
    }


def align_two_scale(frame512: np.ndarray, ref512: np.ndarray):
    """Coarse ECC at 256 px (the scale Phase 0.3 is tuned and validated at),
    then ECC refinement at 512 initialised from it; the refinement is kept
    only if band-passed NCC improves. Returns (W frame512 -> ref512, ncc)."""
    half = ALIGN_PX // 2
    f256 = cv2.resize(frame512, (half, half), interpolation=cv2.INTER_AREA)
    r256 = cv2.resize(ref512, (half, half), interpolation=cv2.INTER_AREA)
    _, score, W256 = align_reference_to_frame(to_bgr(f256), to_bgr(r256), ncc_thresh=0.0, return_warp=True)
    S = np.diag([2.0, 2.0, 1.0])
    W = (S @ W256.astype(np.float64) @ np.linalg.inv(S)).astype(np.float32)
    fg = cv2.cvtColor(frame512, cv2.COLOR_RGB2GRAY)
    fb, rb = bandpass(fg), bandpass(cv2.cvtColor(ref512, cv2.COLOR_RGB2GRAY))

    def score_of(Wx):
        warped = cv2.warpPerspective(rb, Wx, (ALIGN_PX, ALIGN_PX), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return ncc(fb, warped)

    s0 = score_of(W)
    W_ref = ecc_refine(fb, rb, np.linalg.inv(W.astype(np.float64)))  # ecc_refine takes REF -> FRAME init
    s1 = score_of(W_ref)
    return (W_ref, s1) if s1 > s0 else (W, s0)


def perspective_stats(q: np.ndarray) -> Dict[str, float]:
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    angs = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        angs.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    return {"side_ratio": max(sides) / min(sides), "max_angle_dev": max(abs(x - 90) for x in angs)}


def threshold_variants(means: np.ndarray, n: int):
    """{'global': (bits, conf), 'local': (bits, conf)} from module means."""
    u8 = np.clip(means, 0, 255).astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = {"global": ((means <= thr).astype(np.uint8), np.abs(means - thr))}
    local = cv2.GaussianBlur(means.astype(np.float32), (0, 0), sigmaX=4.0, borderType=cv2.BORDER_REFLECT)
    out["local"] = ((means <= local).astype(np.uint8), np.abs(means - local))
    return out


def block_errors(bits, gt, v, e, m) -> List[int]:
    cw, gcw = read_codewords(bits, v, e, m), read_codewords(gt, v, e, m)
    return [int((cw[list(ix)] != gcw[list(ix)]).sum()) for ix in interleave_map(v, e)]


def status(res, payload) -> str:
    if not res.ok:
        return "none"
    return "match" if res.payload == payload else "wrong"


def run(args) -> None:
    sam3 = load_sam3_boxes()
    baseline = collections.defaultdict(bool)
    blur_of: Dict[tuple, float] = {}
    for r in csv.DictReader(open(config.PHASE0_DIR / "decode_baseline.csv", encoding="utf-8")):
        k = (r["session_id"], int(r["frame_idx"]))
        if r["variant"] == "full-native":
            baseline[k] |= r["status"] == "match"
            blur_of[k] = float(r["blur_score"])

    sessions = sorted({p.stem for p in config.SESSIONS_DIR.glob("*.json") if (config.FRAMES_DIR / p.stem).is_dir()})
    refs: Dict[str, dict] = {}
    rows = []
    t0 = time.time()
    for sid in sessions:
        qr_id = sid.split("_", 1)[0]
        if qr_id not in refs:
            refs[qr_id] = prepare_reference(qr_id, sam3)
        R = refs[qr_id]
        v, e, mk = R["meta"]["version"], R["meta"]["error"], R["meta"]["mask"]
        n = modules_for_version(v)
        nsym = min(t - d for t, d in block_layout(v, e))
        fixed, fvals = fixed_template(v, e, mk)
        data_region = ~fixed
        frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
        if args.sample and args.sample < len(frames):
            random.seed(args.seed)
            frames = sorted(random.sample(frames, args.sample))
        print(f"[{sid}] {len(frames)} frames, v{v}-{e} mask {mk}, nsym/block {nsym}")
        for i, fp in enumerate(frames):
            rec = sam3.get(f"frames/{sid}/{fp.name}")
            box = pick_square_box(rec["boxes"]) if rec else None
            if box is None:
                continue
            fidx = int(FRAME_RE.match(fp.name).group("idx"))
            F = np.asarray(crop_square(Image.open(fp).convert("RGB"), box, pad_frac=PAD))
            s_f = F.shape[0]
            F512 = cv2.resize(F, (ALIGN_PX, ALIGN_PX), interpolation=cv2.INTER_AREA)
            W, score = align_two_scale(F512, R["ref512"])
            score = float(score)
            row = {"session_id": sid, "qr_id": qr_id, "frame_idx": fidx,
                   "blur_score": blur_of.get((sid, fidx), ""), "align_ncc": round(score, 3),
                   "aligned": score >= args.ncc_thresh,
                   "pyzbar_opencv_native": baseline.get((sid, fidx), False)}
            if row["aligned"]:
                q = cv2.perspectiveTransform(R["quad512"].reshape(-1, 1, 2).astype(np.float64),
                                             np.linalg.inv(W).astype(np.float64)).reshape(4, 2)
                q_native = q * (s_f / ALIGN_PX)
                row.update({k: round(x, 3) for k, x in perspective_stats(q_native).items()})
                row["qr_side_px"] = round(float(np.mean([np.linalg.norm(q_native[(j + 1) % 4] - q_native[j])
                                                         for j in range(4)])), 1)
                sq = rectify(F, q_native, n * 12, reorder=False)
                _, means = sample_grid(sq, n)
                means = np.rot90(means, R["rotation"])
                for name, (bits, conf) in threshold_variants(means, n).items():
                    be = block_errors(bits, R["gt"], v, e, mk)
                    hard = decode_grid(bits, v, e, mk)
                    soft, k_used = decode_soft(bits, conf, v, e, mk)
                    row.update({
                        f"{name}_fixed_agree": round(float((bits[fixed] == fvals[fixed]).mean()), 4),
                        f"{name}_module_err": round(float((bits[data_region] != R["gt"][data_region]).mean()), 4),
                        f"{name}_max_block_cw_err": max(be),
                        f"{name}_hard": status(hard, R["meta"]["payload"]),
                        f"{name}_soft": status(soft, R["meta"]["payload"]),
                        f"{name}_soft_k": k_used if soft.ok else "",
                        f"{name}_oracle_erasure_ok": max(be) <= nsym,
                    })
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(frames)}  elapsed {time.time() - t0:.0f}s")

    out_csv = config.PHASE0_DIR / "grid_errors.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")
    write_summary(rows, nsym, config.PHASE0_DIR / "grid_errors_summary.md")


def write_summary(rows: List[dict], nsym: int, out_path: Path) -> None:
    by_sess = collections.defaultdict(list)
    for r in rows:
        by_sess[r["session_id"]].append(r)
    for sess_rows in by_sess.values():
        order = sorted(sess_rows, key=lambda r: float(r["blur_score"]) if r["blur_score"] != "" else 0.0)
        m = len(order)
        for rank, r in enumerate(order):
            r["tertile"] = "blurriest" if rank < m / 3 else ("medium" if rank < 2 * m / 3 else "sharpest")
    terts = ("blurriest", "medium", "sharpest")

    def cell(sub, pred):
        k = sum(1 for r in sub if pred(r))
        return f"{100 * k / len(sub):.0f}% ({k}/{len(sub)})" if sub else "-"

    def line(label, pred, sub_rows):
        cells = [cell(sub_rows, pred)] + [cell([r for r in sub_rows if r["tertile"] == t], pred) for t in terts]
        return f"| {label} | " + " | ".join(cells) + " |"

    aligned = [r for r in rows if r["aligned"]]
    L = ["# Phase 0.4 - Module-grid errors and RS decoding on real frames", "",
         f"Frames with a SAM3 box: {len(rows)}. Aligned (NCC >= 0.5): {len(aligned)}.",
         f"RS budget per block: {nsym} EC codewords -> hard decode needs <= {nsym // 2} wrong codewords "
         f"in every block; perfect erasures allow <= {nsym}.", "",
         "Oracle geometry (reference alignment) and oracle version/EC/mask. Decode rates are over ALL",
         "frames with a box (unaligned frames count as failures) so they are comparable to Phase 0.1.", "",
         "## Decode rate", "",
         "| method | overall | blurriest | medium | sharpest |", "|---|---|---|---|---|",
         line("pyzbar/opencv any, full-native (Phase 0.1)", lambda r: r["pyzbar_opencv_native"], rows)]
    for th in ("global", "local"):
        L.append(line(f"grid + RS hard ({th} thr)", lambda r, th=th: r.get(f"{th}_hard") == "match", rows))
        L.append(line(f"grid + RS soft erasures ({th} thr)", lambda r, th=th: r.get(f"{th}_soft") == "match", rows))
        L.append(line(f"oracle erasures, ceiling ({th} thr)", lambda r, th=th: bool(r.get(f"{th}_oracle_erasure_ok")), rows))
    L.append(line("union: baseline OR best soft", lambda r: r["pyzbar_opencv_native"] or r.get("global_soft") == "match"
                  or r.get("local_soft") == "match", rows))
    wrong = {th: sum(r.get(f"{th}_{m}") == "wrong" for r in rows for m in ("hard", "soft")) for th in ("global", "local")}
    L += ["", f"Wrong decodes (RS mis-correction to a non-GT payload): global {wrong['global']}, local {wrong['local']}.", ""]

    L += ["## Module and codeword error rates (aligned frames, local threshold)", "",
          "| tertile | n | median module err | p25 | p75 | median max-block wrong codewords | median fixed-pattern agreement |",
          "|---|---|---|---|---|---|---|"]
    for t in terts:
        sub = [r for r in aligned if r["tertile"] == t]
        if not sub:
            continue
        me = np.array([r["local_module_err"] for r in sub])
        cw = np.array([r["local_max_block_cw_err"] for r in sub])
        fa = np.array([r["local_fixed_agree"] for r in sub])
        L.append(f"| {t} | {len(sub)} | {np.median(me):.3f} | {np.percentile(me, 25):.3f} | {np.percentile(me, 75):.3f} | "
                 f"{np.median(cw):.0f} | {np.median(fa):.3f} |")

    if aligned:
        sr = np.array([r["side_ratio"] for r in aligned])
        ad = np.array([r["max_angle_dev"] for r in aligned])
        L += ["", "## Perspective of the QR in frame (aligned frames)", "",
              "| metric | p5 | p50 | p95 | max |", "|---|---|---|---|---|",
              f"| side-length ratio (max/min) | {np.percentile(sr, 5):.3f} | {np.median(sr):.3f} | {np.percentile(sr, 95):.3f} | {sr.max():.3f} |",
              f"| max corner-angle deviation from 90 deg | {np.percentile(ad, 5):.1f} | {np.median(ad):.1f} | {np.percentile(ad, 95):.1f} | {ad.max():.1f} |"]
    L += ["", "## Reading",
          "- `oracle erasures` is the most any confidence-based erasure scheme can reach from THIS sampled",
          "  grid; the gap between `soft` and it is what better per-module confidence (e.g. a diffusion",
          "  posterior, C3) could still recover. Above the ceiling only a better grid estimate helps",
          "  (restoration, C2, or multi-frame fusion, C4).",
          "- `fixed-pattern agreement` near 1 means geometry is right; low values flag misregistration",
          "  rather than blur.", ""]
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncc-thresh", type=float, default=0.5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
