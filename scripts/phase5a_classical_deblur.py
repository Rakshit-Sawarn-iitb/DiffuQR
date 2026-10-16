"""phase5a_classical_deblur.py

Decode-rate measurement for classical, kernel-based single-image
deconvolution baselines. This is the FLOOR the learned restorer must
clear: if Wiener/Richardson-Lucy with the oracle (per-frame estimated)
kernel already lifts the blurriest tertile from 1% to, say, 60%, then
the paper's story is kernel estimation + classical deconvolution +
RS-aware decoding, and the diffusion contribution has to earn its place
on top of that.

Setup for internal consistency with the kernel bank:
  - 256 px crop size (same as Phase 0.3 kernel estimation).
  - Three methods: oracle-kernel Wiener, oracle-kernel Richardson-Lucy,
    bank-mean-kernel Wiener (blind-ish: no per-frame kernel info).
  - Raw 256-crop decode is reported as the baseline to compare against.
    Native-resolution decode (from Phase 0.1) is reported separately for
    context; it is a different experimental regime, not a direct baseline.

Outputs (artifacts/phase5a/):
  classical_deblur.csv            per (frame, method, decoder) row
  classical_deblur_summary.md     grouped decode rates + blur-tertile breakdown
  classical_deblur.png            bar chart

Usage:
  python -m scripts.phase5a_classical_deblur
  python -m scripts.phase5a_classical_deblur --rl-iters 30 --nsr 0.005
  python -m scripts.phase5a_classical_deblur --save-samples 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from diffuqr import config
from diffuqr.deblur import deblur_rgb
from diffuqr.decode import available_decoders
from diffuqr.image import crop_square, laplacian_variance, pil_to_rgb_array


FRAME_RE = re.compile(r"^(?P<session>[a-z0-9]+_\d{8}_\d{2})_f(?P<idx>\d+)\.jpg$")

PHASE5A_DIR = config.ARTIFACTS_DIR / "phase5a"


def load_sam3_boxes() -> Dict[str, dict]:
    path = config.ANNOTATIONS_DIR / "all_boxes.jsonl"
    out: Dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                out[r["file_name"]] = r
    return out


def pick_square_box(boxes) -> Optional[List[float]]:
    ok = []
    for b in boxes:
        x1, y1, x2, y2 = b["box_xyxy"]
        w, h = x2 - x1, y2 - y1
        if w > 8 and h > 8 and 0.5 <= w / h <= 2.0:
            ok.append(b)
    if not ok:
        return None
    return max(ok, key=lambda b: b["score"])["box_xyxy"]


def extract_gt_per_qrid(decoders) -> Dict[str, str]:
    gt: Dict[str, str] = {}
    for ref_path in sorted((config.RAW_DIR / "reference_photos").glob("*.jpg")):
        qr_id = ref_path.stem
        bgr = cv2.imread(str(ref_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        found = None
        for size in (None, 1024, 768, 512, 384, 256):
            probe = rgb if size is None else cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            for name, fn in decoders.items():
                try:
                    s = fn(probe)
                except Exception:
                    s = None
                if s:
                    found = s
                    break
            if found:
                break
        if found:
            gt[qr_id] = found
    return gt


def status_for(decoded: Optional[str], gt: Optional[str]) -> str:
    if gt is None:
        return "no_gt"
    if not decoded:
        return "none"
    return "match" if decoded == gt else "wrong"


def crop_to_size(pil_img: Image.Image, box: List[float], size: int, pad_frac: float = 0.15) -> np.ndarray:
    sq = crop_square(pil_img, box, pad_frac=pad_frac)
    bgr = cv2.cvtColor(np.asarray(sq), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run(args) -> None:
    decoders = available_decoders()
    if not decoders:
        raise SystemExit("No QR decoders available.")
    print(f"Decoders: {list(decoders)}")

    bank_path = config.CALIBRATION_DIR / "kernels.npz"
    if not bank_path.exists():
        raise SystemExit(f"Kernel bank not found at {bank_path}. Run phase0_kernel_estimation first.")
    d = np.load(bank_path, allow_pickle=True)
    kernels = d["kernels"].astype(np.float32)
    meta_raw = [json.loads(m) for m in d["meta"]]
    print(f"Kernel bank: {kernels.shape}, mean kernel from bank will serve as 'blind' baseline.")

    # Clip to non-negative and renormalise before use. Ringing lobes with negative
    # mass are an artefact of Fourier division; a convolution kernel for deblurring
    # must be non-negative (and in RL's case, strictly so).
    kernels_pos = np.clip(kernels, 0, None)
    sums = kernels_pos.reshape(len(kernels_pos), -1).sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-9, 1.0, sums).reshape(-1, 1, 1)
    kernels_pos = kernels_pos / sums
    mean_kernel = kernels_pos.mean(axis=0)
    mean_kernel = mean_kernel / mean_kernel.sum()

    key_to_kernel = {(m["session_id"], int(m["frame_idx"])): k for m, k in zip(meta_raw, kernels_pos)}

    sam3 = load_sam3_boxes()
    gt_by_qrid = extract_gt_per_qrid(decoders)
    print(f"GT established for qr_ids: {list(gt_by_qrid)}")

    rows: List[dict] = []
    PHASE5A_DIR.mkdir(parents=True, exist_ok=True)
    samples_saved = 0
    t0 = time.time()

    sessions = sorted({k[0] for k in key_to_kernel})
    for sid in sessions:
        qr_id = sid.split("_", 1)[0]
        gt = gt_by_qrid.get(qr_id)
        print(f"\n[{sid}] qr_id={qr_id} gt={'yes' if gt else 'MISSING'}")
        frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
        processed = 0
        for fp in frames:
            m = FRAME_RE.match(fp.name)
            if not m:
                continue
            frame_idx = int(m.group("idx"))
            if (sid, frame_idx) not in key_to_kernel:
                continue
            rel = f"frames/{sid}/{fp.name}"
            rec = sam3.get(rel)
            if rec is None:
                continue
            box = pick_square_box(rec["boxes"])
            if box is None:
                continue
            try:
                pil = Image.open(fp).convert("RGB")
            except Exception:
                continue
            frame_rgb = crop_to_size(pil, box, args.crop_size)
            blur = laplacian_variance(frame_rgb)
            per_frame_kernel = key_to_kernel[(sid, frame_idx)]

            # Six (method, kernel) combinations, pre-computed once per frame.
            variants: Dict[str, np.ndarray] = {
                "raw": frame_rgb,
                "oracle-wiener": deblur_rgb(frame_rgb, per_frame_kernel, "wiener", nsr=args.nsr),
                "oracle-rl": deblur_rgb(frame_rgb, per_frame_kernel, "rl", iterations=args.rl_iters),
                "bank-mean-wiener": deblur_rgb(frame_rgb, mean_kernel, "wiener", nsr=args.nsr),
                "bank-mean-rl": deblur_rgb(frame_rgb, mean_kernel, "rl", iterations=args.rl_iters),
            }

            for v_name, v_img in variants.items():
                for d_name, d_fn in decoders.items():
                    t1 = time.time()
                    try:
                        decoded = d_fn(v_img)
                    except Exception:
                        decoded = None
                    dt = time.time() - t1
                    rows.append({
                        "session_id": sid, "qr_id": qr_id, "frame_idx": frame_idx,
                        "method": v_name, "decoder": d_name,
                        "status": status_for(decoded, gt),
                        "decode_time_s": round(dt, 4),
                        "blur_score": round(blur, 2),
                        "k_length_256": round(float(np.sqrt(((per_frame_kernel.sum(axis=0) *
                            np.arange(per_frame_kernel.shape[1])).sum() -
                            per_frame_kernel.sum() * per_frame_kernel.shape[1] / 2) ** 2) + 1e-9), 3),
                    })

            # Optional qualitative samples: save a side-by-side strip for the
            # first few frames of each session so we can eyeball the output.
            if samples_saved < args.save_samples:
                strip = np.concatenate([variants["raw"], variants["oracle-wiener"],
                                        variants["oracle-rl"]], axis=1)
                out_path = PHASE5A_DIR / f"sample_{sid}_f{frame_idx:06d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
                samples_saved += 1

            processed += 1
            if processed % 25 == 0:
                print(f"  ... {processed}  elapsed {time.time() - t0:.1f}s")

    out_csv = PHASE5A_DIR / "classical_deblur.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    write_summary(rows, PHASE5A_DIR / "classical_deblur_summary.md")
    try:
        plot_summary(rows, PHASE5A_DIR / "classical_deblur.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")


def write_summary(rows: List[dict], out_path: Path) -> None:
    import collections

    # Blur tertile per (session, frame_idx), shared across all method rows.
    by_frame = collections.defaultdict(list)
    for r in rows:
        by_frame[(r["session_id"], r["frame_idx"])].append(r)
    tertile_of = {}
    for sid in {k[0] for k in by_frame}:
        frames = sorted({k[1] for k in by_frame if k[0] == sid})
        blurs = {fi: by_frame[(sid, fi)][0]["blur_score"] for fi in frames}
        order = sorted(frames, key=lambda f: blurs[f])
        n = len(order)
        for rank, fi in enumerate(order):
            bucket = "blurriest" if rank < n / 3 else ("medium" if rank < 2 * n / 3 else "sharpest")
            tertile_of[(sid, fi)] = bucket
    for r in rows:
        r["blur_tertile"] = tertile_of[(r["session_id"], r["frame_idx"])]

    methods = ["raw", "oracle-wiener", "oracle-rl", "bank-mean-wiener", "bank-mean-rl"]
    decoders = sorted({r["decoder"] for r in rows})
    tertiles = ("blurriest", "medium", "sharpest")

    def _rate(subset):
        n = len(subset)
        if n == 0:
            return (0, 0, "-")
        m = sum(r["status"] == "match" for r in subset)
        return (m, n, f"{100 * m / n:.0f}%")

    lines = []
    lines.append("# Phase 5a - Classical Deconvolution Baselines")
    lines.append("")
    lines.append(f"Rows: {len(rows)}.  Methods: {', '.join(methods)}.  Decoders: {', '.join(decoders)}.")
    lines.append("All measurements at 256-px crop size (matches the kernel bank's reference frame).")
    lines.append("")

    lines.append("## Decode rate by (method, decoder), overall + blur tertile")
    lines.append("")
    lines.append("| method | decoder | overall | blurriest | medium | sharpest | wrong | ms/frame |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mth in methods:
        for d in decoders:
            sub = [r for r in rows if r["method"] == mth and r["decoder"] == d]
            m, n, overall = _rate(sub)
            cells = [overall]
            for t in tertiles:
                _, _, c = _rate([r for r in sub if r["blur_tertile"] == t])
                cells.append(c)
            wrong = sum(r["status"] == "wrong" for r in sub)
            ms = 1000 * sum(r["decode_time_s"] for r in sub) / max(1, n)
            lines.append(f"| {mth} | {d} | {cells[0]} ({m}/{n}) | {cells[1]} | {cells[2]} | {cells[3]} | {wrong} | {ms:.0f} |")

    lines.append("")
    lines.append("## Any-decoder match per method (OR over decoders) - the real ceiling")
    lines.append("")
    lines.append("| method | overall | blurriest | medium | sharpest |")
    lines.append("|---|---|---|---|---|")
    for mth in methods:
        by_frame_match: Dict[tuple, bool] = {}
        tert_by_frame: Dict[tuple, str] = {}
        for r in rows:
            if r["method"] != mth:
                continue
            key = (r["session_id"], r["frame_idx"])
            by_frame_match[key] = by_frame_match.get(key, False) or (r["status"] == "match")
            tert_by_frame[key] = r["blur_tertile"]
        total = len(by_frame_match)
        hits = sum(by_frame_match.values())
        per_tert = {t: [0, 0] for t in tertiles}
        for k, v in by_frame_match.items():
            per_tert[tert_by_frame[k]][1] += 1
            per_tert[tert_by_frame[k]][0] += int(v)
        cells = []
        for t in tertiles:
            hh, nn = per_tert[t]
            cells.append(f"{100 * hh / nn:.0f}% ({hh}/{nn})" if nn else "-")
        lines.append(f"| {mth} | {100 * hits / total:.0f}% ({hits}/{total}) | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} |")

    lines.append("")
    lines.append("## Reading")
    lines.append("- `raw` is the baseline at 256-crop (matches the training notebook's eval regime).")
    lines.append("  Compare the `any-decoder` cell for each other method against this row.")
    lines.append("- `oracle-*` uses the kernel estimated from the aligned reference; it is the ceiling")
    lines.append("  of any per-frame kernel-estimation method. If a learned restorer cannot beat this,")
    lines.append("  the real research contribution is in kernel estimation or decoding, not restoration.")
    lines.append("- `bank-mean-*` is a weak blind baseline: no per-frame kernel information.")
    lines.append("- Wrong-decode counts stay at zero in a well-behaved experiment; nonzero means a")
    lines.append("  decoder is confidently returning garbled payloads (unlikely with Reed-Solomon).")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def plot_summary(rows: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["raw", "oracle-wiener", "oracle-rl", "bank-mean-wiener", "bank-mean-rl"]
    tertiles = ("blurriest", "medium", "sharpest")

    # Any-decoder match per (method, tertile).
    rates: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    by_frame: Dict[tuple, Dict[str, bool]] = {}
    for r in rows:
        key = (r["session_id"], r["frame_idx"])
        by_frame.setdefault(key, {})
        by_frame[key][r["method"]] = by_frame[key].get(r["method"], False) or (r["status"] == "match")
    # Tertile per key.
    key_tert = {k: v[0]["blur_tertile"] for k, v in {}.items()}
    import collections
    buckets = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        key = (r["session_id"], r["frame_idx"])
        buckets[r["method"]][r["blur_tertile"]]  # init
    for key, mdict in by_frame.items():
        tert = next(r["blur_tertile"] for r in rows if (r["session_id"], r["frame_idx"]) == key)
        for m, matched in mdict.items():
            buckets[m][tert][1] += 1
            if matched:
                buckets[m][tert][0] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.25
    colors = {"blurriest": "#C44E52", "medium": "#DD8452", "sharpest": "#4C72B0"}
    for i, t in enumerate(tertiles):
        heights = []
        for m in methods:
            hh, nn = buckets[m][t]
            heights.append(100 * hh / nn if nn else 0)
        ax.bar(x + (i - 1) * width, heights, width, label=t, color=colors[t])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Decode rate (any decoder, %)")
    ax.set_title("Phase 5a - classical deconvolution vs 256-crop baseline, by blur tertile")
    ax.legend(title="blur tertile")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--crop-size", type=int, default=256,
                   help="Must match the crop size the kernel bank was fit at (256 by default).")
    p.add_argument("--nsr", type=float, default=0.01)
    p.add_argument("--rl-iters", type=int, default=20)
    p.add_argument("--save-samples", type=int, default=0,
                   help="Save N (raw | wiener | rl) side-by-side strips for eyeballing.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
