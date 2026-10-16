"""phase0_decode_baseline.py

Native-resolution decode baseline: how often does the raw frame already
decode, before any restoration model touches it? This is the only number
that justifies the paper's problem statement.

For every extracted frame we try several (crop, preprocessing, decoder)
combinations and record match / wrong-decode / no-decode plus timing and
per-frame blur score and QR pixel size.

Outputs (artifacts/phase0/):
  decode_baseline.csv            long format, one row per (frame, variant, decoder)
  decode_baseline_summary.md     grouped decode rates, blur-tertile stratified
  decode_baseline.png            bar chart of per-variant decode rate

Usage:
  python -m scripts.phase0_decode_baseline
  python -m scripts.phase0_decode_baseline --sessions buggy01_20260917_01 --sample 60
  python -m scripts.phase0_decode_baseline --variants full-native sam3-up512
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from diffuqr import config
from diffuqr.decode import available_decoders
from diffuqr.image import (
    crop_square,
    laplacian_variance,
    pil_to_rgb_array,
    unsharp,
    upscale_lanczos,
)


FRAME_RE = re.compile(r"^(?P<session>[a-z0-9]+_\d{8}_\d{2})_f(?P<idx>\d+)\.jpg$")


# -- Discovery -------------------------------------------------------------

def discover_sessions(filter_ids: Optional[List[str]]) -> List[str]:
    sess: List[str] = []
    for p in sorted(config.SESSIONS_DIR.glob("*.json")):
        sid = p.stem
        if filter_ids and sid not in filter_ids:
            continue
        if (config.FRAMES_DIR / sid).is_dir():
            sess.append(sid)
    return sess


def discover_frames(session_id: str) -> List[Path]:
    return sorted((config.FRAMES_DIR / session_id).glob(f"{session_id}_f*.jpg"))


def frame_qr_id(session_id: str) -> str:
    return session_id.split("_", 1)[0]


def load_sam3_boxes() -> Dict[str, dict]:
    """{file_name_in_annotations: record} from annotations/all_boxes.jsonl, if present.
    `file_name` as stored in the HF layout uses forward slashes, e.g.
    `frames/<session>/<session>_fNNNNNN.jpg`."""
    path = config.ANNOTATIONS_DIR / "all_boxes.jsonl"
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
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


# -- Ground truth ----------------------------------------------------------

def extract_gt_per_qrid(decoders: Dict[str, callable]) -> Dict[str, str]:
    """Decode each reference photo at successively smaller sizes until any
    decoder returns a payload. One payload per qr_id (the physical sticker)."""
    gt: Dict[str, str] = {}
    ref_dir = config.RAW_DIR / "reference_photos"
    for ref_path in sorted(ref_dir.glob("*.jpg")):
        qr_id = ref_path.stem
        bgr = cv2.imread(str(ref_path))
        if bgr is None:
            print(f"  [GT] {qr_id}: could not open {ref_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        found = None
        for size in (None, 1024, 768, 512, 384, 256):
            if size is None:
                probe = rgb
            else:
                probe = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            for name, fn in decoders.items():
                try:
                    s = fn(probe)
                except Exception:
                    s = None
                if s:
                    found = (s, name, size)
                    break
            if found:
                break
        if found:
            s, name, size = found
            print(f"  [GT] {qr_id}: via {name} at size={size} -- {s[:60]}{'...' if len(s) > 60 else ''}")
            gt[qr_id] = s
        else:
            print(f"  [GT] {qr_id}: WARNING -- reference photo does not decode at any size")
    return gt


# -- Variants --------------------------------------------------------------

def build_variants(frame_rgb: np.ndarray, sam3_box: Optional[List[float]], selected: List[str]) -> List[tuple]:
    """Yield (variant_name, image_rgb) for each requested variant that can be
    built from this frame."""
    pil = Image.fromarray(frame_rgb)
    out = []
    if "full-native" in selected:
        out.append(("full-native", frame_rgb))
    if sam3_box is not None:
        crop_pil = crop_square(pil, sam3_box, pad_frac=0.15)
        crop = pil_to_rgb_array(crop_pil)
        if "sam3-native" in selected:
            out.append(("sam3-native", crop))
        if "sam3-up512" in selected:
            out.append(("sam3-up512", upscale_lanczos(crop, 512)))
        if "sam3-up512-sharp" in selected:
            out.append(("sam3-up512-sharp", unsharp(upscale_lanczos(crop, 512))))
    return out


# -- Main loop -------------------------------------------------------------

def status_for(decoded: Optional[str], gt: Optional[str]) -> str:
    if gt is None:
        return "no_gt"
    if decoded is None or decoded == "":
        return "none"
    return "match" if decoded == gt else "wrong"


def run(args) -> None:
    random.seed(args.seed)
    decoders = available_decoders()
    if not decoders:
        raise SystemExit("No QR decoders available. Install pyzbar: `python -m pip install pyzbar`")
    print(f"Decoders available: {list(decoders)}")

    sessions = discover_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"No sessions found under {config.SESSIONS_DIR}. Did you run pull_dataset?")
    print(f"Sessions: {sessions}")

    sam3 = load_sam3_boxes()
    if sam3:
        print(f"SAM3 annotations loaded: {len(sam3)} records")
    else:
        print("No SAM3 annotations (annotations/all_boxes.jsonl missing). "
              "Running full-frame variants only.")
    variants = [v for v in args.variants if sam3 or v == "full-native"]
    if not variants:
        raise SystemExit("No variants selected. With no SAM3 annotations, pass --variants full-native.")
    print(f"Variants: {variants}")

    print("Decoding reference photos for ground truth...")
    gt_by_qrid = extract_gt_per_qrid(decoders)

    rows: List[dict] = []
    t0 = time.time()
    for sid in sessions:
        qr_id = frame_qr_id(sid)
        gt = gt_by_qrid.get(qr_id)
        frames = discover_frames(sid)
        if args.sample is not None and args.sample < len(frames):
            frames = random.sample(frames, args.sample)
            frames.sort()
        print(f"\n[{sid}] qr_id={qr_id}  gt={'yes' if gt else 'MISSING'}  "
              f"frames={len(frames)}")
        for i, fp in enumerate(frames):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            blur = laplacian_variance(rgb)
            rel_name = f"frames/{sid}/{fp.name}"
            rec = sam3.get(rel_name)
            box = pick_square_box(rec["boxes"]) if rec else None
            qr_side = None
            if box is not None:
                qr_side = float(max(box[2] - box[0], box[3] - box[1]))

            m = FRAME_RE.match(fp.name)
            frame_idx = int(m.group("idx")) if m else -1

            for v_name, v_img in build_variants(rgb, box, variants):
                for d_name, d_fn in decoders.items():
                    t1 = time.time()
                    try:
                        decoded = d_fn(v_img)
                    except Exception:
                        decoded = None
                    dt = time.time() - t1
                    rows.append({
                        "session_id": sid,
                        "qr_id": qr_id,
                        "frame_idx": frame_idx,
                        "variant": v_name,
                        "decoder": d_name,
                        "status": status_for(decoded, gt),
                        "decode_time_s": round(dt, 4),
                        "blur_score": round(blur, 2),
                        "qr_side_px": round(qr_side, 1) if qr_side else "",
                        "decoded_prefix": (decoded or "")[:40],
                    })
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(frames)}  (elapsed {time.time() - t0:.1f}s)")

    out_csv = config.PHASE0_DIR / "decode_baseline.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    write_summary(rows, config.PHASE0_DIR / "decode_baseline_summary.md")
    try:
        plot_summary(rows, config.PHASE0_DIR / "decode_baseline.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# -- Reporting -------------------------------------------------------------

def write_summary(rows: List[dict], out_path: Path) -> None:
    """Grouped decode-rate table + blur-tertile breakdown, in Markdown."""
    import collections

    def _rate(subset):
        n = len(subset)
        if n == 0:
            return (0, 0, "-")
        m = sum(r["status"] == "match" for r in subset)
        return (m, n, f"{100 * m / n:.0f}%")

    # Frame-level blur tertile labels (per-session blur distribution).
    # Each (session, frame_idx) appears in rows across many variants/decoders;
    # we tag the rows with a shared tertile so the same frame lands in the
    # same bucket regardless of which variant produced the row.
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

    variants = sorted({r["variant"] for r in rows})
    decoders = sorted({r["decoder"] for r in rows})
    tertiles = ("blurriest", "medium", "sharpest")

    lines = []
    lines.append("# Phase 0.1 - Decode Baseline")
    lines.append("")
    lines.append(f"Rows: {len(rows)}.  Variants: {', '.join(variants)}.  "
                 f"Decoders: {', '.join(decoders)}.")
    lines.append("")
    lines.append("## Decode rate by (variant, decoder), overall and by blur tertile")
    lines.append("")
    lines.append("| variant | decoder | overall | blurriest | medium | sharpest | wrong-decode | avg time (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in variants:
        for d in decoders:
            subset = [r for r in rows if r["variant"] == v and r["decoder"] == d]
            m, n, overall = _rate(subset)
            cells = [overall]
            for tert in tertiles:
                _, _, c = _rate([r for r in subset if r["blur_tertile"] == tert])
                cells.append(c)
            wrong = sum(r["status"] == "wrong" for r in subset)
            avg_ms = 1000 * sum(r["decode_time_s"] for r in subset) / max(1, n)
            lines.append(f"| {v} | {d} | {cells[0]} ({m}/{n}) | {cells[1]} | {cells[2]} | {cells[3]} | "
                         f"{wrong} | {avg_ms:.0f} |")

    # Any-decoder-wins per variant (OR over decoders): the real ceiling.
    lines.append("")
    lines.append("## Any-decoder match per variant (OR over all decoders)")
    lines.append("")
    lines.append("| variant | overall | blurriest | medium | sharpest |")
    lines.append("|---|---|---|---|---|")
    for v in variants:
        by_frame_match: Dict[tuple, bool] = {}
        tertile_by_frame: Dict[tuple, str] = {}
        for r in rows:
            if r["variant"] != v:
                continue
            key = (r["session_id"], r["frame_idx"])
            by_frame_match[key] = by_frame_match.get(key, False) or (r["status"] == "match")
            tertile_by_frame[key] = r["blur_tertile"]
        total = len(by_frame_match)
        if total == 0:
            continue
        hits = sum(by_frame_match.values())
        per_tert = {t: [0, 0] for t in tertiles}
        for k, matched in by_frame_match.items():
            per_tert[tertile_by_frame[k]][1] += 1
            per_tert[tertile_by_frame[k]][0] += int(matched)
        cells = []
        for t in tertiles:
            hh, nn = per_tert[t]
            cells.append(f"{100 * hh / nn:.0f}% ({hh}/{nn})" if nn else "-")
        lines.append(f"| {v} | {100 * hits / total:.0f}% ({hits}/{total}) | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- `match` = decoded string equals the GT string from the reference photo.")
    lines.append("- `wrong` = decoder returned something, but not the GT. Reed-Solomon false positives.")
    lines.append("- Blur tertiles computed per session (Laplacian variance; low variance = blurrier).")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def plot_summary(rows: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = sorted({(r["variant"], r["decoder"]) for r in rows})
    labels, rates = [], []
    for v, d in pairs:
        sub = [r for r in rows if r["variant"] == v and r["decoder"] == d]
        n = len(sub)
        rate = 100.0 * sum(r["status"] == "match" for r in sub) / max(1, n)
        labels.append(f"{v} / {d}")
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(labels) + 1.5))
    ax.barh(labels, rates)
    ax.set_xlabel("Decode rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Phase 0.1 - Native-resolution decode baseline")
    for i, r in enumerate(rates):
        ax.text(r + 1, i, f"{r:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


# -- CLI -------------------------------------------------------------------

ALL_VARIANTS = ("full-native", "sam3-native", "sam3-up512", "sam3-up512-sharp")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Session ids to include; default = all under data/sessions/")
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=ALL_VARIANTS,
                   help="Which (crop, preproc) variants to run")
    p.add_argument("--sample", type=int, default=None,
                   help="Decode only N random frames per session (quick smoke test)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
