"""phase0_pixel_module_stats.py

For every frame with a SAM3 box, measure the QR side in pixels and
report pixels-per-module under each plausible QR version.

Why: the information-theoretic ceiling of any single-frame restorer is set
by px/module. Below ~2 px/module the module centres are undersampled; by
~3-4 px/module a well-trained decoder recovers almost everything. This
measurement tells us which blur regime is even solvable in principle.

Outputs (artifacts/phase0/):
  pixel_module_stats.csv            one row per frame
  pixel_module_stats_summary.md     candidate-version table + stratification
  pixel_module_stats.png            histogram of px/module (EC-L default)

Usage:
  python -m scripts.phase0_pixel_module_stats
  python -m scripts.phase0_pixel_module_stats --sessions buggy01_20260917_01
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from diffuqr import config
from diffuqr.decode import available_decoders
from diffuqr.image import laplacian_variance


FRAME_RE = re.compile(r"^(?P<session>[a-z0-9]+_\d{8}_\d{2})_f(?P<idx>\d+)\.jpg$")

# UPI payment QRs are typically EC-L for payload-size efficiency (version 11
# in our case). The other EC levels are reported for completeness so the
# measurement doesn't quietly assume one.
CANDIDATE_VERSIONS = {
    "EC-L (likely for UPI)": 11,
    "EC-M": 12,
    "EC-Q": 15,
    "EC-H": 18,
}
MODULES_PER_SIDE = {v: 17 + 4 * v for v in CANDIDATE_VERSIONS.values()}


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


def load_sam3_boxes() -> Dict[str, dict]:
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
    ok = []
    for b in boxes:
        x1, y1, x2, y2 = b["box_xyxy"]
        w, h = x2 - x1, y2 - y1
        if w > 8 and h > 8 and 0.5 <= w / h <= 2.0:
            ok.append(b)
    if not ok:
        return None
    return max(ok, key=lambda b: b["score"])["box_xyxy"]


# -- Reference measurement -------------------------------------------------

def measure_reference_px_per_module(qr_id: str) -> Optional[float]:
    """Px-per-module of the clean reference photo under version=11.
    This is the ideal upper bound: any real frame has fewer px/module."""
    sam3 = load_sam3_boxes()
    ref_name = f"raw/reference_photos/{qr_id}.jpg"
    rec = sam3.get(ref_name)
    if rec is None:
        return None
    box = pick_square_box(rec["boxes"])
    if box is None:
        return None
    side = max(box[2] - box[0], box[3] - box[1])
    return side / MODULES_PER_SIDE[11]


# -- Main ------------------------------------------------------------------

def run(args) -> None:
    sam3 = load_sam3_boxes()
    if not sam3:
        raise SystemExit("No SAM3 annotations. Run `python -m scripts.pull_dataset --skip-videos` first.")

    sessions = discover_sessions(args.sessions)
    print(f"Sessions: {sessions}")

    rows: List[dict] = []
    for sid in sessions:
        qr_id = sid.split("_", 1)[0]
        ref_pxm = measure_reference_px_per_module(qr_id)
        if ref_pxm is not None:
            print(f"  [{qr_id}] reference px/module (ver=11) = {ref_pxm:.1f}")
        frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
        if args.sample is not None and args.sample < len(frames):
            import random
            random.seed(args.seed)
            frames = sorted(random.sample(frames, args.sample))
        for fp in frames:
            rel = f"frames/{sid}/{fp.name}"
            rec = sam3.get(rel)
            if rec is None:
                continue
            box = pick_square_box(rec["boxes"])
            if box is None:
                continue
            side = float(max(box[2] - box[0], box[3] - box[1]))
            m = FRAME_RE.match(fp.name)
            frame_idx = int(m.group("idx")) if m else -1

            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            blur = laplacian_variance(rgb)

            row = {
                "session_id": sid, "qr_id": qr_id, "frame_idx": frame_idx,
                "frame_width": rec["image_width"], "frame_height": rec["image_height"],
                "qr_side_px": round(side, 1),
                "blur_score": round(blur, 2),
            }
            for name, ver in CANDIDATE_VERSIONS.items():
                row[f"px_per_module_v{ver}"] = round(side / MODULES_PER_SIDE[ver], 2)
            rows.append(row)

    out_csv = config.PHASE0_DIR / "pixel_module_stats.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    write_summary(rows, config.PHASE0_DIR / "pixel_module_stats_summary.md")
    try:
        plot_distribution(rows, config.PHASE0_DIR / "pixel_module_stats.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# -- Reporting -------------------------------------------------------------

def write_summary(rows: List[dict], out_path: Path) -> None:
    sides = np.array([r["qr_side_px"] for r in rows])
    blurs = np.array([r["blur_score"] for r in rows])

    # Blur tertiles for stratification (same convention as decode baseline).
    order = np.argsort(blurs)
    n = len(blurs)
    tert = np.empty(n, dtype=object)
    tert[order[: n // 3]] = "blurriest"
    tert[order[n // 3 : 2 * n // 3]] = "medium"
    tert[order[2 * n // 3 :]] = "sharpest"

    def _pct(a, pcts=(5, 25, 50, 75, 95)):
        return {p: float(np.percentile(a, p)) for p in pcts}

    lines = []
    lines.append("# Phase 0.2 - Pixel/Module Statistics")
    lines.append("")
    lines.append(f"Frames with a SAM3 box: {len(rows)}.")
    lines.append("")
    lines.append("## QR side in pixels (SAM3 box longer side)")
    lines.append("")
    p = _pct(sides)
    lines.append(f"- Percentiles (5/25/50/75/95): {p[5]:.0f} / {p[25]:.0f} / {p[50]:.0f} / {p[75]:.0f} / {p[95]:.0f} px")
    lines.append(f"- Min / mean / max: {sides.min():.0f} / {sides.mean():.1f} / {sides.max():.0f} px")
    lines.append("")

    lines.append("## Pixels per module, per candidate version")
    lines.append("")
    lines.append("| EC / version | modules/side | median px/mod | p25 | p75 | % >= 2 px/mod | % >= 3 px/mod | % >= 4 px/mod |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, ver in CANDIDATE_VERSIONS.items():
        pxm = sides / MODULES_PER_SIDE[ver]
        lines.append(
            f"| {name} (v{ver}) | {MODULES_PER_SIDE[ver]} | {np.median(pxm):.2f} | "
            f"{np.percentile(pxm, 25):.2f} | {np.percentile(pxm, 75):.2f} | "
            f"{100*(pxm>=2).mean():.0f}% | {100*(pxm>=3).mean():.0f}% | {100*(pxm>=4).mean():.0f}% |"
        )
    lines.append("")

    lines.append("## Px/module (version 11) stratified by blur tertile")
    lines.append("")
    pxm11 = sides / MODULES_PER_SIDE[11]
    lines.append("| blur tertile | n | median px/mod | p25 | p75 | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in ("blurriest", "medium", "sharpest"):
        mask = tert == t
        v = pxm11[mask]
        if len(v) == 0:
            continue
        lines.append(
            f"| {t} | {len(v)} | {np.median(v):.2f} | {np.percentile(v,25):.2f} | "
            f"{np.percentile(v,75):.2f} | {v.min():.2f} | {v.max():.2f} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("- The industry rule of thumb: a decoder needs >= 2 px/module to even")
    lines.append("  attempt bit-reading, and ~3-4 px/module for reliable Reed-Solomon decode.")
    lines.append("- Any frame below ~2 px/module cannot be decoded from a single frame,")
    lines.append("  no matter how good the restorer. Multi-frame fusion is the only option.")
    lines.append("- The reference photo (ideal upper bound) is reported in the terminal.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def plot_distribution(rows: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sides = np.array([r["qr_side_px"] for r in rows])
    pxm_by_ver = {v: sides / MODULES_PER_SIDE[v] for v in CANDIDATE_VERSIONS.values()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(sides, bins=30, color="#4C72B0", alpha=0.85)
    ax1.set_xlabel("QR side (pixels)")
    ax1.set_ylabel("Frames")
    ax1.set_title("SAM3 box longer-side distribution")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for (name, ver), c in zip(CANDIDATE_VERSIONS.items(), colors):
        ax2.hist(pxm_by_ver[ver], bins=30, alpha=0.5, label=f"{name} (v{ver})", color=c)
    for thr in (2.0, 3.0, 4.0):
        ax2.axvline(thr, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.text(thr, ax2.get_ylim()[1] * 0.9, f"{thr}", ha="center", fontsize=8)
    ax2.set_xlabel("Pixels per module")
    ax2.set_ylabel("Frames")
    ax2.set_title("Pixels per module (per candidate QR version)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


# -- CLI -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
