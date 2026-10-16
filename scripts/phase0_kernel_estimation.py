"""phase0_kernel_estimation.py

Fit a small motion-blur PSF to each frame by solving `frame ~= k * reference`
in the Fourier domain (Tikhonov-regularised division) after aligning the
reference onto the frame via ECC. Produces an empirical kernel bank the
Phase 1 degradation simulator will sample from, and the length/angle
distributions that bound what Wiener/Richardson-Lucy baselines can do.

Outputs (artifacts/phase0/ and data/calibration/):
  data/calibration/kernels.npz             (N, k, k) float32 kernel bank + metadata
  artifacts/phase0/kernels_summary.md      per-kernel length/angle percentiles
  artifacts/phase0/kernels.csv             per-frame kernel stats
  artifacts/phase0/kernels_grid.png        tiled visualisation of sampled kernels
  artifacts/phase0/kernels_distribution.png length + angle histograms

Usage:
  python -m scripts.phase0_kernel_estimation --crop-size 256 --kernel-size 25
  python -m scripts.phase0_kernel_estimation --sample 40    # smoke test
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
from diffuqr.align import align_reference_to_frame
from diffuqr.image import crop_square, laplacian_variance, pil_to_rgb_array
from diffuqr.kernel import clean_kernel, estimate_kernel_fft, kernel_grid_image, kernel_moments


FRAME_RE = re.compile(r"^(?P<session>[a-z0-9]+_\d{8}_\d{2})_f(?P<idx>\d+)\.jpg$")


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


def session_qr_id(session_id: str) -> str:
    return session_id.split("_", 1)[0]


def crop_to_size(pil_img: Image.Image, box: List[float], size: int, pad_frac: float = 0.15) -> np.ndarray:
    """SAM3 square crop -> INTER_AREA downsample to `size` -> RGB uint8."""
    sq = crop_square(pil_img, box, pad_frac=pad_frac)
    bgr = cv2.cvtColor(np.asarray(sq), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run(args) -> None:
    sam3 = load_sam3_boxes()
    if not sam3:
        raise SystemExit("No SAM3 annotations. Run `python -m scripts.pull_dataset --skip-videos` first.")

    sessions = sorted({p.stem for p in config.SESSIONS_DIR.glob("*.json")
                       if (config.FRAMES_DIR / p.stem).is_dir()})
    if args.sessions:
        sessions = [s for s in sessions if s in args.sessions]
    print(f"Sessions: {sessions}")

    # Pre-load one reference crop per qr_id (box from the reference's own annotation).
    ref_crops: Dict[str, np.ndarray] = {}
    for sid in sessions:
        qr_id = session_qr_id(sid)
        if qr_id in ref_crops:
            continue
        ref_rec = sam3.get(f"raw/reference_photos/{qr_id}.jpg")
        if ref_rec is None:
            print(f"  [{qr_id}] no reference annotation, skipping")
            continue
        ref_box = pick_square_box(ref_rec["boxes"])
        if ref_box is None:
            print(f"  [{qr_id}] no usable reference box, skipping")
            continue
        ref_pil = Image.open(config.RAW_DIR / "reference_photos" / f"{qr_id}.jpg").convert("RGB")
        ref_rgb = crop_to_size(ref_pil, ref_box, args.crop_size)
        ref_crops[qr_id] = ref_rgb
        print(f"  [{qr_id}] reference crop prepared at {args.crop_size} px")

    rows: List[dict] = []
    kept_kernels: List[np.ndarray] = []
    kept_raw: List[np.ndarray] = []
    kept_meta: List[dict] = []
    t_start = time.time()
    n_total = n_aligned = n_kept = 0

    for sid in sessions:
        qr_id = session_qr_id(sid)
        if qr_id not in ref_crops:
            continue
        ref_rgb = ref_crops[qr_id]

        frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
        if args.sample is not None and args.sample < len(frames):
            import random
            random.seed(args.seed)
            frames = sorted(random.sample(frames, args.sample))
        print(f"\n[{sid}] processing {len(frames)} frames...")

        for i, fp in enumerate(frames):
            rel = f"frames/{sid}/{fp.name}"
            rec = sam3.get(rel)
            if rec is None:
                continue
            box = pick_square_box(rec["boxes"])
            if box is None:
                continue
            n_total += 1

            try:
                frame_pil = Image.open(fp).convert("RGB")
            except Exception:
                continue
            frame_rgb = crop_to_size(frame_pil, box, args.crop_size)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

            warped_bgr, score = align_reference_to_frame(frame_bgr, ref_bgr, ncc_thresh=args.ncc_thresh)
            if warped_bgr is None:
                continue
            n_aligned += 1

            frame_g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            ref_warp_g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            k_raw = estimate_kernel_fft(frame_g, ref_warp_g, kernel_size=args.kernel_size, eps=args.eps)
            k = clean_kernel(k_raw)  # main lobe only; the diffuse floor is illumination mismatch, not blur
            stats = kernel_moments(k_raw)
            # Residual with a global gain/offset (illumination) fitted, as the
            # degradation model applies illumination separately from blur.
            pred = cv2.filter2D(ref_warp_g, -1, k).ravel()
            A = np.stack([pred, np.ones_like(pred)], 1)
            coef, *_ = np.linalg.lstsq(A, frame_g.ravel(), rcond=None)
            resid = float(np.mean((frame_g.ravel() - A @ coef) ** 2))
            blur = laplacian_variance(frame_rgb)

            m = FRAME_RE.match(fp.name)
            frame_idx = int(m.group("idx")) if m else -1
            rows.append({
                "session_id": sid, "qr_id": qr_id, "frame_idx": frame_idx,
                "align_ncc": round(score, 3), "resid_mse": round(resid, 6),
                "blur_score": round(blur, 2),
                "k_length": round(stats["length"], 3),
                "k_thickness": round(stats["thickness"], 3),
                "k_aspect": round(stats["aspect"], 3),
                "k_angle_deg": round(stats["angle_deg"], 2),
            })
            kept_kernels.append(k.astype(np.float32))
            kept_raw.append(k_raw.astype(np.float32))
            kept_meta.append({
                "session_id": sid, "qr_id": qr_id, "frame_idx": frame_idx,
                "align_ncc": float(score), "k_length": stats["length"], "k_angle_deg": stats["angle_deg"],
            })
            n_kept += 1

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(frames)}  aligned={n_aligned} kept={n_kept}  "
                      f"elapsed {time.time() - t_start:.1f}s")

    if n_kept == 0:
        raise SystemExit("No kernels fit (no frames aligned above NCC threshold).")

    kernels_np = np.stack(kept_kernels, axis=0)
    npz_path = config.CALIBRATION_DIR / "kernels.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # `kernels` = cleaned main-lobe PSFs (what every consumer should use);
    # `kernels_raw` = raw Fourier-division estimates, kept for reference.
    np.savez_compressed(npz_path, kernels=kernels_np, kernels_raw=np.stack(kept_raw, axis=0),
                        meta=np.array([json.dumps(m) for m in kept_meta], dtype=object))
    print(f"\nSaved {n_kept} kernels -> {npz_path}  (shape {kernels_np.shape})")

    out_csv = config.PHASE0_DIR / "kernels.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    write_summary(rows, n_total, n_aligned, config.PHASE0_DIR / "kernels_summary.md")
    try:
        plot_distributions(rows, config.PHASE0_DIR / "kernels_distribution.png")
    except Exception as e:
        print(f"  (distribution plot skipped: {e})")
    try:
        n_sample = min(64, len(kept_kernels))
        import random
        random.seed(args.seed)
        idx = sorted(random.sample(range(len(kept_kernels)), n_sample))
        grid = kernel_grid_image(np.stack([kept_kernels[i] for i in idx], axis=0), cols=8, upscale=5)
        out_grid = config.PHASE0_DIR / "kernels_grid.png"
        cv2.imwrite(str(out_grid), grid)
        print(f"Wrote kernel grid -> {out_grid}")
    except Exception as e:
        print(f"  (grid plot skipped: {e})")


# -- Reporting -------------------------------------------------------------

def write_summary(rows: List[dict], n_total: int, n_aligned: int, out_path: Path) -> None:
    lengths = np.array([r["k_length"] for r in rows])
    thick = np.array([r["k_thickness"] for r in rows])
    aspect = np.array([r["k_aspect"] for r in rows])
    angle = np.array([r["k_angle_deg"] for r in rows])
    resid = np.array([r["resid_mse"] for r in rows])
    blur = np.array([r["blur_score"] for r in rows])

    order = np.argsort(blur)
    n = len(blur)
    tert = np.empty(n, dtype=object)
    tert[order[: n // 3]] = "blurriest"
    tert[order[n // 3 : 2 * n // 3]] = "medium"
    tert[order[2 * n // 3 :]] = "sharpest"

    def _p(a, pcts=(5, 25, 50, 75, 95)):
        return {p: float(np.percentile(a, p)) for p in pcts}

    L = _p(lengths)
    T = _p(thick)
    A = _p(angle)
    R = _p(resid)

    lines = []
    lines.append("# Phase 0.3 - Blur Kernel Estimation")
    lines.append("")
    lines.append(f"Frames seen: {n_total}.  Aligned (NCC >= 0.5): {n_aligned}.  Kernels kept: {n}.")
    lines.append("")
    lines.append("All kernels are 25 x 25 px fit at 256 px crop size (configurable).")
    lines.append("`length`/`thickness` are 2 * sqrt(eigenvalue) of the kernel's second-moment ellipse.")
    lines.append("`angle_deg` is the orientation of the major axis, in [0, 180).")
    lines.append("")
    lines.append("## Kernel size percentiles (pixels, in the 256-px crop)")
    lines.append("")
    lines.append("| metric | p5 | p25 | p50 | p75 | p95 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| length    | {L[5]:.2f} | {L[25]:.2f} | {L[50]:.2f} | {L[75]:.2f} | {L[95]:.2f} |")
    lines.append(f"| thickness | {T[5]:.2f} | {T[25]:.2f} | {T[50]:.2f} | {T[75]:.2f} | {T[95]:.2f} |")
    lines.append(f"| angle_deg | {A[5]:.1f} | {A[25]:.1f} | {A[50]:.1f} | {A[75]:.1f} | {A[95]:.1f} |")
    lines.append(f"| resid MSE | {R[5]:.4f} | {R[25]:.4f} | {R[50]:.4f} | {R[75]:.4f} | {R[95]:.4f} |")
    lines.append("")

    lines.append("## Stratified by blur tertile (Laplacian variance)")
    lines.append("")
    lines.append("| tertile | n | median length | median thickness | median aspect | median resid MSE |")
    lines.append("|---|---|---|---|---|---|")
    for t in ("blurriest", "medium", "sharpest"):
        mask = tert == t
        if not mask.any():
            continue
        lines.append(
            f"| {t} | {int(mask.sum())} | {np.median(lengths[mask]):.2f} | "
            f"{np.median(thick[mask]):.2f} | {np.median(aspect[mask]):.2f} | "
            f"{np.median(resid[mask]):.4f} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("- A short, isotropic kernel means motion blur is mild; Wiener/RL deconvolution")
    lines.append("  with the estimated kernel will close much of the gap.")
    lines.append("- A long, directional kernel (high length, high aspect) means heavy camera motion;")
    lines.append("  Wiener struggles, learned restoration earns its place.")
    lines.append("- Resid MSE after convolving the aligned reference by k is a kernel-quality floor:")
    lines.append("  if resid is large, either the model is misspecified (not just linear blur) or the")
    lines.append("  alignment was poor despite passing NCC. Treat the top-decile-resid kernels with care.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def plot_distributions(rows: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = np.array([r["k_length"] for r in rows])
    angle = np.array([r["k_angle_deg"] for r in rows])
    aspect = np.array([r["k_aspect"] for r in rows])
    blur = np.array([r["blur_score"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.hist(lengths, bins=30, color="#4C72B0")
    ax.set_xlabel("Kernel length (px, 25-tap PSF in 256-crop)")
    ax.set_ylabel("Frames")
    ax.set_title("Motion-blur length distribution")

    ax = axes[0, 1]
    ax.hist(angle, bins=36, color="#55A868")
    ax.set_xlabel("Kernel angle (deg, [0, 180))")
    ax.set_ylabel("Frames")
    ax.set_title("Motion direction distribution")

    ax = axes[1, 0]
    ax.hist(np.log10(aspect + 1e-6), bins=30, color="#C44E52")
    ax.set_xlabel("log10(aspect ratio)")
    ax.set_ylabel("Frames")
    ax.set_title("Kernel anisotropy (higher = more directional)")

    ax = axes[1, 1]
    ax.scatter(blur, lengths, alpha=0.4, s=10, color="#8172B2")
    ax.set_xlabel("Laplacian variance (blur score, lower = blurrier)")
    ax.set_ylabel("Kernel length (px)")
    ax.set_title("Does the blur metric agree with kernel length?")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote distribution plot -> {out_path}")


# -- CLI -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--sample", type=int, default=None, help="N random frames per session (smoke test)")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--kernel-size", type=int, default=25)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--ncc-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
