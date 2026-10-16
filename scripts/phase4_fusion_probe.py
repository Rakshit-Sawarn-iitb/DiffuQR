"""phase4_fusion_probe.py

Learning-free test of contribution C4 (multi-frame evidence fusion): does
combining per-module evidence from neighbouring frames drive the module
error of blurry frames under the Reed-Solomon cliff?

Per frame (same oracle geometry as phase0_grid_errors): sample module means,
convert to signed evidence z = (local_threshold - mean) / robust_scale
(positive = dark), and cache it. Then, for each target frame and window
half-width w (frames within +-w saved frames, i.e. +-3w video frames,
+-0.05w s at 60 fps):

  any-single   any frame in the window decodes on its own (soft RS).
               The trivial multi-frame baseline: fusion must beat it.
  fused        sum of z over the window, weighted by each frame's
               fixed-pattern agreement (finder/timing/alignment/format
               modules are known once the version is known, so this
               weight needs no payload ground truth). Then soft RS.

Two settings:
  all-neighbours  windows may contain sharper frames (realistic burst).
  blurry-only     targets AND window members restricted to the blurriest
                  tertile: does blurry evidence add up by itself?

ORACLE CAVEATS: geometry from reference alignment; version/EC/mask from
the reference. Frames that fail alignment contribute no evidence.

Outputs:
  artifacts/phase4/fusion_evidence_<session>.npz   cached per-frame evidence
  artifacts/phase4/fusion_probe.csv
  artifacts/phase4/fusion_probe_summary.md

Usage:
  python -m scripts.phase4_fusion_probe
  python -m scripts.phase4_fusion_probe --recompute
"""
from __future__ import annotations

import argparse
import collections
import csv
import time
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from diffuqr import config
from diffuqr.annotations import load_sam3_boxes, pick_square_box
from diffuqr.image import crop_square
from diffuqr.qrcodec import decode_soft
from diffuqr.qrgrid import fixed_template, modules_for_version, rectify, sample_grid
from scripts.phase0_grid_errors import ALIGN_PX, FRAME_RE, PAD, align_two_scale, prepare_reference

PHASE4_DIR = config.ARTIFACTS_DIR / "phase4"
WINDOWS = (0, 1, 2, 3, 5, 8)


def frame_evidence(means: np.ndarray) -> np.ndarray:
    local = cv2.GaussianBlur(means.astype(np.float32), (0, 0), sigmaX=4.0, borderType=cv2.BORDER_REFLECT)
    d = local - means
    scale = float(np.median(np.abs(d))) + 1e-3
    return (d / scale).astype(np.float32)


def compute_evidence(sid: str, R: dict, sam3, ncc_thresh: float) -> dict:
    v, e, mk = R["meta"]["version"], R["meta"]["error"], R["meta"]["mask"]
    n = modules_for_version(v)
    fixed, fvals = fixed_template(v, e, mk)
    idxs, zs, fas, blurs = [], [], [], []
    t0 = time.time()
    frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
    for i, fp in enumerate(frames):
        rec = sam3.get(f"frames/{sid}/{fp.name}")
        box = pick_square_box(rec["boxes"]) if rec else None
        if box is None:
            continue
        rgb_full = np.asarray(Image.open(fp).convert("RGB"))
        blur = float(cv2.Laplacian(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        F = np.asarray(crop_square(Image.fromarray(rgb_full), box, pad_frac=PAD))
        s_f = F.shape[0]
        F512 = cv2.resize(F, (ALIGN_PX, ALIGN_PX), interpolation=cv2.INTER_AREA)
        W, score = align_two_scale(F512, R["ref512"])
        idxs.append(int(FRAME_RE.match(fp.name).group("idx")))
        blurs.append(blur)
        if score < ncc_thresh:
            zs.append(np.zeros((n, n), np.float32))
            fas.append(0.0)
            continue
        q = cv2.perspectiveTransform(R["quad512"].reshape(-1, 1, 2).astype(np.float64),
                                     np.linalg.inv(W).astype(np.float64)).reshape(4, 2) * (s_f / ALIGN_PX)
        _, means = sample_grid(rectify(F, q, n * 12, reorder=False), n)
        z = frame_evidence(np.rot90(means, R["rotation"]))
        zs.append(z)
        fas.append(float(((z > 0).astype(np.uint8)[fixed] == fvals[fixed]).mean()))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(frames)}  elapsed {time.time() - t0:.0f}s")
    return {"frame_idx": np.array(idxs), "z": np.stack(zs), "fixed_agree": np.array(fas), "blur": np.array(blurs)}


def weight_of(fa: float) -> float:
    # 0.5 agreement = chance on binary modules; scale so near-perfect frames dominate.
    return max(0.0, (fa - 0.5) / 0.5) ** 2


def run(args) -> None:
    sam3 = load_sam3_boxes()
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted({p.stem for p in config.SESSIONS_DIR.glob("*.json") if (config.FRAMES_DIR / p.stem).is_dir()})
    rows: List[dict] = []
    for sid in sessions:
        qr_id = sid.split("_", 1)[0]
        R = prepare_reference(qr_id, sam3)
        v, e, mk = R["meta"]["version"], R["meta"]["error"], R["meta"]["mask"]
        payload, gt = R["meta"]["payload"], R["gt"]
        fixed, _ = fixed_template(v, e, mk)
        cache = PHASE4_DIR / f"fusion_evidence_{sid}.npz"
        if cache.exists() and not args.recompute:
            ev = dict(np.load(cache))
            print(f"[{sid}] loaded cached evidence ({len(ev['frame_idx'])} frames)")
        else:
            print(f"[{sid}] computing per-frame evidence...")
            ev = compute_evidence(sid, R, sam3, args.ncc_thresh)
            np.savez_compressed(cache, **ev)

        order = np.argsort(ev["frame_idx"])
        fidx, Z, FA, BL = ev["frame_idx"][order], ev["z"][order], ev["fixed_agree"][order], ev["blur"][order]
        N = len(fidx)
        rank = np.argsort(np.argsort(BL))
        tert = np.where(rank < N / 3, "blurriest", np.where(rank < 2 * N / 3, "medium", "sharpest"))
        step = int(np.median(np.diff(fidx)))

        single_ok = np.zeros(N, bool)
        for i in range(N):
            if FA[i] > 0:
                res, _ = decode_soft((Z[i] > 0).astype(np.uint8), np.abs(Z[i]), v, e, mk)
                single_ok[i] = res.ok and res.payload == payload

        for setting in ("all-neighbours", "blurry-only"):
            for i in range(N):
                if setting == "blurry-only" and tert[i] != "blurriest":
                    continue
                for w in WINDOWS:
                    member = np.abs(fidx - fidx[i]) <= w * step
                    if setting == "blurry-only":
                        member &= tert == "blurriest"
                    member &= FA > 0
                    k = int(member.sum())
                    any_single = bool(single_ok[member].any()) if k else False
                    status, err = "none", ""
                    if k:
                        wts = np.array([weight_of(f) for f in FA[member]], np.float32)
                        zf = np.tensordot(wts if wts.sum() > 0 else np.ones_like(wts), Z[member], axes=1)
                        bits = (zf > 0).astype(np.uint8)
                        res, _ = decode_soft(bits, np.abs(zf), v, e, mk)
                        status = "match" if (res.ok and res.payload == payload) else ("wrong" if res.ok else "none")
                        err = round(float((bits[~fixed] != gt[~fixed]).mean()), 4)
                    rows.append({"session_id": sid, "frame_idx": int(fidx[i]), "tertile": tert[i],
                                 "setting": setting, "w": w, "n_frames_used": k,
                                 "any_single": any_single, "fused": status, "fused_module_err": err})
        print(f"[{sid}] done")

    out_csv = PHASE4_DIR / "fusion_probe.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")
    write_summary(rows)


def write_summary(rows: List[dict]) -> None:
    def pct(sub, pred):
        k = sum(1 for r in sub if pred(r))
        return f"{100 * k / len(sub):.0f}% ({k}/{len(sub)})" if sub else "-"

    L = ["# Phase 4 probe - multi-frame evidence fusion (learning-free)", "",
         "Oracle geometry + oracle version/EC/mask (see script docstring). `w` = window half-width in",
         "saved frames (saved every 3rd video frame; w=1 is +-0.05 s at 60 fps). Decode = soft RS, 4",
         "check symbols reserved per block.", ""]
    for setting, terts in (("all-neighbours", ("blurriest", "medium", "sharpest")), ("blurry-only", ("blurriest",))):
        sub = [r for r in rows if r["setting"] == setting]
        L += [f"## {setting}", "",
              "| w | span (s) | target tertile | n | any single frame decodes | fused decodes | fused-only gain | median fused module err |",
              "|---|---|---|---|---|---|---|---|"]
        for w in WINDOWS:
            for t in terts:
                s = [r for r in sub if r["w"] == w and r["tertile"] == t]
                if not s:
                    continue
                gain = sum(1 for r in s if r["fused"] == "match" and not r["any_single"])
                errs = [r["fused_module_err"] for r in s if r["fused_module_err"] != ""]
                L.append(f"| {w} | {2 * w * 0.05:.2f} | {t} | {len(s)} | {pct(s, lambda r: r['any_single'])} | "
                         f"{pct(s, lambda r: r['fused'] == 'match')} | {gain} | "
                         f"{np.median(errs):.3f} |" if errs else "| - |")
        wrong = sum(r["fused"] == "wrong" for r in sub)
        L += ["", f"Wrong decodes ({setting}): {wrong}.", ""]
    L += ["## Reading",
          "- `fused-only gain` counts targets where fusion decodes but no single frame in the window",
          "  does: the evidence that fusion adds something beyond 'try every frame'.",
          "- `blurry-only` is the decisive C4 test: no sharp frame is available to lean on.",
          "- RS cliff for v11-L: <= 10 wrong codewords per block (~1-2% module error when errors",
          "  are spread), <= 20 with perfect erasures.", ""]
    (PHASE4_DIR / "fusion_probe_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote summary -> {PHASE4_DIR / 'fusion_probe_summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--ncc-thresh", type=float, default=0.5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
