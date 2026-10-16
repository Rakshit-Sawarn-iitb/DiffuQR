"""phase2_build_module_dataset.py

Build module-space data for the Phase 2 restorer (see diffuqr/modspace.py):

  synth   procedural pairs, sharded by version:
            data/modspace/<tag>/synth_v<V>_<shard>.npz
              inputs  (B, S, S) uint8,  S = (N + 2*margin) * px_per_module
              targets (B, N, N) uint8
              meta    (B,) JSON strings (payload, EC, mask, camera_ppm, kernel length)
  real    every real frame in the same format, geometry from the ORACLE
          reference alignment (as Phase 0.4), target = bit-exact GT grid:
            data/modspace/real_<session>.npz
  check   realism check: the learning-free `baseline_read` + soft RS on
          synth vs real, module-error distribution and decode rate:
            artifacts/phase2/calibration_<tag>.md

Usage:
  python -m scripts.phase2_build_module_dataset synth --n 2000 --tag v1
  python -m scripts.phase2_build_module_dataset real
  python -m scripts.phase2_build_module_dataset check --tag v1
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from diffuqr import config
from diffuqr.annotations import load_sam3_boxes, pick_square_box
from diffuqr.image import crop_square
from diffuqr.modspace import KernelBank, ModuleSampleConfig, baseline_read, make_synthetic_sample, rectify_quad
from diffuqr.qrcodec import decode_soft
from diffuqr.qrgrid import modules_for_version
from scripts.phase0_grid_errors import ALIGN_PX, FRAME_RE, PAD, align_two_scale, prepare_reference

MODSPACE_DIR = config.DATA_DIR / "modspace"
PHASE2_DIR = config.ARTIFACTS_DIR / "phase2"


def build_synth(args) -> None:
    cfg = ModuleSampleConfig()
    bank = KernelBank.load(config.CALIBRATION_DIR / "kernels.npz")
    out = MODSPACE_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    buckets = collections.defaultdict(lambda: {"inputs": [], "targets": [], "meta": []})
    shard_no = collections.Counter()

    def flush(v):
        b = buckets[v]
        if not b["inputs"]:
            return
        np.savez_compressed(out / f"synth_v{v}_{shard_no[v]:03d}.npz", inputs=np.stack(b["inputs"]),
                            targets=np.stack(b["targets"]), meta=np.array(b["meta"]))
        shard_no[v] += 1
        buckets[v] = {"inputs": [], "targets": [], "meta": []}

    t0 = time.time()
    for i in range(args.n):
        s = make_synthetic_sample(rng, bank, cfg)
        b = buckets[s["version"]]
        b["inputs"].append(s["input"])
        b["targets"].append(s["target"])
        b["meta"].append(json.dumps({k: s[k] for k in ("version", "error", "mask", "payload",
                                                        "camera_ppm", "kernel_len_modules")}))
        if len(b["inputs"]) >= args.shard_size:
            flush(s["version"])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.n}  elapsed {time.time() - t0:.0f}s")
    for v in list(buckets):
        flush(v)
    print(f"Wrote {args.n} synthetic samples -> {out}")


def build_real(args) -> None:
    cfg = ModuleSampleConfig()
    sam3 = load_sam3_boxes()
    MODSPACE_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted({p.stem for p in config.SESSIONS_DIR.glob("*.json") if (config.FRAMES_DIR / p.stem).is_dir()})
    for sid in sessions:
        R = prepare_reference(sid.split("_", 1)[0], sam3)
        n = modules_for_version(R["meta"]["version"])
        side = (n + 2 * cfg.margin) * cfg.px_per_module
        inputs, idxs, blurs, aligned = [], [], [], []
        frames = sorted((config.FRAMES_DIR / sid).glob(f"{sid}_f*.jpg"))
        t0 = time.time()
        for i, fp in enumerate(frames):
            rec = sam3.get(f"frames/{sid}/{fp.name}")
            box = pick_square_box(rec["boxes"]) if rec else None
            if box is None:
                continue
            full = np.asarray(Image.open(fp).convert("RGB"))
            blurs.append(float(cv2.Laplacian(cv2.cvtColor(full, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
            idxs.append(int(FRAME_RE.match(fp.name).group("idx")))
            F = np.asarray(crop_square(Image.fromarray(full), box, pad_frac=PAD))
            F512 = cv2.resize(F, (ALIGN_PX, ALIGN_PX), interpolation=cv2.INTER_AREA)
            W, score = align_two_scale(F512, R["ref512"])
            if score < args.ncc_thresh:
                inputs.append(np.zeros((side, side), np.uint8))
                aligned.append(False)
                continue
            q = cv2.perspectiveTransform(R["quad512"].reshape(-1, 1, 2).astype(np.float64),
                                         np.linalg.inv(W).astype(np.float64)).reshape(4, 2) * (F.shape[0] / ALIGN_PX)
            gray = cv2.cvtColor(F, cv2.COLOR_RGB2GRAY)
            inp = rectify_quad(gray, q, n, cfg.px_per_module, cfg.margin)
            inputs.append(np.ascontiguousarray(np.rot90(inp, R["rotation"])))
            aligned.append(True)
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(frames)}  elapsed {time.time() - t0:.0f}s")
        blurs = np.array(blurs)
        rank = np.argsort(np.argsort(blurs))
        N = len(blurs)
        tert = np.where(rank < N / 3, "blurriest", np.where(rank < 2 * N / 3, "medium", "sharpest"))
        meta = R["meta"]
        np.savez_compressed(MODSPACE_DIR / f"real_{sid}.npz", inputs=np.stack(inputs), target=R["gt"],
                            frame_idx=np.array(idxs), blur=blurs, tertile=tert, aligned=np.array(aligned),
                            meta=json.dumps({k: meta[k] for k in ("version", "error", "mask", "payload")}))
        print(f"[{sid}] wrote {N} frames ({int(np.sum(aligned))} aligned) -> real_{sid}.npz")


def _eval(inputs, target_of, meta_of, cfg):
    errs, ok = [], []
    for i, inp in enumerate(inputs):
        tgt, m = target_of(i), meta_of(i)
        n = tgt.shape[0]
        bits, conf = baseline_read(inp, n, cfg.px_per_module, cfg.margin)
        errs.append(float((bits != tgt).mean()))
        res, _ = decode_soft(bits, conf, m["version"], m["error"], m["mask"])
        ok.append(bool(res.ok and res.payload == m["payload"]))
    return np.array(errs), np.array(ok)


def check(args) -> None:
    cfg = ModuleSampleConfig()
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    L = [f"# Phase 2 data realism check - `{args.tag}`", "",
         "Same learning-free reader (`modspace.baseline_read`, local threshold) + soft RS on both.",
         "Module error over ALL modules (incl. function patterns). Real = oracle geometry.", "",
         "| data | subset | n | module err p25 | p50 | p75 | p90 | decode rate |", "|---|---|---|---|---|---|---|---|"]

    def row(name, sub, e, o):
        if len(e) == 0:
            return
        q = np.percentile(e, [25, 50, 75, 90])
        L.append(f"| {name} | {sub} | {len(e)} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {q[3]:.3f} | "
                 f"{100 * o.mean():.0f}% ({int(o.sum())}/{len(o)}) |")

    for p in sorted(MODSPACE_DIR.glob("real_*.npz")):
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        keep = d["aligned"]
        e, o = _eval(d["inputs"][keep], lambda i: d["target"], lambda i: meta, cfg)
        tert = d["tertile"][keep]
        row(p.stem, "all aligned", e, o)
        for t in ("blurriest", "medium", "sharpest"):
            row(p.stem, t, e[tert == t], o[tert == t])

    shards = sorted((MODSPACE_DIR / args.tag).glob("synth_v*.npz"))
    E, O, KL = [], [], []
    for p in shards:
        d = np.load(p, allow_pickle=True)
        metas = [json.loads(m) for m in d["meta"]]
        targets = d["targets"]
        e, o = _eval(d["inputs"], lambda i: targets[i], lambda i: metas[i], cfg)
        E.append(e); O.append(o); KL.append([m["kernel_len_modules"] for m in metas])
    if shards:
        E, O, KL = np.concatenate(E), np.concatenate(O), np.concatenate(KL)
        row(f"synth/{args.tag}", "all", E, O)
        cuts = np.percentile(KL, [100 / 3, 200 / 3])
        for name, msk in (("kernel-length longest third", KL >= cuts[1]),
                          ("kernel-length middle third", (KL >= cuts[0]) & (KL < cuts[1])),
                          ("kernel-length shortest third", KL < cuts[0])):
            row(f"synth/{args.tag}", name, E[msk], O[msk])
        L += ["", f"Synthetic kernel length (modules): p50 {np.median(KL):.2f}, p90 {np.percentile(KL, 90):.2f}."]
    L += ["", "## Reading",
          "- Synthetic should span the real distribution, including the blurriest tertile's error",
          "  level (real median ~0.26). Much easier synth -> the model never sees real-hard cases;",
          "  much harder -> wasted capacity, but safer than too easy.",
          "- Synth kernel-length thirds are not the same split as real Laplacian tertiles; compare",
          "  the overall shape, not row by row.", ""]
    out = PHASE2_DIR / f"calibration_{args.tag}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--tag", default="v1")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--shard-size", type=int, default=256)
    r = sub.add_parser("real")
    r.add_argument("--ncc-thresh", type=float, default=0.5)
    c = sub.add_parser("check")
    c.add_argument("--tag", default="v1")
    args = p.parse_args()
    {"synth": build_synth, "real": build_real, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
