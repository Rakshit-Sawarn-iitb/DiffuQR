"""phase2_eval_real.py

Evaluate a trained module-space diffusion checkpoint on one real session
(built by `phase2_build_module_dataset real`), locally, next to the two
reference readers:

  pyzbar      pyzbar / OpenCV on the native camera frame (no geometry help)
  reader      learning-free module reader + soft RS on the SAME rectified input
  model K=k   diffusion sampler, `verifier` / `posterior` decode (diffusion.decode_k)

Reports per blur tertile x visibility (QR fully inside the camera frame or not),
wrong decodes, and the "wait for a good frame" comparison: for each frame of
the blurriest tertile, does ANY frame within +-w seconds decode?

Writes artifacts/phase2/eval_<session>_<run>.{csv,md} (gitignored: the CSV
holds the real payload).

Usage:
  python -m scripts.phase2_eval_real --session buggy02_20261018_01
  python -m scripts.phase2_eval_real --session buggy01_20260917_01 --limit 8 --steps 4   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from diffuqr import config
from diffuqr.diffusion import ModelConfig, ModuleDiffusion, run_eval
from diffuqr.modspace import baseline_read
from diffuqr.qrcodec import decode_soft
from diffuqr.qrgrid import modules_for_version, read_format

MODEL_REPO = "RakshitAi/IITB-BuggyQR-diffusion"
PHASE2_DIR = config.ARTIFACTS_DIR / "phase2"
WINDOWS_S = (0.0, 0.1, 0.25, 0.5, 1.0)


def load_model(ckpt: str, device) -> tuple:
    path = hf_hub_download(MODEL_REPO, ckpt, token=config.HF_TOKEN, local_dir=str(config.CHECKPOINTS_DIR))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ModuleDiffusion(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["ema"])
    return model.to(device).eval(), ck


def native_decodes(fp: Path, payload: str) -> bool:
    """pyzbar or OpenCV on the full camera frame (what an off-the-shelf scanner would do)."""
    bgr = cv2.imread(str(fp))  # frames have no EXIF orientation
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if any(r.data.decode("utf-8", "replace") == payload for r in zbar_decode(gray)):
        return True
    txt, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    return txt == payload


def reader_status(it) -> str:
    n = modules_for_version(it["version"])
    bits, conf = baseline_read(it["input"], n, 8, 2)
    e, m, _ = read_format(bits, it["version"])
    r, _ = decode_soft(bits, conf, it["version"], e, m)
    return "match" if (r.ok and r.payload == it["payload"]) else ("wrong" if r.ok else "none")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--ckpt", default="modspace_diff_v1/last.pt", help="path in the model repo")
    p.add_argument("--k", type=int, nargs="+", default=[1], help="K values (CPU: ~6 s/frame per sample at 20 steps)")
    p.add_argument("--steps", type=int, default=20, help="DDIM steps")
    p.add_argument("--limit", type=int, default=None, help="only the first N aligned frames (smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = np.load(config.DATA_DIR / "modspace" / f"real_{args.session}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    inputs, aligned, fidx, tert = d["inputs"], d["aligned"], d["frame_idx"], d["tertile"]
    vis = d["visible_frac"] if "visible_frac" in d.files else np.full(len(fidx), np.nan)
    items = [{"input": inputs[i], "grid": d["target"], "version": meta["version"], "payload": meta["payload"],
              "split": args.session, "blur": str(tert[i]), "frame_idx": int(fidx[i]), "visible_frac": float(vis[i])}
             for i in np.flatnonzero(aligned)][: args.limit]
    print(f"[{args.session}] v{meta['version']} EC-{meta['error']}: {len(fidx)} frames with a SAM3 box, "
          f"{int(aligned.sum())} aligned, evaluating {len(items)}")

    # Native decoder on EVERY saved frame (also those without a box / alignment): the realistic fast path.
    frames = sorted((config.FRAMES_DIR / args.session).glob(f"{args.session}_f*.jpg"))
    if args.limit:
        keep = {it["frame_idx"] for it in items}
        frames = [f for f in frames if int(f.stem.rsplit("_f", 1)[1]) in keep]
    native = {int(f.stem.rsplit("_f", 1)[1]): native_decodes(f, meta["payload"]) for f in frames}
    print(f"  native pyzbar/OpenCV: {sum(native.values())}/{len(native)} saved frames decode")

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    print(f"  model {args.ckpt} step {ck['step']} on {device}; K={args.k}, {args.steps} DDIM steps")
    t0 = time.time()
    rows = run_eval(model, items, args.k, args.steps, device, mode=ck.get("mode", "diffusion"))
    print(f"  sampled {len(items)} frames in {(time.time() - t0) / 60:.1f} min")
    res = pd.DataFrame(rows)
    res["reader"] = [reader_status(it) for it in items]
    res["pyzbar"] = np.where(res.frame_idx.map(native), "match", "none")
    res["visibility"] = np.where(res.visible_frac.isna(), "unknown",
                                 np.where(res.visible_frac >= 0.99, "fully in frame", "cut off"))

    run = Path(args.ckpt).parent.name or "model"
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    out = PHASE2_DIR / f"eval_{args.session}_{run}"
    res.to_csv(out.with_suffix(".csv"), index=False)

    cols = ["pyzbar", "reader"] + [f"{m}_k{k}" for k in args.k for m in ("verifier", "posterior")]
    pct = lambda s: f"{100 * (s == 'match').mean():.0f}% ({int((s == 'match').sum())}/{len(s)})"
    L = [f"# Real-session eval: `{args.session}` with `{args.ckpt}` (step {ck['step']})", "",
         f"QR v{meta['version']} EC-{meta['error']}; {len(items)} aligned frames; oracle geometry for reader and model;",
         f"pyzbar runs on the native frame. K={args.k}, {args.steps} DDIM steps, device {device}.", "",
         "| blur | visibility | n | median module err (K=1) | " + " | ".join(cols) + " |",
         "|---|---|---|---|" + "---|" * len(cols)]
    groups = [("all", "all", res)]
    for t in ("blurriest", "medium", "sharpest"):
        s = res[res.blur == t]
        groups += [(t, "all", s)] + [(t, v, s[s.visibility == v]) for v in ("fully in frame", "cut off")]
    for t, v, s in groups:
        if len(s):
            L.append(f"| {t} | {v} | {len(s)} | {s.module_err_k1.median():.4f} | " + " | ".join(pct(s[c]) for c in cols) + " |")
    wrong = {c: int((res[c] == "wrong").sum()) for c in cols}
    L += ["", f"Wrong decodes: {wrong}", "",
          f"Sampling: {res.ms_per_item_allK.median():.0f} ms/frame for K={max(args.k)} on {device}", ""]

    # "Wait for a good frame": success if any saved frame within +-w s decodes.
    fps = json.loads((config.SESSIONS_DIR / f"{args.session}.json").read_text(encoding="utf-8"))["video"]["fps"]
    idx = np.array(sorted(native))
    z = np.array([native[f] for f in idx])
    model_ok = dict(zip(res.frame_idx, res[f"verifier_k{args.k[0]}"] == "match"))
    reader_ok = dict(zip(res.frame_idx, res.reader == "match"))
    m = np.array([model_ok.get(f, False) for f in idx])
    b = np.array([reader_ok.get(f, False) for f in idx])
    L += [f"## Wait for a good frame (targets: blurriest tertile; model = verifier K={args.k[0]})", "",
          "| targets | window | pyzbar | pyzbar + reader | pyzbar + model |", "|---|---|---|---|---|"]
    tg = res[res.blur == "blurriest"]
    for name, T in (("all blurriest", tg), ("blurriest, fully in frame", tg[tg.visibility == "fully in frame"])):
        if not len(T):
            continue
        for w in WINDOWS_S:
            hit = lambda dec: 100 * np.mean([dec[np.abs(idx - f) <= w * fps].any() for f in T.frame_idx])
            L.append(f"| {name} ({len(T)}) | +-{w:.2f} s | {hit(z):.0f}% | {hit(z | b):.0f}% | {hit(z | m):.0f}% |")

    def longest_gap(dec):
        e = np.r_[idx[0], idx[dec], idx[-1]]
        return np.diff(e).max() / fps
    L += ["", f"Longest stretch with no decode: pyzbar {longest_gap(z):.2f} s, pyzbar + reader "
              f"{longest_gap(z | b):.2f} s, pyzbar + model {longest_gap(z | m):.2f} s", ""]
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {out}.csv / .md")


if __name__ == "__main__":
    main()
