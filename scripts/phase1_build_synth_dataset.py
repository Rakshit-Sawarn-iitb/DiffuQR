"""phase1_build_synth_dataset.py

Build a procedural (clean, degraded) QR dataset calibrated on real buggy
motion kernels. Phase 2 training will consume this directly.

Each sample is one directory entry under:
  data/synth/<tag>/
      clean/<id>.png          PNG of the rendered QR
      degraded/<id>.png       PNG of the degraded version
      manifest.jsonl          one JSON line per sample

Usage:
  python -m scripts.phase1_build_synth_dataset --n 1000
  python -m scripts.phase1_build_synth_dataset --n 100 --size 256 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from diffuqr import config
from diffuqr.degrade import calibrated_degrade, load_kernel_bank
from diffuqr.synth import random_qr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--min-version", type=int, default=10)
    p.add_argument("--max-version", type=int, default=18)
    p.add_argument("--errors", nargs="+", default=["L", "M"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None,
                   help="Output subdirectory name under data/synth/. Default: n<N>_size<S>_v<MIN>-<MAX>.")
    p.add_argument("--kernel-bank", default=None,
                   help="Path to kernels.npz; defaults to data/calibration/kernels.npz.")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tag = args.tag or f"n{args.n}_size{args.size}_v{args.min_version}-{args.max_version}"
    out_dir = config.SYNTH_DIR / tag
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    (out_dir / "degraded").mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}")

    kb_path = Path(args.kernel_bank) if args.kernel_bank else (config.CALIBRATION_DIR / "kernels.npz")
    target_support = max(25, int(round(25 * args.size / 256)))
    if target_support % 2 == 0:
        target_support += 1
    kb = load_kernel_bank(kb_path, target_support=target_support)
    print(f"Loaded kernel bank {kb.shape} (rescaled to {target_support}x{target_support} for size {args.size})")

    manifest_path = out_dir / "manifest.jsonl"
    t0 = time.time()
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i in range(args.n):
            qr = random_qr(
                min_version=args.min_version,
                max_version=args.max_version,
                errors=tuple(args.errors),
                img_size=args.size,
            )
            degraded = calibrated_degrade(qr.image_rgb, kb)
            cv2.imwrite(str(out_dir / "clean" / f"{i:06d}.png"),
                        cv2.cvtColor(qr.image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "degraded" / f"{i:06d}.png"),
                        cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
            mf.write(json.dumps({
                "id": f"{i:06d}",
                "version": qr.version,
                "error": qr.error,
                "payload": qr.payload,
                "px_per_module": qr.px_per_module,
                "modules_per_side": int(qr.modules.shape[0]),
            }) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.n}  elapsed {time.time() - t0:.1f}s")

    print(f"\nWrote {args.n} pairs to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
