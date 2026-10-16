"""phase1_decode_synth.py

Decode-rate sanity check on a synthetic dataset built by
`phase1_build_synth_dataset`. Verifies two things:
  1. Clean renders decode near-100% (if not, the generator is broken).
  2. Degraded renders decode in a sensible range that matches the real
     buggy01 any-decoder baseline (~1% blurriest, 55% sharpest). If
     synthetic decoding is way easier, the degradation is too mild; way
     harder, too harsh.

Writes:
  artifacts/phase1/<tag>_decode.csv            per-sample row
  artifacts/phase1/<tag>_decode_summary.md     per-version breakdown
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from diffuqr import config
from diffuqr.decode import available_decoders


PHASE1_DIR = config.ARTIFACTS_DIR / "phase1"


def decode_rgb(decoders, rgb) -> str | None:
    for fn in decoders.values():
        try:
            s = fn(rgb)
        except Exception:
            s = None
        if s:
            return s
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Synthetic dataset dir (data/synth/<tag>)")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()

    decoders = available_decoders()
    if not decoders:
        raise SystemExit("No QR decoders available.")
    print(f"Decoders: {list(decoders)}")

    d = Path(args.dir)
    manifest = [json.loads(line) for line in open(d / "manifest.jsonl", encoding="utf-8")]
    if args.sample:
        manifest = manifest[: args.sample]
    print(f"Decoding {len(manifest)} synthetic pairs from {d}")

    rows = []
    t0 = time.time()
    for i, m in enumerate(manifest):
        clean = cv2.cvtColor(cv2.imread(str(d / "clean" / f"{m['id']}.png")), cv2.COLOR_BGR2RGB)
        deg = cv2.cvtColor(cv2.imread(str(d / "degraded" / f"{m['id']}.png")), cv2.COLOR_BGR2RGB)
        c = decode_rgb(decoders, clean)
        g = decode_rgb(decoders, deg)
        rows.append({
            "id": m["id"], "version": m["version"], "error": m["error"],
            "payload_len": len(m["payload"]), "px_per_module": m["px_per_module"],
            "modules_per_side": m["modules_per_side"],
            "clean_ok": c == m["payload"], "degraded_ok": g == m["payload"],
            "clean_decoded": "yes" if c else "no", "degraded_decoded": "yes" if g else "no",
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(manifest)}  elapsed {time.time() - t0:.1f}s")

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    tag = d.name
    out_csv = PHASE1_DIR / f"{tag}_decode.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    write_summary(rows, tag)


def write_summary(rows, tag):
    import collections

    n = len(rows)
    clean_m = sum(r["clean_ok"] for r in rows)
    deg_m = sum(r["degraded_ok"] for r in rows)

    lines = [f"# Phase 1 synthetic decode - {tag}",
             "",
             f"- n = {n}",
             f"- clean decode:    {clean_m}/{n} ({100*clean_m/n:.1f}%)",
             f"- degraded decode: {deg_m}/{n} ({100*deg_m/n:.1f}%)",
             "",
             "## By version",
             "",
             "| version | n | clean | degraded |",
             "|---|---|---|---|"]
    by_v = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        by_v[r["version"]][0] += 1
        by_v[r["version"]][1] += int(r["clean_ok"])
        by_v[r["version"]][2] += int(r["degraded_ok"])
    for v in sorted(by_v):
        nn, cc, dd = by_v[v]
        lines.append(f"| {v} | {nn} | {cc}/{nn} ({100*cc/nn:.0f}%) | {dd}/{nn} ({100*dd/nn:.0f}%) |")

    lines += ["",
              "## Reading",
              "- Clean decode must be ~100%; anything lower means the generator renders QRs",
              "  that pyzbar/opencv cannot read (version too high, scale too small, etc.).",
              "- The degraded decode rate's target range comes from Phase 0.1: real buggy01",
              "  at 256-crop was 26% any-decoder (1% blurriest, 46% sharpest). The synthetic",
              "  dataset should land in roughly that aggregate ballpark across versions.",
              "- Big drop with version means px/module is dipping below ~3 for high versions",
              "  at the current rendering size.",
              ""]
    (PHASE1_DIR / f"{tag}_decode_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary -> {PHASE1_DIR / f'{tag}_decode_summary.md'}")


if __name__ == "__main__":
    main()
