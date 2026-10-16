# Phase 1 Findings - Data Engine

Written 2026-10-15. Smoke-tested 100 pairs; full details in
`artifacts/phase1/smoke_decode_summary.md`.

## What landed

- `diffuqr/synth.py` - procedural QR generator (segno-backed), random
  payloads across EMV-like / URL / text, version 10-18, EC L or M. Returns
  both an RGB image and the clean module grid for Phase 2 module-space
  targets.
- `diffuqr/degrade.py` - calibrated degradation. Motion blur sampled from
  the Phase 0.3 kernel bank (328 real-buggy kernels, bilinearly rescaled
  to the generator's rendering size). Plus perspective, defocus,
  illumination + vignette, JPEG, sensor noise. All seedable.
- `scripts/phase1_build_synth_dataset.py` - batch generator; writes PNG
  pairs + `manifest.jsonl` under `data/synth/<tag>/`.
- `scripts/phase1_decode_synth.py` - decode-rate sanity check.

## Smoke-test numbers (n = 100, size = 512, seed 0)

Re-run 2026-10-16 after the quiet-zone fix below.

| | result |
|---|---|
| Clean decode rate | **100%** |
| Degraded decode rate | **31%** (was 22% before the fix) |
| Real buggy01 (Phase 0.1 native any-decoder / Phase 5a 256-crop) | 28% / 26% |

Aggregate degraded decode is in the real-data ballpark. n=100 is too
small to call it "calibrated" beyond that; per-version cells have n=3-37.

## Fixed: v11 = 0% was a generator bug (quiet zone)

The original 22% run showed v11 at 0% degraded. Cause: `synth._render`
sized modules as `img_size // n` and centred the leftover, so the white
border varied by version (v11: 1.5 modules, v14: 0, v13: 2) while the
perspective warp moves corners up to ~31 px. Codes were pushed off-canvas.
Perspective-only test (no blur), 120 random codes: 66/120 decoded before,
119/120 after. `_render` now always reserves a 4-module quiet zone (spec).
Side effect: px/module drops by ~1 for some versions (v11 at 512: 8 -> 7).

| version | degraded before | degraded after |
|---|---|---|
| 10 | 35% | 32% |
| 11 | **0%** | **35%** |
| 12 | 38% | 44% |
| 13 | 11% | 11% |

## Immediate uses

1. Build a training set once more QR videos are added:
   ```
   python -m scripts.phase1_build_synth_dataset --n 10000 --size 512
   python -m scripts.phase1_decode_synth --dir data/synth/n10000_size512_v10-18
   ```
2. Phase 2 training can import `random_qr()` directly for on-the-fly
   generation (no pre-saved files; faster for Kaggle).

## Artifacts

```
data/synth/smoke/
    clean/*.png                       # 100 renders
    degraded/*.png                    # 100 calibrated-degraded renders
    manifest.jsonl                    # payload, version, EC per sample
artifacts/phase1/
    smoke_decode.csv
    smoke_decode_summary.md
```
