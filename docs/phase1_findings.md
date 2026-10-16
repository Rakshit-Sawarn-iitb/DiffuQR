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

## Smoke-test numbers (n = 100, size = 512)

| | result |
|---|---|
| Clean decode rate | **100%** |
| Degraded decode rate | **22%** |
| Real buggy01 target (Phase 0.1 at 256-crop) | 26% |

Degraded is within noise of real; calibration doesn't need re-tuning
before Phase 2 training. If anything the synth is 4 points harder on
aggregate, which is the right side to err on.

## Known caveat

Per-version decode rate is non-monotonic in the 100-sample smoke:

| version | clean | degraded |
|---|---|---|
| 10 | 100% | 35% |
| 11 | 100% | 0% |
| 12 | 100% | 38% |

v11 at 0% stands out. Likely a segno byte-mode-margin interaction (the
rendered QR fits but is close to the decoder's edge; our mild perspective
warp pushes it over). Doesn't block Phase 2 — module-space training uses
the module grid directly, not the pixel decode. Investigate when it
starts to matter.

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
