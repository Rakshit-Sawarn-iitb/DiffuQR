# Phase 2 Data Engine - Module-space Pairs, 2026-10-16

Code: `diffuqr/modspace.py`, `scripts/phase2_build_module_dataset.py`.
Realism table: `artifacts/phase2/calibration_v1.md`.

## Format (what the restorer sees)

- input: rectified grayscale crop, 8 px/module, 2-module margin
  -> side (N + 4) * 8 (v11: 520 px)
- target: N x N module grid (1 = dark) + version / EC / mask
- synthetic: `data/modspace/<tag>/synth_v<V>_<shard>.npz` (sharded by version)
- real: `data/modspace/real_<session>.npz`, all frames, oracle-geometry
  rectification, bit-exact GT grid, blur tertile, aligned flag

## Generator (calibrated, not guessed)

| component | setting | calibrated against |
|---|---|---|
| camera scale | 7-16 px/module | real frames 7.0-15.9 (Phase 0.2) |
| perspective | corner jitter amp U(0.05, 0.17) | real side ratio 1.18/1.48, angle 7.6/18 deg (Phase 0.4) |
| motion blur | real bank kernels (cleaned lobe), rescaled in MODULE units (bank: 3.65 px/module) | Phase 0.3 |
| hard sampling | 50% of kernels from the longest third | design choice: train on the hard cases |
| corner error | up to 0.4 module per corner before rectification | real pipeline has no reference |
| also | defocus (30%), gain/gamma/gradient illumination, noise, JPEG 40-90 | plausible ranges |

Geometry check: with blur/noise/perspective/corner-error off, the reader
recovers the grid with 0 errors (also with perspective on).

## Realism check (same learning-free reader + soft RS on both)

| subset | median module err | decode |
|---|---|---|
| real blurriest tertile (n=86) | 0.261 | 1% |
| synth longest-kernel third (n=334) | 0.260 | 0% |
| real sharpest tertile (n=115) | 0.001 | 88% |
| synth shortest-kernel third (n=333) | 0.001 | 86% |
| real all aligned (n=318) | 0.032 | 52% |
| synth all (n=1000) | 0.188 | 29% |

The extremes match closely with no tuning beyond the measured kernels.
Synth is harder in aggregate by design (hard-kernel oversampling).
Synth kernel-length thirds and real Laplacian tertiles are different
splits; only the extremes are a like-for-like comparison.

## Kernel-bank correction made on the way

Raw Fourier-division kernels held ~67% of their mass in a diffuse floor
(illumination mismatch absorbed into the PSF). `diffuqr.kernel.clean_kernel`
keeps the main lobe; with a fitted gain/offset it explains real frames 3x
better (resid 0.0057 vs 0.0178). `kernels.npz` now stores cleaned
`kernels` + `kernels_raw`. Phase 5a was re-run (headline unchanged, the
"RL hurts" claim retracted; see phase5a_findings.md). The old pixel-space
Phase 1 smoke (`diffuqr/degrade.py`) now degrades to 57% decodable vs real
28%, i.e. too easy; it is superseded by this module-space generator.

## Cost

~0.31 s/sample on one CPU core (1000 samples ~5 min). A 50k training set
is ~4 h single-core: generate in parallel (multiprocessing) or on Kaggle's
CPUs before training.

## Not yet done

- No model trained. Next: direct CNN baseline (input -> N x N logits),
  then module-space diffusion; both on Kaggle.
- Real eval uses oracle geometry; a reference-free corner source (SAM3
  masks or a corner regressor) is still needed for deployment numbers.
