# Phase 0 Findings

Measurement phase closed 2026-10-15. One session (`buggy01_20260917_01`, 362
frames, 1 physical QR, 60 fps clip). Numbers are honest ground-truth; nothing
here depends on a trained model.

## Headline numbers

| finding | number | source |
|---|---|---|
| Native-resolution decode rate (any decoder) | **28%** (100/362) | `artifacts/phase0/decode_baseline_summary.md` |
| Decode rate, blurriest tertile | **1%** (1/121) | same |
| Decode rate, sharpest tertile | **55%** (66/120) | same |
| Wrong-decode rate across 2866 attempts | **0** | same (Reed-Solomon never fakes a hit) |
| QR size percentiles (p5/p50/p95) | 457 / 616 / 841 px | `pixel_module_stats_summary.md` |
| px/module (version 11, EC-L) median | **10.1** | same |
| buggy01 version / EC / mask (measured, not assumed) | **v11 (61x61), EC-L, mask 2** | `reference_version.md` |
| Frames with >= 4 px/module (any candidate version) | **100%** | same |
| Reference-photo px/module (ideal upper bound, v=11) | 32.1 | same |
| Alignment success (NCC >= 0.5) | 94% (335/357; was 328 before the 2026-10-16 rotation fix) | `kernels_summary.md` |
| Motion-blur kernel length, blurriest tertile (256-crop px) | median 6.2, p95 ~10 | same |
| Motion-blur kernel length, sharpest tertile | median 1.2 (near point) | same |
| Kernel aspect ratio, blurriest tertile | median ~6.5x (strongly directional) | same |
| Kernel angle distribution | ~uniform in [0, 180) deg | same |

## What this changes

### 1. The problem is motion blur, not low resolution.
The handoff estimated 3.5 px/module. That was 73 modules at the **256-px
downsampled training crop**, not reality. At native resolution we are at
**~10 px/module**. Every frame has at least 7 px/module. The industry
threshold is 2-4; we are comfortably above it. This kills the "we lack
resolution" argument. The restorer's only job is deblurring.

### 2. The paper's target subset is the blurriest tertile.
Medium and sharpest tertiles already decode at 27% and 55% respectively.
The method must earn its place on the blurriest third, where classical
decoders deliver 1% and median kernel length is ~6 px (native ~15 px).
Reporting an aggregate decode-rate win is defensible only when it holds up
on this slice.

### 3. The "8% baseline" the training notebook reported is misleading.
`notebooks/training.ipynb` evaluates at 256 px after `INTER_AREA`
downsampling, which handicaps the baseline against itself. The true raw
baseline is 28% (3.5x higher). All future reporting must use native-res
or a defensible fixed-size that preserves >= 4 px/module.

### 4. Deconvolution baselines are now dangerous.
Real motion kernels are short (median 2.5 px, p95 8.6 px in 256-crop;
~6 and ~20 px in native coordinates), highly directional (aspect >5x in
blurry frames), and have known shapes once estimated. Wiener and
Richardson-Lucy with an oracle kernel might close most of the gap on their
own. The paper MUST report:
  - **Blind Wiener** (no kernel info) as a reasonable baseline.
  - **Oracle-kernel Wiener** (kernel estimated from the aligned reference)
    as the per-frame upper bound of any kernel-estimation method.
If the learned restorer only beats blind Wiener but not oracle-kernel,
the real contribution is in kernel estimation + verifier-guided decoding,
not in the diffusion architecture.

### 5. Procedural training with calibrated kernels is now the right move.
We have a 328-kernel empirical bank at `data/calibration/kernels.npz`.
Phase 1 degradation samples kernels from this bank; synthetic training
data becomes realistic by construction, and this is a defensible "real-to-
sim" contribution on its own.

### 6. Version 11 is measured, not just assumed (2026-10-16).
`scripts/phase0_reference_version.py` rectifies the reference QR, samples
the grid and matches the payload-independent modules (function patterns +
format + version info): v11 / EC-L / mask 2 at 99.8% of 489 modules, best
other version 78.7%. The sampled grid, re-rendered, decodes to the GT
payload; it is saved as `data/calibration/buggy01_modules.npy`, a
near-exact GT bit grid for module-space evaluation (Phase 2/3). The
handoff's "~73 modules / v14" was wrong. Note: reference JPEGs carry EXIF
orientation 6; SAM3 boxes are in raw (PIL) pixel coordinates, so never
pair them with `cv2.imread`, which auto-rotates.

### 7. Kernel bank was biased against blurry frames; partly fixed (2026-10-16).
The reference crop is stored 90 deg rotated (EXIF) relative to the upright
frames, so alignment depended entirely on ORB, which fails on blur: 20 of
the 29 dropped frames were in the blurriest tertile. `align.py` now also
tries 90/180/270 deg initialisations: 335 kernels (was 328), blurriest
drops 20 -> 13. The remaining 22 failures are genuine misalignments (NCC
0.0-0.17: strong perspective, or QR at the frame edge). Old bank backed up
in `artifacts/phase0/_before_rotfix/`. Phase 5a was measured on the old
bank and has not been re-run.

### 8. Module-grid errors and our own RS decoder on real frames (Phase 0.4, 2026-10-16).
`scripts/phase0_grid_errors.py`, full table in `artifacts/phase0/grid_errors_summary.md`.
Geometry is ORACLE (reference alignment), as are version/EC/mask, so these
are upper bounds for a module reader, not a deployable pipeline.

| method (357 frames) | overall | blurriest | medium | sharpest |
|---|---|---|---|---|
| pyzbar/opencv, full-native (Phase 0.1) | 28% | 1% | 28% | 55% |
| sampled grid + RS hard (local threshold) | 43% | 1% | 47% | 80% |
| + soft erasures, no learning | 45% | 1% | 50% | 85% |
| perfect-erasure ceiling for this grid | 49% | 2% | 59% | 87% |

Wrong decodes: 0 (after reserving 4 check symbols per block; spending
the full RS budget on erasures produced garbage "successes").

- **RS is a cliff, not a slope.** A block survives <= 10 wrong codewords
  (<= 20 erased) of 101. Blurriest frames have median module error 26%
  -> ~92 of 101 codewords wrong in the worst block. Erasures cannot touch
  that; the module error must drop ~10x (to ~2-3%) before RS helps.
- **C3 (soft erasures) gives a real but small gain where frames are near
  the cliff:** +9 frames with intensity-only confidence, and the perfect-
  erasure ceiling is +15 more. A better per-module posterior has at most
  ~4 points overall to win here, almost none on the blurriest tertile.
- **The blurriest tertile needs a better grid estimate, not better
  decoding:** restoration (C2) or multi-frame fusion (C4). Alignment
  itself fails on 33/119 blurriest frames.
- **Perspective is real:** side-length ratio median 1.18 (p95 1.48),
  corner-angle deviation median 7.6 deg (p95 18 deg). Box crops are not
  flat; rectification (or a model trained on skewed crops) is needed.
- Part of the 28% -> 43% gain is the oracle geometry; it is not a
  baseline a decoder without the reference could reach.

## What is still open

- **n=1 QR.** All conclusions are from one physical sticker. The owner has
  agreed to film more. Target: 8-15 distinct dense codes, versions 10-20,
  EC levels L/M/Q/H, from a moving buggy. This is non-negotiable for a
  publishable eval.
- **SAM3 masks/polygons.** Currently only boxes are saved. Masks are
  needed for proper rectification (quad -> canonical square). Rerun SAM3
  on Kaggle saving mask polygons; a short notebook cell.
- **ZXing baseline.** Only pyzbar + OpenCV currently installed locally.
  WeChatQR needs opencv-contrib-python; ZXing has flaky Python bindings.
  Decide later whether these two add anything or whether we skip them.
- **Blind vs oracle kernel.** This measurement estimated kernels with the
  reference available (oracle-ish). A production inference path has no
  reference; blind kernel estimation (Krishnan et al., Pan et al.) must
  be benchmarked before claiming Wiener as a baseline.

## Artifacts produced

```
data/calibration/kernels.npz           # 328 x 25 x 25 float32 kernels + metadata
artifacts/phase0/
    decode_baseline.csv                # 2866 rows, (frame, variant, decoder)
    decode_baseline_summary.md
    decode_baseline.png
    pixel_module_stats.csv             # 357 rows, (frame, qr_side_px, px/module per version)
    pixel_module_stats_summary.md
    pixel_module_stats.png
    kernels.csv                        # 328 rows, (frame, kernel stats, align score)
    kernels_summary.md
    kernels_distribution.png
    kernels_grid.png                   # tiled visualisation of 64 sampled kernels
```

## Next steps

1. **New data collection** (owner, this week): 8-15 distinct QR stickers,
   1-2 short sessions each, from a moving buggy. Known GT payload per
   sticker. Push to `RakshitAi/IITB-BuggyQR` using the existing
   `push_dataset.py`.
2. **Re-run SAM3 saving masks** (short Kaggle cell). Updates
   `annotations/all_boxes.jsonl` to carry mask polygons; downstream code
   gets rectification for free.
3. **Phase 1 - data engine.** `diffuqr/synth.py` (procedural QR generator)
   + `diffuqr/degrade.py` (calibrated degradation using the kernel bank).
   Smoke test by generating 1000 synthetic (clean, blurred) pairs and
   decoding both.
4. **Phase 5a - blind + oracle Wiener baselines.** Before training
   anything, measure what a classical kernel-based method can achieve on
   our data. This is the floor the learned model has to clear.
