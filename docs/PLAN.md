# DiffuQR: Plan

Written 2026-10-15. Owner: Rakshit Sawarn (23B0426), IE643, IIT Bombay.
Grounded in a direct read of `notebooks/training.ipynb` (the Kaggle source of truth) and its last recorded run; not in the handoff doc.

---

## 1. What we are chasing

Two deliverables, same project, overlapping work.

### 1a. Course outcome: AA in IE643
Non-negotiable artefacts, in priority order:
1. A working end-to-end pipeline: raw frame -> detect (SAM3) -> rectify -> restore -> decode, runnable as one notebook.
2. A measurement report with real numbers (native-resolution decode baseline, pixels/module distribution, blur-kernel statistics) that justify every modelling choice.
3. A **real** detector comparison: SAM3 vs a YOLO fine-tuned on SAM3 pseudo-labels (not SAM3 vs OpenCV).
4. A model whose headline metric (**decode rate on held-out frames**) is better than the raw baseline. Not PSNR. Not SSIM.
5. Written limitations section that pre-empts the obvious questions (n=1 QR, synthetic degradation gap, 256 px handicap).

### 1b. Paper outcome: a workshop/short-paper submission
Additional requirements on top of 1a:
1. More than one physical QR in the test set. 5-10 distinct dense QRs is enough.
2. One sharp, defensible novelty. Not three. Candidate (section 3): **module-space, verifier-guided diffusion with RS-aware soft/erasure decoding, trained on procedurally generated codes under a real-calibrated degradation model.**
3. Baselines: no restoration + multi-library decode; Lanczos + unsharp; Wiener/RL deconv with estimated kernel; EG-Restormer (arXiv 2510.12098); Real-ESRGAN (Edula et al. ICCCNT 2023).
4. One figure that sells it: **decode rate vs K samples** or **decode rate vs frames fused** (test-time compute scaling).

Target venues (short-paper friendly, India-accessible): NCVPRIPG, ICIP, CVPR workshops, BMVC workshops.

---

## 2. Where we actually stand (evidence)

From the last recorded Kaggle run in `notebooks/training.ipynb`, trained to epoch 100 (resumed from 40):

| metric | raw -> restored (49 held-out frames) |
|---|---|
| decode rate ALL | 4/49 (8%) -> **3/49 (6%)** |
| decode rate sharpest 16 | 3/16 (19%) -> **2/16 (12%)** |
| PSNR | 13.9 -> 21.0 dB |
| SSIM | 0.73 -> 0.82 |
| inference | ~992 ms/frame (10 DDIM steps, 1 GPU) |

**Reading:** PSNR/SSIM look fine, but decode went down. The sharpest tertile lost a decode. This is consistent with the model drifting toward "the reference photo's tone and macro structure" rather than reconstructing module edges. Training further on this formulation will not fix it.

**Root causes (all three are real; none is a tuning issue):**
1. Dataset is effectively one target image. The optimum of the conditional model is "emit that image." No regulariser recovers from this.
2. Output space is RGB pixels. The thing we want is a binary module grid that passes Reed-Solomon. The loss doesn't know that.
3. Resolution: 73 modules at 256 px is ~3.5 px/module. The 16 px patch tile covers ~4.5 modules at an arbitrary offset. The "patch = module" motivation does not actually hold in the code.

---

## 3. The strategy

**Thesis:** restore QR codes as *structured, self-verifying discrete objects*. The restoration network diffuses over a module-space grid conditioned on a rectified blurry crop; training data is procedurally generated with a real-calibrated degradation model; decoding uses the per-module posterior as soft/erasure input to a Reed-Solomon decoder. Multi-frame posterior fusion is the natural extension if time permits.

This formulation addresses all three root causes above by construction:
- **Memorisation:** gone, because training QRs are drawn from an unbounded procedural distribution. Real frames are used only for kernel calibration and evaluation.
- **Output space:** the network outputs N x N module bits (or analog-bits), not RGB. Finder/timing patterns are clamped during sampling because they are known for a given version.
- **Resolution:** 73 x 73 is the native representation. Nothing is wasted on pixel grid.
- **"Lightweight" claim:** actually true. 73 x 73 x 1 vs 256 x 256 x 3 is ~30x fewer output dimensions.
- **Novelty is non-mix-and-match:** diffusion posterior + QR's own ECC used as a soft-decision decoder is, to my knowledge, new (verify before writing paper).

Fallback if module-space is too ambitious under the deadline: keep pixel-space diffusion, train on procedural QRs, add verifier-guided best-of-K sampling + RS erasure decoding. This still breaks the one-QR problem and keeps the "verifier-guided" story.

---

## 4. Phased plan

Each phase has a hard deliverable. Don't move to the next without it.

### Phase 0 - Measure reality (this week, no training, CPU/local)
Nothing we build is credible until we know the actual difficulty of the task.

- [ ] **Native-resolution decode baseline.** Decode all 362 raw frames at full resolution with pyzbar + OpenCV `QRCodeDetector` + WeChatQR (if available) + ZXing. Both full-frame and SAM3-crop. With and without Lanczos upscale + unsharp. Stratify by Laplacian-variance blur tertile and QR pixel size. **Deliverable:** `docs/phase0_decode_baseline.md` + a plot.
- [ ] **Pixels/module distribution.** From SAM3 boxes in `annotations/all_boxes.jsonl`, histogram the QR side length in pixels; divide by estimated version (~73 modules). **Deliverable:** one plot.
- [ ] **Real motion-blur kernel bank.** For each aligned (frame, reference) pair with NCC >= 0.5, solve `frame ~= k * ref` with regularised least squares for an ~11 x 11 kernel. Dump length/angle histograms + save kernels to `data/calibration/kernels.npz`.
- [ ] **SAM3 masks/polygons.** Re-run SAM3 saving masks (not just boxes); write 4-corner quads to `annotations/all_quads.jsonl`. This unlocks rectification.
- [ ] **Owner decision:** can additional distinct QRs be filmed? If yes, print 8-15 dense codes (versions 10-20, EC levels L/M/Q/H, random EMV-like payloads), tape in buggy, 1-2 short sessions each. Deadline for this decision: end of Phase 0.

Deliverable gate: a 2-3 page measurement note. If the native-resolution baseline is already >= 60%, pivot: the paper becomes about *the hard subset*, not the whole set.

### Phase 1 - Data engine (1 week)

- [ ] `scripts/synth_qr.py`: random payload + random version + random EC -> clean 73 x 73 (or variable-N padded to fixed size) module grid + rasterised reference image. Use `segno` or `qrcode` (not only `cv2.QRCodeEncoder`, which is restrictive).
- [ ] `scripts/degrade.py`: apply calibrated motion kernel (sampled from the Phase 0 bank) + defocus + perspective + illumination + JPEG + sensor noise. Deterministic seed per call for reproducibility.
- [ ] `scripts/rectify.py`: SAM3 mask quad -> canonical square, estimate module count N from finder-pattern spacing / timing pattern. Report rectification failure rate on real frames.
- [ ] CPU smoke test: one batch, two DDIM steps, 32 x 32 toy -> runs end-to-end locally without touching Kaggle.

Deliverable gate: `python -m scripts.synth_qr --n 1000 --out data/synth/` + `python -m scripts.degrade` on real reference crops produces visually plausible blurred QRs; held-out *synthetic* decode baseline logged.

### Phase 2 - Module-space diffusion (2 weeks on Kaggle)

- [ ] `scripts/train_module_diffusion.py`: diffusers UNet2DModel on an N x N (padded to 80 x 80 so all common versions fit) single-channel grid. Condition = rectified blurry crop downsampled to 80 x 80 x 3 (or feature embedding). Clamp finder/timing/alignment patterns during sampling. v-prediction, cosine, T=100, 10 DDIM steps, EMA, DataParallel.
- [ ] Loss: BCE on module bits (soft target during diffusion via analog-bits trick), weighted by `alpha_bar`.
- [ ] Train on **procedural data only**. Real frames go into eval, never train.
- [ ] Sanity gate before anything else: decode rate on held-out *synthetic* QRs under the calibrated degradation. Must be strictly above a bicubic + decode baseline. If not, the formulation is broken and more training won't fix it.

Deliverable gate: module-space model with synthetic held-out decode >> baseline; **real-frame decode rate at least matches raw baseline** (not worse).

### Phase 3 - RS-aware inference (3-5 days)

- [ ] Sample K restorations per frame -> stack per-module Bernoulli posterior.
- [ ] Soft/erasure decode: threshold confident modules to hard bits, mark low-confidence as erasures (`reedsolo` library supports erasures), feed to RS decoder. Erasures cost half the ECC budget of errors, so this is the main lever.
- [ ] Verifier selection: among K candidates, pick the one whose RS decode succeeds (free ground-truth-less verifier).
- [ ] **Headline figure:** decode rate vs K (1, 2, 4, 8, 16, 32).

Deliverable gate: this plot. If K=16 beats K=1 by any margin, we have a paper hook.

### Phase 4 - Multi-frame fusion (optional, 2-3 days, do only if Phase 3 works)

- [ ] Align/rectify neighbouring frames -> fuse per-module log-odds across frames -> RS-decode.
- [ ] **Second headline figure:** decode rate vs frames fused.

### Phase 5 - Baselines and evaluation (1 week)

- [ ] No restoration + multi-library decode (already in Phase 0, re-run on test split).
- [ ] Lanczos upscale + unsharp.
- [ ] Wiener / Richardson-Lucy with estimated kernel.
- [ ] Real-ESRGAN (Edula et al. pipeline).
- [ ] EG-Restormer (arXiv 2510.12098, Oct 2025) if their code runs - closest recent prior art, must be cited.
- [ ] YOLOv8 fine-tuned on SAM3 pseudo-labels -> fair "SAM3 vs YOLO" row.
- [ ] Stage-wise reporting (detect -> restore -> decode) by blur tertile and pixel/module size, EgoQR-style.
- [ ] Ablations, one change at a time: module-space vs pixel-space, procedural-only vs +real-aligned, patch-loss on/off, K=1 vs K=16, soft RS vs hard RS, single-frame vs multi-frame, v-pred vs eps.

### Phase 6 - Report and demo (last week)

- [ ] Course report: pipeline figure, Phase 0 measurement plots, decode rate vs K figure, ablation table, honest limitations section.
- [ ] Kaggle-notebook demo: 2-3 example frames end-to-end, rendered inline.
- [ ] If filing a paper: cut to 4 pages (workshop format), move ablations to supplementary.

---

## 5. Risks and honest limitations we will state

- If frames routinely have < 2 px/module or extreme blur, no single-frame method can recover the payload. Only multi-frame might.
- Procedural-to-real gap: calibrated degradation may still miss real artefacts (rolling shutter, specular highlights on glossy stickers). Phase 2's real-frame decode check is the honest test.
- Test set is one physical QR unless the owner can film more. Section 1b.1 depends on this.
- "Lightweight" claim only becomes real with the module-space formulation. Pixel-space 256 x 256 UNet is not lightweight by any paper's definition.

---

## 6. What stops now

- No more epochs of the current pixel-space UNet. The last run converged and decode went down. Further training is wasted compute.
- No more "SAM3 vs OpenCV" claims. The detection story is already won; the paper needs a YOLO comparison or nothing.
- No more PSNR/SSIM as headline metrics. Decode rate is the only number that matters.

---

## 7. Immediate next action

Phase 0.1: write `scripts/decode_baseline.py` to decode all 362 raw frames at native resolution with four decoders. One afternoon of work; the result reframes the entire paper.
