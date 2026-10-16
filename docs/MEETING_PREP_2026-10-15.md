# Guide Meeting Prep - 2026-10-15

What to show, in what order, with the two-sentence version of each point.
Target length: 15 minutes talking + Q&A.

---

## The 30-second opener (say this verbatim if useful)

> "Before training more, I spent this week measuring what the actual
> problem is. Three measurements changed my direction. The one-line
> version: the real bottleneck is **non-stationary motion blur**, not
> resolution; the handoff's '8% baseline' was a 256-pixel downsampling
> artifact; and **classical deconvolution with the oracle kernel gives
> zero improvement on the hard subset**, so there's genuine room for a
> learned restorer. I've scoped three real contributions, built the
> data engine, and I'm filming more QRs today."

---

## Why the old training was retired (2 minutes)

Open `notebooks/training.ipynb`, scroll to the eval cell output.

| | before (handoff claim) | after 100 epochs (verified) |
|---|---|---|
| ALL (49 frames) | 4/49 decode (8%) | **3/49 (6%)** |
| sharpest 16 | 3/16 (19%) | **2/16 (12%)** |
| PSNR | 13.9 dB | 21.0 dB |
| SSIM | 0.73 | 0.82 |

The model trained to convergence, PSNR/SSIM went up a lot, and **decode
rate went down**. On the sharpest tertile, we lost a decode. This is the
classic sign of a model learning the **reference image's tone** instead
of restoring QR module structure.

Root cause: **one physical QR**. The dataset's optimum is "emit the
reference image." No amount of regularisation fixes a dataset of size 1.
That's why I pivoted to measurement-first.

---

## Three measurements that redirect the paper (5 min)

Open `docs/phase0_findings.md` and `artifacts/phase0/decode_baseline.png`.

### 1. Native-resolution decode baseline

```
raw frames, no restoration, two decoders (pyzbar + opencv):
  ALL         28% (100/362)
  blurriest     1% (1/121)
  medium        27% (33/121)
  sharpest     55% (66/120)
```

- The old "8%" number was decoded at 256 px after INTER_AREA
  downsampling. At native resolution the baseline is **28%**.
- The hard subset is the blurriest tertile. **1% at native.** That's
  what the paper has to improve.
- 0 wrong-decodes across 2866 attempts: Reed-Solomon never fakes a hit.
  Free fact for the paper.

### 2. Pixels per module

Open `artifacts/phase0/pixel_module_stats_summary.md`.

- Every frame has **>= 7 px/module** (median 10). Reference has 32.
- The industry threshold for reliable decoding is 2-4. **We're above it.**
- The handoff's "3.5 px/mod" was the 256-crop figure, not reality.
- **Reading:** we don't have a resolution problem. This is a pure
  deblurring problem. That narrows the technical scope.

### 3. Real motion-blur kernels

Open `artifacts/phase0/kernels_grid.png` (shows 64 kernels visually).
Open `artifacts/phase0/kernels_summary.md`.

I estimated a per-frame PSF by solving `frame ~= k * aligned_reference`
in Fourier space. 328 of 357 frames aligned above NCC 0.5 - saved to
`data/calibration/kernels.npz`.

Measured properties:
- Length: median 2.5 px in 256-crop (~6 px native); p95 ~8.6 px in
  256-crop. **Short.**
- Aspect ratio: median 5x in blurry frames. **Highly directional.**
- Angle: isotropic across [0, 180). **Camera moves in all directions.**
- Blur-tertile stratification is validated: blurriest median length 6.2
  px, sharpest 1.2 px (near-point).

**Reading:** This is a well-defined, non-stationary motion-blur
distribution. Phase 1's degradation simulator samples from this bank
instead of guessing kernel parameters.

---

## The scary baseline that turned out safe (3 min)

Open `docs/phase5a_findings.md`.

I was worried that Wiener/Richardson-Lucy with the oracle kernel would
eat the paper. **It doesn't.**

| method | overall | **blurriest** | medium | sharpest |
|---|---|---|---|---|
| raw @ 256-crop | 26% | 1% | 30% | 46% |
| oracle-kernel Wiener | 27% | **1%** | 32% | 49% |
| oracle Richardson-Lucy | 13% ↓ | 0% | 18% | 21% |

Even with the ground-truth kernel (estimated from the aligned reference,
not blind), Wiener gains **+1 point overall, +0 on the hard subset**.
RL actively hurts. Reasons: the degradation isn't purely linear
convolution; there's illumination change, perspective, JPEG, and
non-stationary blur.

**This is a paper-defining negative result:**
- Classical kernel-based deconvolution is proven bounded.
- Any decode-rate gain the learned restorer delivers on the blurriest
  tertile is genuinely novel work.
- Single-frame methods have a ceiling; **multi-frame fusion** becomes
  the obvious next lever.

---

## The new direction - three contributions, each tied to a measurement (3 min)

Open `docs/PLAN.md`.

### C1. Procedural QR training + real-calibrated degradation (the data answer)
- Motivation: measurement 1 (one physical QR). Procedural generation
  removes memorisation by construction.
- Build status: **done today.** `diffuqr/synth.py` + `diffuqr/degrade.py`.
  Smoke test (`docs/phase1_findings.md`): clean synth decodes 100%,
  degraded synth decodes 22% vs real 26%. Calibration is tuned right.

### C2. Module-space diffusion with structured output
- Motivation: measurement 2 (we don't need super-resolution). The
  output we want is a 73x73 bit grid, not a 256x256x3 image. Pixel
  losses reward tone, not modules. Function patterns (finders, timing,
  alignment) are known for a given version and can be clamped during
  sampling.
- "Lightweight" claim gets real: 73x73 vs 256x256x3 = ~30x fewer output
  dimensions.
- Not built yet - Phase 2, needs more real data for evaluation.

### C3. Verifier-guided, Reed-Solomon-aware decoding of the diffusion posterior
- Motivation: measurement 4 (classical is bounded). QR provides a free
  verifier at test time: a decode either passes RS checksum or doesn't.
  Sample K restorations, use per-module posterior as soft/erasure input
  to RS decoder, select by checksum.
- Headline figure for the paper: **decode rate vs K (test-time compute
  scaling)**.
- Prior-art check (not exhaustive, still verifying): no diffusion+RS-
  soft-decode work found.

### Optional C4. Multi-frame posterior fusion
- Fuse per-module log-odds across neighbouring frames before RS. The
  60 fps data is tailor-made for it. Add only if C1-C3 beat the raw
  baseline; otherwise it's noise.

---

## Status + ask (2 min)

**Done this week:**
- Phase 0 (3 measurement scripts, 328-kernel bank).
- Phase 5a (classical baselines).
- Phase 1 (data engine).
- Full repo refactor: `diffuqr/` library package, phase-prefixed scripts,
  all outputs under `artifacts/`, docs under `docs/`.

**In progress:**
- Filming 8-15 more QR stickers today - breaks n=1 and makes the eval
  publishable.

**Blocked / needs your input:**
1. **Is the direction OK?** C1 + C2 + C3 - not a mix-and-match, each
   contribution has a reason tied to a measurement. Any contribution
   you'd cut or any you'd add?
2. **Venue + deadline?** Short-paper options I'm aware of: NCVPRIPG,
   ICIP, CVPR workshops, BMVC workshops. Which is realistic given the
   course deadline?
3. **Scope:** keep it to buggies-only, or expand to any dense sticker
   QR filmed in motion (hostel, cafeteria)? Latter helps with n but
   dilutes the "buggy" story.

**Not asking you to decide:** architecture choice, loss formulation,
training hyperparameters - those are my job.

---

## Likely questions, prepared answers

**Q: "Why did you throw away two months of training?"**
A: It wasn't throwing away - the training ran to convergence and showed
the method does not improve decode rate (3/49 at epoch 100 vs 4/49 raw;
one fewer on the sharpest tertile). That's a conclusive negative result
on the pixel-space-UNet + patch-loss formulation. Continuing was waste.

**Q: "Is this really novel enough for a paper?"**
A: Each contribution is defensible on its own:
- C1 is a "real-to-sim" calibration story using actually-measured
  kernels, not synthetic ones.
- C2 is a different output space (module grid, not RGB) with known
  function patterns clamped - no prior QR restoration paper does this.
- C3 is coupling a generative posterior to the code's own ECC - no
  diffusion+RS work found in my literature check (not exhaustive,
  will verify).
What makes it non-mix-and-match is the combination: diffusion in module
space + RS-aware decoding + real-calibrated training.

**Q: "How confident are you in the measurements?"**
A: 362 frames, two independent decoders, 0 wrong-decodes across 2866
attempts. 328 kernels fit at NCC >= 0.5. The conclusions are
measurement-limited by **one physical QR**. That's the known gap, and
filming today addresses it directly.

**Q: "What if your method doesn't beat 28%?"**
A: 28% is the overall native-res baseline. The paper's claim is on the
**blurriest tertile (1%)**. Anything meaningful there is a gain. If
the method doesn't beat 1%, the paper's contribution shifts to the
measurement + classical-baseline + negative-result story, which is
still publishable (just as a workshop short paper).

**Q: "Why not just fine-tune Real-ESRGAN?"**
A: Edula et al. tried that. Their pipeline is YOLOv8 + Real-ESRGAN +
multi-library decode. We will compare against that as a baseline.
Real-ESRGAN is trained for natural images; it has no notion of QR
structure, module grid, or Reed-Solomon. It operates in pixel space,
which measurement 2 (we don't need super-resolution) shows is the wrong
space for this problem.

**Q: "SAM3 vs YOLO - didn't you claim SAM3 wins?"**
A: Current benchmark is SAM3 vs OpenCV, not vs YOLO (no QR-trained
YOLO weights exist). The "SAM3 vs YOLO" row in the paper needs YOLO
fine-tuned on SAM3 pseudo-labels. That's one afternoon of work,
scheduled for Phase 5.

**Q: "What are you doing about the one-QR problem?"**
A: Three parallel moves:
1. Filming additional distinct QRs today (committed).
2. Procedural training (C1) breaks memorisation by construction.
3. Explicit unseen-QR holdout in evaluation once we have 2+ QRs.

---

## Files to have open on the laptop

1. `docs/PLAN.md` - the master strategy doc.
2. `docs/phase0_findings.md` - all measurement numbers in one place.
3. `docs/phase5a_findings.md` - the classical-baseline result.
4. `docs/phase1_findings.md` - data engine readiness.
5. `artifacts/phase0/decode_baseline.png` - plot.
6. `artifacts/phase0/kernels_grid.png` - visualisation of 64 real kernels.
7. `artifacts/phase5a/sample_*.png` - raw | wiener | rl side-by-side.
8. `notebooks/training.ipynb` eval cell output - "why the old
   formulation didn't work."

---

## One slide if you need it (ascii)

```
Problem:  decode UPI QR stickers filmed from moving IIT-B buggies.
Baseline: 28% native-res, 1% on blurriest tertile. 0 wrong-decodes.
Finding:  not resolution-limited (10 px/mod); motion-blur-limited.
Classical kernel-based deconv w/ oracle kernel: +0 on hard subset.

Approach: three contributions, each from a measurement:
  C1: procedural-QR training + real-calibrated kernel bank (328 PSFs)
  C2: module-space diffusion (73x73 bits, not 256x256 RGB)
  C3: verifier-guided RS-aware decoding of the diffusion posterior

Status:   Phase 0 measurements + Phase 1 data engine shipped today.
          Filming more QRs now. Phase 2 training queued.
```
