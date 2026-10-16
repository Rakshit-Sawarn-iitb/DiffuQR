# Guide-Meeting Glossary: What You're Actually Talking About

Plain-English version of every technical term in your project, in the order
your guide is most likely to ask about it. Each entry has: what it is,
why it matters, and the one sentence you say if asked.

---

## 1. QR code basics - the structure you're decoding

A QR code is a square grid of little black-and-white squares. Each square
is called a **module**. A "version 11" QR is 61 modules across (formula:
17 + 4 x version). Higher version = more modules = holds more data.

The code has three fixed landmarks called **finder patterns** - the big
square boxes in three corners. They never change; the decoder uses them
to find and orient the code. There's also a **timing pattern** (a dotted
line of alternating black-and-white between the finders) and **alignment
patterns** (smaller squares inside) that help with perspective correction.
All of these are **known** for a given QR version - we don't have to
"restore" them, we just paste them in.

The code has **error correction built in**. QR specifies four levels:
L (7%), M (15%), Q (25%), H (30%). An EC-L code can lose up to 7% of its
modules and still decode because of **Reed-Solomon coding** - see below.

> **In your words:** "Each QR is a grid of binary squares called modules.
> Version 11 is 61x61; some positions are fixed function patterns we know
> in advance, and the whole thing is protected by a Reed-Solomon code."

---

## 2. Reed-Solomon error correction - the "built-in checksum"

Reed-Solomon (RS) is the error-correcting code QR codes use. Think of it
like a checksum that doesn't just detect errors - it **fixes them**. It
adds some redundancy, and the decoder can recover the original payload
even if some modules are wrong.

Two important facts:
- RS **either returns the correct string or returns nothing**. It does
  not return garbled text. This is why your 0 wrong-decodes across 2866
  attempts is a free gift - every "match" in your table is a true match.
- RS treats two kinds of errors differently:
  - **Errors** (we don't know which module is wrong) cost 2 units of
    error correction budget each.
  - **Erasures** (we know a module is unreadable but not which value it
    should be) cost only 1 unit each.
  **This is the key to contribution C3.** If your diffusion model can
  say "I'm unsure about this module," you mark it as an erasure and the
  decoder gets twice the error-correction budget.

> **In your words:** "Reed-Solomon is QR's built-in error-correcting code.
> It either returns the right payload or nothing - never garbage. And it
> treats 'unsure' modules (erasures) at half the cost of 'wrong' modules,
> which is exactly what my diffusion posterior can tell it."

---

## 3. Motion-blur kernel / Point-Spread Function (PSF)

When you take a photo of something while the camera is moving, each point
in the real scene gets smeared out across several pixels. The **shape of
that smear** is called the **point-spread function** or **blur kernel**.

Mathematically: `blurry_image = kernel * clean_image + noise` (where `*`
means 2D convolution). The kernel is a small image (say 25x25) that says
"each clean pixel contributes to these neighbouring pixels, with these
weights."

A horizontal camera motion produces a horizontal **streak** kernel. A
circular shake makes a disk-shaped kernel. For QR codes shot from a
moving buggy, the kernel is a short line (median ~6 pixels long native,
very directional - aspect ratio 5x in your blurry frames).

> **In your words:** "A blur kernel is the point-spread function of the
> camera motion - a small image that says how each clean pixel got
> smeared. I estimated one per frame and saved them to a bank; the
> blurriest frames have ~6-pixel directional streaks, the sharpest are
> near-point."

---

## 4. How you estimated the kernels (FFT division)

For each frame, you have the blurry image AND the aligned clean reference
(since you own the reference photo and can warp it to match the frame).
That gives you both sides of `B = k * R + noise` - and you want to find
`k`.

In the Fourier domain, convolution becomes multiplication, so the
equation becomes `B(f) = K(f) . R(f)`. Rearranging: `K(f) = B(f) / R(f)`.
You take the inverse FFT and keep only the central patch (say 25x25) to
get the kernel.

Division by small `R(f)` values blows up noise - that's why you add a
small regulariser: `K(f) = conj(R) . B / (|R|^2 + eps)`. This is called
**Tikhonov regularisation** or **regularised deconvolution**.

> **In your words:** "I estimate the kernel by Fourier-domain division of
> the blurry frame by the aligned reference, with Tikhonov regularisation.
> Not blind - it uses the reference as an oracle, which is why we call
> it 'oracle kernel' in the baseline results."

---

## 5. Image alignment: ORB, ECC, NCC

To solve for the kernel you need the reference pixel-aligned with the
frame. The clean reference photo was shot separately, so it has a
different scale, rotation, perspective. You line them up:

- **ORB** (Oriented FAST and Rotated BRIEF): a feature detector/matcher.
  Finds "corners" in both images and matches them, giving a rough
  **homography** (3x3 projective transform matrix) that maps one onto
  the other. Fast but unreliable on blurry textures - which is why you
  fall back to ECC.
- **ECC** (Enhanced Correlation Coefficient): refines the homography by
  gradient descent on a correlation objective. Robust to illumination
  changes if you run it on **band-pass filtered** images (so you ignore
  slow brightness drifts).
- **NCC** (Normalised Cross-Correlation): the alignment-quality score.
  Values near 1 = good alignment, near 0 = unrelated images. You gate
  on NCC >= 0.5 and drop pairs below that threshold.

> **In your words:** "I use ECC on band-pass-filtered images, initialised
> with an ORB homography, and gate by NCC >= 0.5. 328 of 357 frames
> aligned above threshold, which seeded the kernel bank."

---

## 6. Wiener filter and Richardson-Lucy - classical deblurring

Both try to invert the blur: given a blurry image and a kernel, produce
a sharp image.

- **Wiener** is a one-shot filter in the Fourier domain:
  `sharp(f) = conj(K) / (|K|^2 + NSR) . blurry(f)`.
  Fast, but assumes linear blur and additive Gaussian noise. The
  `NSR` (noise-to-signal ratio) is a hyperparameter.
- **Richardson-Lucy (RL)** is iterative: it updates an estimate of the
  sharp image by comparing a re-blurred version against the actual
  blurry input, like EM for Poisson noise. Can over-sharpen into
  **ringing / speckle**, which is why it hurt your QR decode rate.

Your Phase 5a result: oracle-kernel Wiener gained +1 point overall and
+0 on the blurriest tertile. RL dropped from 26% to 13%. This is the
"classical floor is zero" result that motivates learned restoration.

> **In your words:** "Wiener is a scalar-NSR Fourier-domain deconvolution,
> Richardson-Lucy is the iterative Poisson-noise cousin. Both with the
> oracle kernel. Wiener does nothing; RL diverges on sharp binary edges."

---

## 7. Diffusion models - what they are, briefly

A **diffusion model** is a generative model that learns to produce images
by **gradually denoising**. Training: take a clean image, add Gaussian
noise in T steps until it's pure noise. The model learns to reverse one
step at a time. Sampling: start from pure noise, run T denoising steps
to get a clean image.

For restoration (which is what you're doing), you **condition** on the
blurry input: at every step the model sees `(noisy_target, blurry_input)`
and predicts how to denoise toward the clean target.

Modern tricks you're using:
- **v-prediction** instead of noise prediction: numerically stabler at
  high noise levels. Makes the x0 (clean image) estimate recoverable at
  every step without dividing by small numbers.
- **Cosine beta schedule**: distributes noise more evenly across steps
  than the original linear schedule.
- **DDIM sampling**: deterministic, faster alternative to DDPM
  (10 steps instead of 100).
- **EMA**: exponential moving average of model weights for cleaner
  samples at inference.

Your training notebook uses `diffusers.UNet2DModel` as the backbone,
which is the standard HuggingFace diffusion UNet. Nothing custom there -
that's infrastructure, not contribution.

> **In your words:** "Standard conditional diffusion restoration with
> v-prediction and DDIM sampling via HuggingFace diffusers. UNet and
> scheduler are infrastructure, not novelty - the novelty lives in the
> output space and the decoding."

---

## 8. Module-space diffusion vs pixel-space diffusion (your contribution C2)

**Pixel-space** (the current notebook): the model outputs a 256x256x3 RGB
image. The loss compares pixels. The problem: pixel losses reward
reproducing the reference photo's **tone and brightness**, not reading
the modules. This is why PSNR went up but decode rate went down.

**Module-space** (your proposal): the model outputs a 61x61x1 (or 73x73,
or whichever version) **binary module grid**. The target is bits, not
pixels. You condition on the blurry crop as a feature embedding. The
known function patterns (finders, timing, alignment) are **clamped**
during sampling - the model doesn't waste capacity re-generating them.

Why this is a real contribution:
- **Different output space.** Not just "we used diffusion for QR" (that
  would be mix-and-match). The thing being generated is the error-
  correcting code's own symbol alphabet.
- **Lightweight becomes true.** 61x61x1 vs 256x256x3 = ~50x fewer output
  dims. That is a defensible "lightweight" claim.
- **Loss aligns with the goal.** Binary cross-entropy on module bits is
  exactly what the decoder reads next, with no tone-leakage pathway.

> **In your words:** "Instead of diffusing in 256x256x3 pixel space, I
> diffuse directly on the 61x61 module grid conditioned on the blurry
> crop. Function patterns are known and clamped during sampling. The
> loss is binary cross-entropy on the modules themselves, which is the
> correct output space for a Reed-Solomon-protected bit grid."

---

## 9. Verifier-guided + RS-aware decoding (your contribution C3)

This is the most defensibly novel bit. At inference time:

1. **Sample K times.** Diffusion is stochastic - running it K times with
   different random seeds gives K candidate module grids.
2. **Aggregate the posterior.** For each module position, you now have K
   votes. The fraction voting "dark" is your posterior probability that
   the module is dark. **Modules with high agreement (e.g. >95% voting
   one way) are confident; modules near 50-50 are uncertain.**
3. **Soft / erasure decode.** Instead of thresholding every module to a
   hard bit and feeding to RS, you tell the RS decoder "these N uncertain
   positions are erasures." RS can tolerate 2x more erasures than errors.
4. **Verifier selection.** Across the K candidates, pick the one whose
   RS decode succeeds - the checksum is a free verifier that needs no
   ground truth.

**Headline figure for the paper: decode rate vs K.** The curve should go
up as K increases, giving you a "test-time compute scaling" story.

Why novel:
- Nobody (per your literature check so far) couples a **generative
  diffusion posterior** to **Reed-Solomon soft/erasure decoding**.
- The verifier is QR-specific but the pattern (any error-correcting
  code can act as its own verifier) generalises.

> **In your words:** "I sample K times from the diffusion model, treat
> the per-module agreement as a posterior, mark low-confidence modules
> as erasures for the Reed-Solomon decoder, and select the candidate
> whose checksum passes. The decode-rate-vs-K curve is my headline figure."

---

## 10. Procedural data + calibrated degradation (your contribution C1)

**The problem:** you only have one physical QR sticker so far. A model
trained on that one QR will memorise it. More regularisation does not fix
a dataset of size 1.

**The fix:**
- **Procedural QRs:** generate thousands of synthetic QRs with random
  payloads, versions, and EC levels using the `segno` library. The model
  never sees the same QR twice, so there's nothing to memorise.
- **Calibrated degradation:** the synthetic QRs need to **look blurry
  like real buggy frames**, not like textbook Gaussian blur. You already
  built a 328-kernel bank from real frames. At degradation time, you
  sample one of those real kernels (rescaled to the generator's image
  size) and convolve the synthetic QR with it. Plus perspective, defocus,
  illumination, JPEG, sensor noise.
- **Validation:** your clean synth decodes at 100% (generator sound);
  degraded synth decodes at 22% vs real-buggy01 at 26%. The calibration
  landed in the right ballpark on first try.

Why this is a real contribution and not "we just used synthetic data":
- The degradation **is empirically calibrated** - the motion kernels are
  the actual kernels your camera produces, not plausible-looking fakes.
- It's a defensible "real-to-sim" story: real frames serve as the
  calibration anchor, synthetic data provides scale.

> **In your words:** "I generate QRs procedurally with segno for scale,
> and degrade them by sampling motion kernels from an empirical bank fit
> on real buggy frames. Real data calibrates; synthetic data trains.
> This removes memorisation by construction."

---

## 11. Decode rate, blur tertile, native resolution - eval terms

- **Decode rate:** fraction of frames whose restored (or raw) image a
  decoder reads correctly. **The only metric that matters.** PSNR and
  SSIM are reported for reference but we are explicit that they are
  not the goal.
- **Blur tertile:** you sort frames by **Laplacian variance** (a classic
  blur score: a sharp image has lots of high-frequency content, so the
  Laplacian - which highlights edges - has high variance; a blurry image
  has low variance). You split into thirds: blurriest, medium, sharpest.
  Reporting is stratified so the hard subset doesn't hide in the average.
- **Native resolution:** the frame at its original pixel size, before
  any resize. The training notebook was evaluating at 256-crop which
  understates the baseline; you now report native-res as the real
  reference.

> **In your words:** "Headline metric is decode rate at native resolution.
> I stratify by blur tertile - Laplacian variance - so wins on the easy
> frames don't hide failures on the hard subset. PSNR/SSIM are reported
> but secondary."

---

## 12. Honest limitations to say out loud

Your guide rewards honesty. State these before being asked:

- **One physical QR.** All current numbers are from one sticker. Filming
  more today.
- **"Oracle" kernel means the reference was used to estimate it.** At
  inference on a brand new frame we don't have a reference. So "oracle
  Wiener does nothing" is a strong result (even the best classical case
  fails), but blind kernel estimation on a new frame is a separate
  problem we haven't benchmarked yet.
- **Alignment is best-effort.** 328/357 pairs aligned above NCC 0.5. The
  other 29 are dropped from kernel estimation and from training.
- **ZXing / WeChatQR baselines not run.** Only pyzbar and OpenCV
  installed. Adds a row to the Phase 0 table once installed.
- **No multi-frame comparison yet.** 60 fps data makes multi-frame
  natural; Phase 4 covers this if Phase 2-3 land first.

---

## 13. The two things to NEVER do in this meeting

1. **Never claim a result from the handoff doc as yours.** If the
   professor asks about something you don't recognise, say "I'd have to
   verify that - the handoff from my prior sessions had some claims that
   the measurements didn't support, which is why I spent this week
   measuring."
2. **Never say "the AI did it" or "I used Claude."** Code and research
   decisions are yours; you ran, verified, and own them. If asked how
   you built it that fast: "I set up the structure deliberately so each
   measurement script took a day. The thinking was mine, the typing was
   fast."
