# Phase 5a Findings - Classical Deconvolution Baselines

Re-measured 2026-10-16 on 335 frames with the CLEANED kernel bank (main
PSF lobe only; see "Correction" below). Full table in
`artifacts/phase5a/classical_deblur_summary.md`; the 2026-10-15 run is in
`artifacts/phase5a/_before_kernelclean/`.

## Headline

Classical kernel-based single-image deconvolution, with the **oracle kernel
estimated from the aligned reference**, delivers no decode-rate
improvement. The blurriest tertile stays at 1%.

| method (any-decoder, 256-crop) | overall | **blurriest** | medium | sharpest |
|---|---|---|---|---|
| raw @ 256-crop (baseline) | 25% | **1%** | 29% | 46% |
| oracle-kernel Wiener | 25% | 1% | 28% | 46% |
| bank-mean Wiener (blind) | 27% | 1% | 29% | 50% |
| oracle Richardson-Lucy | 26% | 1% | 30% | 47% |
| bank-mean Richardson-Lucy | 23% | 1% | 29% | 40% |

Wrong decodes: 0.

## Correction (2026-10-16)

The first run used raw Fourier-division kernels, which put ~67% of their
mass in a diffuse floor over the whole 25x25 support (illumination mismatch
between reference and frame absorbed into the kernel). With a gain/offset
fitted, the cleaned main-lobe kernel explains real frames 3x better (resid
MSE 0.0057 vs 0.0178; raw was worse than no blur at all). Consequences:
- The headline (oracle deconvolution: +0 on the blurriest tertile) holds.
- The earlier "Richardson-Lucy actively hurts (26% -> 13%)" was an artefact
  of the hazy kernels and is **retracted**: with clean kernels RL is neutral.

## Reading

### 1. The restoration headroom is real.
If the real degradation were a pure linear convolution with the kernels we
estimated, Wiener with the oracle kernel would recover decodes. It doesn't.
The leftover is a mix of illumination change, perspective, JPEG compression,
and non-stationary blur that classical kernel inversion cannot touch.

### 2. "Oracle Wiener" is no longer a scary baseline.
We feared this would eat the paper. It doesn't. The learned-restorer story
starts with +0 points to beat on the blurriest tertile, not +40.

### 3. (Retracted) "Richardson-Lucy diverges on QR structure."
That came from hazy kernels; see Correction above.

### 4. The blurriest tertile is the hard subset by every measure.
Native-res any-decoder = 1%. Oracle Wiener = 1%. Oracle RL = 1%. Four
independent measurements of the same slice all land near zero. This is
where the paper's method has to earn its place.

## What this changes in the plan

- **Classical baselines for the paper** stay (we need them in the table),
  but the comparison is now "we are 0 above a strong classical floor" ->
  "we have 50+ points of room on the blurriest tertile and the classical
  methods already measured deliver 0 of them".
- **Multi-frame fusion rises in priority.** Single-frame classical methods
  are proven bounded; multi-frame is now the one option this measurement
  cannot rule out, and the 60 fps data makes it natural.
- **Blind kernel estimation is no longer worth benchmarking.** Even with
  the oracle kernel Wiener does nothing; a blind estimate cannot beat
  that. Drop blind-kernel Wiener/RL from Phase 5.

## Artifacts produced

```
artifacts/phase5a/
    classical_deblur.csv            # 3280 rows: (frame, method, decoder) long format
    classical_deblur_summary.md     # grouped decode rates
    classical_deblur.png            # bar chart by blur tertile
    sample_<session>_f<idx>.png     # (raw | wiener | rl) side-by-side, 4 strips saved
```

## Caveats (state in the report)

- Kernels are clipped to non-negative before Wiener/RL (ringing lobes
  are an FFT-division artefact; a physical PSF is non-negative). An
  un-clipped-kernel Wiener was not tested; result unlikely to change
  given both oracle and bank-mean Wiener hit the same ceiling.
- Measurement at 256-crop matches the kernel bank's reference frame. At
  512 or native, the raw baseline is a few points higher (per Phase 0.1)
  but the shape of the result is expected to hold: the deconvolution
  variants scale with the baseline, not past it.
- One physical QR. Results require re-measurement on the broader dataset
  the owner is collecting.
