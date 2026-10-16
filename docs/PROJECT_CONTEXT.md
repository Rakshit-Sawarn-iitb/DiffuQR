# DiffuQR: Research Handoff Context

Last updated: 2026-10-15. Written for a model taking over research + implementation.
Sources: owner's brief, direct read of the repo at commit `a4eb9da` (scripts, 3 notebooks, `benchmarking.txt`, session JSON), and a prior-art web check.
Tags: **[VERIFIED]** = seen in repo/outputs this session. **[CLAIMED]** = from the owner's brief, not visible in the repo. **[HYPOTHESIS]** = my reasoning, untested.

---

## 0. TL;DR

- **Project:** restore/decode payment QR stickers filmed from moving IIT-B buggies. Pipeline: detect (SAM3, zero-shot) -> restore (diffusion) -> decode. IE643 course project, but a research paper is the target.
- **Constraint from the professor:** not a mix-and-match of existing papers. Each claimed contribution needs a genuine idea.
- **Contribution 1 (done):** SAM3 zero-shot QR localisation. 363/363 images found vs OpenCV 41.6% **[VERIFIED, `benchmarking.txt`]**. The "vs YOLO" comparison is still not real (no QR-trained YOLO exists).
- **Contribution 2 (this is the open research question):** the diffusion model. Today it is a stock diffusers `UNet2DModel` + a patch L1/SSIM loss. That is not enough novelty, and it has a fundamental data problem: **one QR, one video**.
- **Core diagnosis [HYPOTHESIS, strong]:** we treat a QR as a natural image and train on one code, so overfitting/memorisation is guaranteed. A QR is a **discrete, structured, self-verifying object** (binary module grid, fixed finder/timing patterns, Reed-Solomon checksum). The novelty should come from exploiting that, not from the UNet. See section 7.
- **Recommended thesis:** *structure-aware, verifier-guided diffusion for QR restoration*, trained on **procedurally generated QR codes** with a **degradation simulator calibrated from real buggy frames**; real data is used only for calibration and evaluation. This removes the memorisation problem by construction.
- **Do first (Phase 0, no GPU training):** measure the real problem (section 8). We do not know the native-resolution decode rate of the raw frames, and the existing "8% decode" is at a handicapped 256 px.

---

## 1. Problem and deliverables

**Title:** A Lightweight Diffusion-Based QR Code Detection while Travelling in IIT-B Buggies. Student: Rakshit Sawarn (23B0426), IE643 (IIT Bombay). Repo: `Rakshit-Sawarn-iitb/DiffuQR`.

**Problem:** UPI payment QR stickers inside buggies are filmed from a moving buggy: motion blur, defocus, perspective distortion, illumination change, noise. Standard decoders fail.

**Promised deliverables (proposal):** (1) Buggy Motion QR dataset, (2) degradation + preprocessing pipeline, (3) lightweight diffusion restoration model, (4) detection + decoding module, (5) quantitative/qualitative evaluation vs baselines (PSNR/SSIM, IoU, decode rate, localisation, inference time).

**Reference papers (from slides):**
- Edula et al., ICCCNT 2023: YOLOv8 + Real-ESRGAN + multi-library decode (closest existing pipeline).
- EgoQR (arXiv 2410.05497): stage-wise, size-stratified evaluation.
- FideDiff (arXiv 2510.01641): one-step consistency-model deblurring ("lightweight" reference).
- Chen & Liu (arXiv 2401.05907): diffusion prior + Swintormer.
- Meta SAM3.

**Phases (slides):** P1 data + SAM3 vs YOLO benchmark; P2 synthetic degradation + lightweight diffusion training (Kaggle T4/P100); P3 integrate detector + restoration + multi-library decoder, stage-wise evaluation; P4 Kaggle-notebook interface, mobile demo as stretch, report.

---

## 2. Constraints and working style

- **Compute:** Kaggle only, 2x T4 (or P100), 16 GB, Python 3.12. Local: Windows, PowerShell, Python 3.10 venv at `venv/`. **Always `python -m pip`**, never bare `pip`.
- **Hugging Face:** dataset repo `RakshitAi/IITB-BuggyQR` (private). Model repo `RakshitAi/IITB-BuggyQR-diffusion` for checkpoints. **Never push checkpoints to the dataset repo.** Tokens are in `.env` / Kaggle secret `HF_TOKEN`; never print/commit them (`.env` is gitignored; do not read it). `facebook/sam3` is gated and needs the token passed explicitly.
- **Owner preferences:** direct, no fluff; targeted edits, not whole-file rewrites; **verify before presenting** (they are tired of "that was a real bug" follow-ups); honest framing of limitations, no overselling. Say clearly what was not run.
- **Professor:** novelty must be real. Architecture/scheduler stay standard (diffusers) and are infrastructure, not the contribution. Don't hand-roll them.
- **Out of scope for the next model:** pushing, committing, or running Kaggle jobs without the owner. Training happens on Kaggle in notebooks; hand the owner cells to paste.

---

## 3. Repo state (what is actually here) [VERIFIED]

```
README.md                    2 lines (title only)
benchmarking.txt             full SAM3 vs OpenCV benchmark output (363 images)
scripts/config.py            paths, .env loading, parse_session_id (regex <qr_id>_<YYYYMMDD>_<NN>)
scripts/extract_frames.py    video -> every-Nth frame JPGs + metadata.jsonl (keeps blurry frames)
scripts/push_dataset.py      extract + push video, ref photo, frames, manifest in ONE HF commit
scripts/pull_dataset.py      snapshot_download, --skip-videos
notebooks/segmentation.ipynb          = annotate_and_push.py (SAM3 boxes for all images -> HF)
notebooks/benchmarking-sam3-and-yolo.ipynb   SAM3 vs OpenCV benchmark (4 MB, mostly plot outputs)
notebooks/training.ipynb              = train_qr_diffusion.py, OLDER version (see below)
data/ (gitignored)           1 session: raw video + ref photo + frames + sessions/*.json
```

**Important discrepancies between the brief and the repo:**
1. **`notebooks/training.ipynb` is the OLD version.** It does NOT contain: random-QR mix (`random_qr_pair`, `SYNTH_QR_PROB`), `REF_PAD_FRAC`/`MIN_VALID_FRAC` coverage filter, `decode_qr_robust`, ceiling row, binary-pixel agreement, `memorization_probe`, unseen-QR holdout. The brief calls the newer file "the source of truth", but **it is not in the repo**. Ask the owner to add it (e.g. `scripts/train_qr_diffusion.py`) before building on it. Treat every claim about the "latest version" as [CLAIMED].
2. **Benchmark scale:** the brief says the SAM3-vs-OpenCV benchmark ran on 9 images. `benchmarking.txt` shows it ran on **all 363** (see section 5). The 9-image run was an earlier one.
3. **Checkpoint push:** the old notebook has `PUSH_CHECKPOINTS=True` / `RESUME_FROM_HUB=True` to the separate *model* repo (allowed). The brief says the latest has both `False`.
4. The three Kaggle scripts live only inside notebooks; `scripts/` has only the data-prep scripts. `README.md` is empty of content.
5. `qr_payload` in `data/sessions/buggy01_20260917_01.json` is `null`. The GT string comes from decoding the reference photo.

---

## 4. Data [VERIFIED unless noted]

- **One QR (`buggy01`), one session (`buggy01_20260917_01`).** Video: 1080x1920, 60 fps, 1085 frames (~18 s), every 3rd frame saved -> 362 frames. Reference photo 5712x4284 (SAM3 box ~1868x1961 px **[CLAIMED]**).
- **QR payload [CLAIMED]:** dense EMVCo-style UPI merchant string (starts `000201`), ~73x73 modules (~version 14). At 256 px that is only ~3.5 px/module, which is why strict decode at 256 px is resolution-limited.
- **Identity scheme:** `qr_id` = physical QR/buggy (one reference photo each); `session_id` = `<qr_id>_<YYYYMMDD>_<NN>`; frames `<session_id>_fNNNNNN.jpg` embed the original video frame index.
- **HF dataset layout:** `raw/reference_photos/`, `raw/videos/`, `frames/{session}/` (+`metadata.jsonl`), `sessions/{session}.json`, `annotations/{reference_photos,frames/{session},all_boxes}.jsonl`.
- **Annotation row:** `file_name, qr_id, session_id, source (reference|frame), image_width/height, detector="sam3", num_boxes, boxes:[{box_xyxy, box_xyxyn, score}]`. Note: only **boxes** are saved, not SAM3 masks. Masks would give a 4-corner quad for rectification (section 7), so re-running SAM3 to save masks/polygons is a cheap, useful step.
- Real-data limitation: the dataset is effectively **one training example**. Neighbouring frames are near-duplicates (0.05 s apart).

---

## 5. What was run and measured

### Detection [VERIFIED, `benchmarking.txt`, 363 images]
| method | detection rate | avg conf | s/image |
|---|---|---|---|
| OpenCV QR detector (multi-scale/CLAHE) | 0.416 | 1.0 (constant, meaningless) | 0.96 |
| SAM3 ("QR Code" prompt, thr 0.5) | 1.000 (avg 1.006 detections) | 0.984 | 2.46 |

Caveats: the "YOLO" column was never YOLO (no QR-trained weights); it is OpenCV. "Detected" means a box exists, **not** that the box is tight; there is no IoU ground truth. SAM3 at ~2.5 s/image is not "lightweight" and the slides' claim "SAM3 outperforms YOLO" is unsupported until a real YOLO baseline exists (idea: fine-tune YOLO on SAM3 pseudo-labels).

### Restoration [VERIFIED, old notebook outputs; 100 epochs, resumed from epoch 40]
Config: 256 px, T=100, 10 DDIM steps, cosine, v-prediction, EMA, DataParallel on 2 GPUs. Final train v-loss ~0.045, patch loss ~0.036. Alignment: 329/357 pairs kept; train 275 / test 49 (temporal split).

| split (n) | strict decode raw -> restored | PSNR raw -> restored | SSIM raw -> restored |
|---|---|---|---|
| ALL (49) | 4 -> 3 (8% -> 6%) | 13.92 -> 20.99 | 0.726 -> 0.817 |
| blurriest (17) | 0 -> 0 | 12.18 -> 15.31 | 0.533 -> 0.647 |
| medium (16) | 1 -> 1 | 14.44 -> 22.28 | 0.776 -> 0.886 |
| sharpest (16) | 3 -> 2 | 15.23 -> 25.73 | 0.880 -> 0.930 |

~990 ms/frame (10 DDIM steps, 1 GPU).

**Correct interpretation:**
- PSNR/SSIM gains are real but **decode did not improve** (differences of 1 frame in 49 are noise). Part of the PSNR gain is likely the model learning the *reference photo's tone/illumination* rather than sharpening modules.
- The test set is one QR, the last 15% of one video. A model that outputs "the" QR regardless of input would also score well on PSNR. **These numbers cannot support any claim about restoration quality.**
- The raw baseline (8%) is decoded at **256 px after INTER_AREA downsampling**, which hurts a 73-module code. The native-resolution decode rate of the frames is **unknown** and may be much higher. Nothing is established about how hard the real task is until this is measured.
- `memorization_probe`, `distill_one_step`, robust decode, and the random-QR mix have never been run on a real model [CLAIMED/never run].

---

## 6. Current architecture (old notebook, [VERIFIED])

- **Pairing:** SAM3 box (highest-score, roughly square) -> square crop (window shifted inside the image, never clipped) -> resize to 256 with `INTER_AREA`. Reference is warped onto each frame by ORB-homography/identity init + ECC on band-passed images; accept if NCC >= 0.5. GT string = decode of the original full-res reference.
- **Split:** temporal per session, last 15% test, 5-frame gap. (Brief: unseen-QR holdout once >= 4 QR ids.)
- **Dataset mix:** real (frame, aligned ref); 50% of samples replace the input by `synthetic_degrade(target)` (motion kernel 3-25 px, defocus sigma 0.5-2, gamma/contrast/brightness, noise).
- **Model:** `UNet2DModel`, in=6 (noisy target + condition concat), out=3, channels (32,64,128), layers_per_block 2, one attention level. `DDPMScheduler`/`DDIMScheduler`, `squaredcos_cap_v2`, v-prediction, `trailing` spacing, T=100, 10 sampling steps. AdamW 2e-4, grad-clip 1, EMA 0.995.
- **Loss:** `MSE(v) + lambda * mean_over_batch(alpha_bar * PatchL1SSIM(x0_pred, x0))`; tiles 16x16, per-tile L1 + (1 - SSIM with tile as one window), optional log-MSE (patch-PSNR) term, random tile offset.
- **Eval:** strict decode (pyzbar then OpenCV), PSNR/SSIM vs aligned target, blur tertiles, ms/frame.
- Observations: the SSIM term with a whole 16x16 tile as one window is essentially a per-tile mean/variance/covariance statistic. The loss is pixel-space and ignores that the target is binary. The "patch = module grid" motivation does not hold in the code: at 256 px a module is ~3.5 px, so a 16 px tile covers ~4.5 modules at an arbitrary offset, not aligned to the grid.

---

## 7. Research ideation: where the real novelty can come from

### 7.1 Why the current formulation is weak (first principles) [HYPOTHESIS]
1. **Memorisation is structural, not a tuning issue.** With one target image, the optimum of the conditional model is "emit that image". More regularisation will not fix a dataset of size ~1. The only real fix is a training distribution containing many distinct QR patterns.
2. **Wrong output space.** The goal is a *bit matrix that passes Reed-Solomon*, not an image with high PSNR. Pixel losses reward tone and edge sharpness, which decoders don't need.
3. **Wasted structure.** QR provides: fixed function patterns (finders, timing, alignment) for a given version; BCH-protected format info; RS error correction (7/15/25/30%); a built-in **verifier** (a decode either passes the checksum or not).
4. **Resolution mismatch.** 73 modules at 256 px is below a comfortable ~4-5 px/module. A module-space representation sidesteps this.

### 7.2 Candidate contributions (ordered by recommended priority)

**A. Procedural-QR training + real-calibrated degradation simulator (the data answer).** Generate unlimited QRs (random payloads, versions, EC levels, EMV-like long strings; use `qrcode`/`segno`, not only `cv2.QRCodeEncoder`). Degrade with a physical model: motion kernel, defocus, perspective, rolling shutter, illumination, JPEG, sensor noise. **Calibrate the distribution from real frames:** since we have an aligned clean reference per frame, the blur kernel can be *estimated* (regularised least squares for `frame ≈ k * ref`), giving an empirical kernel/illumination bank from real buggy motion. Real data then serves only for calibration and evaluation, never for memorisable supervision. This makes the overfitting fear moot and is itself a defensible "real-to-sim" contribution.

**B. Module-space (rectified) conditional diffusion with a structured output.** Rectify the SAM3 crop to a canonical square using the SAM3 mask quad (needs masks/polygons saved); estimate version/grid size N from finder patterns/timing; then diffuse over the **N x N module grid** (continuous "analog-bits" style or discrete diffusion), conditioned on the rectified blurry crop. Fixed function patterns are known -> clamp/inpaint them during sampling. Benefits: tiny model (a real "lightweight" claim: 73x73 vs 256x256x3), direct bit output, module-aligned loss (this is what makes the patch loss principled: tile = module), and no tone leakage. Risk: rectification/version estimation can fail on heavy blur; needs a fallback.

**C. Verifier-guided, RS-aware decoding of the diffusion posterior (strongest "not a mix-and-match" idea).** Sample K restorations; each gives per-module bit estimates. (i) Use the empirical per-module posterior as **soft information**: mark low-confidence modules as *erasures* for the RS decoder (erasures cost half the redundancy of errors, so this materially raises recoverable damage). (ii) Pick among candidates by the QR checksum (free ground-truth-less verifier) -> **test-time compute scaling curve: decode rate vs K**. This couples the generative model to the code's error-correcting structure, which neither Real-ESRGAN-style pipelines nor existing QR deblur networks do. Our prior-art check found no diffusion+RS-soft-decode work (not exhaustive; verify).

**D. Multi-frame evidence fusion.** The video gives many frames of the same code. Align/rectify neighbours, fuse per-module posteriors across frames (log-odds sum), then RS-decode. Handles the case where every single frame is hopeless but the sequence isn't. Fits "while travelling" better than single-frame, and the data already supports it (60 fps). Natural extension of B + C.

**E. Weak/secondary:** the patch loss alone (keep it only if it becomes module-aligned); one-step distillation (plain output distillation is not FideDiff-style consistency training and is not a novelty; use it only as an engineering step toward the "lightweight" claim).

### 7.3 Suggested thesis
> *Restoring QR codes as structured, self-verifying discrete objects: module-space diffusion trained purely on procedurally generated codes under a real-calibrated motion-blur model, decoded with posterior-driven Reed-Solomon erasure decoding, verifier-guided sampling, and (optionally) multi-frame fusion.*

SAM3 localisation (done) is contribution 1; the A+B+C(+D) formulation is contribution 2; headline metric is decode success stratified by blur and module-size, EgoQR-style.

### 7.4 Prior art (quick check, not a systematic survey; the next model should verify)
- No diffusion-based QR restoration found.
- EG-Restormer / "Adaptive Edge-Guided Dual-Network Framework for Fast QR Code Motion Deblurring" (arXiv 2510.12098, Oct 2025): non-diffusion, edge-prior Transformer + light net + blur-adaptive switch. **Must be cited and ideally compared against**; it notes a lack of real dynamic-scene QR datasets.
- Binary-constrained multi-frame SR for QR (Nagoya, ~2011); NEC QR-specific SR (2010); QRCNN/QRGAN (GCCE 2022); Levinson/Markel/Triantafillou SIIMS 2023 (binary prior recovers band-limited deblurring uniquely). These cover "binary prior" and "multi-frame", so novelty must rest on the **diffusion posterior + RS soft/erasure decoding + procedural-trained, real-calibrated** combination, not on "binary prior" alone.
- Also worth searching: discrete/bit diffusion (D3PM, Analog Bits), diffusion posterior sampling for inverse problems (DPS), soft-decision RS decoding, barcode/QR super-resolution 2024-2026, ZXing/WeChat QR decoder as strong baselines.

### 7.5 Risks to be honest about
- Information limit: if frames have < ~2 px/module or extreme blur, nothing recovers the payload; only multi-frame (D) might.
- Real-to-sim gap: synthetic degradations may not match real ones; calibration (A) mitigates but must be demonstrated on real frames.
- Test set is one physical QR. A paper-grade claim needs more distinct physical codes (see section 9).
- Strong classical baselines may close much of the gap (native-res decoding with multiple libraries, deconvolution with an estimated kernel, WeChat QR detector). Report them.

---

## 8. Prioritised plan

**Phase 0: measure reality (no training, CPU/local, hours).**
1. Decode rate of all 362 raw frames at **native resolution** with pyzbar, OpenCV (QRCodeDetector, WeChat if available), ZXing; both full-frame and SAM3-crop, with/without simple upscaling/sharpening. Stratify by blur (Laplacian variance) and QR pixel size.
2. Measure QR size in pixels and **pixels/module** across frames (from SAM3 boxes in `annotations/all_boxes.jsonl`). This bounds what any method can achieve.
3. Estimate real blur kernels from the aligned (frame, reference) pairs; summarise length/angle distributions (feeds A).
4. Re-run SAM3 saving **masks/quads** (cheap, ~15 min on Kaggle) for rectification.
5. Ask whether different buggies carry different QR layouts/payloads (open question, section 9).

**Phase 1: data engine.** Procedural QR generator + calibrated degradation simulator; held-out *synthetic* and *real* evaluation. Rectification + version/grid estimation, with failure rate reported.

**Phase 2: module-space diffusion (B).** Reuse diffusers; small UNet on the N x N (or fixed-size padded) grid with function-pattern clamping. Train from scratch on procedural data. Sanity gates before anything fancy: unseen-payload decode rate on synthetic; then real frames.

**Phase 3: RS-aware inference (C), then multi-frame (D).** Posterior sampling, erasure decoding, verifier-guided best-of-K; plot decode rate vs K and vs steps. Then fusion.

**Phase 4: evaluation + baselines.** No restoration; multi-library decode; bicubic/Lanczos + unsharp; Wiener/Richardson-Lucy with estimated kernel; EG-Restormer if code (leejianping/ADNet) runs; Real-ESRGAN (Edula et al.); a QR-trained YOLO on SAM3 pseudo-labels for the detector comparison. Stage-wise reporting (detect -> restore -> decode) by size and blur. Ablations (one change at a time): module-aligned vs generic patch loss, procedural-only vs +real, K, steps, soft vs hard RS, single vs multi-frame, v-pred vs eps.

**Phase 5: lightweight claim.** Params/FLOPs/latency vs the 256 px image UNet; optional one-step distillation; Kaggle-notebook demo.

---

## 9. Open questions for the owner
1. Do different buggies carry different QR payloads/layouts, or the same sticker? (Decides whether real data ever gives variety.)
2. Can you print and film **additional distinct dense QR codes** (different payloads/versions, known GT) from a moving buggy? Even 5-10 codes would move the evaluation from n=1 to something publishable and costs one afternoon.
3. Where is the latest `train_qr_diffusion.py`? Add it under `scripts/`.
4. Deadline and target venue? (Course deadline vs a workshop paper changes how much of C/D to attempt.)
5. Is mobile/on-device a firm requirement or a stretch goal? It affects how hard to push the "lightweight" framing.

---

## 10. Bugs and lessons already paid for (don't reintroduce)

1. Gated SAM3 needs the token passed explicitly; a dataset-scoped fine-grained token can't read it.
2. Use `python -m pip`, not `pip`.
3. Linear betas (1e-4..0.02) are for T=1000; at T=100 sampling-from-noise is out-of-distribution and outputs noise. Use cosine + v-prediction.
4. Don't resize cached images twice (a same-size PIL resize still degrades fine modules).
5. Downsample QR with `cv2.INTER_AREA`; fixed-size decoding of dense QR is non-monotonic in size.
6. OpenCV QR detector is weak on dense codes; use pyzbar first (needs system libzbar; falls back silently).
7. ORB matching is unreliable on QR texture/blur; use ECC on band-passed images and gate by NCC.
8. `BORDER_REPLICATE` over large warp regions teaches artifacts; require coverage, give the reference a big margin, build the coverage mask in the reference's frame.
9. A frame crop is axis-aligned; for rotated frames it shows more margin than a tight reference crop (naive 95% coverage drops almost everything).
10. Random train/test split leaks (adjacent frames are 0.05 s apart): temporal block split with a gap.
11. GT string must come from the full-resolution reference, not the downscaled target.
12. Patch loss on x0 is ill-conditioned at high noise and `clamp` kills gradients: use v-pred and weight by alpha_bar.
13. `DataParallel` can't gather diffusers' output object: wrap the UNet to return a tensor (`UNetForDP`).
14. Never push checkpoints to the dataset repo.
15. Jupyter prints the last expression: use `_ = evaluate(...)`.
16. Windows console `cp1252` raised a `UnicodeEncodeError` in a benchmark notebook cell: avoid non-ASCII in print/log strings on Windows.

---

## 11. Known limitations to state in the report
One physical QR / one session so far; alignment is best-effort (ECC + NCC gate) and only synthetically validated; synthetic QRs are cleaner than real stickers; decode at 256 px is partly resolution-limited; SAM3 is slow (~2.5 s/image) and not tight-box validated; "SAM3 > YOLO" is currently SAM3 > OpenCV; a one-step student would be distillation, not consistency training.

## 12. Housekeeping backlog
Move the three notebook scripts into `scripts/` with local+Kaggle path handling; real README; keep `.env`, `data/`, `venv/` gitignored (they are); consider `requirements.txt`; add a small CPU smoke test (tiny image, 1 batch, 2 DDIM steps) before any Kaggle run; synthetic-data test harness (generate QRs -> warp/blur -> fake `all_boxes.jsonl` -> `prepare_training_pairs`).
