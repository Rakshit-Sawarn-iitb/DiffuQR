# Phase 4 Probe - Multi-frame Fusion (learning-free), 2026-10-16

Script: `scripts/phase4_fusion_probe.py`. Full table:
`artifacts/phase4/fusion_probe_summary.md`. One session (buggy01, 357 frames).
Oracle geometry (reference alignment) and oracle version/EC/mask.

Method: per frame, signed module evidence z = (local threshold - module
mean) / robust scale; fuse by a weighted sum over a window of +-w saved
frames (+-0.05w s), weight from fixed-pattern agreement (no payload GT
needed); soft RS decode with 4 check symbols reserved. Sanity: at w=0
fused == single-frame and matches Phase 0.4 (1% / 50% / 85%).

## Result 1 - blurry evidence does not add up to a decode

Fusing ONLY blurriest-tertile frames:

| window | frames | median module error | decodes |
|---|---|---|---|
| single | 1 | 25.7% | 1% |
| +-0.15 s | up to 7 | 17.0% | 0% |
| +-0.40 s | up to 17 | 12.8% | 0% |

The RS cliff for v11-L is ~1-2% module error. Averaging 17 blurry frames
only halves the error: the errors are systematic (deterministic smear
from similar motion in consecutive frames), not independent noise, so
averaging cannot remove them. **Naive evidence fusion (C4) does not solve
the hard subset.**

## Result 2 - naive fusion is worse than "try every frame"

Windows that may include sharper neighbours (blurriest-tertile targets):

| window | any single frame decodes (our grid reader) | weighted fusion decodes | plain pyzbar/OpenCV, any frame (deployable) |
|---|---|---|---|
| +-0.05 s | 27% | 17% | 16% |
| +-0.15 s | 59% | 34% | 34% |
| +-0.25 s | 79% | 50% | 50% |
| +-0.40 s | 97% | 68% | 64% |

Fusion dilutes the one good frame with bad ones; it beat "try every
frame" on only 0-3 targets per setting. Wrong decodes: 0.

## Result 3 - the trivial video baseline is strong

With no restoration and no reference, plain pyzbar/OpenCV run on every
frame finds a decodable frame within +-0.25 s of a blurriest frame 50% of
the time (+-0.4 s: 64%). In this clip, a scanner that just keeps trying
succeeds quickly.

## What this means

1. Contributions C3 (erasures) and C4 (naive fusion) do not crack the
   blurriest frames. Only a method that genuinely inverts the blur can:
   module error must go from ~26% to <~2% from a single (or few) blurry
   frame(s). Classical deconvolution (Phase 5a) and averaging (here) both
   failed, which is the case for a learned, structure-aware restorer (C2).
2. The paper's evaluation must include the "try every frame" video
   baseline, or a reviewer will. A per-frame decode-rate gain on blurry
   frames is only meaningful in a setting where sharp frames are rare.
   Candidate honest metrics: per-frame decode rate on the blurriest
   tertile, and **time-to-first-decode** (how much video is needed before
   the payload is read).
3. This clip has sharp frames every fraction of a second. Whether the
   single-frame problem matters in practice depends on clips with long
   all-blurry stretches (rougher roads, faster buggies, worse light),
   which the current data does not contain.

## Caveats

- One physical QR, one 18 s clip.
- Oracle geometry; misaligned frames (33/119 blurriest) contribute nothing.
- Fusion was the simplest form (weighted sum of per-frame evidence). A
  smarter fusion (e.g. deconvolving jointly across frames with different
  blur directions) is not ruled out; this probe only rules out averaging.
