# DiffuQR: Repo Structure

Written 2026-10-15. Set up at the start of Phase 0.

## Layout

```
DiffuQR/
├── diffuqr/            # importable library (one source of truth for logic)
│   ├── __init__.py
│   ├── config.py       # paths, .env, session-id parsing
│   ├── qrgrid.py       # rectify quad, sample module grid, identify version/EC/mask
│                       # future: decode.py, blur.py, kernel.py, detect.py,
│                       #         rectify.py, degrade.py, synth.py, models.py
│
├── scripts/            # thin CLI entry points, one task per file
│   ├── __init__.py
│   ├── extract_frames.py / push_dataset.py / pull_dataset.py   (data pipeline)
│   └── phase0_*.py, phase1_*.py, ...                            (new code, phase-prefixed)
│
├── notebooks/          # Kaggle workspace
│   ├── segmentation.ipynb
│   ├── benchmarking-sam3-and-yolo.ipynb
│   └── training.ipynb
│
├── data/               # gitignored, INPUTS only
│   ├── raw/            raw/{reference_photos,videos}/
│   ├── frames/         frames/{session_id}/*.jpg + metadata.jsonl
│   ├── sessions/       sessions/{session_id}.json
│   ├── annotations/    SAM3 boxes/masks
│   ├── calibration/    Phase 0.3: kernels.npz, blur stats
│   └── synth/          Phase 1: procedurally generated QRs
│
├── artifacts/          # gitignored, OUTPUTS only (.gitkeep tracked)
│   ├── phase0/         measurement CSVs + plots
│   ├── figures/        paper-ready figures
│   └── checkpoints/    local training checkpoints (Kaggle still pushes to HF)
│
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── PLAN.md
│   ├── REPO_STRUCTURE.md         (this file)
│   └── phase0_*.md, phase1_*.md  (written as each phase lands)
│
├── pyproject.toml      # `python -m pip install -e .` makes `diffuqr` importable
├── requirements.txt    # concrete deps for a fresh venv or Kaggle cell
├── .gitignore          # ignores venv/, data/, artifacts/* (keeps .gitkeep)
└── .env                # gitignored, HF_TOKEN etc.
```

## Rules

1. **Library in `diffuqr/`, entry points in `scripts/`.** No logic lives in two
   places. If a script and a notebook both need the same code, it goes in
   `diffuqr/`.
2. **Scripts are phase-prefixed** (`phase0_decode_baseline.py` etc.) so a
   directory listing sorts by phase.
3. **`data/` = inputs, `artifacts/` = outputs.** Never mixed. Both gitignored.
4. **Notebooks stay in `notebooks/`.** Training stays on Kaggle. On Kaggle,
   `pip install git+https://github.com/Rakshit-Sawarn-iitb/DiffuQR` makes
   `diffuqr` importable; measurement scripts still run locally.
5. **No new top-level directories** without updating this file.

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e .[decode]      # core + Phase 0 decoders
python -m pip install -r requirements.txt   # alternatively, install everything flat
```

## Running scripts

From the project root:

```powershell
python -m scripts.phase0_decode_baseline --help
python -m scripts.extract_frames --video ... --session-id ...
```

`python -m scripts.foo` is preferred over `python scripts/foo.py` because the
module form sets `sys.path` to the project root, so `from diffuqr.config
import ...` resolves without an editable install.
