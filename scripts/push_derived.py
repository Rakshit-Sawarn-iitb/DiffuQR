"""
push_derived.py

Pushes DERIVED files that Kaggle training/eval needs to the Hugging Face
dataset repo, in ONE commit:

    data/calibration/kernels.npz              -> calibration/kernels.npz
    data/calibration/<qr_id>_modules.npy      -> calibration/<qr_id>_modules.npy
    data/calibration/<qr_id>_qrmeta.json      -> calibration/<qr_id>_qrmeta.json
    data/modspace/real_<session>.npz          -> modspace/real_<session>.npz

Synthetic training data (data/modspace/<tag>/) is NOT pushed: it is large
(~0.3 MB/sample, incompressible) and is regenerated on Kaggle from the code
+ kernel bank. Checkpoints never go here (model repo only).

Note: <qr_id>_qrmeta.json contains the decoded payload; keep the repo private.

Usage:
    python -m scripts.push_derived --dry-run
    python -m scripts.push_derived

On Kaggle, fetch only these:
    snapshot_download(repo_id, repo_type="dataset", token=...,
                      allow_patterns=["calibration/*", "modspace/real_*.npz"])
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from diffuqr.config import CALIBRATION_DIR, DATA_DIR, HF_REPO_ID, HF_TOKEN


def collect() -> list:
    files = []
    for p in sorted(CALIBRATION_DIR.glob("kernels.npz")):
        files.append((p, f"calibration/{p.name}"))
    for pattern in ("*_modules.npy", "*_qrmeta.json"):
        for p in sorted(CALIBRATION_DIR.glob(pattern)):
            files.append((p, f"calibration/{p.name}"))
    for p in sorted((DATA_DIR / "modspace").glob("real_*.npz")):
        files.append((p, f"modspace/{p.name}"))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--token", default=None, help="Override HF_TOKEN from .env")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded; send nothing")
    args = parser.parse_args()

    files = collect()
    if not files:
        sys.exit("Nothing to push: run phase0_kernel_estimation, phase0_reference_version and "
                 "phase2_build_module_dataset real first.")
    total = sum(p.stat().st_size for p, _ in files)
    print(f"Target dataset repo: {args.repo_id}")
    for p, dst in files:
        print(f"  {p.relative_to(DATA_DIR.parent)}  ->  {dst}  ({p.stat().st_size / 1e6:.2f} MB)")
    print(f"Total: {total / 1e6:.1f} MB")
    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return

    token = args.token or HF_TOKEN
    if not token:
        sys.exit("No Hugging Face token found. Set HF_TOKEN in .env or pass --token.")
    api = HfApi(token=token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if not getattr(info, "private", False):
        sys.exit(f"{args.repo_id} is PUBLIC. Refusing to push derived files (qrmeta contains the payload).")
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(p)) for p, dst in files]
    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                      commit_message=f"Add derived calibration + real module-space eval ({len(files)} files)")
    print(f"Pushed {len(files)} files to {args.repo_id}")


if __name__ == "__main__":
    main()
