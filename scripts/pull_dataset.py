"""
pull_dataset.py

Downloads/updates the local `data/` directory from the Hugging Face dataset repo.
Creates `data/` if it doesn't exist; only downloads new/changed files if it does.
(Files deleted on the Hub are NOT deleted locally.)

Usage:
    python scripts/pull_dataset.py
    python scripts/pull_dataset.py --skip-videos      # frames + photos + manifests only

HF token is read from HF_TOKEN in the .env file at the project root.

Requires:
    pip install huggingface_hub python-dotenv
"""

import argparse
import sys

from huggingface_hub import snapshot_download

from config import DATA_DIR, HF_REPO_ID, HF_TOKEN

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", default=HF_REPO_ID)
parser.add_argument("--local-dir", default=str(DATA_DIR))
parser.add_argument("--skip-videos", action="store_true", help="Don't download the large raw videos")
parser.add_argument("--token", default=None, help="Override HF_TOKEN from .env")
args = parser.parse_args()

token = args.token or HF_TOKEN
if not token:
    sys.exit("No Hugging Face token found. Set HF_TOKEN in .env or pass --token.")

snapshot_download(
    repo_id=args.repo_id,
    repo_type="dataset",
    local_dir=args.local_dir,
    token=token,
    ignore_patterns=["raw/videos/*"] if args.skip_videos else None,
)

print(f"{args.local_dir} is now up to date with {args.repo_id}")