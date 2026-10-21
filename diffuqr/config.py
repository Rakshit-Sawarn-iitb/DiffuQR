"""
Project-wide paths, .env loading, and session-id parsing.

Anything that rarely changes lives here (or in .env) instead of being passed
as a CLI argument on every run.

.env example (project root):
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    HF_REPO_ID=your-team/buggy-qr-dataset
    EVERY_N_FRAMES=3
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Anchor everything to the project root, not the current working directory.
# diffuqr/config.py -> parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
SESSIONS_DIR = DATA_DIR / "sessions"
RAW_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CALIBRATION_DIR = DATA_DIR / "calibration"
SYNTH_DIR = DATA_DIR / "synth"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PHASE0_DIR = ARTIFACTS_DIR / "phase0"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-team/buggy-qr-dataset")
EVERY_N_FRAMES = int(os.environ.get("EVERY_N_FRAMES", "3"))


def reference_photo(qr_id: str) -> Path:
    """The ground-truth photo of `qr_id` under raw/reference_photos/ (.jpg, .jpeg or .png)."""
    hits = [p for p in sorted((RAW_DIR / "reference_photos").glob(f"{qr_id}.*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one reference photo for '{qr_id}', found {[p.name for p in hits]}")
    return hits[0]

# session_id = <qr_id>_<YYYYMMDD>_<take>   e.g. buggy03_20260918_01
#   qr_id : the physical QR code / vehicle (one ground-truth photo per qr_id)
#   date  : recording date
#   take  : 2-digit counter for multiple videos of the same QR on the same day
_SESSION_RE = re.compile(r"^(?P<qr_id>[a-z0-9]+)_(?P<date>\d{8})_(?P<take>\d{2})$")


def parse_session_id(session_id: str) -> dict:
    """Validate a session id and return its parts: {'qr_id', 'date', 'take'}."""
    m = _SESSION_RE.match(session_id)
    if not m:
        raise ValueError(
            f"Invalid session id '{session_id}'. Expected <qr_id>_<YYYYMMDD>_<NN>, "
            "lowercase letters/digits only in qr_id, e.g. buggy03_20260918_01"
        )
    return m.groupdict()
