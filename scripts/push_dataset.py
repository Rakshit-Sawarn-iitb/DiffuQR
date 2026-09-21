"""
push_dataset.py

Extracts frames from a video, then pushes the video, the ground-truth
reference photo, the frames and a session manifest to a Hugging Face
dataset repo in ONE commit.

Repo layout produced:
    raw/reference_photos/{qr_id}.jpg          one ground truth per physical QR
    raw/videos/{session_id}.mp4               one per recording
    frames/{session_id}/{session_id}_fNNNNNN.jpg
    frames/{session_id}/metadata.jsonl        frame -> session, qr_id, reference photo
    sessions/{session_id}.json                manifest (hashes, fps, counts, ...)

Usage:
    python scripts/push_dataset.py \
        --video data/raw/videos/IMG_1534.mp4 \
        --photo data/raw/reference_photos/IMG_1533.jpg \
        --session-id buggy03_20260918_01

HF token is read from HF_TOKEN in the .env file at the project root.

Requires:
    pip install huggingface_hub opencv-python python-dotenv
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
)

from config import DATA_DIR, EVERY_N_FRAMES, FRAMES_DIR, HF_REPO_ID, HF_TOKEN, parse_session_id
from extract_frames import extract_frames


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the raw video file")
    parser.add_argument("--photo", required=True, help="Path to the ground-truth reference photo")
    parser.add_argument("--session-id", required=True, help="<qr_id>_<YYYYMMDD>_<NN>, e.g. buggy03_20260918_01")
    parser.add_argument("--qr-payload", default=None, help="Optional: the decoded text of the QR code (ground-truth label)")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--every-n-frames", type=int, default=EVERY_N_FRAMES)
    parser.add_argument("--overwrite", action="store_true", help="Replace this session if it already exists in the repo")
    parser.add_argument("--replace-photo", action="store_true", help="Replace the reference photo if this qr_id already has one")
    parser.add_argument("--token", default=None, help="Override HF_TOKEN from .env")
    args = parser.parse_args()

    video_path, photo_path = Path(args.video), Path(args.photo)
    for p in (video_path, photo_path):
        if not p.is_file():
            sys.exit(f"File not found: {p}")

    sid = args.session_id
    qr_id = parse_session_id(sid)["qr_id"]

    token = args.token or HF_TOKEN
    if not token:
        sys.exit("No Hugging Face token found. Set HF_TOKEN in .env or pass --token.")

    video_repo = f"raw/videos/{sid}{video_path.suffix.lower()}"
    photo_repo = f"raw/reference_photos/{qr_id}{photo_path.suffix.lower()}"
    frames_repo = f"frames/{sid}"
    manifest_repo = f"sessions/{sid}.json"

    # --- Check what's already in the repo BEFORE doing any work ---------------
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))

    old_frames = {f for f in existing if f.startswith(frames_repo + "/")}
    session_exists = bool(old_frames) or manifest_repo in existing
    if session_exists and not args.overwrite:
        sys.exit(f"Session '{sid}' already exists in {args.repo_id}. Use a new take number or pass --overwrite.")

    photo_exists = photo_repo in existing
    upload_photo = (not photo_exists) or args.replace_photo
    if photo_exists and not args.replace_photo:
        print(f"Reference photo for '{qr_id}' already in repo -> keeping existing ({photo_repo}). "
              "Use --replace-photo to override.")

    # --- Extract frames (also writes metadata.jsonl) ---------------------------
    frames_dir, info = extract_frames(
        video_path,
        sid,
        out_root=FRAMES_DIR,
        every_n_frames=args.every_n_frames,
        extra={"reference_photo": photo_repo, "video": video_repo, "qr_payload": args.qr_payload},
    )

    # --- Session manifest -------------------------------------------------------
    manifest = {
        "session_id": sid,
        "qr_id": qr_id,
        "qr_payload": args.qr_payload,
        "video": {
            "path": video_repo,
            "original_filename": video_path.name,
            "sha256": sha256_of(video_path),
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "total_frames": info["total_frames_read"],
        },
        "reference_photo": {
            "path": photo_repo,
            "original_filename": photo_path.name,
            "sha256": sha256_of(photo_path),
        },
        "every_n_frames": info["every_n_frames"],
        "frames_saved": info["saved"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_local = DATA_DIR / "sessions" / f"{sid}.json"
    manifest_local.parent.mkdir(parents=True, exist_ok=True)
    manifest_local.write_text(json.dumps(manifest, indent=2))

    # --- Build ONE commit: video + photo + manifest + frames --------------------
    ops = [
        CommitOperationAdd(path_in_repo=video_repo, path_or_fileobj=str(video_path)),
        CommitOperationAdd(path_in_repo=manifest_repo, path_or_fileobj=str(manifest_local)),
    ]
    if upload_photo:
        ops.append(CommitOperationAdd(path_in_repo=photo_repo, path_or_fileobj=str(photo_path)))

    new_frame_paths = set()
    for f in sorted(frames_dir.iterdir()):
        if f.is_file():
            repo_path = f"{frames_repo}/{f.name}"
            new_frame_paths.add(repo_path)
            ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(f)))

    # On --overwrite, remove remote frames that no longer exist locally
    for stale in old_frames - new_frame_paths:
        ops.append(CommitOperationDelete(path_in_repo=stale))

    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Add session {sid} ({info['saved']} frames)",
    )

    print(f"Pushed video, photo, manifest and {info['saved']} frames for session '{sid}' to {args.repo_id}")


if __name__ == "__main__":
    main()