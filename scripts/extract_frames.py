"""
extract_frames.py

Splits a video into frames and writes a metadata.jsonl next to them.
Can be imported (extract_frames function) or run standalone.

Output layout:
    data/frames/{session_id}/{session_id}_f000000.jpg
    data/frames/{session_id}/{session_id}_f000003.jpg   <- number = ORIGINAL video frame index
    data/frames/{session_id}/metadata.jsonl              <- one JSON line per frame

Usage (standalone):
    python scripts/extract_frames.py --video data/raw/videos/IMG_1534.mp4 \
        --session-id buggy03_20260918_01
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

from config import EVERY_N_FRAMES, FRAMES_DIR, parse_session_id


def extract_frames(
    video_path,
    session_id: str,
    out_root=FRAMES_DIR,
    every_n_frames: int = EVERY_N_FRAMES,
    jpeg_quality: int = 95,
    extra: dict | None = None,
):
    """
    Extract every Nth frame of `video_path` into {out_root}/{session_id}/.

    - Filenames embed session_id and the original frame index, so a frame can
      always be traced back to its video and timestamp, even if moved around.
    - `extra` fields (e.g. reference_photo path) are copied into every
      metadata.jsonl row, linking each frame to its ground truth.
    - Blurry frames are intentionally NOT filtered out.

    Returns (frames_dir, info) where info has fps / size / counts.
    """
    qr_id = parse_session_id(session_id)["qr_id"]
    video_path = Path(video_path)
    frames_dir = Path(out_root) / session_id

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frames are derived data: start clean so re-runs with a different
    # every_n_frames never leave stale frames from a previous run behind.
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            name = f"{session_id}_f{frame_idx:06d}.jpg"
            cv2.imwrite(
                str(frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            records.append(
                {
                    "file_name": name,  # column name expected by HF "imagefolder" loader
                    "session_id": session_id,
                    "qr_id": qr_id,
                    "frame_idx": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 4) if fps else None,
                    **(extra or {}),
                }
            )
        frame_idx += 1
    cap.release()

    if not records:
        raise RuntimeError(f"No frames could be read from {video_path}")

    with open(frames_dir / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    info = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_read": frame_idx,
        "saved": len(records),
        "every_n_frames": every_n_frames,
    }
    print(f"Extracted {info['saved']} of {frame_idx} frames -> {frames_dir}")
    return frames_dir, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the raw video file")
    parser.add_argument("--session-id", required=True, help="e.g. buggy03_20260918_01")
    parser.add_argument("--out-root", default=str(FRAMES_DIR))
    parser.add_argument("--every-n-frames", type=int, default=EVERY_N_FRAMES)
    args = parser.parse_args()

    extract_frames(args.video, args.session_id, args.out_root, args.every_n_frames)