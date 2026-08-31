"""Capture evenly spaced camera frames for a manually labelled YOLO batch.

The output images are intentionally unlabelled. Open the directory in
X-AnyLabeling, annotate every visible mouse and cup, then import the matching
JSON files into the audited dataset pipeline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return capture


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset_work" / "incoming_camera_v9" / "images",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_fps <= 0:
        raise ValueError("seconds and sample-fps must both be positive")

    capture = open_camera(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / args.sample_fps
    started = time.perf_counter()
    next_sample = started
    saved = 0
    try:
        while time.perf_counter() - started < args.seconds:
            ok, frame = capture.read()
            if not ok:
                continue
            now = time.perf_counter()
            if now < next_sample:
                continue
            output = args.output_dir / f"camera{args.camera}_{saved:04d}.jpg"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Could not write {output}")
            saved += 1
            next_sample += interval
    finally:
        capture.release()

    print(f"camera_index={args.camera}")
    print(f"frames_saved={saved}")
    print(f"images={args.output_dir.resolve()}")
    print("next_step=annotate the images in X-AnyLabeling before importing them")


if __name__ == "__main__":
    main()
