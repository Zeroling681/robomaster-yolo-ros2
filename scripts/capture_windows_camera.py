"""Capture a short Windows camera clip for WSL YOLO testing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "pc_camera_input.avi",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open Windows camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video {args.output}")

    started = time.perf_counter()
    frames = 0
    try:
        while time.perf_counter() - started < args.seconds:
            ok, frame = capture.read()
            if not ok:
                continue
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    print(f"camera_index={args.camera}")
    print(f"resolution={width}x{height}")
    print(f"frames={frames}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
