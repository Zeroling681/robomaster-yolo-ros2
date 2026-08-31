"""List camera indices that OpenCV can open on Windows.

Use this before a live YOLO run when USB cameras have been reconnected or their
index has changed.
"""

from __future__ import annotations

import argparse

import cv2


def try_open(index: int) -> tuple[bool, str]:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        return False, "unavailable"
    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not ok or frame is None:
        return False, "opened but did not return a frame"
    return True, f"{width}x{height}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=6)
    args = parser.parse_args()
    if args.max_index < 0:
        raise ValueError("max-index must be non-negative")

    found = 0
    for index in range(args.max_index + 1):
        available, detail = try_open(index)
        if available:
            found += 1
            print(f"camera {index}: available ({detail})")
        else:
            print(f"camera {index}: {detail}")
    if not found:
        raise RuntimeError("No usable camera was found. Reconnect the USB camera and retry.")


if __name__ == "__main__":
    main()
