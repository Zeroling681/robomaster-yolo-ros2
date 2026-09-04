"""Build the final v13 twenty-angle and horizontal-cup evidence sheets."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REVIEW = RESULTS / "v13_submission_review"
ANGLE_CSV = REVIEW / "v13_20_angle_results_with_horizontal_cup.csv"
ANGLE_FRAMES = REVIEW / "twenty_angle_evidence_final"


def read_frame(video_path: Path, timestamp_s: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {video_path.name} at {timestamp_s:.1f}s")
    return frame


def fit_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_caption(frame: np.ndarray, first: str, second: str, passed: bool) -> np.ndarray:
    caption = np.full((80, frame.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        caption,
        first,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        caption,
        second,
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 135, 20) if passed else (20, 20, 210),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([frame, caption])


def build_twenty_angle_sheet() -> None:
    with ANGLE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("test_id", "").isdigit()]
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 scored scenes, found {len(rows)}")

    ANGLE_FRAMES.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    for row in rows:
        timestamp = float(row["timestamp_s"])
        frame = fit_frame(read_frame(RESULTS / row["video"], timestamp))
        passed = row["result"] == "correct"
        test_id = int(row["test_id"])
        first = f"#{test_id:02d} {row['expected']} | {row['angle_description']}"
        second = f"{timestamp:.1f}s | {'PASS' if passed else 'MISS'} | {row['platform']}"
        tile = add_caption(frame, first, second, passed)
        frame_path = ANGLE_FRAMES / f"{test_id:02d}_{Path(row['video']).stem}_{timestamp:.1f}s.jpg"
        cv2.imwrite(str(frame_path), tile)
        tiles.append(tile)

    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, 20, 4)]
    output = REVIEW / "v13_20_angle_evidence_with_horizontal_cup.jpg"
    cv2.imwrite(str(output), np.vstack(rows))
    print(output)


def build_horizontal_cup_sheet() -> None:
    video = RESULTS / "v13_horizontal_cup_success_detected.avi"
    timestamps = [2.5, 3.5, 4.5, 5.0, 5.5, 6.5, 7.5, 8.5, 11.5, 12.5]
    tiles = []
    for timestamp in timestamps:
        frame = fit_frame(read_frame(video, timestamp), width=480, height=360)
        tile = add_caption(
            frame,
            f"Horizontal cup | {timestamp:.1f}s",
            "v13 ONNX | cup detected",
            True,
        )
        tiles.append(tile)

    rows = [np.hstack(tiles[index : index + 5]) for index in range(0, 10, 5)]
    output = REVIEW / "v13_horizontal_cup_success_detected_contact_sheet.jpg"
    cv2.imwrite(str(output), np.vstack(rows))
    print(output)


if __name__ == "__main__":
    build_twenty_angle_sheet()
    build_horizontal_cup_sheet()
