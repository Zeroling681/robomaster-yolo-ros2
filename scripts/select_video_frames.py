from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    timestamp: float
    frame: np.ndarray
    sharpness: float
    brightness: float
    perceptual_hash: int


def difference_hash(gray: np.ndarray) -> int:
    reduced = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = reduced[:, 1:] > reduced[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), byteorder="big")


def hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def make_candidate(frame: np.ndarray, timestamp: float) -> Candidate:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return Candidate(
        timestamp=timestamp,
        frame=frame,
        sharpness=sharpness,
        brightness=brightness,
        perceptual_hash=difference_hash(gray),
    )


def choose_frames(video: dict[str, Any], target: int) -> list[Candidate]:
    capture = cv2.VideoCapture(video["path"])
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video['path']}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    margin = min(0.6, duration * 0.02)
    boundaries = np.linspace(margin, duration - margin, target + 1)
    sample_stride = max(1, round(fps / 8))
    bins: list[list[Candidate]] = [[] for _ in range(target)]

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        timestamp = frame_index / fps
        frame_index += 1
        if frame_index % sample_stride != 0:
            continue
        if timestamp < margin or timestamp >= duration - margin:
            continue
        bin_index = min(
            target - 1,
            int((timestamp - margin) / (duration - 2 * margin) * target),
        )
        bins[bin_index].append(make_candidate(frame, timestamp))

    capture.release()
    selected: list[Candidate] = []
    for bin_index, (start, end) in enumerate(pairwise(boundaries)):
        available = bins[bin_index]
        if not available:
            raise RuntimeError(f"时间段 {start:.2f}-{end:.2f}s 无法解码")

        available.sort(
            key=lambda item: (
                20.0 <= item.brightness <= 235.0,
                item.sharpness,
            ),
            reverse=True,
        )
        distinct = [
            item
            for item in available
            if all(
                hash_distance(item.perceptual_hash, prior.perceptual_hash) >= 5
                for prior in selected
            )
        ]
        selected.append(distinct[0] if distinct else available[0])
    return selected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-video", type=int, default=32)
    args = parser.parse_args()

    videos = json.loads(args.manifest.read_text(encoding="utf-8"))
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if any(images_dir.glob("*.jpg")):
        raise RuntimeError(f"输出目录已有图片，拒绝覆盖: {images_dir}")

    rows: list[dict[str, str]] = []
    for video in videos:
        selected = choose_frames(video, args.per_video)
        for index, candidate in enumerate(selected, start=1):
            timestamp_ms = round(candidate.timestamp * 1000)
            filename = f"{video['id']}_{index:03d}_t{timestamp_ms:06d}.jpg"
            output_path = images_dir / filename
            if not cv2.imwrite(str(output_path), candidate.frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"写入失败: {output_path}")
            rows.append(
                {
                    "filename": filename,
                    "class_name": video["class_name"],
                    "source_video": video["id"],
                    "timestamp_seconds": f"{candidate.timestamp:.3f}",
                    "sharpness": f"{candidate.sharpness:.2f}",
                    "brightness": f"{candidate.brightness:.2f}",
                    "sha256": file_sha256(output_path),
                }
            )
        print(f"{video['id']}: selected={len(selected)}")

    metadata_path = args.output / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"total={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
