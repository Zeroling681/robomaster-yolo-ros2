"""Extract review frames from new videos and create cautious YOLO prelabels.

The prelabels are only an editing aid. Every exported image must be checked in
X-AnyLabeling before it is used for training, especially when the confidence
is low or the expected class is not detected.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_video(value: str) -> tuple[str, Path]:
    try:
        label, path_text = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("video must use LABEL=PATH") from exc
    if label not in {"mouse", "cup"}:
        raise argparse.ArgumentTypeError("label must be mouse or cup")
    return label, Path(path_text)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        action="append",
        type=parse_video,
        required=True,
        help="Video mapping, for example cup=/mnt/e/data/cup.mp4 (repeatable)",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_work" / "video_review_v1",
    )
    parser.add_argument("--frames-per-video", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.05)
    return parser.parse_args()


def choose_expected_box(result, class_id: int) -> tuple[list[float] | None, float]:
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    candidates = [
        (float(confidence), [float(value) for value in box])
        for box, cls, confidence in zip(boxes, classes, confidences)
        if int(cls) == class_id
    ]
    if not candidates:
        return None, 0.0
    confidence, box = max(candidates, key=lambda item: item[0])
    return box, confidence


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def yolo_line(class_id: int, box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    images_dir = output / "images"
    labels_dir = output / "prelabels_yolo"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.frames_per_video < 1:
        raise ValueError("frames-per-video must be positive")

    model = YOLO(str(args.model.resolve()))
    rows: list[dict[str, str]] = []
    for video_number, (expected_label, video_path) in enumerate(args.video):
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        indices = [
            round(index * (frame_count - 1) / max(args.frames_per_video - 1, 1))
            for index in range(min(args.frames_per_video, frame_count))
        ]
        class_id = {"mouse": 0, "cup": 1}[expected_label]
        for sample_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            result = next(
                model.predict(
                    source=image,
                    imgsz=640,
                    conf=args.confidence,
                    device=0,
                    verbose=False,
                    stream=True,
                )
            )
            height, width = image.shape[:2]
            box, confidence = choose_expected_box(result, class_id)
            if box is not None:
                box = clamp_box(box, width, height)
            stem = f"{expected_label}_video{video_number + 1:02d}_{sample_index:04d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {image_path}")
            label_path.write_text(
                (yolo_line(class_id, box, width, height) + "\n") if box else "",
                encoding="utf-8",
            )
            rows.append(
                {
                    "filename": image_path.name,
                    "source_video": str(video_path),
                    "expected_class": expected_label,
                    "frame_index": str(frame_index),
                    "timestamp_seconds": f"{frame_index / fps:.3f}",
                    "prelabel_confidence": f"{confidence:.6f}",
                    "review_required": "yes" if box is None or confidence < 0.50 else "yes",
                    "prelabel_status": "box_found" if box else "missing_expected_box",
                }
            )
        capture.release()

    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"review_images={len(rows)}")
    print(f"output={output}")
    print("PRELABEL_REVIEW_REQUIRED=YES")


if __name__ == "__main__":
    main()
