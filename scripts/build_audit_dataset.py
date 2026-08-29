"""Build one flat X-AnyLabeling audit set from the current YOLO dataset.

The current training set contains reviewed originals, video prelabels and
synthetic augmentations.  This script keeps those origins visible in one
manifest and adds every difficult-video frame that was previously excluded
because no prelabel was found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "mouse", 1: "cup"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", type=Path, default=root / "dataset_work" / "yolo_dataset_v4_combined")
    parser.add_argument("--review-set", type=Path, default=root / "dataset_work" / "video_review_v1")
    parser.add_argument("--reviewed-set", type=Path, default=root / "dataset_work" / "anylabeling_dataset")
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "audit_dataset")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    if not label_path.is_file():
        return shapes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        class_id = int(values[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unknown class id {class_id} at {label_path}:{line_number}")
        center_x, center_y, box_width, box_height = map(float, values[1:])
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"Out-of-range YOLO box at {label_path}:{line_number}")
        x1 = max(0.0, (center_x - box_width / 2.0) * width)
        y1 = max(0.0, (center_y - box_height / 2.0) * height)
        x2 = min(float(width), (center_x + box_width / 2.0) * width)
        y2 = min(float(height), (center_y + box_height / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate YOLO box at {label_path}:{line_number}")
        shapes.append(
            {
                "label": CLASS_NAMES[class_id],
                "shape_type": "rectangle",
                "flags": {},
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": None,
                "difficult": False,
                "attributes": {},
            }
        )
    return shapes


def write_xanylabeling(annotation_path: Path, image_name: str, width: int, height: int, shapes: list[dict[str, object]]) -> None:
    document = {
        "version": "4.0.3",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    annotation_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classify_origin(filename: str, reviewed_names: set[str], review_names: set[str]) -> tuple[str, str]:
    if filename.startswith("aug_"):
        return "synthetic_augmentation", "review_required"
    if filename in reviewed_names:
        return "human_reviewed_original", "recheck_recommended"
    if filename in review_names:
        return "difficult_video_prelabel", "review_required"
    return "current_training_unknown", "review_required"


def main() -> None:
    args = parse_args()
    training_set = args.training_set.resolve()
    review_set = args.review_set.resolve()
    reviewed_set = args.reviewed_set.resolve()
    output = args.output.resolve()
    images_output = output / "images"
    annotations_output = output / "annotations"
    images_output.mkdir(parents=True, exist_ok=True)
    annotations_output.mkdir(parents=True, exist_ok=True)
    if any(images_output.iterdir()) or any(annotations_output.iterdir()):
        raise FileExistsError(f"Audit output is not empty: {output}")

    reviewed_names = {path.name for path in (reviewed_set / "images").glob("*.jpg")}
    review_names = {path.name for path in (review_set / "images").glob("*.jpg")}
    review_metadata: dict[str, dict[str, str]] = {}
    with (review_set / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review_metadata[row["filename"]] = row

    records: list[dict[str, str]] = []
    copied_names: set[str] = set()
    class_counts: Counter[str] = Counter()
    empty_annotations = 0

    for split in ("train", "val", "test"):
        for image_path in sorted((training_set / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image_path.name in copied_names:
                raise ValueError(f"Duplicate filename in training set: {image_path.name}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            label_path = training_set / "labels" / split / f"{image_path.stem}.txt"
            shapes = read_yolo_boxes(label_path, width, height)
            for shape in shapes:
                class_counts[str(shape["label"])] += 1
            if not shapes:
                empty_annotations += 1
            shutil.copy2(image_path, images_output / image_path.name)
            write_xanylabeling(annotations_output / f"{image_path.stem}.json", image_path.name, width, height, shapes)
            origin, audit_status = classify_origin(image_path.name, reviewed_names, review_names)
            review_row = review_metadata.get(image_path.name, {})
            records.append(
                {
                    "filename": image_path.name,
                    "dataset_split": split,
                    "data_origin": origin,
                    "expected_class": review_row.get("expected_class", "derived_from_label"),
                    "annotation_origin": "yolo_training_label",
                    "box_count": str(len(shapes)),
                    "audit_status": audit_status,
                    "source_video": Path(review_row.get("source_video", "")).name,
                    "source_frame": review_row.get("frame_index", ""),
                    "timestamp_seconds": review_row.get("timestamp_seconds", ""),
                    "sha256": sha256(images_output / image_path.name),
                }
            )
            copied_names.add(image_path.name)

    # Include frames excluded from training because the model found no expected box.
    for image_path in sorted((review_set / "images").glob("*.jpg")):
        if image_path.name in copied_names:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        label_path = review_set / "prelabels_yolo" / f"{image_path.stem}.txt"
        shapes = read_yolo_boxes(label_path, width, height)
        for shape in shapes:
            class_counts[str(shape["label"])] += 1
        if not shapes:
            empty_annotations += 1
        shutil.copy2(image_path, images_output / image_path.name)
        write_xanylabeling(annotations_output / f"{image_path.stem}.json", image_path.name, width, height, shapes)
        review_row = review_metadata[image_path.name]
        records.append(
            {
                "filename": image_path.name,
                "dataset_split": "not_assigned",
                "data_origin": "difficult_video_excluded_frame",
                "expected_class": review_row["expected_class"],
                "annotation_origin": "empty_or_low_confidence_prelabel",
                "box_count": str(len(shapes)),
                "audit_status": "annotation_required" if not shapes else "review_required",
                "source_video": Path(review_row["source_video"]).name,
                "source_frame": review_row["frame_index"],
                "timestamp_seconds": review_row["timestamp_seconds"],
                "sha256": sha256(images_output / image_path.name),
            }
        )
        copied_names.add(image_path.name)

    with (output / "audit_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    origin_counts = Counter(row["data_origin"] for row in records)
    status_counts = Counter(row["audit_status"] for row in records)
    report = {
        "images": len(records),
        "annotations": len(records),
        "boxes_by_class": dict(sorted(class_counts.items())),
        "empty_annotations": empty_annotations,
        "origin_counts": dict(sorted(origin_counts.items())),
        "audit_status_counts": dict(sorted(status_counts.items())),
        "xanylabeling_images": str(images_output),
        "xanylabeling_output": str(annotations_output),
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
