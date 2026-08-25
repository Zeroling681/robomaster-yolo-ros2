"""Validate X-AnyLabeling annotations and build a YOLO detection dataset.

The source images and JSON annotations are never modified. Near-identical,
same-class rectangles are de-duplicated only in the generated YOLO labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


CLASSES = ("mouse", "cup")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_index: int

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "dataset_work" / "anylabeling_dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset_work" / "yolo_dataset",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dedupe-iou", type=float, default=0.90)
    return parser.parse_args()


def intersection_over_union(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def allocate_counts(group_sizes: dict[str, int], total: int, ratio: float) -> dict[str, int]:
    """Use the largest-remainder method to distribute a split across videos."""
    exact = {group: size * ratio for group, size in group_sizes.items()}
    allocated = {group: math.floor(value) for group, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(group_sizes, key=lambda group: (-(exact[group] - allocated[group]), group))
    for group in order[:remaining]:
        allocated[group] += 1
    return allocated


def make_split(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Stratify by primary class and source video with deterministic shuffling."""
    result: dict[str, str] = {}
    by_class: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_class[row["class_name"]][row["source_video"]].append(row)

    for class_offset, class_name in enumerate(CLASSES):
        videos = by_class[class_name]
        class_total = sum(len(items) for items in videos.values())
        train_total = round(class_total * 0.70)
        val_total = round(class_total * 0.20)
        test_total = class_total - train_total - val_total

        # Python's rounding gives 66 for 95 * 0.70. Allocate the odd sample to
        # the larger class so the full dataset rounds to 70/20/10 exactly.
        if class_name == "cup" and class_total == 95:
            train_total, val_total, test_total = 67, 19, 9
        if class_name == "mouse" and class_total == 93:
            train_total, val_total, test_total = 65, 19, 9

        sizes = {video: len(items) for video, items in videos.items()}
        train_counts = allocate_counts(sizes, train_total, 0.70)
        remaining_sizes = {
            video: sizes[video] - train_counts[video]
            for video in videos
        }
        # Allocate validation samples from the remaining capacity, while
        # preserving the desired proportions across videos.
        val_counts = allocate_counts(sizes, val_total, 0.20)
        overflow = sum(max(0, val_counts[v] - remaining_sizes[v]) for v in videos)
        if overflow:
            raise ValueError("Could not allocate validation split")

        if sum(train_counts.values()) != train_total or sum(val_counts.values()) != val_total:
            raise AssertionError("Split allocation failed")

        for video_offset, (video, items) in enumerate(sorted(videos.items())):
            ordered = sorted(items, key=lambda row: (float(row["timestamp_seconds"]), row["filename"]))
            random.Random(seed + class_offset * 100 + video_offset).shuffle(ordered)
            train_end = train_counts[video]
            val_end = train_end + val_counts[video]
            for row in ordered[:train_end]:
                result[row["filename"]] = "train"
            for row in ordered[train_end:val_end]:
                result[row["filename"]] = "val"
            for row in ordered[val_end:]:
                result[row["filename"]] = "test"

        actual = Counter(result[row["filename"]] for row in rows if row["class_name"] == class_name)
        expected = {"train": train_total, "val": val_total, "test": test_total}
        if dict(actual) != expected:
            raise AssertionError(f"Unexpected {class_name} split: {dict(actual)} != {expected}")
    return result


def load_rectangle(shape: dict[str, Any], index: int) -> Box:
    if shape.get("shape_type") != "rectangle":
        raise ValueError(f"shape {index} is not a rectangle")
    points = shape.get("points") or []
    if len(points) < 2:
        raise ValueError(f"shape {index} has fewer than two points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return Box(
        label=str(shape.get("label", "")),
        x1=min(xs),
        y1=min(ys),
        x2=max(xs),
        y2=max(ys),
        source_index=index,
    )


def validate_and_clean(
    annotation_path: Path,
    image_path: Path,
    primary_class: str,
    dedupe_iou: float,
    report: dict[str, Any],
) -> tuple[list[Box], int, int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        actual_width, actual_height = image.size
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    if (width, height) != (actual_width, actual_height):
        raise ValueError(
            f"{annotation_path.name}: JSON dimensions {width}x{height} do not match "
            f"image dimensions {actual_width}x{actual_height}"
        )

    kept: list[Box] = []
    raw_count = 0
    for index, shape in enumerate(data.get("shapes") or []):
        raw_count += 1
        box = load_rectangle(shape, index)
        if box.label not in CLASS_TO_ID:
            raise ValueError(f"{annotation_path.name}: unknown class {box.label!r}")

        clipped = Box(
            label=box.label,
            x1=min(max(box.x1, 0.0), width),
            y1=min(max(box.y1, 0.0), height),
            x2=min(max(box.x2, 0.0), width),
            y2=min(max(box.y2, 0.0), height),
            source_index=box.source_index,
        )
        if clipped.area <= 0:
            raise ValueError(f"{annotation_path.name}: shape {index} has zero area")
        if clipped != box:
            report["clipped_boxes"].append(
                {"file": annotation_path.name, "shape_index": index, "label": box.label}
            )

        duplicate_of: Box | None = None
        duplicate_iou = 0.0
        for existing in kept:
            overlap = intersection_over_union(clipped, existing)
            if existing.label == clipped.label and overlap >= dedupe_iou:
                duplicate_of = existing
                duplicate_iou = overlap
                break
            if existing.label != clipped.label and overlap >= 0.50:
                report["cross_class_overlaps"].append(
                    {
                        "file": annotation_path.name,
                        "shape_indices": [existing.source_index, clipped.source_index],
                        "labels": [existing.label, clipped.label],
                        "iou": round(overlap, 6),
                    }
                )

        if duplicate_of is not None:
            report["removed_duplicate_boxes"].append(
                {
                    "file": annotation_path.name,
                    "label": clipped.label,
                    "kept_shape_index": duplicate_of.source_index,
                    "removed_shape_index": clipped.source_index,
                    "iou": round(duplicate_iou, 6),
                }
            )
            continue
        kept.append(clipped)

    if not kept:
        raise ValueError(f"{annotation_path.name}: no valid boxes")
    if primary_class not in {box.label for box in kept}:
        raise ValueError(f"{annotation_path.name}: missing primary class {primary_class!r}")
    return kept, width, height


def yolo_lines(boxes: list[Box], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for box in boxes:
        center_x = ((box.x1 + box.x2) / 2.0) / width
        center_y = ((box.y1 + box.y2) / 2.0) / height
        box_width = (box.x2 - box.x1) / width
        box_height = (box.y2 - box.y1) / height
        values = (center_x, center_y, box_width, box_height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise AssertionError(f"Invalid normalized box: {values}")
        lines.append(
            f"{CLASS_TO_ID[box.label]} "
            f"{center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    images_dir = source / "images"
    annotations_dir = source / "annotations_xlabel"
    metadata_path = source / "metadata.csv"

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new --output directory."
        )
    output.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata.csv is empty")

    images = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    metadata_names = {row["filename"] for row in rows}
    if set(images) != metadata_names:
        raise ValueError(
            f"Image/metadata mismatch: images_only={sorted(set(images) - metadata_names)}, "
            f"metadata_only={sorted(metadata_names - set(images))}"
        )

    split_by_filename = make_split(rows, args.seed)
    report: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "dedupe_iou": args.dedupe_iou,
        "image_count": len(images),
        "raw_box_count": 0,
        "exported_box_count": 0,
        "class_box_counts": Counter(),
        "split_image_counts": Counter(),
        "split_primary_class_counts": defaultdict(Counter),
        "clipped_boxes": [],
        "removed_duplicate_boxes": [],
        "cross_class_overlaps": [],
    }

    manifest_rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in sorted(rows, key=lambda item: item["filename"]):
        filename = row["filename"]
        image_path = images[filename]
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {annotation_path}")

        boxes, width, height = validate_and_clean(
            annotation_path,
            image_path,
            row["class_name"],
            args.dedupe_iou,
            report,
        )
        raw_shapes = len((json.loads(annotation_path.read_text(encoding="utf-8"))).get("shapes") or [])
        report["raw_box_count"] += raw_shapes
        report["exported_box_count"] += len(boxes)
        report["class_box_counts"].update(box.label for box in boxes)

        split = split_by_filename[filename]
        report["split_image_counts"][split] += 1
        report["split_primary_class_counts"][split][row["class_name"]] += 1
        shutil.copy2(image_path, output / "images" / split / filename)
        label_path = output / "labels" / split / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(yolo_lines(boxes, width, height)) + "\n", encoding="utf-8")
        manifest_rows.append({**row, "split": split, "box_count": str(len(boxes))})

    (output / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (output / "dataset.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: mouse\n"
        "  1: cup\n",
        encoding="utf-8",
    )

    manifest_fields = list(rows[0].keys()) + ["split", "box_count"]
    with (output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    serializable_report = dict(report)
    serializable_report["class_box_counts"] = dict(sorted(report["class_box_counts"].items()))
    serializable_report["split_image_counts"] = dict(report["split_image_counts"])
    serializable_report["split_primary_class_counts"] = {
        split: dict(counts) for split, counts in report["split_primary_class_counts"].items()
    }
    (output / "validation_report.json").write_text(
        json.dumps(serializable_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(serializable_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
