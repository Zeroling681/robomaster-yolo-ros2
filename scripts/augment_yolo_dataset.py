"""Expand the reviewed YOLO dataset with deterministic image augmentations.

Validation and test images are copied without augmentation. Only the training
split is expanded, so augmented copies cannot leak into evaluation.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "dataset_work" / "yolo_dataset_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_work" / "yolo_dataset_v3_augmented",
    )
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def read_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label: {label_path}")
        class_id, cx, cy, box_width, box_height = map(float, values)
        boxes.append(
            Box(
                class_id=int(class_id),
                x1=(cx - box_width / 2) * width,
                y1=(cy - box_height / 2) * height,
                x2=(cx + box_width / 2) * width,
                y2=(cy + box_height / 2) * height,
            )
        )
    if not boxes:
        raise ValueError(f"Empty label file cannot be augmented: {label_path}")
    return boxes


def write_boxes(label_path: Path, boxes: list[Box], width: int, height: int) -> None:
    lines: list[str] = []
    for box in boxes:
        x1 = max(0.0, min(float(width), box.x1))
        y1 = max(0.0, min(float(height), box.y1))
        x2 = max(0.0, min(float(width), box.x2))
        y2 = max(0.0, min(float(height), box.y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        raise ValueError(f"Augmentation removed every box: {label_path}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rotate(image: np.ndarray, boxes: list[Box], angle: float) -> tuple[np.ndarray, list[Box]]:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed: list[Box] = []
    for box in boxes:
        corners = np.array(
            [[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]],
            dtype=np.float32,
        )
        points = np.column_stack((corners, np.ones(4, dtype=np.float32))) @ matrix.T
        transformed.append(
            Box(box.class_id, points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
        )
    return rotated, transformed


def horizontal_flip(image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[Box]]:
    height, width = image.shape[:2]
    flipped = cv2.flip(image, 1)
    transformed = [Box(box.class_id, width - box.x2, box.y1, width - box.x1, box.y2) for box in boxes]
    return flipped, transformed


def apply_augmentation(
    image: np.ndarray,
    boxes: list[Box],
    rng: random.Random,
) -> tuple[np.ndarray, list[Box], list[str]]:
    result = image.copy()
    result_boxes = [Box(box.class_id, box.x1, box.y1, box.x2, box.y2) for box in boxes]
    operations: list[str] = []

    if rng.random() < 0.70:
        angle = rng.uniform(-12.0, 12.0)
        result, result_boxes = rotate(result, result_boxes, angle)
        operations.append(f"rotate={angle:.1f}")
    if rng.random() < 0.35:
        result, result_boxes = horizontal_flip(result, result_boxes)
        operations.append("hflip")
    if rng.random() < 0.45:
        kernel = rng.choice([3, 5, 7])
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)
        operations.append(f"gaussian_blur={kernel}")
    if rng.random() < 0.35:
        kernel_size = rng.choice([5, 7, 9])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        result = cv2.filter2D(result, -1, kernel)
        operations.append(f"motion_blur={kernel_size}")
    if rng.random() < 0.65:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.randint(-25, 25)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        operations.append(f"brightness_contrast={alpha:.2f},{beta}")
    if rng.random() < 0.30:
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 5, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        operations.append("gaussian_noise")
    if rng.random() < 0.35:
        height, width = result.shape[:2]
        for _ in range(rng.randint(1, 2)):
            box_width = rng.randint(max(10, width // 12), max(11, width // 4))
            box_height = rng.randint(max(10, height // 12), max(11, height // 4))
            x1 = rng.randint(0, max(0, width - box_width))
            y1 = rng.randint(0, max(0, height - box_height))
            color = np.array([rng.randrange(256) for _ in range(3)], dtype=np.uint8)
            result[y1 : y1 + box_height, x1 : x1 + box_width] = color
        operations.append("random_occlusion")
    return result, result_boxes, operations


def image_records(source: Path) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for split in ("train", "val", "test"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append((split, image_path, label_dir / f"{image_path.stem}.txt"))
    return records


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.target_total < 1:
        raise ValueError("target-total must be positive")

    records = image_records(source)
    if not records:
        raise ValueError(f"No images found in {source}")
    if args.target_total < len(records):
        raise ValueError(f"target-total={args.target_total} is below current count={len(records)}")
    splits = {split: [record for record in records if record[0] == split] for split in ("train", "val", "test")}
    rng = random.Random(args.seed)
    manifest: list[dict[str, str]] = []

    for split, split_records in splits.items():
        for _, image_path, label_path in split_records:
            destination_image = output / "images" / split / image_path.name
            destination_label = output / "labels" / split / label_path.name
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)
            manifest.append({"filename": image_path.name, "split": split, "source": str(image_path), "augmentation": "original"})

    extra_count = args.target_total - len(records)
    train_records = splits["train"]
    for index in range(extra_count):
        _, source_image, source_label = train_records[index % len(train_records)]
        image = cv2.imread(str(source_image))
        if image is None:
            raise RuntimeError(f"Could not read {source_image}")
        height, width = image.shape[:2]
        boxes = read_boxes(source_label, width, height)
        augmented, augmented_boxes, operations = apply_augmentation(image, boxes, rng)
        stem = f"aug_{index + 1:04d}_{source_image.stem}"
        destination_image = output / "images" / "train" / f"{stem}.jpg"
        destination_label = output / "labels" / "train" / f"{stem}.txt"
        if not cv2.imwrite(str(destination_image), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {destination_image}")
        write_boxes(destination_label, augmented_boxes, width, height)
        manifest.append({"filename": destination_image.name, "split": "train", "source": str(source_image), "augmentation": ";".join(operations)})

    shutil.copy2(source / "classes.txt", output / "classes.txt")
    shutil.copy2(source / "dataset.yaml", output / "dataset.yaml")
    with (output / "augmentation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"original_images={len(records)}")
    print(f"synthetic_train_images={extra_count}")
    print(f"total_images={len(manifest)}")
    print(f"output={output}")
    print("AUGMENTATION_DATASET=READY")


if __name__ == "__main__":
    main()
