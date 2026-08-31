"""Merge a reviewed external-camera batch into a new audited dataset version.

Run this only after all images in the batch have been checked in
X-AnyLabeling.  The script copies (or hard-links) sources; it never modifies
the preceding audited dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


CLASSES = {"mouse", "cup"}
FORMAL_SPLITS = {"train", "val", "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def validate_annotation(
    annotation: dict[str, Any], image_path: Path, filename: str
) -> tuple[list[str], Counter[str], int]:
    image = cv2.imread(str(image_path))
    if image is None:
        return [f"无法读取图片: {filename}"], Counter(), 0
    height, width = image.shape[:2]
    errors: list[str] = []
    classes: Counter[str] = Counter()
    clipped_count = 0
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
        errors.append(f"{filename}: JSON尺寸与图片不一致")
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, shape in enumerate(annotation.get("shapes") or []):
        label = str(shape.get("label", ""))
        points = shape.get("points") or []
        if label not in CLASSES:
            errors.append(f"{filename}: shape {index} 类别 {label!r} 无效")
            continue
        if shape.get("shape_type") != "rectangle" or len(points) < 2:
            errors.append(f"{filename}: shape {index} 不是有效矩形框")
            continue
        xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
        raw_x1, raw_x2, raw_y1, raw_y2 = min(xs), max(xs), min(ys), max(ys)
        x1, x2 = max(0.0, raw_x1), min(float(width), raw_x2)
        y1, y2 = max(0.0, raw_y1), min(float(height), raw_y2)
        if (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2):
            shape["points"] = [[x1, y1], [x2, y2]]
            clipped_count += 1
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{filename}: shape {index} 面积为零")
            continue
        boxes.append((label, (x1, y1, x2, y2)))
        classes[label] += 1
    for left_index, (left_label, left_box) in enumerate(boxes):
        for right_label, right_box in boxes[left_index + 1 :]:
            overlap = iou(left_box, right_box)
            if left_label == right_label and overlap >= 0.90:
                errors.append(f"{filename}: 同类重复框 IoU={overlap:.3f}")
            if left_label != right_label and overlap >= 0.50:
                errors.append(f"{filename}: 跨类重叠框 IoU={overlap:.3f}")
    return errors, classes, clipped_count


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset_v8")
    parser.add_argument(
        "--batch", type=Path, default=root / "dataset_work" / "camera_v9_annotation_batch"
    )
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "audit_dataset_v9")
    parser.add_argument(
        "--confirm-all-reviewed",
        action="store_true",
        help=(
            "Accept every batch JSON as human reviewed even if X-AnyLabeling did not "
            "persist checked=true. Use only after explicit human confirmation."
        ),
    )
    return parser.parse_args()


def json_path(folder: Path, image_name: str) -> Path:
    return folder / f"{Path(image_name).stem}.json"


def main() -> None:
    args = parse_args()
    audit, batch, output = args.audit.resolve(), args.batch.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")
    for required in (audit / "audit_manifest.csv", batch / "batch_manifest.csv"):
        if not required.is_file():
            raise FileNotFoundError(required)

    with (audit / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with (batch / "batch_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))
    if not source_rows or not batch_rows:
        raise ValueError("审计集或相机批次为空")
    if len({row["filename"] for row in batch_rows}) != len(batch_rows):
        raise ValueError("相机批次存在重复文件名")

    output_images = output / "images"
    output_annotations = output / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    merged_rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    clipped_boxes: list[dict[str, object]] = []
    confirmed_unchecked: list[str] = []
    source_names = {row["filename"] for row in source_rows}
    source_hashes = {row.get("sha256", "") for row in source_rows}

    def add_pair(row: dict[str, str], image_source: Path, annotation_source: Path) -> None:
        name = row["filename"]
        if not image_source.is_file() or not annotation_source.is_file():
            errors.append(f"缺少图片或标注: {name}")
            return
        image_destination = output_images / name
        annotation_destination = output_annotations / f"{Path(name).stem}.json"
        link_or_copy(image_source, image_destination)
        annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
        annotation["imagePath"] = name
        annotation["imageData"] = None
        annotation["checked"] = True
        pair_errors, pair_classes, clipped_count = validate_annotation(annotation, image_destination, name)
        errors.extend(pair_errors)
        if clipped_count:
            clipped_boxes.append({"filename": name, "count": clipped_count})
        annotation_destination.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row["box_count"] = str(sum(pair_classes.values()))
        labels = set(pair_classes)
        row["expected_class"] = "none" if not labels else sorted(labels)[0]
        row["audit_status"] = "human_checked" if labels else "human_checked_negative"
        row["sha256"] = sha256(image_destination)
        merged_rows.append(row)
        split_counts[row["dataset_split"]] += 1
        class_counts.update(pair_classes)

    for original in source_rows:
        row = dict(original)
        # v8 deliberately retains excluded/candidate images as an audit trail.
        # Keep those records unchanged; export_audited_yolo.py will exclude them
        # from the train/val/test materialization.
        add_pair(row, audit / "images" / row["filename"], json_path(audit / "annotations", row["filename"]))

    for batch_row in batch_rows:
        name = batch_row["filename"]
        image_source = batch / name
        annotation_source = json_path(batch, name)
        if name in source_names:
            errors.append(f"相机批次文件名冲突: {name}")
            continue
        if not image_source.is_file() or not annotation_source.is_file():
            errors.append(f"相机批次缺少文件: {name}")
            continue
        annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
        if annotation.get("checked") is not True:
            if not args.confirm_all_reviewed:
                errors.append(f"相机样本尚未在 AnyLabeling 中确认: {name}")
                continue
            confirmed_unchecked.append(name)
        image_hash = sha256(image_source)
        if image_hash in source_hashes:
            errors.append(f"相机批次与基础数据存在内容重复: {name}")
            continue
        row = {
            "filename": name,
            "dataset_split": "train",
            "data_origin": batch_row.get("source_video", "external_camera"),
            "expected_class": "none",
            "annotation_origin": "xanylabeling_human_review",
            "box_count": "0",
            "audit_status": "annotation_required",
            "source_video": batch_row.get("source_video", "external_camera_20260831"),
            "source_frame": name.rsplit("_t", 1)[-1].removesuffix(".jpg"),
            "timestamp_seconds": batch_row.get("timestamp_seconds", ""),
            "sha256": "",
        }
        add_pair(row, image_source, annotation_source)

    if errors:
        shutil.rmtree(output)
        raise ValueError("审计失败:\n" + "\n".join(errors))

    fields = list(source_rows[0])
    with (output / "audit_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged_rows)
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    report = {
        "images_total": len(merged_rows),
        "camera_images_added": len(batch_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "boxes_by_class": dict(sorted(class_counts.items())),
        "empty_annotations": sum(row["box_count"] == "0" for row in merged_rows),
        "clipped_boxes": clipped_boxes,
        "confirmed_unchecked_images": confirmed_unchecked,
        "audit_errors": [],
        "source_audit": str(audit),
        "camera_batch": str(batch),
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 外接摄像头增量审计数据集\n\n"
        f"基础审计集：`{audit.name}`。新增人工复核批次：`{batch.name}`。"
        "相机样本仅进入训练集；"
        "验证和测试集保持与该录制视频独立。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
