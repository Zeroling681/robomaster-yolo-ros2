"""Merge a completed X-AnyLabeling batch into a new, auditable dataset version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CLASSES = {"mouse", "cup"}
FORMAL_SPLITS = {"train", "val", "test"}
SCENE_PATTERN = re.compile(r"(?:^|_)(mouse_v[123]|cup_v[123]|cup_video01|mouse_video02)(?:_|\.)")


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
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def scene_from_name(filename: str) -> str:
    match = SCENE_PATTERN.search(filename)
    return match.group(1) if match else ""


def planned_split(row: dict[str, str]) -> str:
    scene = scene_from_name(row["filename"])
    if scene in {"cup_video01", "mouse_video02"}:
        return "excluded" if row["data_origin"] == "synthetic_augmentation" else "test"
    if scene.endswith("v3"):
        return "excluded" if row["data_origin"] == "synthetic_augmentation" else "val"
    if scene.endswith("v1") or scene.endswith("v2"):
        return "train"
    if row["data_origin"] in {"user_captured_hard_negative", "user_captured_new_mouse_angle"}:
        return "train"
    return row["dataset_split"] if row["dataset_split"] in FORMAL_SPLITS else "train"


def validate_annotation(
    annotation: dict[str, Any], image_path: Path, filename: str
) -> tuple[list[str], Counter[str], int]:
    errors: list[str] = []
    classes: Counter[str] = Counter()
    clipped_count = 0
    with Image.open(image_path) as image:
        width, height = image.size
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
        errors.append(f"{filename}: JSON尺寸与图片不一致")

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, shape in enumerate(annotation.get("shapes") or []):
        label = str(shape.get("label", ""))
        if label not in CLASSES:
            errors.append(f"{filename}: shape {index} 类别 {label!r} 无效")
            continue
        if shape.get("shape_type") != "rectangle":
            errors.append(f"{filename}: shape {index} 不是矩形框")
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            errors.append(f"{filename}: shape {index} 点不足")
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        raw_x1, raw_x2 = min(xs), max(xs)
        raw_y1, raw_y2 = min(ys), max(ys)
        x1, x2 = max(0.0, raw_x1), min(float(width), raw_x2)
        y1, y2 = max(0.0, raw_y1), min(float(height), raw_y2)
        if (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2):
            shape["points"] = [[x1, y1], [x2, y2]]
            clipped_count += 1
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{filename}: shape {index} 面积为零")
        else:
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


def expected_class(previous: str, annotation: dict[str, Any]) -> str:
    labels = {str(shape.get("label", "")) for shape in annotation.get("shapes") or []}
    if not labels:
        return "none"
    if previous in labels:
        return previous
    return sorted(labels)[0]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument(
        "--batch",
        type=Path,
        default=root / "dataset_work" / "manual_annotation_batch_20260830",
    )
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "audit_dataset_v6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = args.audit.resolve()
    batch = args.batch.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")
    if not (batch / "batch_manifest.csv").is_file():
        raise FileNotFoundError(batch / "batch_manifest.csv")

    with (audit / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (batch / "batch_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))

    batch_by_name = {row["filename"]: row for row in batch_rows if not row["duplicate_of"]}
    existing_names = {row["filename"] for row in rows}
    new_rows = [row for row in batch_rows if row["batch_source"] == "new_user_image" and not row["duplicate_of"]]
    duplicate_rows = [row for row in batch_rows if row["duplicate_of"]]

    output_images = output / "images"
    output_annotations = output / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    errors: list[str] = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    clipped_boxes: list[dict[str, object]] = []

    def add_pair(row: dict[str, str], image_source: Path, annotation_source: Path) -> None:
        image_destination = output_images / row["filename"]
        annotation_destination = output_annotations / f"{Path(row['filename']).stem}.json"
        link_or_copy(image_source, image_destination)
        annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
        annotation["imagePath"] = row["filename"]
        annotation["imageData"] = None
        annotation["checked"] = True
        pair_errors, pair_classes, clipped_count = validate_annotation(
            annotation, image_destination, row["filename"]
        )
        errors.extend(pair_errors)
        if clipped_count:
            clipped_boxes.append({"filename": row["filename"], "count": clipped_count})
        annotation_destination.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row["box_count"] = str(sum(pair_classes.values()))
        row["expected_class"] = expected_class(row["expected_class"], annotation)
        row["audit_status"] = "human_checked" if pair_classes else "human_checked_negative"
        row["annotation_origin"] = "xanylabeling_human_review"
        row["dataset_split"] = planned_split(row)
        row["source_video"] = scene_from_name(row["filename"]) or row.get("source_video", "")
        row["sha256"] = sha256(image_destination)
        merged_rows.append(row)
        class_counts.update(pair_classes)
        split_counts[row["dataset_split"]] += 1

    for source_row in rows:
        row = dict(source_row)
        name = row["filename"]
        if name in batch_by_name:
            image_source = batch / name
            annotation_source = batch / f"{Path(name).stem}.json"
        else:
            image_source = audit / "images" / name
            annotation_source = audit / "annotations" / f"{Path(name).stem}.json"
        if not image_source.is_file() or not annotation_source.is_file():
            errors.append(f"缺少图片或标注: {name}")
            continue
        add_pair(row, image_source, annotation_source)

    for batch_row in new_rows:
        name = batch_row["filename"]
        if name in existing_names:
            errors.append(f"新增文件名与现有数据冲突: {name}")
            continue
        row = {
            "filename": name,
            "dataset_split": "train_candidate",
            "data_origin": "user_captured_new_mouse_angle",
            "expected_class": "mouse",
            "annotation_origin": "xanylabeling_human_review",
            "box_count": "0",
            "audit_status": "annotation_required",
            "source_video": "new_mouse_angles_20260830",
            "source_frame": "",
            "timestamp_seconds": "",
            "sha256": "",
        }
        add_pair(row, batch / name, batch / f"{Path(name).stem}.json")
        if row["expected_class"] != "mouse":
            errors.append(f"{name}: 新增鼠标样本缺少 mouse 标注")

    if errors:
        shutil.rmtree(output)
        raise ValueError("审计失败:\n" + "\n".join(errors))

    fields = list(rows[0].keys())
    with (output / "audit_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged_rows)
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    report = {
        "images_total": len(merged_rows),
        "new_unique_images_merged": len(new_rows),
        "duplicate_images_excluded": [
            {"filename": row["filename"], "duplicate_of": row["duplicate_of"]}
            for row in duplicate_rows
        ],
        "split_counts": dict(sorted(split_counts.items())),
        "boxes_by_class": dict(sorted(class_counts.items())),
        "empty_annotations": sum(int(row["box_count"]) == 0 for row in merged_rows),
        "clipped_boxes": clipped_boxes,
        "audit_errors": [],
        "source_audit": str(audit),
        "manual_batch": str(batch),
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 审计数据集 v6\n\n"
        "由已完成的人工标注批次合并生成。`excluded` 样本保留在审计目录中，"
        "但不会由导出脚本写入训练、验证或测试集。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
