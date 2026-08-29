"""Export the unified, human-audited X-AnyLabeling set to YOLO format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2


CLASS_TO_ID = {"mouse": 0, "cup": 1}
READY_STATUSES = {"human_checked", "human_checked_negative"}


def yolo_lines(annotation: dict[str, object], width: int, height: int, source: Path) -> list[str]:
    lines: list[str] = []
    for index, shape in enumerate(annotation.get("shapes") or []):
        label = str(shape.get("label", ""))
        if label not in CLASS_TO_ID:
            raise ValueError(f"未知类别 {label!r}: {source}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"仅支持矩形框: {source} shape={index}")
        points = shape.get("points") or []
        if len(points) < 2:
            raise ValueError(f"矩形框点不足: {source} shape={index}")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"无效矩形框: {source} shape={index}")
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{CLASS_TO_ID[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    audit = args.audit.resolve()
    output = (args.output or audit / "yolo_export").resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"输出目录已有内容: {output}")
        shutil.rmtree(output)

    rows = list(csv.DictReader((audit / "audit_manifest.csv").open(encoding="utf-8-sig", newline="")))
    exported: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    negative_count = 0
    for row in rows:
        if row["audit_status"] not in READY_STATUSES:
            excluded.append({"filename": row["filename"], "reason": row["audit_status"]})
            continue
        split = row["dataset_split"] if row["dataset_split"] in {"train", "val", "test"} else "train"
        image_path = audit / "images" / row["filename"]
        annotation_path = audit / "annotations" / f"{Path(row['filename']).stem}.json"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        lines = yolo_lines(annotation, width, height, annotation_path)
        for shape in annotation.get("shapes") or []:
            class_counts[str(shape["label"])] += 1
        if not lines:
            negative_count += 1
        destination_image = output / "images" / split / image_path.name
        destination_label = output / "labels" / split / f"{image_path.stem}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(image_path, destination_image)
        destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported.append({"filename": row["filename"], "split": split, "boxes": str(len(lines)), "audit_status": row["audit_status"]})
        split_counts[split] += 1

    (output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: mouse\n  1: cup\n",
        encoding="utf-8",
    )
    with (output / "export_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0].keys()))
        writer.writeheader()
        writer.writerows(exported)
    with (output / "excluded_pending.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    report = {
        "exported_images": len(exported),
        "negative_images": negative_count,
        "excluded_pending": len(excluded),
        "split_counts": dict(split_counts),
        "boxes_by_class": dict(class_counts),
    }
    (output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
