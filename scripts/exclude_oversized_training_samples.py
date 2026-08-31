"""Create a new audit version that excludes oversized training targets.

Large close-up targets are valid annotations, but they are unsuitable for the
desktop real-time deployment profile.  This script never deletes samples: it
marks affected training rows as ``excluded_oversized_target`` and preserves a
JSON report listing every decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def box_area_ratio(annotation: dict[str, object]) -> float:
    width = float(annotation["imageWidth"])
    height = float(annotation["imageHeight"])
    largest = 0.0
    for shape in annotation.get("shapes") or []:
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        largest = max(largest, (max(xs) - min(xs)) * (max(ys) - min(ys)) / (width * height))
    return largest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset_v7")
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "audit_dataset_v8")
    parser.add_argument("--max-train-box-area", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.max_train_box_area < 1:
        raise ValueError("--max-train-box-area 必须在 0 和 1 之间")
    audit = args.audit.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")

    with (audit / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0])
    (output / "images").mkdir(parents=True)
    (output / "annotations").mkdir(parents=True)

    excluded: list[dict[str, object]] = []
    for row in rows:
        filename = row["filename"]
        image_source = audit / "images" / filename
        annotation_source = audit / "annotations" / f"{Path(filename).stem}.json"
        link_or_copy(image_source, output / "images" / filename)
        shutil.copy2(annotation_source, output / "annotations" / annotation_source.name)
        annotation = json.loads(annotation_source.read_text(encoding="utf-8"))
        area = box_area_ratio(annotation)
        if row["dataset_split"] == "train" and area > args.max_train_box_area:
            excluded.append(
                {
                    "filename": filename,
                    "max_box_area_ratio": round(area, 6),
                    "data_origin": row["data_origin"],
                    "expected_class": row["expected_class"],
                }
            )
            row["dataset_split"] = "excluded"
            row["audit_status"] = "excluded_oversized_target"

    with (output / "audit_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    report = {
        "source_audit": str(audit),
        "max_train_box_area": args.max_train_box_area,
        "excluded_oversized_training_samples": excluded,
        "split_counts_after_cleaning": dict(sorted(Counter(row["dataset_split"] for row in rows).items())),
        "audit_errors": [],
    }
    (output / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 审计数据集 v8\n\n"
        "从 v7 生成。为匹配桌面实时检测距离，训练集中目标框面积超过 65% 的图片"
        "被标为 `excluded_oversized_target`；图片和标注仍完整保留以便追溯。"
        "验证集与测试集未作改动。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
