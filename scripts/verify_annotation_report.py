"""Check the generated annotation report before training or release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors: list[str] = []
    # Support both the original audit report schema and the current YOLO export
    # report schema.  They describe the same invariants with different names.
    image_count = int(report.get("image_count", report.get("images_total", report.get("exported_images", 0))))
    exported_box_count = int(
        report.get("exported_box_count", sum(int(value) for value in report.get("boxes_by_class", {}).values()))
    )
    split_counts = report.get("split_image_counts", report.get("split_counts", {}))

    if image_count <= 0 or exported_box_count <= 0:
        errors.append("没有可用于训练的图片或标注框")
    if report.get("cross_class_overlaps"):
        errors.append("存在不同类别的重叠框，需要人工复核")
    if sum(int(value) for value in split_counts.values()) != image_count:
        errors.append("训练/验证/测试图片数与总数不一致")
    if errors:
        for error in errors:
            print(f"ANNOTATION_QC=FAIL: {error}")
        raise SystemExit(1)

    print("ANNOTATION_QC=PASS")
    print(f"images={image_count}")
    print(f"exported_boxes={exported_box_count}")
    print(f"clipped_boxes={len(report.get('clipped_boxes', []))}")
    print(f"removed_duplicate_boxes={len(report.get('removed_duplicate_boxes', []))}")


if __name__ == "__main__":
    main()
