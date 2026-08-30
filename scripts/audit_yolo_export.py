"""Validate a generated YOLO dataset before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASSES = {0, 1}
SCENE_PATTERN = re.compile(r"(?:^|_)(mouse_v[123]|cup_v[123]|cup_video01|mouse_video02)(?:_|\.)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=root / "dataset_work" / "yolo_export_v6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    negative_count = 0
    seen_names: set[str] = set()
    hash_to_splits: dict[str, set[str]] = {}
    scene_to_splits: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"缺少 {split} 图像或标签目录")
            continue
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            if image_path.name in seen_names:
                errors.append(f"图片跨集合重复: {image_path.name}")
            seen_names.add(image_path.name)
            image_hash = sha256(image_path)
            hash_to_splits.setdefault(image_hash, set()).add(split)
            scene_match = SCENE_PATTERN.search(image_path.name)
            if scene_match:
                scene_to_splits.setdefault(scene_match.group(1), set()).add(split)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"缺少标签: {label_path}")
                continue
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                negative_count += 1
            for line_number, line in enumerate(lines, start=1):
                values = line.split()
                if len(values) != 5:
                    errors.append(f"标签列数错误: {label_path}:{line_number}")
                    continue
                try:
                    class_id = int(values[0])
                    cx, cy, width, height = map(float, values[1:])
                except ValueError:
                    errors.append(f"标签数值错误: {label_path}:{line_number}")
                    continue
                if class_id not in VALID_CLASSES:
                    errors.append(f"未知类别: {label_path}:{line_number}")
                if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
                    errors.append(f"标签坐标越界: {label_path}:{line_number}")
                class_counts[str(class_id)] += 1
            split_counts[split] += 1

    manifest = dataset / "export_manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        if len(manifest_rows) != sum(split_counts.values()):
            errors.append("导出清单图片数与目录图片数不一致")
    else:
        errors.append("缺少 export_manifest.csv")

    duplicate_content = {
        image_hash: sorted(splits)
        for image_hash, splits in hash_to_splits.items()
        if len(splits) > 1
    }
    if duplicate_content:
        errors.append("存在跨集合的重复图片内容")
    scene_leaks = {
        scene: sorted(splits)
        for scene, splits in scene_to_splits.items()
        if len(splits) > 1
    }
    if scene_leaks:
        errors.append("存在同源场景跨集合泄漏")

    report = {
        "dataset": str(dataset),
        "images_by_split": dict(sorted(split_counts.items())),
        "boxes_by_class_id": dict(sorted(class_counts.items())),
        "negative_images": negative_count,
        "scene_splits": {scene: sorted(splits) for scene, splits in sorted(scene_to_splits.items())},
        "duplicate_content_across_splits": duplicate_content,
        "scene_leaks": scene_leaks,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    (dataset / "dataset_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
