"""Import external images into the audit set with explicit label status."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


def parse_item(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("使用 NAME=PATH 格式") from exc
    return name, Path(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_empty_json(path: Path, image_name: str, width: int, height: int, checked: bool) -> None:
    data = {
        "version": "4.0.3",
        "flags": {},
        "checked": checked,
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument("--negative", action="append", type=parse_item, default=[])
    parser.add_argument("--needs-label", action="append", type=parse_item, default=[])
    args = parser.parse_args()
    if not args.negative and not args.needs_label:
        raise ValueError("至少指定一个 --negative 或 --needs-label")

    audit = args.audit.resolve()
    images_dir = audit / "images"
    annotations_dir = audit / "annotations"
    manifest_path = audit / "audit_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    known_names = {row["filename"] for row in rows}
    known_hashes = {row["sha256"].lower() for row in rows}

    items = [("human_checked_negative", name, path) for name, path in args.negative]
    items += [("annotation_required", name, path) for name, path in args.needs_label]
    for status, stem, source in items:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        filename = f"{stem}{source.suffix.lower()}"
        if filename in known_names:
            raise FileExistsError(f"文件名已存在: {filename}")
        source_hash = sha256(source)
        if source_hash.lower() in known_hashes:
            raise ValueError(f"图片内容已存在: {source}")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        height, width = image.shape[:2]
        destination = images_dir / filename
        shutil.copy2(source, destination)
        annotation = annotations_dir / f"{stem}.json"
        write_empty_json(annotation, filename, width, height, checked=status == "human_checked_negative")
        rows.append(
            {
                "filename": filename,
                "dataset_split": "train_candidate",
                "data_origin": "user_captured_hard_negative",
                "expected_class": "none" if status == "human_checked_negative" else "mouse",
                "annotation_origin": "human_confirmation" if status == "human_checked_negative" else "pending_human_annotation",
                "box_count": "0",
                "audit_status": status,
                "source_video": "",
                "source_frame": "",
                "timestamp_seconds": "",
                "sha256": source_hash,
            }
        )
        known_names.add(filename)
        known_hashes.add(source_hash.lower())
        print(f"{filename}: {status}")

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = audit / "audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["images"] = len(rows)
    report["annotations"] = len(rows)
    report["empty_annotations"] = sum(int(row["box_count"]) == 0 for row in rows)
    report["origin_counts"] = dict(Counter(row["data_origin"] for row in rows))
    report["audit_status_counts"] = dict(Counter(row["audit_status"] for row in rows))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
