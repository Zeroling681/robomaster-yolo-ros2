"""Create one self-contained X-AnyLabeling batch for manual review.

The batch keeps JPG/JSON pairs in the same folder, which lets X-AnyLabeling
load existing boxes and save edits beside the image.  Source audit data is
copied, never moved, so an unfinished review cannot break the training set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blank_annotation(image_path: Path, image_name: str) -> dict[str, object]:
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "version": "4.0.3",
        "flags": {},
        "checked": False,
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_work" / "manual_annotation_batch_20260830",
    )
    parser.add_argument(
        "--new-image",
        type=Path,
        action="append",
        default=[],
        help="New image to add as an unannotated mouse candidate. Repeat as needed.",
    )
    return parser.parse_args()


def write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        "# 本批待人工标注图片\n\n"
        "本目录用于 X-AnyLabeling 单独复核。图片和同名 JSON 已放在一起；请直接打开本目录。\n\n"
        "- 类别只有 `mouse` 和 `cup`。\n"
        "- 每个可见鼠标或杯子各画一个矩形框；背景物体不要标。\n"
        "- 保留原文件名，不要移动或删除文件。\n"
        "- 标注后保存 JSON；新图片默认没有标注框，需要人工绘制。\n"
        "- `batch_manifest.csv` 中 `duplicate_of` 非空的图片是精确重复图，可只标原图。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = audit / "audit_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    pending_rows = [
        row
        for row in audit_rows
        if row["dataset_split"] in {"not_assigned", "train_candidate"}
    ]

    records: list[dict[str, str]] = []
    images_dir = audit / "images"
    annotations_dir = audit / "annotations"
    for row in pending_rows:
        image_name = row["filename"]
        source_image = images_dir / image_name
        source_annotation = annotations_dir / f"{Path(image_name).stem}.json"
        if not source_image.is_file() or not source_annotation.is_file():
            raise FileNotFoundError(f"Missing audit pair for {image_name}")
        shutil.copy2(source_image, output / image_name)
        shutil.copy2(source_annotation, output / source_annotation.name)
        records.append(
            {
                "filename": image_name,
                "batch_source": "existing_audit_pending",
                "source_image": str(source_image),
                "expected_class": row["expected_class"],
                "initial_split": row["dataset_split"],
                "initial_status": row["audit_status"],
                "duplicate_of": "",
                "annotation_action": "review_existing_json",
                "sha256": sha256(source_image),
            }
        )

    seen_hashes: dict[str, str] = {}
    for index, source_image in enumerate(args.new_image, start=1):
        source_image = source_image.resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        destination_name = f"new_mouse_angle_{index:02d}{source_image.suffix.lower()}"
        destination_image = output / destination_name
        image_hash = sha256(source_image)
        duplicate_of = seen_hashes.get(image_hash, "")
        seen_hashes.setdefault(image_hash, destination_name)
        shutil.copy2(source_image, destination_image)
        annotation_path = output / f"{destination_image.stem}.json"
        annotation_path.write_text(
            json.dumps(blank_annotation(destination_image, destination_name), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "filename": destination_name,
                "batch_source": "new_user_image",
                "source_image": str(source_image),
                "expected_class": "mouse",
                "initial_split": "train_candidate",
                "initial_status": "annotation_required",
                "duplicate_of": duplicate_of,
                "annotation_action": "draw_mouse_box" if not duplicate_of else "duplicate_no_need_to_label",
                "sha256": image_hash,
            }
        )

    with (output / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    write_readme(output)
    print(f"BATCH_READY={output}")
    print(f"existing_pending={len(pending_rows)}")
    print(f"new_images={len(args.new_image)}")
    print(f"total_images={len(records)}")


if __name__ == "__main__":
    main()
