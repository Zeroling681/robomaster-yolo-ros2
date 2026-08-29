"""Create lightweight views of the audit set for X-AnyLabeling review."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument("--force", action="store_true", help="remove only previously generated view directories")
    args = parser.parse_args()
    audit = args.audit.resolve()
    manifest_path = audit / "audit_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    source_images = audit / "images"
    source_annotations = audit / "annotations"
    for name, predicate in (("review_with_boxes", lambda row: int(row["box_count"]) > 0),
                            ("review_missing", lambda row: int(row["box_count"]) == 0)):
        view = audit / name
        if view.exists() and any(view.iterdir()):
            if not args.force:
                raise FileExistsError(f"目录已有内容，拒绝覆盖: {view}")
            shutil.rmtree(view)
        (view / "images").mkdir(parents=True, exist_ok=True)
        (view / "annotations").mkdir(parents=True, exist_ok=True)
        selected = [row for row in rows if predicate(row)]
        for row in selected:
            image = source_images / row["filename"]
            annotation = source_annotations / f"{Path(row['filename']).stem}.json"
            if not image.is_file() or not annotation.is_file():
                raise FileNotFoundError(f"缺少配对文件: {image} / {annotation}")
            # Hard links avoid duplicating ~150 MB locally; editing JSON in a view
            # intentionally edits the canonical audit annotation as well.
            (view / "images" / image.name).hardlink_to(image)
            (view / "annotations" / annotation.name).hardlink_to(annotation)
        with (view / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
        (view / "README.md").write_text(
            ("# 有框标注复核\n\n" if name == "review_with_boxes" else "# 漏标/空框复核\n\n")
            + f"图片数：{len(selected)}。在 X-AnyLabeling 中打开本目录的 `images/`，标注目录选择 `annotations/`。\n"
            + ("这些图片已有至少一个框，但仍需人工确认类别、边界和遮挡情况。\n"
               if name == "review_with_boxes" else
               "这些图片当前没有框。确认画面中有鼠标或水杯时补框；确认没有目标时保留空标注。\n"),
            encoding="utf-8",
        )
        print(f"{name}: {len(selected)}")


if __name__ == "__main__":
    main()
