"""Create a new audited dataset version with manually confirmed hard negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2


# The black mouse visible in c388...jpg is deliberately omitted: an empty label
# for a real target would teach the detector that a mouse is background.
NEGATIVE_SOURCES = [
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\c4bdfdb264da47601602cf4274bbc465.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\40e607c71f1b2c214ffaef13da524b37.jpg"),
    Path(r"C:\Users\tonyt\AppData\Local\Temp\codex-clipboard-a033fa69-b6df-4b75-afed-74d84e8cb06b.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\aefc0a72ee3d147c136cd0f153721266.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\a0502d682ffe12e20bde539a51c2165a.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\3aa3fdb27cf0a0a13fededb7137db81e.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\2dbca3bc1b672e215dd15beec577352b.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\71410589ed5a88efd3f3c9577e1bac3c.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\74db370fbf6ff541488a1afa0a27c5a7.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\9d98f9a9135ef6ceb1ad7917cd25c459.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\7b84a5f2620e49977e96a82223074825.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\365986cd3317b7949d7ff684955a3e36.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\49ee334aa98ec9bdf78c17d944fa9139.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\8893070d2bd7db127c36f316496dcd13.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\46a350f0075fb611816e4f87265b8fbd.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\46e1f0efbdefcf967ad37dd95173823a.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\d7b294e2c2fb77148765462538430ad4.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\ede22d53234c31fec3ca74c603fdaf16.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\f0c03dcb5cc8ea6ade943c9832258633.jpg"),
    Path(r"E:\xwechat_files\wxid_9ong8xwpxo7f22_51c6\temp\RWTemp\2026-08\9e20f478899dc29eb19741386f9343c8\aa33738f5579828e4bf475cdf3d13f2c.jpg"),
]
EXCLUDED_SOURCE = "c388d96805d52dd6e8cee9daea07f2ed.jpg"


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset_v6")
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "audit_dataset_v7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")

    rows = list(csv.DictReader((audit / "audit_manifest.csv").open(encoding="utf-8-sig", newline="")))
    fields = list(rows[0])
    known_hashes = {row["sha256"] for row in rows}
    images_dir = output / "images"
    annotations_dir = output / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for row in rows:
        filename = row["filename"]
        link_or_copy(audit / "images" / filename, images_dir / filename)
        shutil.copy2(audit / "annotations" / f"{Path(filename).stem}.json", annotations_dir / f"{Path(filename).stem}.json")

    imported: list[dict[str, str]] = []
    for index, source in enumerate(NEGATIVE_SOURCES, start=1):
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256(source)
        if digest in known_hashes:
            raise ValueError(f"新增负样本与既有图片重复: {source.name}")
        filename = f"hard_negative_lab_{index:02d}.jpg"
        destination = images_dir / filename
        shutil.copy2(source, destination)
        image = cv2.imread(str(destination))
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        height, width = image.shape[:2]
        annotation = {
            "version": "4.0.3",
            "flags": {},
            "checked": True,
            "shapes": [],
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        (annotations_dir / f"{Path(filename).stem}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row = {
            "filename": filename,
            "dataset_split": "train",
            "data_origin": "user_captured_hard_negative",
            "expected_class": "none",
            "annotation_origin": "human_negative_review",
            "box_count": "0",
            "audit_status": "human_checked_negative",
            "source_video": "lab_negative_20260830",
            "source_frame": str(index),
            "timestamp_seconds": "",
            "sha256": digest,
        }
        rows.append(row)
        known_hashes.add(digest)
        imported.append({"filename": filename, "source_filename": source.name})

    with (output / "audit_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")

    split_counts = Counter(row["dataset_split"] for row in rows)
    report = {
        "images_total": len(rows),
        "new_hard_negatives_imported": len(imported),
        "negative_sources": imported,
        "excluded_from_import": {
            "source_filename": EXCLUDED_SOURCE,
            "reason": "contains_visible_mouse; must not receive an empty label",
        },
        "split_counts": dict(sorted(split_counts.items())),
        "empty_annotations": sum(row["box_count"] == "0" for row in rows),
        "audit_errors": [],
        "source_audit": str(audit),
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 审计数据集 v7\n\n"
        "在 v6 基础上新增 20 张人工确认的困难负样本，均为 `mouse` 与 `cup` 的空标注。"
        "含有可见鼠标的 `c388d96805d52dd6e8cee9daea07f2ed.jpg` 已排除，未写入数据集。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
