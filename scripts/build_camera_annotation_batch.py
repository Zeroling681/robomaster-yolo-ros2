"""Create a clean X-AnyLabeling batch from an external-camera recording."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

from select_video_frames import choose_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_work" / "camera_v9_annotation_batch",
    )
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--source-id", default="external_camera_20260831")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = choose_frames({"path": str(args.video)}, args.count)
    rows: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        timestamp_ms = round(candidate.timestamp * 1000)
        filename = f"camera_v9_{index:03d}_t{timestamp_ms:06d}.jpg"
        image_path = args.output / filename
        if not cv2.imwrite(str(image_path), candidate.frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"could not write {image_path}")
        height, width = candidate.frame.shape[:2]
        annotation = {
            "version": "4.0.3",
            "flags": {},
            "checked": False,
            "shapes": [],
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "description": "",
        }
        (args.output / f"{image_path.stem}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "filename": filename,
                "source_video": args.source_id,
                "timestamp_seconds": f"{candidate.timestamp:.3f}",
                "sharpness": f"{candidate.sharpness:.2f}",
                "brightness": f"{candidate.brightness:.2f}",
                "sha256": sha256(image_path),
                "annotation_status": "annotation_required",
                "label_policy": "draw every visible mouse and cup; leave no-target frames empty",
            }
        )

    with (args.output / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# 外接摄像头 v9 标注批次\n\n"
        "在 X-AnyLabeling 中直接打开本目录。图片与同名 JSON 必须保留在同一目录。\n\n"
        "- 类别：`mouse`、`cup`。\n"
        "- 标注每个可见鼠标和水杯/保温杯；被手部分遮挡时仍标可见物体的完整外接框。\n"
        "- 没有目标的画面保持空 JSON，作为困难负样本。\n"
        "- 不要把显示器、人物、桌面、线缆、笔记本或普通瓶子标为目标。\n"
        "- 完成后不要改文件名，回复“标注完成”。\n",
        encoding="utf-8",
    )
    print(f"BATCH_READY={args.output.resolve()}")
    print(f"images={len(rows)}")


if __name__ == "__main__":
    main()
