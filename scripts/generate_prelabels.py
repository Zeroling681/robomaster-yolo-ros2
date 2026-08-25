from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

CLASS_IDS = {"mouse": 0, "cup": 1}
CLASS_ALIASES = {"mouse": "mouse", "cup": "cup", "bottle": "cup"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()

    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"没有找到图片: {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("*.txt")):
        raise RuntimeError(f"输出目录已有标签，拒绝覆盖: {args.output}")

    model = YOLO(args.model)
    detections_by_class: Counter[str] = Counter()
    images_with_detections = 0
    review_rows: list[dict[str, str]] = []

    results = model.predict(
        source=[str(path) for path in image_paths],
        device=0,
        conf=args.confidence,
        iou=0.5,
        verbose=False,
        stream=True,
    )
    for image_path, result in zip(image_paths, results, strict=True):
        label_lines: list[str] = []
        for box in result.boxes:
            predicted_name = result.names[int(box.cls.item())]
            target_name = CLASS_ALIASES.get(predicted_name)
            if target_name is None:
                continue
            confidence = float(box.conf.item())
            center_x, center_y, width, height = box.xywhn[0].tolist()
            label_lines.append(
                f"{CLASS_IDS[target_name]} {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            detections_by_class[target_name] += 1
            review_rows.append(
                {
                    "filename": image_path.name,
                    "mapped_class": target_name,
                    "original_class": predicted_name,
                    "confidence": f"{confidence:.4f}",
                }
            )
        if label_lines:
            images_with_detections += 1
        (args.output / f"{image_path.stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

    with (args.output / "prelabel_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "mapped_class", "original_class", "confidence"],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"images={len(image_paths)}")
    print(f"images_with_detections={images_with_detections}")
    print(f"detections={dict(detections_by_class)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

