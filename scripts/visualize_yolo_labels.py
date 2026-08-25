from __future__ import annotations

import argparse
from pathlib import Path

import cv2

COLORS = [(0, 220, 0), (255, 160, 0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    class_names = args.classes.read_text(encoding="utf-8").splitlines()
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(args.images.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取: {image_path}")
        height, width = image.shape[:2]
        label_path = args.labels / f"{image_path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            class_id_text, x_text, y_text, w_text, h_text = line.split()
            class_id = int(class_id_text)
            center_x, center_y = float(x_text) * width, float(y_text) * height
            box_width, box_height = float(w_text) * width, float(h_text) * height
            left = round(center_x - box_width / 2)
            top = round(center_y - box_height / 2)
            right = round(center_x + box_width / 2)
            bottom = round(center_y + box_height / 2)
            color = COLORS[class_id]
            cv2.rectangle(image, (left, top), (right, bottom), color, 4)
            cv2.putText(
                image,
                class_names[class_id],
                (max(0, left), max(30, top)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA,
            )
        if not lines:
            cv2.putText(
                image,
                "NO PRELABEL - MANUAL BOX REQUIRED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        output_path = args.output / image_path.name
        if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"写入失败: {output_path}")


if __name__ == "__main__":
    main()

