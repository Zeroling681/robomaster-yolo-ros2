from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(mouse_v\d|cup_v\d)_")
    for path in sorted(args.images.glob("*.jpg")):
        match = pattern.match(path.name)
        if match:
            grouped[match.group(1)].append(path)

    args.output.mkdir(parents=True, exist_ok=True)
    columns, rows = 8, 4
    cell_width, cell_height = 240, 320
    for source_id, paths in sorted(grouped.items()):
        canvas = np.zeros(
            (rows * cell_height, columns * cell_width, 3), dtype=np.uint8
        )
        for index, path in enumerate(paths[: columns * rows]):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"无法读取: {path}")
            cell = fit_image(image, cell_width, cell_height)
            cv2.putText(
                cell,
                f"{index + 1:02d}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row, column = divmod(index, columns)
            top, left = row * cell_height, column * cell_width
            canvas[top : top + cell_height, left : left + cell_width] = cell

        output_path = args.output / f"{source_id}_selected.jpg"
        if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"写入失败: {output_path}")
        print(f"{source_id}: images={len(paths)}, sheet={output_path}")


if __name__ == "__main__":
    main()

