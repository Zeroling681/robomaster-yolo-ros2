"""Render a labelled contact sheet for quick manual annotation review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--draw-annotations",
        action="store_true",
        help="overlay matching LabelMe/X-AnyLabeling rectangle annotations",
    )
    args = parser.parse_args()
    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no JPG files found in {args.images}")
    if args.columns < 1 or args.width < 64:
        raise ValueError("columns and width must be positive")

    first_image = cv2.imread(str(paths[0]))
    if first_image is None:
        raise RuntimeError(f"could not read {paths[0]}")
    source_height, source_width = first_image.shape[:2]
    image_height = round(args.width * source_height / source_width)
    title_height = 26
    cells: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        resized = cv2.resize(image, (args.width, image_height), interpolation=cv2.INTER_AREA)
        if args.draw_annotations:
            annotation_path = path.with_suffix(".json")
            if annotation_path.exists():
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                scale_x = args.width / annotation["imageWidth"]
                scale_y = image_height / annotation["imageHeight"]
                for item in annotation.get("shapes", []):
                    points = item.get("points", [])
                    if item.get("shape_type") != "rectangle" or len(points) < 2:
                        continue
                    # X-AnyLabeling may serialize a rectangle as either two
                    # diagonal points or all four corners.  Use the full extent
                    # so both representations render correctly.
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    p1 = (round(x1 * scale_x), round(y1 * scale_y))
                    p2 = (round(x2 * scale_x), round(y2 * scale_y))
                    color = (0, 220, 0) if item.get("label") == "mouse" else (0, 180, 255)
                    cv2.rectangle(resized, p1, p2, color, 2)
                    cv2.putText(resized, item.get("label", "?"), (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cell = np.zeros((image_height + title_height, args.width, 3), dtype=np.uint8)
        cell[:title_height] = (0, 0, 0)
        cell[title_height:] = resized
        cv2.putText(cell, path.stem, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    blank = np.full_like(cells[0], 240)
    rows = [cells[index : index + args.columns] for index in range(0, len(cells), args.columns)]
    rendered_rows = [np.hstack(row + [blank] * (args.columns - len(row))) for row in rows]
    contact_sheet = np.vstack(rendered_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), contact_sheet):
        raise RuntimeError(f"could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
