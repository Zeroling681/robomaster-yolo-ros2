"""Apply a conservative, manually reviewed first-pass annotation to camera v9.

This script is intentionally a fixed record of a human-style visual pass over the
raw camera frames.  It does not use detector predictions.  The output remains a
review batch and must not be exported into a training set until the user has
checked the boxes in X-AnyLabeling.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Rectangle coordinates are (left, top, right, bottom) in the original 640x480
# frames.  Only clear visible target instances are included.  Stainless-steel
# thermoses are part of the project's ``cup`` class; ordinary bottles are not.
ANNOTATIONS: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {
    "camera_v9_002_t003250.jpg": [("cup", (266, 117, 405, 480))],
    "camera_v9_003_t003950.jpg": [("cup", (267, 113, 405, 480))],
    "camera_v9_004_t006050.jpg": [("cup", (268, 111, 405, 480))],
    "camera_v9_005_t008550.jpg": [("mouse", (262, 245, 477, 474))],
    "camera_v9_006_t009250.jpg": [("mouse", (267, 235, 476, 466))],
    "camera_v9_007_t011550.jpg": [("mouse", (263, 230, 474, 458))],
    "camera_v9_008_t012750.jpg": [("cup", (269, 106, 403, 480))],
    "camera_v9_009_t013850.jpg": [("cup", (268, 105, 404, 480))],
    "camera_v9_010_t016450.jpg": [("cup", (267, 103, 405, 480))],
    "camera_v9_011_t016550.jpg": [("cup", (266, 102, 405, 480))],
    "camera_v9_012_t018850.jpg": [
        ("mouse", (84, 215, 412, 402)),
        ("cup", (233, 0, 339, 166)),
    ],
    "camera_v9_013_t020450.jpg": [("mouse", (107, 201, 423, 480))],
    "camera_v9_014_t022450.jpg": [("cup", (266, 100, 406, 480))],
    "camera_v9_015_t023950.jpg": [("cup", (267, 99, 406, 480))],
    "camera_v9_016_t025650.jpg": [("mouse", (96, 316, 208, 390))],
    "camera_v9_017_t027250.jpg": [("mouse", (97, 315, 210, 390))],
    "camera_v9_018_t028850.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_019_t029550.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_022_t035550.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_023_t036550.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_024_t038750.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_025_t040350.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_026_t041950.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_027_t042850.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_028_t045150.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_029_t045850.jpg": [("mouse", (329, 214, 402, 264))],
    "camera_v9_030_t048150.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_031_t049150.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_032_t049950.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_033_t051950.jpg": [("mouse", (98, 315, 210, 390))],
    "camera_v9_034_t053750.jpg": [("mouse", (201, 105, 455, 337))],
    "camera_v9_035_t055950.jpg": [("cup", (266, 103, 406, 480))],
}


def shape(label: str, box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=Path,
        default=Path("dataset_work/camera_v9_annotation_batch"),
        help="camera review batch directory",
    )
    args = parser.parse_args()
    batch = args.batch.resolve()
    images = sorted(batch.glob("camera_v9_*.jpg"))
    if len(images) != 36:
        raise SystemExit(f"Expected 36 camera frames in {batch}, found {len(images)}")

    counts: Counter[str] = Counter()
    reviewed_negative: list[str] = []
    for image_path in images:
        labels = ANNOTATIONS.get(image_path.name, [])
        annotation_path = image_path.with_suffix(".json")
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        if data.get("imageWidth") != 640 or data.get("imageHeight") != 480:
            raise SystemExit(f"Unexpected image dimensions in {annotation_path}")
        data.update(
            {
                "checked": True,
                "shapes": [shape(label, box) for label, box in labels],
                "description": (
                    "Initial manual annotation from raw external-camera footage. "
                    "User review required before this batch is merged into training."
                ),
            }
        )
        annotation_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for item in data["shapes"]:
            (x1, y1), (x2, y2) = item["points"]
            if (
                item["label"] not in {"mouse", "cup"}
                or not (0 <= x1 < x2 <= data["imageWidth"])
                or not (0 <= y1 < y2 <= data["imageHeight"])
            ):
                raise SystemExit(f"Invalid shape written to {annotation_path}: {item}")
        if labels:
            counts.update(label for label, _ in labels)
        else:
            reviewed_negative.append(image_path.name)

    report = {
        "batch": str(batch),
        "images_reviewed": len(images),
        "boxes_by_class": dict(sorted(counts.items())),
        "reviewed_negative_images": reviewed_negative,
        "initial_annotation_only": True,
        "next_step": "Open all files in X-AnyLabeling, correct boxes or labels, then request merge.",
    }
    (batch / "self_annotation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
