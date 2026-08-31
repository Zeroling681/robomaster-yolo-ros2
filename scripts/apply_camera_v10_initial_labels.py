"""Apply a conservative first-pass annotation to the camera v10 review batch.

The boxes below come from a visual pass over raw external-camera frames.  They
deliberately focus on the blue insulated cup and mouse views that were missed
by the live detector.  Detector predictions are not used as annotation data.
Frames without a confident first-pass box remain unchecked for review in
X-AnyLabeling; they are not negative samples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Coordinates are (left, top, right, bottom) in the original 640x480 frames.
# The green bottle in frames 42--54 is intentionally excluded: it is not part
# of this project's cup class.  The blue insulated cup is class ``cup``.
ANNOTATIONS: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {
    "camera_v10_026_t063850.jpg": [("mouse", (318, 327, 508, 463))],
    "camera_v10_027_t066150.jpg": [("mouse", (233, 113, 469, 299))],
    "camera_v10_028_t067550.jpg": [("mouse", (236, 199, 455, 358))],
    "camera_v10_029_t070650.jpg": [("mouse", (250, 190, 415, 348))],
    "camera_v10_030_t073250.jpg": [("mouse", (224, 206, 474, 363))],
    "camera_v10_031_t075350.jpg": [("mouse", (264, 169, 486, 402))],
    "camera_v10_032_t078750.jpg": [("mouse", (377, 248, 509, 321))],
    "camera_v10_033_t080150.jpg": [
        ("mouse", (107, 281, 238, 371)),
        ("mouse", (392, 252, 511, 319)),
    ],
    "camera_v10_034_t081550.jpg": [
        ("mouse", (53, 163, 235, 278)),
        ("mouse", (411, 118, 521, 193)),
    ],
    "camera_v10_035_t084950.jpg": [
        ("mouse", (165, 163, 302, 302)),
        ("mouse", (330, 118, 453, 186)),
    ],
    "camera_v10_036_t088750.jpg": [("mouse", (247, 158, 398, 220))],
    "camera_v10_037_t089650.jpg": [("mouse", (245, 238, 433, 306))],
    "camera_v10_038_t091950.jpg": [("mouse", (174, 157, 516, 365))],
    "camera_v10_039_t095050.jpg": [("mouse", (220, 151, 488, 375))],
    "camera_v10_040_t096750.jpg": [("mouse", (210, 96, 427, 391))],
    "camera_v10_041_t099850.jpg": [("cup", (260, 3, 398, 465))],
    "camera_v10_042_t103050.jpg": [
        ("mouse", (111, 200, 250, 264)),
        ("mouse", (228, 233, 367, 302)),
    ],
    "camera_v10_043_t104450.jpg": [
        ("mouse", (112, 201, 251, 264)),
        ("mouse", (226, 233, 367, 303)),
    ],
    "camera_v10_044_t106450.jpg": [
        ("mouse", (111, 203, 251, 264)),
        ("mouse", (227, 234, 368, 303)),
    ],
    "camera_v10_045_t110250.jpg": [("mouse", (0, 280, 337, 480))],
    "camera_v10_048_t117750.jpg": [("mouse", (62, 315, 226, 466))],
    "camera_v10_049_t118350.jpg": [("mouse", (56, 352, 209, 480))],
    "camera_v10_051_t124750.jpg": [("cup", (310, 102, 431, 479))],
    "camera_v10_052_t125650.jpg": [("cup", (305, 95, 421, 480))],
    "camera_v10_053_t129050.jpg": [("cup", (298, 92, 416, 480))],
    "camera_v10_054_t130650.jpg": [("cup", (306, 92, 410, 480))],
}


def make_shape(label: str, box: tuple[int, int, int, int]) -> dict:
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
        default=Path("dataset_work/camera_v10_annotation_batch"),
        help="camera review batch directory",
    )
    args = parser.parse_args()
    batch = args.batch.resolve()
    images = sorted(batch.glob("camera_v10_*.jpg"))
    if len(images) != 54:
        raise SystemExit(f"Expected 54 camera frames in {batch}, found {len(images)}")

    counts: Counter[str] = Counter()
    checked: list[str] = []
    pending: list[str] = []
    for image_path in images:
        annotation_path = image_path.with_suffix(".json")
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        if data.get("imageWidth") != 640 or data.get("imageHeight") != 480:
            raise SystemExit(f"Unexpected image dimensions in {annotation_path}")

        labels = ANNOTATIONS.get(image_path.name)
        if labels is None:
            pending.append(image_path.name)
            continue
        data.update(
            {
                "checked": True,
                "shapes": [make_shape(label, box) for label, box in labels],
                "description": (
                    "Initial manual annotation from raw external-camera footage. "
                    "Review each box in X-AnyLabeling before this batch is merged."
                ),
            }
        )
        for shape in data["shapes"]:
            (x1, y1), (x2, y2) = shape["points"]
            if (
                shape["label"] not in {"mouse", "cup"}
                or not (0 <= x1 < x2 <= data["imageWidth"])
                or not (0 <= y1 < y2 <= data["imageHeight"])
            ):
                raise SystemExit(f"Invalid shape written to {annotation_path}: {shape}")
        annotation_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts.update(label for label, _ in labels)
        checked.append(image_path.name)

    report = {
        "batch": str(batch),
        "images_total": len(images),
        "images_initially_annotated": len(checked),
        "images_pending_human_review": len(pending),
        "boxes_by_class": dict(sorted(counts.items())),
        "focus": ["blue insulated cup", "previously missed mouse viewpoints"],
        "pending_filenames": pending,
        "initial_annotation_only": True,
        "next_step": "Review all checked boxes and annotate pending frames in X-AnyLabeling, then request merge.",
    }
    (batch / "self_annotation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
