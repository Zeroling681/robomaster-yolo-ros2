"""Add cautious ONNX prelabels to an X-AnyLabeling camera batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from live_camera_onnx import CLASS_NAMES, letterbox, postprocess


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to PRELABEL_REVIEW.csv inside the batch directory.",
    )
    parser.add_argument(
        "--allow-existing-shapes",
        action="store_true",
        help="Replace existing shapes. Omit this flag for a new annotation batch.",
    )
    return parser.parse_args()


def make_shape(
    class_id: int,
    confidence: float,
    box: tuple[int, int, int, int],
) -> dict[str, object]:
    x, y, width, height = box
    return {
        "label": CLASS_NAMES[class_id],
        "points": [[x, y], [x + width, y + height]],
        "group_id": None,
        "description": f"v12 ONNX prelabel confidence={confidence:.4f}",
        "shape_type": "rectangle",
        "flags": {"prelabel": True},
        "mask": None,
    }


def main() -> None:
    args = parse_args()
    batch = args.batch.resolve()
    model_path = args.model.resolve()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not batch.is_dir():
        raise FileNotFoundError(batch)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    image_paths = sorted(batch.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No JPG images found in {batch}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    shape = model_input.shape
    if not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError(f"The ONNX model must have a static input size, got {shape}")
    if shape[2] != shape[3]:
        raise ValueError(f"Only square ONNX inputs are supported, got {shape}")
    imgsz = shape[2]

    report_rows: list[dict[str, str]] = []
    class_totals: Counter[str] = Counter()
    for image_path in image_paths:
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("shapes") and not args.allow_existing_shapes:
            raise RuntimeError(
                f"Existing shapes found in {annotation_path}; refusing to overwrite"
            )

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        prepared, scale, pad_x, pad_y = letterbox(image, imgsz)
        tensor = (
            prepared[:, :, ::-1]
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255.0
        )
        output = session.run(None, {input_name: tensor})[0]
        detections = postprocess(
            output,
            image.shape[:2],
            scale,
            pad_x,
            pad_y,
            (args.confidence, args.confidence),
            args.iou,
        )

        annotation["checked"] = False
        annotation["shapes"] = [
            make_shape(class_id, confidence, box)
            for class_id, confidence, box in detections
        ]
        annotation["description"] = (
            "Initial v12 ONNX predictions only. Review every visible mouse and cup, "
            "correct inaccurate boxes, add missed horizontal cups, then mark checked."
        )
        annotation_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        counts = Counter(CLASS_NAMES[class_id] for class_id, _, _ in detections)
        class_totals.update(counts)
        max_confidence = max(
            (confidence for _, confidence, _ in detections), default=0.0
        )
        report_rows.append(
            {
                "filename": image_path.name,
                "box_count": str(len(detections)),
                "mouse_count": str(counts["mouse"]),
                "cup_count": str(counts["cup"]),
                "max_confidence": f"{max_confidence:.4f}",
                "prelabel_status": (
                    "model_boxes_require_review" if detections else "empty_requires_manual_boxes"
                ),
                "human_review_required": "yes",
            }
        )

    report_path = args.report.resolve() if args.report else batch / "PRELABEL_REVIEW.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"images={len(image_paths)}")
    print(f"boxes={sum(class_totals.values())}")
    print(f"boxes_by_class={dict(sorted(class_totals.items()))}")
    print(f"empty_images={sum(row['box_count'] == '0' for row in report_rows)}")
    print(f"report={report_path}")
    print("HUMAN_REVIEW_REQUIRED=YES")


if __name__ == "__main__":
    main()
