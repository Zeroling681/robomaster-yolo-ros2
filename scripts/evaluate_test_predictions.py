"""Create per-image test results and saved typical error cases."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


CONFIDENCE_THRESHOLD = 0.50
MATCH_IOU_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on the held-out test split.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset_work" / "yolo_dataset_v2",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "runs" / "detect" / "mouse_cup_yolo11n" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "runs" / "detect" / "mouse_cup_yolo11n_final_test",
    )
    return parser.parse_args()


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, object]]:
    ground_truth: list[dict[str, object]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_text, x_text, y_text, width_text, height_text = line.split()
        class_id = int(class_text)
        center_x = float(x_text) * width
        center_y = float(y_text) * height
        box_width = float(width_text) * width
        box_height = float(height_text) * height
        ground_truth.append(
            {
                "class_id": class_id,
                "xyxy": [
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ],
            }
        )
    return ground_truth


def match_predictions(
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    matches: list[tuple[int, int, float]] = []
    unmatched_ground_truth = set(range(len(ground_truth)))
    unmatched_predictions = set(range(len(predictions)))
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index]["confidence"]),
        reverse=True,
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        candidates: list[tuple[float, int]] = []
        for ground_truth_index in unmatched_ground_truth:
            target = ground_truth[ground_truth_index]
            if prediction["class_id"] != target["class_id"]:
                continue
            overlap = iou(prediction["xyxy"], target["xyxy"])
            candidates.append((overlap, ground_truth_index))
        if not candidates:
            continue
        best_iou, best_ground_truth_index = max(candidates)
        if best_iou >= MATCH_IOU_THRESHOLD:
            matches.append((prediction_index, best_ground_truth_index, best_iou))
            unmatched_predictions.remove(prediction_index)
            unmatched_ground_truth.remove(best_ground_truth_index)
    return matches, sorted(unmatched_predictions), sorted(unmatched_ground_truth)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    image_dir = dataset / "images" / "test"
    label_dir = dataset / "labels" / "test"
    model_path = args.model.resolve()
    output = args.output.resolve()
    all_images_dir = output / "all"
    error_images_dir = output / "errors"

    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    all_images_dir.mkdir(parents=True)
    error_images_dir.mkdir(parents=True)

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_dir),
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        imgsz=640,
        device=0,
        stream=True,
        verbose=False,
    )

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    totals = Counter()
    class_totals: dict[str, Counter] = {"mouse": Counter(), "cup": Counter()}

    for result in results:
        image_path = Path(result.path)
        height, width = result.orig_img.shape[:2]
        ground_truth = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        xyxy_values = result.boxes.xyxy.cpu().tolist()
        class_values = result.boxes.cls.cpu().tolist()
        confidence_values = result.boxes.conf.cpu().tolist()
        predictions = [
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "xyxy": [float(value) for value in xyxy],
            }
            for xyxy, class_id, confidence in zip(xyxy_values, class_values, confidence_values)
        ]
        matches, false_positive_indices, false_negative_indices = match_predictions(
            ground_truth, predictions
        )

        totals.update(
            gt=len(ground_truth),
            predictions=len(predictions),
            tp=len(matches),
            fp=len(false_positive_indices),
            fn=len(false_negative_indices),
            images=1,
        )
        for _, ground_truth_index, _ in matches:
            name = result.names[int(ground_truth[ground_truth_index]["class_id"])]
            class_totals[name].update(gt=1, tp=1)
        for ground_truth_index in false_negative_indices:
            name = result.names[int(ground_truth[ground_truth_index]["class_id"])]
            class_totals[name].update(gt=1, fn=1)
        for prediction_index in false_positive_indices:
            name = result.names[int(predictions[prediction_index]["class_id"])]
            class_totals[name].update(fp=1)

        has_error = bool(false_positive_indices or false_negative_indices)
        rows.append(
            {
                "filename": image_path.name,
                "ground_truth_objects": len(ground_truth),
                "predicted_objects": len(predictions),
                "true_positives": len(matches),
                "false_positives": len(false_positive_indices),
                "false_negatives": len(false_negative_indices),
                "object_correct_rate": round(len(matches) / len(ground_truth), 6),
                "inference_ms": round(float(result.speed["inference"]), 3),
                "has_error": has_error,
            }
        )
        details.append(
            {
                "filename": image_path.name,
                "matches": [
                    {
                        "prediction_index": prediction_index,
                        "ground_truth_index": ground_truth_index,
                        "iou": round(overlap, 6),
                    }
                    for prediction_index, ground_truth_index, overlap in matches
                ],
                "false_positive_indices": false_positive_indices,
                "false_negative_indices": false_negative_indices,
                "predictions": predictions,
                "ground_truth": ground_truth,
            }
        )

        annotated = result.plot()
        for prediction_index in false_positive_indices:
            prediction = predictions[prediction_index]
            x1, y1, x2, y2 = (round(value) for value in prediction["xyxy"])
            label = result.names[int(prediction["class_id"])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 5)
            cv2.putText(
                annotated,
                f"FALSE POSITIVE: {label}",
                (max(0, x1), max(35, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        for ground_truth_index in false_negative_indices:
            target = ground_truth[ground_truth_index]
            x1, y1, x2, y2 = (round(value) for value in target["xyxy"])
            label = result.names[int(target["class_id"])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.putText(
                annotated,
                f"MISSED: {label}",
                (max(0, x1), max(35, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"TP={len(matches)} FP={len(false_positive_indices)} FN={len(false_negative_indices)}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        destination = all_images_dir / image_path.name
        if not cv2.imwrite(str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Failed to save {destination}")
        if has_error:
            shutil.copy2(destination, error_images_dir / image_path.name)

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    summary = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "images": totals["images"],
        "ground_truth_objects": totals["gt"],
        "predicted_objects": totals["predictions"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "object_correct_rate": totals["tp"] / totals["gt"],
        "precision": precision,
        "recall": recall,
        "error_image_count": sum(bool(row["has_error"]) for row in rows),
        "class_counts": {name: dict(counts) for name, counts in class_totals.items()},
    }

    with (output / "test_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "prediction_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
