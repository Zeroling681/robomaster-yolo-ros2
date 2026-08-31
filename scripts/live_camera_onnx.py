"""Run the exported YOLO model on a Windows camera in a live OpenCV window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = ("mouse", "cup")


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def postprocess(
    output: np.ndarray,
    original_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    height, width = original_shape
    predictions = output[0].T if output.ndim == 3 else output.T
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    keep = confidences >= confidence_threshold
    predictions = predictions[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    boxes: list[list[int]] = []
    scores: list[float] = []
    ids: list[int] = []
    for prediction, class_id, confidence in zip(predictions, class_ids, confidences):
        center_x, center_y, box_width, box_height = prediction[:4]
        x1 = (center_x - box_width / 2 - pad_x) / scale
        y1 = (center_y - box_height / 2 - pad_y) / scale
        x2 = (center_x + box_width / 2 - pad_x) / scale
        y2 = (center_y + box_height / 2 - pad_y) / scale
        boxes.append(
            [
                max(0, min(width - 1, round(x1))),
                max(0, min(height - 1, round(y1))),
                max(1, min(width, round(x2) - round(x1))),
                max(1, min(height, round(y2) - round(y1))),
            ]
        )
        scores.append(float(confidence))
        ids.append(int(class_id))
    selected = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    if len(selected) == 0:
        return []
    selected_indices = np.asarray(selected).reshape(-1).tolist()
    return [
        (ids[index], scores[index], tuple(boxes[index]))
        for index in selected_indices
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "results" / "mouse_cup_yolo11n_v4_best.onnx",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--save-raw",
        type=Path,
        default=None,
        help="Optional path for the unannotated camera stream used in later audits.",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    writer = None
    raw_writer = None
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
    previous = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            if raw_writer is None and args.save_raw is not None:
                height, width = frame.shape[:2]
                raw_writer = cv2.VideoWriter(
                    str(args.save_raw),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    20.0,
                    (width, height),
                )
                if not raw_writer.isOpened():
                    raise RuntimeError(f"Could not open raw output video {args.save_raw}")
            if raw_writer is not None:
                raw_writer.write(frame)
            prepared, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
            tensor = prepared[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            output = session.run(None, {input_name: tensor})[0]
            detections = postprocess(
                output,
                frame.shape[:2],
                scale,
                pad_x,
                pad_y,
                args.conf,
                args.iou,
            )
            for class_id, confidence, (x, y, width, height) in detections:
                color = (0, 220, 0) if class_id == 0 else (255, 160, 0)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    f"{CLASS_NAMES[class_id]} {confidence:.2f}",
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = 0.85 * fps + 0.15 * instant if fps else instant
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  detections: {len(detections)}  press Q to quit",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None and args.save is not None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.save),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    20.0,
                    (width, height),
                )
            if writer is not None:
                writer.write(frame)
            cv2.imshow("YOLO mouse/cup - camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if raw_writer is not None:
            raw_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
