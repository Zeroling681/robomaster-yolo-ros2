"""Run a trained YOLO ONNX model on a recorded video.

This is the repeatable counterpart to ``live_camera_onnx.py``.  It keeps the
same preprocessing, class thresholds and temporal smoothing, but reads frames
from a file and writes a new annotated video plus a small JSON run summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from live_camera_onnx import CLASS_NAMES, DetectionSmoother, letterbox, postprocess


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            root
            / "runs"
            / "detect"
            / "mouse_cup_yolo11n_v13_horizontal_cup_768"
            / "weights"
            / "best.onnx"
        ),
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--mouse-conf", type=float, default=None)
    parser.add_argument("--cup-conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    return parser.parse_args()


def draw_detection(
    frame: np.ndarray,
    class_id: int,
    confidence: float,
    box: tuple[int, int, int, int],
) -> None:
    x, y, width, height = box
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


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError("The selected time range is invalid")

    mouse_conf = args.mouse_conf if args.mouse_conf is not None else args.conf
    cup_conf = args.cup_conf if args.cup_conf is not None else args.conf
    class_thresholds = (mouse_conf, cup_conf)
    if any(not 0.0 <= threshold <= 1.0 for threshold in class_thresholds):
        raise ValueError("Confidence thresholds must be between 0 and 1")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    static_height, static_width = model_input.shape[2], model_input.shape[3]
    if isinstance(static_height, int) and isinstance(static_width, int):
        if static_height != static_width:
            raise ValueError(f"Only square ONNX inputs are supported: {model_input.shape}")
        if args.imgsz is not None and args.imgsz != static_height:
            raise ValueError(
                f"--imgsz={args.imgsz} conflicts with the model input {static_height}"
            )
        imgsz = static_height
    else:
        imgsz = args.imgsz or 640

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not 1.0 <= source_fps <= 120.0:
        source_fps = 20.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps if frame_count else 0.0
    end_time = min(args.end, duration) if args.end is not None and duration else args.end
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    smoother = DetectionSmoother(
        confirm_frames=args.confirm_frames,
        hold_frames=args.hold_frames,
        smoothing=args.smooth_alpha,
    )
    processed_frames = 0
    frames_with_cup = 0
    frames_with_mouse = 0
    processing_started = time.perf_counter()

    try:
        while True:
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if end_time is not None and timestamp >= end_time:
                break
            ok, frame = capture.read()
            if not ok:
                break

            prepared, scale, pad_x, pad_y = letterbox(frame, imgsz)
            tensor = prepared[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            output = session.run(None, {input_name: tensor})[0]
            detections = postprocess(
                output,
                frame.shape[:2],
                scale,
                pad_x,
                pad_y,
                class_thresholds,
                args.iou,
            )
            detections = smoother.update(detections)
            classes = {class_id for class_id, _, _ in detections}
            frames_with_mouse += int(0 in classes)
            frames_with_cup += int(1 in classes)
            for detection in detections:
                draw_detection(frame, *detection)

            elapsed = max(time.perf_counter() - processing_started, 1e-6)
            processing_fps = (processed_frames + 1) / elapsed
            cv2.putText(
                frame,
                (
                    f"Offline v13  detections: {len(detections)}  "
                    f"mouse>={mouse_conf:.2f} cup>={cup_conf:.2f}  "
                    f"processing: {processing_fps:.1f} FPS"
                ),
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {args.output}")
            writer.write(frame)
            processed_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = max(time.perf_counter() - processing_started, 1e-6)
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "model": str(args.model.resolve()),
        "source_fps": source_fps,
        "source_duration_seconds": round(duration, 3),
        "selected_start_seconds": args.start,
        "selected_end_seconds": end_time,
        "processed_frames": processed_frames,
        "processing_seconds": round(elapsed, 3),
        "average_processing_fps": round(processed_frames / elapsed, 3),
        "frames_with_mouse": frames_with_mouse,
        "frames_with_cup": frames_with_cup,
        "mouse_confidence_threshold": mouse_conf,
        "cup_confidence_threshold": cup_conf,
        "iou_threshold": args.iou,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
