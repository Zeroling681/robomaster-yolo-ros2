"""Run the exported YOLO model on a Windows camera in a live OpenCV window."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = ("mouse", "cup")
Detection = tuple[int, float, tuple[int, int, int, int]]


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
    second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
    overlap_width = max(0, min(first_x2, second_x2) - max(first[0], second[0]))
    overlap_height = max(0, min(first_y2, second_y2) - max(first[1], second[1]))
    overlap = overlap_width * overlap_height
    union = first[2] * first[3] + second[2] * second[3] - overlap
    return overlap / union if union else 0.0


@dataclass
class StableDetection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    hits: int = 1
    misses: int = 0

    def as_detection(self) -> Detection:
        x, y, width, height = (round(value) for value in self.box)
        confidence = self.confidence * (0.9**self.misses)
        return self.class_id, confidence, (x, y, width, height)


class DetectionSmoother:
    """Keep short dropouts from making boxes flicker in the live view."""

    def __init__(
        self,
        confirm_frames: int = 2,
        hold_frames: int = 3,
        smoothing: float = 0.65,
        match_iou: float = 0.20,
    ) -> None:
        self.confirm_frames = confirm_frames
        self.hold_frames = hold_frames
        self.smoothing = smoothing
        self.match_iou = match_iou
        self.tracks: list[StableDetection] = []

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        possible_matches: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            track_box = tuple(round(value) for value in track.box)
            for detection_index, (class_id, _, box) in enumerate(detections):
                if class_id != track.class_id:
                    continue
                overlap = box_iou(track_box, box)
                if overlap >= self.match_iou:
                    possible_matches.append((overlap, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in sorted(possible_matches, reverse=True):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_index]
            _, confidence, box = detections[detection_index]
            keep = 1.0 - self.smoothing
            track.box = tuple(
                keep * old_value + self.smoothing * new_value
                for old_value, new_value in zip(track.box, box)
            )
            track.confidence = keep * track.confidence + self.smoothing * confidence
            track.hits += 1
            track.misses = 0
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.misses += 1
        self.tracks = [track for track in self.tracks if track.misses <= self.hold_frames]

        for detection_index, (class_id, confidence, box) in enumerate(detections):
            if detection_index not in matched_detections:
                self.tracks.append(
                    StableDetection(class_id, confidence, tuple(float(value) for value in box))
                )

        return [
            track.as_detection()
            for track in self.tracks
            if track.hits >= self.confirm_frames
        ]


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
    confidence_threshold: float | Sequence[float],
    iou_threshold: float,
) -> list[Detection]:
    height, width = original_shape
    predictions = output[0].T if output.ndim == 3 else output.T
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    if isinstance(confidence_threshold, (int, float)):
        class_thresholds = np.full(len(CLASS_NAMES), float(confidence_threshold))
    else:
        class_thresholds = np.asarray(confidence_threshold, dtype=np.float32)
        if len(class_thresholds) != len(CLASS_NAMES):
            raise ValueError("A confidence threshold is required for each class")
    keep = confidences >= class_thresholds[class_ids]
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
        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(0, min(width, round(x2)))
        bottom = max(0, min(height, round(y2)))
        if right <= left or bottom <= top:
            continue
        boxes.append([left, top, right - left, bottom - top])
        scores.append(float(confidence))
        ids.append(int(class_id))

    selected_indices: list[int] = []
    id_array = np.asarray(ids)
    for class_id in sorted(set(ids)):
        class_indices = np.flatnonzero(id_array == class_id)
        class_boxes = [boxes[index] for index in class_indices]
        class_scores_for_nms = [scores[index] for index in class_indices]
        selected = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores_for_nms,
            float(class_thresholds[class_id]),
            iou_threshold,
        )
        if len(selected):
            selected_indices.extend(class_indices[np.asarray(selected).reshape(-1)].tolist())
    if not selected_indices:
        return []
    detections = [
        (ids[index], scores[index], tuple(boxes[index]))
        for index in selected_indices
    ]
    return sorted(detections, key=lambda detection: detection[1], reverse=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "results" / "mouse_cup_yolo11n_v4_best.onnx",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional square input size. Static ONNX models infer this automatically.",
    )
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--mouse-conf", type=float, default=None)
    parser.add_argument("--cup-conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)
    parser.add_argument("--camera-fps", type=float, default=0.0)
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
    if args.confirm_frames < 1 or args.hold_frames < 0:
        raise ValueError("confirm-frames must be positive and hold-frames cannot be negative")
    if not 0.0 < args.smooth_alpha <= 1.0:
        raise ValueError("smooth-alpha must be in the range (0, 1]")

    mouse_conf = args.mouse_conf if args.mouse_conf is not None else args.conf
    cup_conf = args.cup_conf if args.cup_conf is not None else args.conf
    class_thresholds = (mouse_conf, cup_conf)
    if any(not 0.0 <= threshold <= 1.0 for threshold in class_thresholds):
        raise ValueError("confidence thresholds must be between 0 and 1")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    shape = model_input.shape
    static_height, static_width = shape[2], shape[3]
    if isinstance(static_height, int) and isinstance(static_width, int):
        if static_height != static_width:
            raise ValueError(f"Only square ONNX inputs are supported, got {shape}")
        if args.imgsz is not None and args.imgsz != static_height:
            raise ValueError(
                f"--imgsz={args.imgsz} conflicts with the ONNX model input {static_height}x{static_width}"
            )
        imgsz = static_height
    else:
        imgsz = args.imgsz or 640
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    recording_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not 1.0 <= recording_fps <= 120.0:
        recording_fps = args.camera_fps if args.camera_fps > 0 else 20.0
    smoother = DetectionSmoother(
        confirm_frames=args.confirm_frames,
        hold_frames=args.hold_frames,
        smoothing=args.smooth_alpha,
    )

    writer = None
    raw_writer = None
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
    previous = time.perf_counter()
    fps = 0.0
    failed_reads = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 30:
                    raise RuntimeError("Camera stopped returning frames")
                continue
            failed_reads = 0
            if raw_writer is None and args.save_raw is not None:
                height, width = frame.shape[:2]
                raw_writer = cv2.VideoWriter(
                    str(args.save_raw),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    recording_fps,
                    (width, height),
                )
                if not raw_writer.isOpened():
                    raise RuntimeError(f"Could not open raw output video {args.save_raw}")
            if raw_writer is not None:
                raw_writer.write(frame)
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
                (
                    f"FPS: {fps:.1f}  detections: {len(detections)}  "
                    f"mouse>={mouse_conf:.2f} cup>={cup_conf:.2f}  Q: quit"
                ),
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
                    recording_fps,
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
