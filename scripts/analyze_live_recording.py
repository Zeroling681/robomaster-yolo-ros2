"""Sample a recorded camera test and save reproducible ONNX detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from live_camera_onnx import CLASS_NAMES, letterbox, postprocess


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "runs" / "detect" / "mouse_cup_yolo11n_v8_768" / "weights" / "best.onnx",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def draw(frame: np.ndarray, detections: list[tuple[int, float, tuple[int, int, int, int]]]) -> np.ndarray:
    rendered = frame.copy()
    for class_id, confidence, (x, y, width, height) in detections:
        color = (0, 220, 0) if class_id == 0 else (255, 160, 0)
        cv2.rectangle(rendered, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            rendered,
            f"{CLASS_NAMES[class_id]} {confidence:.2f}",
            (x, max(24, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def main() -> None:
    args = parse_args()
    if not args.video.is_file() or not args.model.is_file():
        raise FileNotFoundError("video and model must both exist")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
    indices = np.linspace(0, max(0, frame_count - 1), args.samples, dtype=int)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for sample_index, frame_index in enumerate(indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not decode frame {frame_index}")
        prepared, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
        tensor = prepared[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = session.run(None, {input_name: tensor})[0]
        detections = postprocess(output, frame.shape[:2], scale, pad_x, pad_y, args.conf, args.iou)
        filename = f"sample_{sample_index:02d}_f{frame_index:06d}.jpg"
        if not cv2.imwrite(str(frames_dir / filename), draw(frame, detections)):
            raise RuntimeError(f"could not write {filename}")
        rows.append(
            {
                "filename": filename,
                "frame_index": int(frame_index),
                "timestamp_seconds": round(frame_index / fps, 3),
                "detections": [
                    {
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": round(confidence, 4),
                        "xywh": list(box),
                    }
                    for class_id, confidence, box in detections
                ],
            }
        )
    capture.release()
    report = {
        "video": str(args.video.resolve()),
        "model": str(args.model.resolve()),
        "confidence_threshold": args.conf,
        "sample_count": len(rows),
        "video_frame_count": frame_count,
        "fps": fps,
        "samples": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
