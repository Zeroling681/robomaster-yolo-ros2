"""Run a YOLO PyTorch model on a Jetson camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=root / "results" / "mouse_cup_yolo11n_v4_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头 /dev/video{args.camera}")

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
    previous = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            rendered = result.plot()
            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = 0.85 * fps + 0.15 * instant if fps else instant
            cv2.putText(rendered, f"FPS: {fps:.1f}  detections: {len(result.boxes)}  press Q to quit",
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is None and args.save:
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (width, height))
            if writer is not None:
                writer.write(rendered)
            cv2.imshow("YOLO mouse/cup - Jetson camera", rendered)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
