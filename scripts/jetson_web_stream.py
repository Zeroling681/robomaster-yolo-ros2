"""Run YOLO on a Jetson camera and expose the annotated frames as MJPEG."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Jetson v13 实时检测</title>
  <style>
    body { margin: 0; background: #111827; color: #f8fafc; font: 16px system-ui; }
    main { max-width: 1100px; margin: 24px auto; padding: 0 18px; }
    img { width: 100%; max-width: 960px; border: 2px solid #334155; border-radius: 10px; }
    p { color: #cbd5e1; }
  </style>
</head>
<body><main>
  <h1>Jetson YOLO v13 实时检测</h1>
  <p>类别：mouse / cup　摄像头：CAMERA　推理设备：DEVICE</p>
  <img id="feed" alt="实时检测画面">
</main>
<script>window.addEventListener("load", () => { document.getElementById("feed").src = "/video_feed"; });</script>
</body></html>"""


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开 /dev/video{index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


class Detector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device if args.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
        self.thresholds = {"mouse": args.mouse_conf, "cup": args.cup_conf}
        self.model = YOLO(str(args.model.resolve()))
        self.camera = open_camera(args.camera, args.width, args.height)
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.frames = 0
        self.fps = 0.0
        self.detections = 0
        self.error: str | None = None
        self.writer: cv2.VideoWriter | None = None
        self.worker = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.worker.start()

    def run(self) -> None:
        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError("摄像头读取失败")
                started = time.perf_counter()
                result = self.model.predict(
                    frame,
                    conf=min(self.thresholds.values()),
                    device=self.device,
                    imgsz=self.args.imgsz,
                    verbose=False,
                )[0]

                kept = 0
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = self.model.names[class_id]
                    threshold = self.thresholds.get(class_name)
                    if threshold is None or confidence < threshold:
                        continue
                    kept += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = (0, 210, 255) if class_name == "cup" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                elapsed = max(time.perf_counter() - started, 1e-6)
                instant_fps = 1.0 / elapsed
                self.fps = instant_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instant_fps
                self.detections = kept
                cv2.putText(
                    frame,
                    f"FPS {self.fps:.1f}  detections {kept}",
                    (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if self.args.save and self.writer is None:
                    self.args.save.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    self.writer = cv2.VideoWriter(
                        str(self.args.save),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        self.args.save_fps,
                        (width, height),
                    )
                    if not self.writer.isOpened():
                        raise RuntimeError(f"无法创建录像文件：{self.args.save}")
                if self.writer is not None:
                    self.writer.write(frame)

                encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if encoded_ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.frames += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.camera.release()
            if self.writer is not None:
                self.writer.release()

    def state(self) -> tuple[bytes | None, int]:
        with self.lock:
            return self.jpeg, self.frames


class Handler(BaseHTTPRequestHandler):
    detector: Detector

    def log_message(self, *_args: object) -> None:
        return

    def send_body(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        detector = self.detector
        if self.path == "/":
            page = PAGE.replace("CAMERA", str(detector.args.camera)).replace("DEVICE", detector.device)
            self.send_body(200, "text/html; charset=utf-8", page.encode("utf-8"))
            return
        if self.path == "/health":
            jpeg, frames = detector.state()
            body = json.dumps(
                {
                    "ok": detector.error is None and jpeg is not None,
                    "camera": detector.args.camera,
                    "device": detector.device,
                    "fps": round(detector.fps, 1),
                    "frames": frames,
                    "detections": detector.detections,
                    "error": detector.error,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_body(200, "application/json; charset=utf-8", body)
            return
        if self.path != "/video_feed":
            self.send_body(404, "text/plain; charset=utf-8", b"Not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        previous: bytes | None = None
        try:
            while True:
                frame, _ = detector.state()
                if frame is None or frame == previous:
                    time.sleep(0.01)
                    continue
                previous = frame
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson YOLO 实时检测网页流")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--mouse-conf", type=float, default=0.50)
    parser.add_argument("--cup-conf", type=float, default=0.50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--save-fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    detector = Detector(args)
    detector.start()
    Handler.detector = detector
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"stream=http://{args.host}:{args.port}", flush=True)
    print(f"camera={args.camera} device={detector.device} save={args.save}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
