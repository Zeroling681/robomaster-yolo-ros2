"""Publish mouse and cup detections from a camera as ROS 2 messages."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


CLASS_NAMES = ("mouse", "cup")
COLORS = ((0, 220, 0), (255, 160, 0))


class YoloDetectionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detection_publisher")
        self.declare_parameter("model", "/home/nvidia/jetson_yolo/best.pt")
        self.declare_parameter("camera", 0)
        self.declare_parameter("source", "")
        self.declare_parameter("image_size", 768)
        self.declare_parameter("mouse_confidence", 0.75)
        self.declare_parameter("cup_confidence", 0.75)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("device", "0")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("topic", "/yolo/detections")
        self.declare_parameter("show", True)
        self.declare_parameter("save_path", "")

        model_path = Path(str(self.get_parameter("model").value))
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.image_size = int(self.get_parameter("image_size").value)
        self.mouse_confidence = float(self.get_parameter("mouse_confidence").value)
        self.cup_confidence = float(self.get_parameter("cup_confidence").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.device = str(self.get_parameter("device").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.show = bool(self.get_parameter("show").value)
        save_path_value = str(self.get_parameter("save_path").value).strip()
        self.save_path = Path(save_path_value).expanduser() if save_path_value else None

        topic = str(self.get_parameter("topic").value)
        self.publisher = self.create_publisher(Detection2DArray, topic, 10)
        self.model = YOLO(str(model_path))

        source_value = str(self.get_parameter("source").value).strip()
        camera_index = int(self.get_parameter("camera").value)
        self.is_file_source = bool(source_value)
        if self.is_file_source:
            input_path = Path(source_value).expanduser()
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            capture_source: int | str = str(input_path)
            source_description = str(input_path)
            self.capture = cv2.VideoCapture(capture_source)
        else:
            capture_source = camera_index
            source_description = f"/dev/video{camera_index}"
            self.capture = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open input: {source_description}")

        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= self.source_fps <= 120.0:
            self.source_fps = 20.0
        self.writer: cv2.VideoWriter | None = None
        self.smoothed_fps = 0.0
        self.previous_frame_time = time.perf_counter()
        self.finished = False
        self.timer = self.create_timer(0.001, self.process_frame)
        self.get_logger().info(
            f"Publishing detections on {topic} from {source_description}"
        )

    def process_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            if self.is_file_source:
                self.get_logger().info("Input video finished")
                self.finished = True
                self.timer.cancel()
                return
            self.get_logger().warning("Camera did not return a frame")
            return

        minimum_confidence = min(self.mouse_confidence, self.cup_confidence)
        result = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=minimum_confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        message = Detection2DArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        rendered = frame.copy()

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if class_id >= len(CLASS_NAMES):
                continue
            threshold = self.mouse_confidence if class_id == 0 else self.cup_confidence
            if confidence < threshold:
                continue

            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detection = Detection2D()
            detection.header = message.header
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            detection.id = f"{CLASS_NAMES[class_id]}_{len(message.detections)}"

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = CLASS_NAMES[class_id]
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)
            message.detections.append(detection)

            left, top, right, bottom = map(round, (x1, y1, x2, y2))
            color = COLORS[class_id]
            cv2.rectangle(rendered, (left, top), (right, bottom), color, 2)
            cv2.putText(
                rendered,
                f"{CLASS_NAMES[class_id]} {confidence:.2f}",
                (left, max(25, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

        try:
            self.publisher.publish(message)
        except Exception:
            # Ctrl-C can invalidate the ROS context while this timer callback is
            # finishing an inference. That is a normal shutdown, not a failure.
            if rclpy.ok():
                raise
            return
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.previous_frame_time, 1e-6)
        self.previous_frame_time = now
        self.smoothed_fps = (
            0.85 * self.smoothed_fps + 0.15 * instant_fps
            if self.smoothed_fps
            else instant_fps
        )
        cv2.putText(
            rendered,
            f"FPS: {self.smoothed_fps:.1f}  detections: {len(message.detections)}  ROS 2: ON",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.save_path and self.writer is None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = rendered.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.save_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.source_fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open output video: {self.save_path}")
        if self.writer is not None:
            self.writer.write(rendered)
        if self.show:
            cv2.imshow("YOLO mouse/cup ROS 2", rendered)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.capture.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: YoloDetectionPublisher | None = None
    try:
        node = YoloDetectionPublisher()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
