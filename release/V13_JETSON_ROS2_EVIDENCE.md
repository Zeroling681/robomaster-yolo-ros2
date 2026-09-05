# v13 Jetson ROS 2 runtime evidence

This package is the final physical-board record for Experiment One. It was
captured on the Jetson host `nvidia-desktop` with `/dev/video0`, ROS 2 Humble,
the final v13 `best.pt` model and independent mouse/cup thresholds of 0.75.

The stored `vision_msgs/msg/Detection2DArray` contains a `mouse` detection with
confidence `0.9478288888931274`. The measured `/yolo/detections` publication
rate was approximately 20.57 Hz, above the required 5 FPS. The 15-second AVI
shows the detected class, box, confidence, live FPS and `ROS 2: ON` overlay.

The timestamp embedded in the generated filenames comes from the Jetson board
clock. The files were copied to the PC and verified with SHA-256 checksums on
2026-09-05.

Files:

- `v13_jetson_ros2_camera_evidence.avi`: 15-second annotated camera recording.
- `v13_jetson_ros2_camera_evidence_frame.jpg`: representative overlay frame.
- `v13_ros2_runtime_evidence_20260829_193344.txt`: topic type, publisher,
  non-empty message and topic-rate capture.
- `v13_ros2_detector_20260829_193344.log`: detector startup record.
- `SHA256SUMS.txt`: integrity hashes for the evidence files.
