# Experiment One submission checklist

This checklist records the final evidence available for Experiment One. The
software, camera, ROS 2 and physical Jetson checks are all complete.

## Acceptance requirements

| Requirement | Evidence | Status |
| --- | --- | --- |
| Two desktop classes | `mouse` and `cup` in the v13 YOLO dataset | Complete |
| Self-collected and labelled data | `dataset_work/audit_dataset_v13/yolo_export/` | Complete |
| Audited dataset | 793 paired images/labels; audit status `PASS`; no scene leakage | Complete |
| Trained model | `release/experiment_one_v13_model.zip` | Complete |
| Jetson detector | `scripts/live_camera_pt.py` and saved Jetson result videos | Complete |
| Class, box and confidence overlay | `v13_Jetson_camera1_detected_mouseCUP.avi` inside the video evidence archive | Complete |
| Twenty angles at 80% or more | 18/20 correct, or 90% | Complete |
| Horizontal cup recognition | Two scored horizontal angles plus 207/261 detected video frames | Complete |
| Jetson speed at 5 FPS or more | Selected evidence frames show 15.7-21.2 FPS, mean about 17.9 FPS | Complete |
| Save results and typical errors | Annotated/raw recording support, score CSV and evidence sheets | Complete |
| ROS 2 publisher implementation | `ros2/yolo_detection_ros2/` | Complete |
| ROS 2 runtime message path | Stored WSL `Detection2DArray`, cup confidence 0.8217, about 35 Hz | Complete |
| ROS 2 publication on Jetson camera | `/dev/video0`; mouse confidence 0.9478; `/yolo/detections` about 20.57 Hz | Complete |
| English experiment report | 33-page DOCX and PDF with labelled dataset samples and original Ultralytics training curves | Complete |

## Portable archives

- `release/experiment_one_v13_model.zip` contains PT and ONNX weights, full
  training arguments, epoch metrics, curves and confusion matrix. Archive
  SHA-256: `3feeeecf0c7179606ab58d18fb1cf81ff4bddfb6928a9d574088729d760dbfdd`.
- `release/experiment_one_v13_video_evidence.zip` contains the simultaneous
  mouse/cup Jetson video, horizontal-cup success video, 20-angle CSV, evidence
  sheet, contact sheet and JSON summary. Archive SHA-256:
  `19100ab32796548cce63295e02e6eff226aea102ea2d081e779ccac06adb98d6`.
- The model archive contains `best.pt` with SHA-256
  `14b82fb5b50d8140cebb134f6eb1d57e07902e07a78fbdc5d033ab71d9025792`
  and `best.onnx` with SHA-256
  `42c78972a3a4f7f2a331df22a7758e9140a38d6736735391bc910cbf56f1aea0`.
- `release/experiment_one_v13_jetson_ros2_kit.zip` contains the exact v13 PT
  model, ROS 2 source package, Jetson detector, capture helper and a short
  deployment guide. Archive SHA-256:
  `020718540a31b41c061cdece31dc419c6bea4412a8f028d2b771b4900354e89c`.
- `release/experiment_one_v13_ros2_jetson_evidence.zip` contains the final
  annotated Jetson camera clip, representative frame, detector log, non-empty
  `Detection2DArray` topic record and file checksums. Archive SHA-256:
  `8fe7bf7b4a57bf58e27676fd062099a4606ca87459e3eefaa75485e31a088e75`.

## Reproducible audit

Run the final strict audit before copying the submission:

```powershell
py -3.13 scripts/audit_experiment_one_submission.py `
  --verify-git --require-jetson-ros2
```

The strict audit requires the portable evidence archive to be on `origin/main`
and verifies that it contains a non-empty mouse or cup
`vision_msgs/msg/Detection2DArray` from a Jetson camera.
