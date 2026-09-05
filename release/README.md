# Experiment One portable artifacts

These ZIP files are ordinary Git objects and provide a download fallback when
Git LFS is unavailable on a classroom computer.

| Archive | Contents | SHA-256 |
| --- | --- | --- |
| `experiment_one_v13_model.zip` | v13 PT and ONNX weights, training arguments, metrics and plots | `3feeeecf0c7179606ab58d18fb1cf81ff4bddfb6928a9d574088729d760dbfdd` |
| `experiment_one_v13_video_evidence.zip` | two-class Jetson video, 13-second horizontal-cup video, score CSV, two evidence sheets and JSON summary | `19100ab32796548cce63295e02e6eff226aea102ea2d081e779ccac06adb98d6` |
| `experiment_one_v13_jetson_ros2_kit.zip` | final PT model, Jetson detector, ROS 2 package, capture helper and quick-start guide | `020718540a31b41c061cdece31dc419c6bea4412a8f028d2b771b4900354e89c` |
| `experiment_one_v13_ros2_jetson_evidence.zip` | final Jetson camera AVI, overlay frame, non-empty ROS 2 message, rate capture, log and checksums | `8fe7bf7b4a57bf58e27676fd062099a4606ca87459e3eefaa75485e31a088e75` |

The audited v13 dataset is stored directly under
`dataset_work/audit_dataset_v13/`. The final English report is available in
both DOCX and PDF form under `docs/`. It includes a labelled v13 dataset
sample gallery and the original Ultralytics training curves.

Use `JETSON_ROS2_QUICKSTART.md` with the Jetson kit to reproduce the hardware
test. The completed physical-board record is stored in the Jetson evidence
archive.
