# Jetson ROS 2 final evidence quick start

This kit contains the final v13 PyTorch model, the Jetson camera detector, the
ROS 2 Humble package and the evidence capture helper. The same procedure was
used successfully on the final physical-board check for Experiment One.

## 1. Copy and unpack

Copy `experiment_one_v13_jetson_ros2_kit.zip` to the Jetson, then run:

```bash
mkdir -p ~/jetson_yolo_submission
cd ~/jetson_yolo_submission
unzip -o ~/experiment_one_v13_jetson_ros2_kit.zip
```

All following commands assume the archive was unpacked in that directory.

## 2. Verify the environment

```bash
source /opt/ros/humble/setup.bash
python3 -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
ls -l /dev/video*
```

If the camera is `/dev/video1`, use camera index `1` in the final command.

## 3. Build the ROS 2 package

```bash
mkdir -p ~/yolo_ros2_ws/src
cp -a ros2/yolo_detection_ros2 ~/yolo_ros2_ws/src/
cd ~/yolo_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select yolo_detection_ros2
source install/setup.bash
```

## 4. Capture the final evidence

Return to the unpacked project directory, put a mouse or cup in front of the
camera, and run:

```bash
cd ~/jetson_yolo_submission
bash scripts/capture_jetson_ros2_evidence.sh 0 \
  "$PWD/models/v13/best.pt" \
  "$HOME/yolo_ros2_ws"
```

The helper validates the camera and model, starts the detector, waits for a
non-empty `mouse` or `cup` message, measures `/yolo/detections`, and saves:

- `results/v13_ros2_runtime_evidence_<timestamp>.txt`
- `results/v13_ros2_detector_<timestamp>.log`
- `results/v13_ros2_detected_<timestamp>.avi`

The TXT file must contain `vision_msgs/msg/Detection2DArray`, a `class_id` of
`mouse` or `cup`, and a confidence score. The AVI must display the class,
bounding box, confidence and FPS.

## 5. Copy the proof to the PC

Run this command on the PC, replacing the address if necessary:

```powershell
scp "nvidia@192.168.55.1:~/jetson_yolo_submission/results/v13_ros2_*" `
  "F:\PycharmProjects\robomaster\results\"
```

Then run the strict audit on the PC:

```powershell
cd F:\PycharmProjects\robomaster
py -3.13 scripts/audit_experiment_one_submission.py `
  --verify-git --require-jetson-ros2
```

The completed reference capture is packaged as
`experiment_one_v13_ros2_jetson_evidence.zip`. It records a mouse confidence
of 0.9478 and a `/yolo/detections` rate of about 20.57 Hz on `/dev/video0`.
