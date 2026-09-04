#!/usr/bin/env bash

# Build the ROS 2 package in a temporary workspace and verify real v13 messages
# by replaying a recorded video. This is an integration check, not Jetson proof.

set -Eeuo pipefail

repository_dir="/mnt/f/PycharmProjects/robomaster"
model_file="${1:-${repository_dir}/runs/detect/mouse_cup_yolo11n_v13_horizontal_cup_768/weights/best.pt}"
video_file="${2:-${repository_dir}/results/v13_horizontal_cup_success_detected.avi}"
evidence_file="${3:-${repository_dir}/results/ros2_v13_wsl_video_runtime_evidence.txt}"
rendered_file="${repository_dir}/results/ros2_v13_wsl_video_runtime.avi"
node_log="${repository_dir}/results/ros2_v13_wsl_video_runtime_node.log"
topic_name="/yolo/detections"
runtime_dir="$(mktemp -d /tmp/robomaster_ros2_runtime_XXXXXX)"
detector_pid=""

cleanup() {
  if [[ -n "${detector_pid}" ]] && kill -0 "${detector_pid}" 2>/dev/null; then
    kill -INT "${detector_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${detector_pid}" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "${detector_pid}" 2>/dev/null; then
      kill -TERM "${detector_pid}" 2>/dev/null || true
    fi
    wait "${detector_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${runtime_dir}"
}
trap cleanup EXIT INT TERM

for required_file in \
  /opt/ros/humble/setup.bash \
  /home/tonyt/.venvs/robomaster/bin/activate \
  "${model_file}" \
  "${video_file}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${runtime_dir}/src"
ln -s \
  "${repository_dir}/ros2/yolo_detection_ros2" \
  "${runtime_dir}/src/yolo_detection_ros2"

set +u
source /opt/ros/humble/setup.bash
source /home/tonyt/.venvs/robomaster/bin/activate
set -u

(
  cd "${runtime_dir}"
  colcon build --symlink-install
)

set +u
source "${runtime_dir}/install/setup.bash"
set -u

{
  echo "YOLO v13 ROS 2 recorded-video integration evidence"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "environment=WSL 2 Ubuntu 22.04 ROS 2 Humble"
  echo "model=${model_file}"
  echo "source=${video_file}"
  echo "topic=${topic_name}"
  echo "scope=runtime message verification only; not Jetson hardware evidence"
  echo
} >"${evidence_file}"

python -m yolo_detection_ros2.detector_node --ros-args \
  -p model:="${model_file}" \
  -p source:="${video_file}" \
  -p image_size:=768 \
  -p mouse_confidence:=0.75 \
  -p cup_confidence:=0.75 \
  -p show:=false \
  -p save_path:="${rendered_file}" \
  >"${node_log}" 2>&1 &
detector_pid=$!

topic_ready=0
for _ in $(seq 1 45); do
  if ! kill -0 "${detector_pid}" 2>/dev/null; then
    echo "Detector stopped before topic discovery. See ${node_log}." >&2
    exit 2
  fi
  if ros2 topic list 2>/dev/null | grep -Fxq "${topic_name}"; then
    topic_ready=1
    break
  fi
  sleep 1
done
if [[ "${topic_ready}" -ne 1 ]]; then
  echo "Topic did not appear: ${topic_name}" >&2
  exit 2
fi

{
  echo "--- package executable ---"
  ros2 pkg executables yolo_detection_ros2
  echo
  echo "--- topic info ---"
  ros2 topic info "${topic_name}" --verbose
  echo
  echo "--- cup detection message ---"
} >>"${evidence_file}"

message_file="${runtime_dir}/message.txt"
message_found=0
for _ in $(seq 1 20); do
  timeout 5s ros2 topic echo "${topic_name}" --once >"${message_file}" 2>&1 || true
  if grep -Eq "class_id:[[:space:]]*['\"]?cup" "${message_file}"; then
    cat "${message_file}" >>"${evidence_file}"
    message_found=1
    break
  fi
done

{
  echo
  echo "--- topic rate ---"
  timeout 6s ros2 topic hz "${topic_name}" || true
  echo
  echo "node_log=${node_log}"
  echo "rendered_video=${rendered_file}"
} >>"${evidence_file}"

if [[ "${message_found}" -ne 1 ]]; then
  echo "No cup message was captured from the horizontal-cup video." >&2
  exit 3
fi

echo "Evidence saved: ${evidence_file}"
