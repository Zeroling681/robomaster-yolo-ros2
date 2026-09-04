#!/usr/bin/env bash

# Capture one reproducible ROS 2 runtime proof from the Jetson detector.
# Usage: bash scripts/capture_jetson_ros2_evidence.sh [camera] [model] [ros_workspace]

set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
camera_index="${1:-0}"
default_model="${project_dir}/models/v13/best.pt"
if [[ -f "${project_dir}/best.pt" ]]; then
  default_model="${project_dir}/best.pt"
fi
model_file="${2:-${default_model}}"
ros_workspace="${3:-${HOME}/yolo_ros2_ws}"
topic_name="/yolo/detections"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_dir="${project_dir}/results"
evidence_file="${result_dir}/v13_ros2_runtime_evidence_${timestamp}.txt"
node_log="${result_dir}/v13_ros2_detector_${timestamp}.log"
video_file="${result_dir}/v13_ros2_detected_${timestamp}.avi"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
  exit 1
fi
if [[ ! -f "${ros_workspace}/install/setup.bash" ]]; then
  echo "Built workspace not found: ${ros_workspace}/install/setup.bash" >&2
  exit 1
fi
if [[ ! -f "${model_file}" ]]; then
  echo "Model not found: ${model_file}" >&2
  exit 1
fi
if [[ ! -e "/dev/video${camera_index}" ]]; then
  echo "Camera not found: /dev/video${camera_index}" >&2
  exit 1
fi

mkdir -p "${result_dir}"
source /opt/ros/humble/setup.bash
source "${ros_workspace}/install/setup.bash"

detector_pid=""
stop_detector() {
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
}
trap stop_detector EXIT INT TERM

{
  echo "YOLO v13 ROS 2 runtime evidence"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "camera=/dev/video${camera_index}"
  echo "model=${model_file}"
  echo "topic=${topic_name}"
  echo "mouse_confidence=0.75"
  echo "cup_confidence=0.75"
  echo
} | tee "${evidence_file}"

ros2 run yolo_detection_ros2 detector_node --ros-args \
  -p model:="${model_file}" \
  -p camera:="${camera_index}" \
  -p image_size:=768 \
  -p mouse_confidence:=0.75 \
  -p cup_confidence:=0.75 \
  -p show:=false \
  -p save_path:="${video_file}" \
  >"${node_log}" 2>&1 &
detector_pid=$!

topic_ready=0
for _ in $(seq 1 20); do
  if ! kill -0 "${detector_pid}" 2>/dev/null; then
    echo "Detector stopped during startup. See ${node_log}." | tee -a "${evidence_file}" >&2
    exit 2
  fi
  if ros2 topic list 2>/dev/null | grep -Fxq "${topic_name}"; then
    topic_ready=1
    break
  fi
  sleep 1
done

if [[ "${topic_ready}" -ne 1 ]]; then
  echo "Topic did not appear: ${topic_name}" | tee -a "${evidence_file}" >&2
  exit 2
fi

{
  echo "--- topic info ---"
  ros2 topic info "${topic_name}" --verbose
  echo
  echo "--- first message containing a recognised class ---"
} | tee -a "${evidence_file}"

detection_found=0
message_file="${result_dir}/.v13_ros2_message_${timestamp}.tmp"
for attempt in $(seq 1 15); do
  timeout 4s ros2 topic echo "${topic_name}" --once >"${message_file}" 2>&1 || true
  if grep -Eq "class_id:[[:space:]]*['\"]?(mouse|cup)" "${message_file}"; then
    cat "${message_file}" | tee -a "${evidence_file}"
    detection_found=1
    break
  fi
  echo "attempt_${attempt}=no accepted mouse/cup box" >>"${evidence_file}"
done
rm -f "${message_file}"

{
  echo
  echo "--- topic rate ---"
  timeout 10s ros2 topic hz "${topic_name}" || true
  echo
  echo "node_log=${node_log}"
  echo "annotated_video=${video_file}"
} | tee -a "${evidence_file}"

if [[ "${detection_found}" -ne 1 ]]; then
  echo "No accepted detection was captured. Put a mouse or cup in view and rerun." | tee -a "${evidence_file}" >&2
  exit 3
fi

echo "Evidence saved: ${evidence_file}"
echo "Copy it to the PC with:"
echo "scp nvidia@<JETSON_IP>:${evidence_file} results/"
