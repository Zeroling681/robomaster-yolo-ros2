#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/home/nvidia/jetson_yolo"
ACTIVATE_SCRIPT="/home/nvidia/activate_yolo.sh"
ONNX_MODEL_PATH="${PROJECT_DIR}/results/mouse_cup_yolo11n_v4_best.onnx"
PT_MODEL_PATH="${PROJECT_DIR}/results/mouse_cup_yolo11n_v4_best.pt"
OUTPUT_PATH="${PROJECT_DIR}/results/live_camera_conf060.avi"
CAMERA_INDEX="${1:-0}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "项目目录不存在: ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "环境启动脚本不存在: ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -e "/dev/video${CAMERA_INDEX}" ]]; then
  echo "摄像头不存在: /dev/video${CAMERA_INDEX}" >&2
  echo "当前摄像头设备:"
  ls -l /dev/video* 2>/dev/null || true
  exit 1
fi

cd "${PROJECT_DIR}"
# ROS 2 和 Python 虚拟环境的官方激活脚本并不保证兼容 `set -u`。
# 激活期间暂时关闭 nounset，完成后再恢复本脚本的严格检查。
set +u
source "${ACTIVATE_SCRIPT}"
set -u
mkdir -p "${PROJECT_DIR}/results"

# Jetson 上优先使用 Ultralytics/PyTorch，便于调用 CUDA；只有明确没有 PT
# 模型时才回退到 ONNX。Windows 专用的 CAP_DSHOW 不应成为板端默认路径。
if [[ -f "${PT_MODEL_PATH}" ]]; then
  MODEL_PATH="${PT_MODEL_PATH}"
  INFER_SCRIPT="scripts/live_camera_pt.py"
elif python -c 'import onnxruntime' >/dev/null 2>&1 && [[ -f "${ONNX_MODEL_PATH}" ]]; then
  MODEL_PATH="${ONNX_MODEL_PATH}"
  INFER_SCRIPT="scripts/live_camera_onnx.py"
else
  echo "未找到可用模型: ${PT_MODEL_PATH} 或 ${ONNX_MODEL_PATH}" >&2
  exit 1
fi

echo "摄像头: /dev/video${CAMERA_INDEX}"
echo "模型: ${MODEL_PATH}"
echo "录像: ${OUTPUT_PATH}"
echo "在检测窗口按 Q，或在终端按 Ctrl+C 结束。"

python "${INFER_SCRIPT}" \
  --camera "${CAMERA_INDEX}" \
  --model "${MODEL_PATH}" \
  --conf 0.60 \
  --save "${OUTPUT_PATH}"
