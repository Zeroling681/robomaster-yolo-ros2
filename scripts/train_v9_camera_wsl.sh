#!/usr/bin/env bash

# Fine-tune the validated v8 detector with the human-reviewed external-camera
# batch.  Keep the held-out v8 val/test scenes unchanged for comparison.
set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/audit_dataset_v9/yolo_export

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v8_768/weights/best.pt \
  --name mouse_cup_yolo11n_v9_camera_768 \
  --epochs 50 \
  --patience 15 \
  --batch 16 \
  --imgsz 768 \
  --optimizer SGD \
  --lr0 0.001 \
  --lrf 0.01
