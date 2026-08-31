#!/usr/bin/env bash

# Fine-tune v10 with manually reviewed false-positive and missed-detection
# samples. Validation and test scenes remain unchanged from v10.
set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/audit_dataset_v11/yolo_export

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v10_clean_768/weights/best.pt \
  --name mouse_cup_yolo11n_v11_error_feedback_768 \
  --epochs 30 \
  --patience 10 \
  --batch 16 \
  --imgsz 768 \
  --optimizer SGD \
  --lr0 0.0002 \
  --lrf 0.01
