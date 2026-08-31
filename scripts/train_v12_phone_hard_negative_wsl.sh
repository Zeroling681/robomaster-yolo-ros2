#!/usr/bin/env bash

# Fine-tune v11 with manually reviewed phone hard negatives and additional
# mouse viewpoints. Validation and test scenes remain unchanged from v11.
set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/audit_dataset_v12/yolo_export

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v11_error_feedback_768/weights/best.pt \
  --name mouse_cup_yolo11n_v12_phone_hard_negative_768 \
  --epochs 30 \
  --patience 10 \
  --batch 16 \
  --imgsz 768 \
  --optimizer SGD \
  --lr0 0.0002 \
  --lrf 0.01
