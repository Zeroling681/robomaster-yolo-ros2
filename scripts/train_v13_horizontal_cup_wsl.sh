#!/usr/bin/env bash

# Fine-tune v12 with reviewed horizontal-cup views and additional mouse scenes.
# Validation and test data are kept unchanged so results remain comparable.
set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/audit_dataset_v13/yolo_export

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v12_phone_hard_negative_768/weights/best.pt \
  --name mouse_cup_yolo11n_v13_horizontal_cup_768 \
  --epochs 40 \
  --patience 12 \
  --batch 16 \
  --imgsz 768 \
  --optimizer SGD \
  --lr0 0.00015 \
  --lrf 0.01
