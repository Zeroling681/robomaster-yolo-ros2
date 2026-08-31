#!/usr/bin/env bash

# Fine-tune v9 on the cleaned, human-reviewed v10 dataset. The v10 export
# keeps scene-isolated validation/test splits and excludes pending records.
set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/audit_dataset_v10/yolo_export

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v9_camera_768/weights/best.pt \
  --name mouse_cup_yolo11n_v10_clean_768 \
  --epochs 60 \
  --patience 20 \
  --batch 16 \
  --imgsz 768 \
  --optimizer SGD \
  --lr0 0.0005 \
  --lrf 0.01
