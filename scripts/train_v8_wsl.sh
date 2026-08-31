#!/usr/bin/env bash

set -euo pipefail

source /home/tonyt/.venvs/robomaster/bin/activate
# Ultralytics 8.4 resolves `path: .` in dataset.yaml from the current directory.
cd /mnt/f/PycharmProjects/robomaster/dataset_work/yolo_export_v8

exec python /mnt/f/PycharmProjects/robomaster/scripts/train_yolo.py \
  --data dataset.yaml \
  --model /home/tonyt/models/yolo11n.pt \
  --name mouse_cup_yolo11n_v8_768 \
  --epochs 100 \
  --batch 16 \
  --imgsz 768
