set -euo pipefail
source /home/tonyt/.venvs/robomaster/bin/activate
cd /mnt/f/PycharmProjects/robomaster/dataset_work/yolo_export_v8
python /mnt/f/PycharmProjects/robomaster/scripts/evaluate_test_predictions.py \
  --dataset . \
  --model /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v8_768/weights/best.pt \
  --output /mnt/f/PycharmProjects/robomaster/runs/detect/mouse_cup_yolo11n_v8_768_test_conf035_retry \
  --imgsz 768 \
  --conf 0.35 \
  --iou 0.45 \
  --device 0
