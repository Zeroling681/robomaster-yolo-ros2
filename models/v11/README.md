# YOLO11n v11 模型

v11 使用 `dataset_work/audit_dataset_v11/yolo_export/dataset.yaml`，从 v10 最佳权重
继续微调。输入尺寸为 768，优化器为 SGD，初始学习率为 0.0002。

- `best.pt`：用于继续训练或 Ultralytics PyTorch 推理。
- `best.onnx`：用于 Windows、Jetson 或其他 ONNX Runtime 环境部署。
- `args.yaml`：完整训练参数。
- `results.csv`：逐轮训练和验证指标。

训练在第 19 轮提前停止，最佳权重来自第 9 轮。验证集最佳指标为：

- precision：0.8535
- recall：0.7864
- mAP50：0.8779
- mAP50-95：0.7102

独立测试集指标为 precision 0.763、recall 0.591、mAP50 0.690、mAP50-95 0.373。
