# YOLO11n v13 final model

This folder describes the reproducible deployment artifacts for the final mouse
and cup detector. The local workspace keeps the individual weight files here.
The GitHub submission provides the same files in
`release/experiment_one_v13_model.zip` so downloading does not depend on Git LFS.

- `best.pt`: Ultralytics PyTorch checkpoint used on Jetson and for later training.
- `best.onnx`: ONNX export used by the Windows real-time program.
- `args.yaml`: complete Ultralytics training configuration.
- `results.csv`: per-epoch training and validation metrics.
- `results.png`: training curve overview.
- `confusion_matrix.png`: validation confusion matrix.

Training configuration: YOLO11n, image size 768, batch size 16, SGD, 40 epochs,
patience 12, initial learning rate 0.00015, final learning-rate factor 0.01.
The best checkpoint was selected at epoch 27 with validation precision 0.856,
recall 0.776, mAP50 0.886 and mAP50-95 0.715.

SHA-256:

- `best.pt`: `14b82fb5b50d8140cebb134f6eb1d57e07902e07a78fbdc5d033ab71d9025792`
- `best.onnx`: `42c78972a3a4f7f2a331df22a7758e9140a38d6736735391bc910cbf56f1aea0`
