from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    model_dir = Path.home() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(model_dir)

    model = YOLO("yolo11n.pt")
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    model.predict(image, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()

    iterations = 10
    started = time.perf_counter()
    for _ in range(iterations):
        model.predict(image, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(f"Model: {model_dir / 'yolo11n.pt'}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Smoke-test throughput: {iterations / elapsed:.2f} FPS")
    print("YOLO_GPU_INFERENCE=PASS")


if __name__ == "__main__":
    main()

