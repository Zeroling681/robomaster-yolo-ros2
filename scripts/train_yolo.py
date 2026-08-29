"""Train the mouse/cup YOLO11n model in the configured WSL environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a YOLO mouse/cup detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "dataset_work" / "yolo_dataset_v2" / "dataset.yaml",
    )
    parser.add_argument("--model", type=Path, default=Path.home() / "models" / "yolo11n.pt")
    parser.add_argument("--name", default="mouse_cup_yolo11n")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    pretrained_model = args.model.resolve()
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "runs" / "detect"

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    if not pretrained_model.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model}")

    model = YOLO(str(pretrained_model))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=25,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        cache="ram",
        pretrained=True,
        optimizer="auto",
        seed=20260824,
        deterministic=True,
        close_mosaic=10,
        amp=True,
        plots=True,
        project=str(output_root),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
