"""Train the mouse/cup YOLO11n model in the configured WSL environment."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset_work" / "yolo_dataset_v2" / "dataset.yaml"
    pretrained_model = Path.home() / "models" / "yolo11n.pt"
    output_root = project_root / "runs" / "detect"

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    if not pretrained_model.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model}")

    model = YOLO(str(pretrained_model))
    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=25,
        imgsz=640,
        batch=16,
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
        name="mouse_cup_yolo11n",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
