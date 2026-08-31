"""Build the reviewed v12 phone-negative and mouse-viewpoint batch.

The source recording contains real mice and a phone that v11 sometimes calls
a mouse.  Positive samples keep every visible mouse box.  Phone-only crops are
left empty so that the phone becomes background without accidentally teaching
the model to ignore a real mouse elsewhere in the same frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class PositiveSample:
    frame_index: int
    labels: tuple[tuple[str, tuple[int, int, int, int]], ...]


@dataclass(frozen=True)
class NegativeCrop:
    frame_index: int
    crop: tuple[int, int, int, int]


# Coordinates are (left, top, right, bottom) in the 640x480 raw recording.
# These are conservative first-pass boxes and remain unchecked until reviewed.
POSITIVE_SAMPLES = (
    PositiveSample(201, (("mouse", (288, 115, 440, 273)), ("mouse", (437, 84, 640, 231)))),
    PositiveSample(363, (("mouse", (157, 82, 374, 205)), ("mouse", (202, 105, 456, 346)))),
    PositiveSample(444, (("mouse", (91, 55, 345, 337)), ("mouse", (239, 47, 456, 191)))),
    PositiveSample(565, (("mouse", (61, 41, 197, 144)), ("mouse", (188, 57, 391, 221)))),
    PositiveSample(1494, (("mouse", (26, 81, 203, 193)), ("mouse", (190, 91, 392, 231)))),
    PositiveSample(1534, (("mouse", (82, 105, 301, 231)), ("mouse", (234, 142, 473, 283)))),
    PositiveSample(
        1898,
        (
            ("mouse", (205, 55, 420, 153)),
            ("mouse", (198, 105, 437, 230)),
            ("mouse", (191, 152, 494, 323)),
        ),
    ),
    PositiveSample(1938, (("mouse", (328, 78, 640, 345)), ("mouse", (475, 0, 640, 88)))),
    PositiveSample(2019, (("mouse", (203, 195, 430, 355)), ("mouse", (294, 342, 640, 480)))),
    PositiveSample(2059, (("mouse", (37, 299, 273, 458)), ("cup", (553, 0, 616, 154)))),
)


# Crops deliberately exclude the real mice visible above or beside the phone.
# An empty annotation for the resulting crop is therefore semantically valid.
NEGATIVE_CROPS = (
    NegativeCrop(646, (0, 240, 462, 480)),
    NegativeCrop(767, (5, 145, 510, 325)),
    NegativeCrop(1009, (150, 0, 625, 240)),
    NegativeCrop(1050, (185, 120, 620, 325)),
    NegativeCrop(1171, (140, 70, 410, 480)),
    NegativeCrop(1211, (215, 90, 465, 480)),
    NegativeCrop(1332, (150, 125, 640, 480)),
    NegativeCrop(1575, (120, 185, 640, 410)),
    NegativeCrop(1615, (150, 185, 620, 410)),
    NegativeCrop(1655, (100, 185, 570, 420)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index}")
    return frame


def shape(label: str, box: tuple[int, int, int, int]) -> dict[str, object]:
    left, top, right, bottom = box
    return {
        "label": label,
        "points": [[left, top], [right, bottom]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
    }


def write_pair(
    output: Path,
    filename: str,
    image,
    shapes: list[dict[str, object]],
    description: str,
) -> Path:
    image_path = output / filename
    if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"could not write {image_path}")
    height, width = image.shape[:2]
    annotation = {
        "version": "4.0.3",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": description,
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return image_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=root / "results" / "v11_phone_negative_capture_raw.avi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset_work" / "camera_v12_phone_mouse_annotation_batch",
    )
    args = parser.parse_args()
    video, output = args.video.resolve(), args.output.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
    rows: list[dict[str, str]] = []

    for index, sample in enumerate(POSITIVE_SAMPLES, start=1):
        frame = read_frame(capture, sample.frame_index)
        filename = f"camera_v12_mouse_{index:02d}_f{sample.frame_index:06d}.jpg"
        image_path = write_pair(
            output,
            filename,
            frame,
            [shape(label, box) for label, box in sample.labels],
            "First-pass manual mouse boxes. Phone and other non-target objects stay unlabelled.",
        )
        rows.append(
            {
                "filename": filename,
                "source_video": "v11_phone_negative_capture",
                "timestamp_seconds": f"{sample.frame_index / fps:.3f}",
                "source_frame": str(sample.frame_index),
                "sample_role": "mouse_positive",
                "sha256": sha256(image_path),
                "annotation_status": "review_required",
                "label_policy": "review every visible mouse; do not label the phone",
            }
        )

    for index, sample in enumerate(NEGATIVE_CROPS, start=1):
        frame = read_frame(capture, sample.frame_index)
        left, top, right, bottom = sample.crop
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError(f"empty crop for frame {sample.frame_index}: {sample.crop}")
        filename = f"camera_v12_phone_negative_{index:02d}_f{sample.frame_index:06d}.jpg"
        image_path = write_pair(
            output,
            filename,
            crop,
            [],
            "Phone hard-negative crop. Confirm that no mouse or cup remains in this image.",
        )
        rows.append(
            {
                "filename": filename,
                "source_video": "v11_phone_negative_capture",
                "timestamp_seconds": f"{sample.frame_index / fps:.3f}",
                "source_frame": str(sample.frame_index),
                "sample_role": "phone_hard_negative",
                "sha256": sha256(image_path),
                "annotation_status": "review_required",
                "label_policy": "keep empty only when no mouse or cup is visible",
            }
        )

    capture.release()
    fields = list(rows[0])
    with (output / "batch_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "classes.txt").write_text("mouse\ncup\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# v12 手机难负样本与鼠标新视角\n\n"
        "本批次包含 10 张鼠标正样本和 10 张手机难负样本。正样本已写入第一轮鼠标框；"
        "手机样本经过裁剪，JSON 应保持空框。请在 X-AnyLabeling 中逐张复核：补齐所有"
        "可见鼠标和杯子，确认负样本内没有真实目标。完成前不要合并到正式训练集。\n",
        encoding="utf-8",
    )
    print(f"BATCH_READY={output}")
    print(f"positive_images={len(POSITIVE_SAMPLES)}")
    print(f"phone_negative_images={len(NEGATIVE_CROPS)}")


if __name__ == "__main__":
    main()
