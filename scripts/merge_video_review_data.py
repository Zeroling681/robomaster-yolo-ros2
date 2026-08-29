"""Merge the reviewed base dataset with usable prelabelled video frames.

Video prelabels are recorded as such in the manifest and must be checked in
X-AnyLabeling before a final release. Frames with an empty prelabel are kept
out of the training source instead of being silently treated as negatives.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=root / "dataset_work" / "yolo_dataset_v2")
    parser.add_argument("--review", type=Path, default=root / "dataset_work" / "video_review_v1")
    parser.add_argument("--output", type=Path, default=root / "dataset_work" / "yolo_dataset_combined_base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    review = args.review.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    manifest: list[dict[str, str]] = []

    for split in ("train", "val", "test"):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((base / "images" / split).iterdir()):
            if not image_path.is_file():
                continue
            label_path = base / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(label_path, label_dir / label_path.name)
            manifest.append({"filename": image_path.name, "split": split, "source": str(image_path), "label_status": "human_checked"})

    included = 0
    skipped = 0
    for image_path in sorted((review / "images").glob("*.jpg")):
        label_path = review / "prelabels_yolo" / f"{image_path.stem}.txt"
        if not label_path.is_file() or not label_path.read_text(encoding="utf-8").strip():
            skipped += 1
            manifest.append({"filename": image_path.name, "split": "excluded", "source": str(image_path), "label_status": "missing_prelabel"})
            continue
        shutil.copy2(image_path, output / "images" / "train" / image_path.name)
        shutil.copy2(label_path, output / "labels" / "train" / label_path.name)
        included += 1
        manifest.append({"filename": image_path.name, "split": "train", "source": str(image_path), "label_status": "video_prelabel_requires_review"})

    shutil.copy2(base / "classes.txt", output / "classes.txt")
    shutil.copy2(base / "dataset.yaml", output / "dataset.yaml")
    with (output / "merge_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"base_images={sum(row['label_status'] == 'human_checked' for row in manifest)}")
    print(f"video_prelabels_included={included}")
    print(f"video_frames_excluded_missing_label={skipped}")
    print(f"output={output}")
    print("MERGE_REQUIRES_LABEL_REVIEW=YES")


if __name__ == "__main__":
    main()
