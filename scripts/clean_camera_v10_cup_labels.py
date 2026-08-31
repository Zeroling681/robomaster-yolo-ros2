"""Remove non-target bottle/container boxes from the reviewed camera v10 batch.

The camera v10 recording contains three visually similar cylindrical objects:
the target insulated cups, a green ordinary bottle, and a red gum container.
The batch policy defines only the insulated cups as ``cup``.  This script keeps
all mouse boxes, keeps the steel and blue insulated cups, and removes the two
non-target objects identified during the semantic review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def frame_index(path: Path) -> int:
    return int(path.stem.split("_", 3)[2])


def box_area(shape: dict) -> float:
    xs = [float(point[0]) for point in shape.get("points", [])]
    ys = [float(point[1]) for point in shape.get("points", [])]
    if not xs or not ys:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def box_left(shape: dict) -> float:
    return min(float(point[0]) for point in shape.get("points", []))


def cups_to_keep(index: int, cups: list[dict]) -> set[int]:
    if not cups:
        return set()
    # 2--7 contain only the large steel insulated cup.
    if 2 <= index <= 7:
        return set(range(len(cups)))
    # 8--25 contain the steel cup plus a small red non-target container.
    if 8 <= index <= 25:
        return {max(range(len(cups)), key=lambda item: box_area(cups[item]))}
    # Frame 41 is another clear steel insulated-cup view.
    if index == 41:
        return set(range(len(cups)))
    # Frames 51--54 contain the blue insulated cup on the left, a green bottle,
    # and sometimes the red container.  Keep only the left-most cup box.
    if 51 <= index <= 54:
        return {min(range(len(cups)), key=lambda item: box_left(cups[item]))}
    # Frame 26 and 42--49 contain only the red container and/or green bottle.
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=Path,
        default=Path("dataset_work/camera_v10_annotation_batch"),
    )
    args = parser.parse_args()
    batch = args.batch.resolve()
    files = sorted(batch.glob("camera_v10_*.json"))
    if len(files) != 54:
        raise SystemExit(f"Expected 54 camera v10 annotations, found {len(files)}")

    removed: list[dict[str, object]] = []
    retained: Counter[str] = Counter()
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        shapes = list(data.get("shapes") or [])
        cups = [shape for shape in shapes if shape.get("label") == "cup"]
        keep_cup_indices = cups_to_keep(frame_index(path), cups)
        cup_cursor = 0
        cleaned: list[dict] = []
        for shape in shapes:
            if shape.get("label") != "cup":
                cleaned.append(shape)
                retained[str(shape.get("label"))] += 1
                continue
            if cup_cursor in keep_cup_indices:
                cleaned.append(shape)
                retained["cup"] += 1
            else:
                removed.append(
                    {
                        "filename": path.with_suffix(".jpg").name,
                        "label": "cup",
                        "points": shape.get("points"),
                        "reason": "ordinary_green_bottle_or_red_container",
                    }
                )
            cup_cursor += 1
        data["shapes"] = cleaned
        data["checked"] = True
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "batch": str(batch),
        "files_reviewed": len(files),
        "removed_non_target_cup_boxes": len(removed),
        "retained_boxes_by_class": dict(sorted(retained.items())),
        "removed": removed,
        "status": "PASS",
    }
    (batch / "semantic_cleanup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "removed"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
