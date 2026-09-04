"""Audit the final Experiment One submission without changing project files.

The default audit verifies every portable deliverable and reports the live
Jetson ROS 2 capture separately. Use ``--require-jetson-ros2`` for the final
pre-submission gate after the board has been reconnected and evidence copied
into ``results``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset_work" / "audit_dataset_v13" / "yolo_export"
MODEL_ARCHIVE = ROOT / "release" / "experiment_one_v13_model.zip"
VIDEO_ARCHIVE = ROOT / "release" / "experiment_one_v13_video_evidence.zip"
JETSON_KIT = ROOT / "release" / "experiment_one_v13_jetson_ros2_kit.zip"
ANGLE_CSV = (
    ROOT
    / "results"
    / "v13_submission_review"
    / "v13_20_angle_results_with_horizontal_cup.csv"
)

EXPECTED_MODEL_HASHES = {
    "best.pt": "14b82fb5b50d8140cebb134f6eb1d57e07902e07a78fbdc5d033ab71d9025792",
    "best.onnx": "42c78972a3a4f7f2a331df22a7758e9140a38d6736735391bc910cbf56f1aea0",
}

PORTABLE_PATHS = [
    "dataset_work/audit_dataset_v13/yolo_export/dataset.yaml",
    "dataset_work/audit_dataset_v13/yolo_export/dataset_audit_report.json",
    "release/experiment_one_v13_model.zip",
    "release/experiment_one_v13_video_evidence.zip",
    "release/experiment_one_v13_jetson_ros2_kit.zip",
    "release/JETSON_ROS2_QUICKSTART.md",
    "scripts/live_camera_onnx.py",
    "scripts/live_camera_pt.py",
    "ros2/yolo_detection_ros2/package.xml",
    "results/v13_submission_review/v13_20_angle_results_with_horizontal_cup.csv",
    "results/v13_submission_review/v13_20_angle_evidence_with_horizontal_cup.jpg",
    "results/ros2_v13_wsl_video_runtime_evidence.txt",
    "docs/Experiment_One_Object_Detection_Report.docx",
    "docs/Experiment_One_Object_Detection_Report.pdf",
    "README.md",
]


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, label: str, detail: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {label}: {detail}")
        if not condition:
            self.failures.append(label)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def archive_member_hash(archive: zipfile.ZipFile, member: str) -> str:
    return hashlib.sha256(archive.read(member)).hexdigest()


def audit_dataset(audit: Audit) -> None:
    export_report = load_json(DATASET / "export_report.json")
    dataset_report = load_json(DATASET / "dataset_audit_report.json")
    image_count = sum(1 for path in (DATASET / "images").rglob("*") if path.is_file())
    label_count = sum(1 for path in (DATASET / "labels").rglob("*.txt"))

    audit.check(
        export_report.get("exported_images") == image_count == 793,
        "audited v13 images",
        f"manifest={export_report.get('exported_images')}, files={image_count}",
    )
    audit.check(
        label_count == 793,
        "YOLO label pairing",
        f"labels={label_count}, images={image_count}",
    )
    audit.check(
        export_report.get("split_counts") == {"train": 492, "val": 61, "test": 240},
        "scene-based split",
        str(export_report.get("split_counts")),
    )
    audit.check(
        dataset_report.get("status") == "PASS"
        and not dataset_report.get("errors")
        and not dataset_report.get("scene_leaks")
        and not dataset_report.get("duplicate_content_across_splits"),
        "dataset integrity audit",
        f"status={dataset_report.get('status')}",
    )


def audit_archives(audit: Audit) -> None:
    required_model_members = {
        "best.pt",
        "best.onnx",
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "README.md",
    }
    with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
        members = set(archive.namelist())
        audit.check(
            required_model_members <= members,
            "portable model archive",
            f"members={len(members)}",
        )
        for member, expected_hash in EXPECTED_MODEL_HASHES.items():
            actual_hash = archive_member_hash(archive, member)
            audit.check(
                actual_hash == expected_hash,
                f"{member} identity",
                actual_hash,
            )

    required_video_members = {
        "v13_horizontal_cup_success_detected.avi",
        "v13_Jetson_camera1_detected_mouseCUP.avi",
        "v13_horizontal_cup_success_detected.json",
        "v13_20_angle_results_with_horizontal_cup.csv",
        "v13_20_angle_evidence_with_horizontal_cup.jpg",
        "v13_horizontal_cup_success_detected_contact_sheet.jpg",
    }
    with zipfile.ZipFile(VIDEO_ARCHIVE) as archive:
        members = set(archive.namelist())
        audit.check(
            required_video_members <= members,
            "portable video evidence archive",
            f"members={len(members)}",
        )

    required_jetson_members = {
        "models/v13/best.pt",
        "scripts/live_camera_pt.py",
        "scripts/capture_jetson_ros2_evidence.sh",
        "ros2/yolo_detection_ros2/package.xml",
        "ros2/yolo_detection_ros2/setup.py",
        "ros2/yolo_detection_ros2/yolo_detection_ros2/detector_node.py",
        "release/JETSON_ROS2_QUICKSTART.md",
    }
    with zipfile.ZipFile(JETSON_KIT) as archive:
        members = set(archive.namelist())
        audit.check(
            required_jetson_members <= members,
            "Jetson ROS 2 evidence kit",
            f"members={len(members)}",
        )
        actual_hash = archive_member_hash(archive, "models/v13/best.pt")
        audit.check(
            actual_hash == EXPECTED_MODEL_HASHES["best.pt"],
            "Jetson kit model identity",
            actual_hash,
        )


def audit_acceptance_evidence(audit: Audit) -> None:
    with ANGLE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    test_rows = [row for row in rows if row.get("test_id", "").strip().isdigit()]
    correct = sum(row.get("result") == "correct" for row in test_rows)
    accuracy = 100.0 * correct / len(test_rows) if test_rows else 0.0
    horizontal = sum("horizontal" in row.get("angle_description", "") for row in test_rows)

    audit.check(
        len(test_rows) == 20 and correct == 18 and accuracy >= 80.0,
        "twenty-angle accuracy",
        f"correct={correct}/{len(test_rows)}, accuracy={accuracy:.1f}%",
    )
    audit.check(
        horizontal >= 2,
        "horizontal-cup angles",
        f"horizontal scenes={horizontal}",
    )

    horizontal_report = load_json(ROOT / "results" / "v13_horizontal_cup_success_detected.json")
    processed = int(horizontal_report.get("processed_frames", 0))
    cup_frames = int(horizontal_report.get("frames_with_cup", 0))
    coverage = 100.0 * cup_frames / processed if processed else 0.0
    audit.check(
        processed == 261 and cup_frames == 207,
        "horizontal-cup video detection",
        f"cup frames={cup_frames}/{processed}, coverage={coverage:.1f}%",
    )

    wsl_evidence = (
        ROOT / "results" / "ros2_v13_wsl_video_runtime_evidence.txt"
    ).read_text(encoding="utf-8")
    audit.check(
        "vision_msgs/msg/Detection2DArray" in wsl_evidence
        and "class_id: cup" in wsl_evidence
        and "Publisher count: 1" in wsl_evidence,
        "ROS 2 end-to-end software runtime",
        "non-empty Detection2DArray stored",
    )


def audit_portable_files(audit: Audit, verify_git: bool) -> None:
    missing = [path for path in PORTABLE_PATHS if not (ROOT / path).is_file()]
    audit.check(not missing, "portable deliverables", f"missing={missing or 'none'}")

    if not verify_git:
        return
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "origin/main"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = set(result.stdout.splitlines())
    missing_remote = [path for path in PORTABLE_PATHS if path not in tracked]
    audit.check(
        not missing_remote,
        "GitHub origin/main deliverables",
        f"missing={missing_remote or 'none'}",
    )


def find_jetson_ros2_evidence() -> tuple[Path | None, str]:
    candidates = sorted((ROOT / "results").glob("v13_ros2_runtime_evidence_*.txt"))
    for path in reversed(candidates):
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            "camera=/dev/video" in text
            and "Type: vision_msgs/msg/Detection2DArray" in text
            and ("class_id: mouse" in text or "class_id: cup" in text)
        ):
            return path, "valid live Jetson camera topic record"
    return None, "run capture_jetson_ros2_evidence.sh on the connected Jetson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="also require every portable artifact to exist in origin/main",
    )
    parser.add_argument(
        "--require-jetson-ros2",
        action="store_true",
        help="fail unless a live Jetson camera ROS 2 topic capture is present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = Audit()
    audit_dataset(audit)
    audit_archives(audit)
    audit_acceptance_evidence(audit)
    audit_portable_files(audit, args.verify_git)

    jetson_path, jetson_detail = find_jetson_ros2_evidence()
    if args.require_jetson_ros2:
        audit.check(jetson_path is not None, "live Jetson ROS 2 proof", jetson_detail)
    else:
        status = "PASS" if jetson_path else "PENDING"
        suffix = str(jetson_path.relative_to(ROOT)) if jetson_path else jetson_detail
        print(f"[{status}] live Jetson ROS 2 proof: {suffix}")

    if audit.failures:
        print("\nSubmission audit failed: " + ", ".join(audit.failures))
        return 1
    print("\nPortable submission audit passed.")
    if jetson_path is None:
        print("The strict hardware gate remains pending until Jetson ROS 2 evidence is copied back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
