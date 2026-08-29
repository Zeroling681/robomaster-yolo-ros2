"""Synchronize X-AnyLabeling edits from a filtered audit view into its manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument("--subset", type=Path, default=None, help="filtered view, e.g. review_missing")
    parser.add_argument(
        "--confirmed-empty",
        action="append",
        default=[],
        help="image filename confirmed by a human to contain no target (repeatable)",
    )
    parser.add_argument(
        "--confirm-all-empty",
        action="store_true",
        help="treat every empty annotation in this reviewed subset as a confirmed negative",
    )
    parser.add_argument(
        "--confirm-all-reviewed",
        action="store_true",
        help="mark every non-empty annotation in this subset as human checked",
    )
    args = parser.parse_args()
    audit = args.audit.resolve()
    subset = (args.subset or audit / "review_missing").resolve()
    manifest_path = audit / "audit_manifest.csv"
    subset_manifest_path = subset / "manifest.csv"
    if not manifest_path.is_file() or not subset_manifest_path.is_file():
        raise FileNotFoundError("找不到总清单或子集清单")

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    subset_names = {row["filename"] for row in csv.DictReader(subset_manifest_path.open(encoding="utf-8-sig", newline=""))}
    row_by_name = {row["filename"]: row for row in rows}
    confirmed_empty = set(args.confirmed_empty)
    unknown_empty = confirmed_empty - subset_names
    if unknown_empty:
        raise ValueError(f"confirmed-empty 不在审核子集中: {sorted(unknown_empty)}")
    changed = Counter()
    for name in subset_names:
        row = row_by_name[name]
        subset_annotation = subset / "annotations" / f"{Path(name).stem}.json"
        annotation_path = audit / "annotations" / f"{Path(name).stem}.json"
        if not subset_annotation.is_file():
            raise FileNotFoundError(subset_annotation)
        # X-AnyLabeling saves via file replacement, which can break hard links.
        # Copy the reviewed JSON back explicitly before updating the manifest.
        if not annotation_path.exists() or not subset_annotation.samefile(annotation_path):
            shutil.copy2(subset_annotation, annotation_path)
        data = json.loads(subset_annotation.read_text(encoding="utf-8"))
        box_count = len(data.get("shapes") or [])
        is_confirmed_empty = name in confirmed_empty or (args.confirm_all_empty and box_count == 0)
        row["box_count"] = str(box_count)
        row["annotation_origin"] = "xanylabeling_human_review"
        if is_confirmed_empty:
            if box_count:
                raise ValueError(f"confirmed-empty 文件仍有标注框: {name}")
            row["audit_status"] = "human_checked_negative"
        elif args.confirm_all_reviewed or data.get("checked") is True:
            row["audit_status"] = "human_checked"
        elif box_count:
            row["audit_status"] = "human_edited_pending_check"
        else:
            row["audit_status"] = "empty_candidate_pending_confirmation"
        changed[row["audit_status"]] += 1

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        for row in rows:
            row["sha256"] = sha256(audit / "images" / row["filename"])
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report_path = audit / "audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["audit_status_counts"] = dict(Counter(row["audit_status"] for row in rows))
    report["empty_annotations"] = sum(int(row["box_count"]) == 0 for row in rows)
    class_counts: Counter[str] = Counter()
    for annotation_path in (audit / "annotations").glob("*.json"):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        for shape in annotation.get("shapes") or []:
            class_counts[str(shape.get("label", ""))] += 1
    report["boxes_by_class"] = dict(sorted(class_counts.items()))
    report["human_review_sync"] = {"subset": str(subset), "updated_rows": len(subset_names), "status_counts": dict(changed)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated_rows={len(subset_names)}")
    print(json.dumps(dict(changed), ensure_ascii=False))


if __name__ == "__main__":
    main()
