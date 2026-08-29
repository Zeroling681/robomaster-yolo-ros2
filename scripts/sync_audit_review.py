"""Synchronize X-AnyLabeling edits from a filtered audit view into its manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=root / "dataset_work" / "audit_dataset")
    parser.add_argument("--subset", type=Path, default=None, help="filtered view, e.g. review_missing")
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
    changed = Counter()
    for name in subset_names:
        row = row_by_name[name]
        annotation_path = audit / "annotations" / f"{Path(name).stem}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        box_count = len(data.get("shapes") or [])
        row["box_count"] = str(box_count)
        row["annotation_origin"] = "xanylabeling_human_review"
        if data.get("checked") is True:
            row["audit_status"] = "human_checked"
        elif box_count:
            row["audit_status"] = "human_edited_pending_check"
        else:
            row["audit_status"] = "empty_candidate_pending_confirmation"
        changed[row["audit_status"]] += 1

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report_path = audit / "audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["audit_status_counts"] = dict(Counter(row["audit_status"] for row in rows))
    report["empty_annotations"] = sum(int(row["box_count"]) == 0 for row in rows)
    report["human_review_sync"] = {"subset": str(subset), "updated_rows": len(subset_names), "status_counts": dict(changed)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated_rows={len(subset_names)}")
    print(json.dumps(dict(changed), ensure_ascii=False))


if __name__ == "__main__":
    main()
