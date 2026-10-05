"""Compare saved PCDCR A metrics to the existing official-candidate replay.

This reads existing results only. It performs no model forward or optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def key(row):
    return (row["split"], int(row["image_id"]), int(row["annotation_id"]), int(row["raw_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    history = {key(r): r for r in rows(args.historical)}
    current = rows(args.current)
    missing, differences = [], []
    for row in current:
        k = key(row)
        if k not in history:
            missing.append(k)
            continue
        differences.append({"identity": k, "absolute_iou_error": abs(row["iou_A"] - history[k]["iou_A"])})
    maximum = max((r["absolute_iou_error"] for r in differences), default=None)
    tolerance = 1e-6  # same tolerance as the pre-training smoke
    result = {
        "comparison": "PCDCR A normal-original-image IoU vs historical official TAL baseline on identical keys",
        "scope": "saved results only; no new inference",
        "candidate_count": len(current), "matched_count": len(differences), "missing": missing,
        "absolute_tolerance": tolerance, "maximum_absolute_iou_error": maximum,
        "outside_tolerance": [r for r in differences if r["absolute_iou_error"] > tolerance],
        "passed": not missing and maximum is not None and maximum <= tolerance,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.historical, args.current)},
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("candidate_count", "matched_count", "maximum_absolute_iou_error", "passed")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
