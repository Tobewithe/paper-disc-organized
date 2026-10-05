"""Recompute support counts using COCO masks, OpenCV sums and pixel searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with (args.run_dir / "per_gt_stage.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(int(r["image_id"]), int(r["annotation_id"]), r["stage"]): r for r in rows}
    if len(expected) != len(rows) or len(rows) != 7008:
        raise ValueError("Output keys/count mismatch")
    receipt = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    checked_hashes = 0
    for category in ("input_sha256", "source_sha256", "current_raw_sha256"):
        for path, expected_hash in receipt[category].items():
            with Path(path).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
                if digest is None:
                    h = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(block)
                    digest = h.hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Receipt mismatch: {path}")
            checked_hashes += 1
    trace_file = next(Path(p) for p in receipt["input_sha256"] if Path(p).name == "diagnostic_cache.jsonl")
    manifest_file = next(Path(p) for p in receipt["input_sha256"] if Path(p).name == "manifest.json")
    coco = COCO(str(manifest_file))
    mismatches = []
    checked_rows = 0
    with trace_file.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            image_id = int(trace["image_id"])
            image = coco.imgs[image_id]
            height, width = int(image["height"]), int(image["width"])
            with np.load(trace_file.parent / trace["raw_cache"], allow_pickle=False) as raw:
                boxes, scores, top = raw["boxes_xyxy"], raw["scores"], raw["top_indices"].astype(int)
            # searchsorted follows integer coordinate inequalities without using ceil.
            x0 = np.searchsorted(np.arange(width), boxes[:, 0], side="left")
            x1 = np.searchsorted(np.arange(width), boxes[:, 2], side="left")
            y0 = np.searchsorted(np.arange(height), boxes[:, 1], side="left")
            y1 = np.searchsorted(np.arange(height), boxes[:, 3], side="left")
            empty = (x1 <= x0) | (y1 <= y0)
            stages = {"raw": np.arange(len(boxes)), "topk": top, "conf": top[scores[top] > .05], "final": np.asarray([int(m["source_candidate_id"]) for m in trace["final_pred_to_source_candidate"]], dtype=int)}
            for ann in coco.imgToAnns[image_id]:
                mask = coco.annToMask(ann)
                integral = cv2.integral(mask, sdepth=cv2.CV_64F)
                intersections = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
                intersections[empty] = 0
                coverage = intersections / np.count_nonzero(mask)
                for stage, ids in stages.items():
                    key = (image_id, int(ann["id"]), stage)
                    row = expected[key]
                    best = float(np.max(coverage[ids])) if len(ids) else None
                    stored_best = float(row["best_support"]) if row["best_support"] else None
                    count = int(np.count_nonzero(coverage[ids] >= .75))
                    if count != int(row["support75_count"]) or len(ids) != int(row["candidate_count"]) or best != stored_best:
                        mismatches.append(key)
                    checked_rows += 1
    if checked_rows != len(rows):
        raise ValueError("Incomplete support verification")
    report = {"verdict": "PASS" if not mismatches else "FAIL", "review_independence": "deterministic", "scope": "Support counts and maxima only; not method or annotation-policy acceptance.", "rows_checked": checked_rows, "receipt_files_verified": checked_hashes, "mismatches": mismatches, "implementation": "pycocotools.COCO.annToMask + cv2.integral + np.searchsorted", "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
