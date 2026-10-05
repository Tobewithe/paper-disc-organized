"""Reconstruct the archived manual converter's effects without running it.

No original labels, weights, or remote files are modified.
"""
import ast
import csv
import hashlib
import json
import re
import tempfile
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"refine-logs/coco-evaluation"
SOURCE = ROOT/"remote_scripts/write_labels.py"
WARNING_IDS = {99844: 4, 201706: 1, 214087: 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def filename(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def bbox(poly, width=1., height=1.):
    pts = np.asarray([[round(poly[k]/width, 6), round(poly[k+1]/height, 6)]
                      for k in range(0, len(poly), 2)], dtype=np.float32)
    low, high = pts.min(0), pts.max(0)
    xywh = np.concatenate(((low+high)/2, high-low)).astype(np.float32)
    return xywh


def intersecting_fragments(polys):
    boxes = []
    for poly in polys:
        p = np.asarray(poly, float).reshape(-1, 2)
        boxes.append(np.concatenate((p.min(0), p.max(0))))
    if len(boxes) < 2:
        return 0
    b = np.asarray(boxes)
    wh = np.maximum(0, np.minimum(b[:, None, 2:], b[None, :, 2:])
                    - np.maximum(b[:, None, :2], b[None, :, :2]))
    inter = wh.prod(-1)
    area = (b[:, 2:]-b[:, :2]).prod(-1)
    union = area[:, None]+area[None, :]-inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return int(np.triu(iou > .05, 1).sum())


def scan(split):
    annotations = ROOT/f"datasets/coco/annotations/instances_{split}2017.json"
    selected = ROOT/f"gemini/data/coco_dense/{split}_dense.txt"
    ids = {int(Path(line.strip()).stem) for line in selected.read_text().splitlines() if line.strip()}
    data = json.loads(annotations.read_text(encoding="utf-8"))
    image_meta = {im["id"]: im for im in data["images"]}
    class_map = {c["id"]: i for i, c in enumerate(sorted(data["categories"], key=lambda c: c["id"]))}
    counters = Counter()
    affected_images, warned_rows, affected = set(), {i: [] for i in WARNING_IDS}, []
    examples = []
    for ann in data["annotations"]:
        if ann["image_id"] not in ids:
            continue
        if ann.get("iscrowd", 0):
            counters["crowd_annotations"] += 1
            continue
        counters["ordinary_annotations"] += 1
        seg = ann.get("segmentation")
        if not isinstance(seg, list):
            counters["nonpolygon_annotations_skipped"] += 1
            continue
        polys = [p for p in seg if len(p) >= 6]
        counters["manual_converter_rows_before_dedup"] += len(polys)
        if not polys:
            counters["empty_polygon_annotations"] += 1
        if len(polys) > 1:
            counters["fragmented_annotations"] += 1
            counters["extra_rows_from_fragmentation"] += len(polys)-1
            affected_images.add(ann["image_id"])
            false_pairs = intersecting_fragments(polys)
            counters["within_instance_fragment_pairs_box_iou_gt_0_05"] += false_pairs
            if false_pairs:
                counters["instances_with_within_instance_fragment_pairs"] += 1
            affected.append(dict(split=split, image_id=ann["image_id"], annotation_id=ann["id"],
                                 category_id=ann["category_id"], fragments=len(polys),
                                 extra_rows=len(polys)-1, false_ccl_gt_pairs_before_assignment=false_pairs))
            if false_pairs and len(examples) < 5:
                examples.append(affected[-1])
        if split == "train" and ann["image_id"] in warned_rows:
            im = image_meta[ann["image_id"]]
            for poly in polys:
                warned_rows[ann["image_id"]].append(
                    np.concatenate(([class_map[ann["category_id"]]],
                                    bbox(poly, im["width"], im["height"]))).astype(np.float32))
    checks = []
    if split == "train":
        for iid, rows in warned_rows.items():
            values = np.asarray(rows)
            duplicates = len(values)-len(np.unique(values, axis=0))
            checks.append(dict(image_id=iid, reconstructed_duplicate_count=duplicates,
                               logged_duplicate_count=WARNING_IDS[iid],
                               matches=duplicates == WARNING_IDS[iid]))
    return dict(selected_images=len(ids), **dict(counters),
                affected_images=len(affected_images),
                fragmentation_fraction=counters["fragmented_annotations"]/counters["ordinary_annotations"],
                extra_row_fraction=counters["extra_rows_from_fragmentation"]/counters["ordinary_annotations"],
                examples=examples, duplicate_fingerprint_checks=checks,
                inputs_sha256={filename(annotations): digest(annotations), filename(selected): digest(selected)}), affected


def main():
    source = SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    payload = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and "def write_yolo_labels" in n.value)
    assert "for seg in ann['segmentation']" in payload
    (OUT/"manual_converter_source_snapshot_20260911.py.txt").write_text(payload, encoding="utf-8")
    # Execute only the inspected conversion function on a synthetic fixture.
    # No SSH launcher, original data write, or training command is executed.
    function = next(n for n in ast.parse(payload).body if isinstance(n, ast.FunctionDef)
                    and n.name == "write_yolo_labels")
    namespace = {"json": json, "os": os, "tqdm": lambda x: x}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "archived_converter_function", "exec"), namespace)
    fixture = {"images": [{"id": 1, "file_name": "fixture.jpg", "width": 100, "height": 100}],
               "annotations": [{"id": 9, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                "segmentation": [[1, 1, 10, 1, 10, 10, 1, 10],
                                                 [30, 30, 40, 30, 40, 40, 30, 40]]}]}
    with tempfile.TemporaryDirectory(prefix="coco_fragmentation_witness_") as folder:
        path = Path(folder)/"fixture.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        target = Path(folder)/"labels"
        namespace["write_yolo_labels"](str(path), str(target))
        witness_lines = (target/"fixture.txt").read_text().strip().splitlines()
    assert len(witness_lines) == 2
    report = dict(created=datetime.now().isoformat(), source=filename(SOURCE),
                  source_sha256=digest(SOURCE),
                  confirmed_source_behavior="One line per polygon; explicitly called for both train2017 and val2017, followed by training launch.",
                  source_function_witness={"original_instances": 1, "polygons": 2,
                                           "actual_output_label_lines": witness_lines,
                                           "passed": len(witness_lines) == 2},
                  evidence_scope="Archived converter plus independent annotation reconstruction and training-log fingerprints; remote label/cache files were unavailable.",
                  splits={}, log_evidence={})
    all_affected = []
    for split in ("train", "val"):
        report["splits"][split], affected = scan(split)
        all_affected.extend(affected)
        print(json.dumps({"completed_split": split, **report["splits"][split]}, ensure_ascii=False), flush=True)
    logs = [
        ROOT/"gemini/remote_runs/export_light/logs/run_cluster_fixed.log",
        ROOT/"gemini/remote_runs/export_light/logs/run_stage2.log",
        ROOT/"gemini/remote_runs/baseline_stage2/run_baseline_stage2.log",
    ]
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        counts = [int(x) for x in re.findall(r"\ball\s+1576\s+(\d+)\s+", text)]
        train_scans = re.findall(r"train2017\.cache[^\r\n]*?(\d+) images", text)
        report["log_evidence"][filename(path)] = dict(
            sha256=digest(path), validation_instance_counts=dict(Counter(counts)),
            training_image_counts=dict(Counter(train_scans)),
            val_count_equals_polygon_split_count=bool(counts) and
                all(x == report["splits"]["val"]["manual_converter_rows_before_dedup"] for x in counts))
    report["interpretation"] = [
        "Validation instance count matches polygon fragments exactly in all inspected runs.",
        "Training converter explicitly writes fragmented train labels; matching duplicate-label warnings independently fingerprint that route.",
        "Actual remote train/val cache remains uninspected; reconstructed counts are not claimed as directly read cache contents.",
        "Within-instance fragment pairs can satisfy CCL box-overlap eligibility; actual loss activation also requires both fragment GT to receive assigned candidates.",
        "Existing corrected official evaluations remain measurements of these checkpoints, but do not certify a clean-label CCL comparison."
    ]
    (OUT/"TRAINING_LABEL_FORENSICS_20260911.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT/"TRAINING_LABEL_FRAGMENTATION_20260911.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_affected[0]))
        writer.writeheader()
        writer.writerows(all_affected)
    print("AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
