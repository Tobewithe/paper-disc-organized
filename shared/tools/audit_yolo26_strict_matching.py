"""Audit strict-quality edges and exact simultaneous allocation on frozen traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_yolo26_stage_availability import read_csv, read_json, sha256
from analyze_yolo26_strict_quality import DATASETS, QUALITY, STAGES, quality_flags
from bootstrap_yolo26_gate import bootstrap_rate
from run_yolo26_candidate_perturbation import load_legacy, maximum_cardinality_assignment, write_csv


PROJECT = Path(__file__).resolve().parents[1]


def gt_key(row):
    return row["dataset"], int(row["image_id"]), int(row["annotation_id"])


def match_image(fixed, targets, edges):
    gt_ids = [int(row["annotation_id"]) for row in targets]
    gt_indices = {annotation: index for index, annotation in enumerate(gt_ids)}
    output, components = [], []
    for quality in QUALITY:
        for stage in STAGES:
            options = {annotation: [] for annotation in gt_ids}
            for edge in edges:
                if not int(edge[f"quality_{quality}"]) or (stage != "raw" and not int(edge[f"{stage}_member"])):
                    continue
                options[int(edge["annotation_id"])].append({
                    "source_candidate_id": int(edge["source_candidate_id"]),
                    "raw_mask_iou": float(edge["iou"]), "raw_gt_coverage": float(edge["coverage"]),
                    "score": float(edge["source_score"]), "global_rank": int(edge["global_rank"]),
                })
            for target in targets:
                if len(options[int(target["annotation_id"])]) != int(target[f"{quality}_{stage}_count"]):
                    raise ValueError(f"Edge/count mismatch: {gt_key(target)}/{quality}/{stage}")
            assigned = maximum_cardinality_assignment(options)
            selected_sources = [choice["source_candidate_id"] for choice in assigned.values()]
            if len(selected_sources) != len(set(selected_sources)):
                raise ValueError("Matching reused a candidate source")
            sources = sorted({choice["source_candidate_id"] for choices in options.values() for choice in choices})
            source_indices = {source: index for index, source in enumerate(sources)}
            matrix = np.zeros((len(gt_ids), len(sources)), dtype=bool)
            for annotation, choices in options.items():
                for choice in choices:
                    matrix[gt_indices[annotation], source_indices[choice["source_candidate_id"]]] = True
            groups = fixed.graph_components(matrix)
            groups.extend(([index], []) for index, annotation in enumerate(gt_ids) if not options[annotation])
            component_by_gt = {}
            for component_id, (gs, ps) in enumerate(groups):
                available = sum(bool(options[gt_ids[index]]) for index in gs)
                matched = sum(gt_ids[index] in assigned for index in gs)
                for index in gs:
                    component_by_gt[gt_ids[index]] = component_id
                components.append({
                    "dataset": targets[0]["dataset"], "image_id": targets[0]["image_id"],
                    "quality": quality, "stage": stage, "component_id": component_id,
                    "gt_count": len(gs), "candidate_count": len(ps),
                    "baseline_correct_gt": sum(targets[index]["baseline_class"] == "C" for index in gs),
                    "available_gt": available, "matched_gt": matched,
                    "absent_gt": len(gs) - available, "assignment_deficit": available - matched,
                })
            if set(component_by_gt) != set(gt_ids):
                raise ValueError("Component partition lost GTs")
            for target in targets:
                annotation = int(target["annotation_id"])
                choice = assigned.get(annotation)
                available = int(bool(options[annotation]))
                output.append({
                    **{key: target[key] for key in ("dataset", "image_id", "annotation_id", "baseline_class", "scene_label")},
                    "quality": quality, "stage": stage, "available": available, "matched": int(choice is not None),
                    "assignment_conflict": int(available and choice is None),
                    "selected_source_candidate_id": choice["source_candidate_id"] if choice else None,
                    "component_id": component_by_gt[annotation],
                })
    return output, components


def validate_edge(edge, targets, seen):
    key = gt_key(edge)
    if key not in targets or edge["baseline_class"] != targets[key]["baseline_class"]:
        raise ValueError(f"Edge refers to unknown or changed GT: {key}")
    identity = (*key, int(edge["source_candidate_id"]))
    if identity in seen:
        raise ValueError(f"Duplicate quality edge: {identity}")
    seen.add(identity)
    iou, cov, pur = (float(edge[name]) for name in ("iou", "coverage", "purity"))
    if not all(np.isfinite(value) and 0 < value <= 1 for value in (iou, cov, pur)):
        raise ValueError(f"Invalid edge quality: {identity}")
    if not np.isclose(iou, 1 / (1 / cov + 1 / pur - 1), rtol=1e-9, atol=1e-12):
        raise ValueError(f"Inconsistent IoU/coverage/purity: {identity}")
    flags = quality_flags(iou, cov, pur)
    if not flags["iou50"] or any(int(edge[f"quality_{quality}"]) != flags[quality] for quality in QUALITY):
        raise ValueError(f"Incorrect quality flag: {identity}")
    memberships = [1] + [int(edge[f"{stage}_member"]) for stage in STAGES[1:]]
    if any(value not in (0, 1) for value in memberships) or memberships != sorted(memberships, reverse=True):
        raise ValueError(f"Non-nested source stages: {identity}")


def summarize(rows):
    summary = []
    for dataset in DATASETS:
        for scope in ("ALL_GT", "FAILED_GT", "BASELINE_C"):
            for quality in QUALITY:
                for stage in STAGES:
                    subset = [row for row in rows if row["dataset"] == dataset and row["quality"] == quality and row["stage"] == stage and (scope == "ALL_GT" or (row["baseline_class"] == "C") == (scope == "BASELINE_C"))]
                    if not subset:
                        continue
                    row = {"dataset": dataset, "scope": scope, "quality": quality, "stage": stage, "gt_count": len(subset), "images_with_scoped_gt": len({r["image_id"] for r in subset})}
                    for metric in ("available", "matched", "assignment_conflict"):
                        point, low, high = bootstrap_rate(subset, metric, 3000, 20260905)
                        row.update({f"{metric}_gt": sum(r[metric] for r in subset), f"{metric}_rate": point, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
                    summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    run = read_json(args.quality_dir / "summary.json")
    if run["status"] != "completed" or (not run["full_coverage"] and not args.allow_smoke):
        raise ValueError("A completed full quality run is required unless --allow-smoke is explicit")
    if run["final_source_xor_total"] or run["final_source_xor_max"]:
        raise ValueError("Quality decoder parity did not pass")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    targets = {}
    by_image = defaultdict(list)
    for row in read_csv(args.quality_dir / "per_gt.csv"):
        key = gt_key(row)
        if key in targets:
            raise ValueError(f"Duplicate GT: {key}")
        targets[key] = row
        by_image[key[:2]].append(row)
    if len(targets) != run["gt"] or len(by_image) != run["images"]:
        raise ValueError("Quality summary coverage mismatch")
    edges_by_image, seen = defaultdict(list), set()
    for edge in read_csv(args.quality_dir / "edge_metrics.csv"):
        validate_edge(edge, targets, seen)
        edges_by_image[gt_key(edge)[:2]].append(edge)
    if len(seen) != run["edges"]:
        raise ValueError("Quality summary edge count mismatch")
    previous_path = PROJECT / "experiments/yolo26_stage_availability_full_20260905_v1/stage_gt.csv"
    previous = read_csv(previous_path)
    compared = 0
    for old in previous:
        key = gt_key(old)
        if key not in targets:
            if run["full_coverage"]:
                raise ValueError(f"Missing previously audited failed GT: {key}")
            continue
        for stage in STAGES:
            if int(old[f"{stage}_good_count"]) != int(targets[key][f"iou50_{stage}_count"]):
                raise ValueError(f"Previous IoU-50 count mismatch: {key}/{stage}")
        compared += 1
    fixed = load_legacy()
    rows, components = [], []
    for key, image_targets in by_image.items():
        matched, groups = match_image(fixed, image_targets, edges_by_image[key])
        rows.extend(matched)
        components.extend(groups)
    if len(rows) != len(targets) * len(QUALITY) * len(STAGES):
        raise ValueError("Matching output lost GTs")
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_gt_matching.csv", rows)
    write_csv(args.output_dir / "components.csv", components)
    write_csv(args.output_dir / "summary.csv", summaries)
    audit = {
        "status": "completed", "full_coverage": run["full_coverage"], "images": len(by_image), "gt": len(targets),
        "edges": len(seen), "matching_rows": len(rows), "previous_iou50_gt_compared": compared,
        "definition": "Exact maximum-cardinality assignment over all GT in each image, including baseline-C neighbors. Edges are pairwise quality eligibility only; no exclusion of other-GT contamination or guarantee of class-C output-set recovery.",
        "matching_ties": "Existing R006 deterministic quality/score/rank/source ordering; unmatched identity can depend on ties.",
        "bootstrap": {"draws": 3000, "seed": 20260905, "unit": "image", "scope": "conditional on images with scoped GT"},
        "inputs_sha256": {str(path): sha256(path) for path in [Path(__file__), Path(fixed.__file__), PROJECT / "tools/run_yolo26_candidate_perturbation.py", previous_path, args.quality_dir / "per_gt.csv", args.quality_dir / "edge_metrics.csv", args.quality_dir / "summary.json", args.quality_dir / "run_metadata.json"]},
    }
    audit["outputs_sha256"] = {name: sha256(args.output_dir / name) for name in ("per_gt_matching.csv", "components.csv", "summary.csv")}
    (args.output_dir / "validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "full_coverage", "images", "gt", "edges", "matching_rows", "previous_iou50_gt_compared")}))


if __name__ == "__main__":
    main()
