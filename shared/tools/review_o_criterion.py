"""Review the fixed O criterion with counterexamples and cached-mask sensitivity."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as masks

import run_yolo26_external_full as task


def labels_for(fixed, edge, coverage, purity):
    labels = ["MISS"] * edge.shape[0]
    components = fixed.graph_components(edge)
    for gs, ps in components:
        if len(gs) == len(ps) == 1:
            g, p = gs[0], ps[0]
            label = "C" if coverage[g, p] >= .75 and purity[g, p] >= .75 else "I" if coverage[g, p] < .75 and purity[g, p] >= .75 else "L" if coverage[g, p] >= .75 else "S"
        else:
            label = "O" if len(gs) == 1 else "M" if len(ps) == 1 else "X"
        for g in gs:
            labels[g] = label
    return labels, components


def encode(array):
    value = masks.encode(np.asfortranarray(array, dtype=np.uint8))
    return {"size": value["size"], "counts": value["counts"].decode("ascii")}


def synthetic(fixed):
    gt = np.zeros((10, 30), dtype=np.uint8)
    gt[:, :10] = 1

    def part(start, end):
        value = np.zeros_like(gt)
        view = np.zeros(100, dtype=np.uint8)
        view[start:end] = 1
        value[:, :10] = view.reshape(10, 10)
        return value

    outside = np.zeros_like(gt)
    outside[:, 10:] = 1
    other_gt = np.zeros_like(gt)
    other_gt[:, 10:20] = 1
    cases = [("two_exact_duplicates", [gt], [gt, gt], ["O"]),
        ("disjoint_50_50", [gt], [part(0, 50), part(50, 100)], ["O"]),
        ("disjoint_60_40", [gt], [part(0, 60), part(60, 100)], ["I"]),
        ("disjoint_40_35_25", [gt], [part(0, 40), part(40, 75), part(75, 100)], ["MISS"]),
        ("complete_plus_30_percent_part", [gt], [gt, part(0, 30)], ["C"]),
        ("two_mostly_outside_masks", [gt], [part(0, 60) | outside, part(40, 100) | outside], ["O"]),
        ("cross_gt_bridge", [gt, other_gt], [gt, gt | other_gt], ["X", "X"])]
    results = []
    for name, truths, predictions, expected in cases:
        anns = [{"id": i + 1, "image_id": 1, "source_annotation_id": i + 1, "segmentation": encode(g)} for i, g in enumerate(truths)]
        coco = {"images": [{"id": 1, "source_image_id": 1, "file_name": name, "height": 10, "width": 30}], "annotations": anns}
        cache = {1: {"predictions": [{"mask_rle": encode(p)} for p in predictions]}}
        actual = [r["primary_class"] for r in task.classify(fixed, coco, cache)]
        if actual != expected:
            raise RuntimeError(f"Unexpected actual-classifier example: {name}: {actual}")
        _, cov, pur = fixed.matrices([fixed.rle(encode(g)) for g in truths], [fixed.rle(encode(p)) for p in predictions])
        results.append({"case": name, "actual_classes": actual, "coverage": cov.tolist(), "purity": pur.tolist()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("Review output already exists")
    fixed_path = task.LEGACY / "scripts/analyze_yolo26_diagnostic_cache_full.py"
    fixed = task.load_module("o_review_fixed_classifier", fixed_path)
    inputs = [fixed_path, task.LEGACY / "scripts/classify_public_test_instance_failures.py", Path(task.__file__),
        Path(__file__).resolve(), task.ROOT / "experiments/o_criterion_review_20260907_protocol.md", source / "summary.json"]
    for name in ("FaroPigSeg", "BamaPig2D"):
        inputs.extend(source / p for p in (f"manifests/{name}_all.json", f"{name}_all_predictions.json", f"{name}_all_gt_analysis.csv"))
    hashes = {str(p): task.sha(p) for p in inputs}
    examples = synthetic(fixed)
    output.mkdir(parents=True)
    task.write(output / "synthetic_cases.json", examples)
    rows, summaries, transition_rows, aggregate = [], [], [], []
    for name in ("FaroPigSeg", "BamaPig2D"):
        coco = task.read(source / "manifests" / f"{name}_all.json")
        anns_by_image, preds_by_image = defaultdict(list), defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)
        for pred in task.read(source / f"{name}_all_predictions.json"):
            preds_by_image[pred["image_id"]].append(pred)
        with (source / f"{name}_all_gt_analysis.csv").open(encoding="utf-8", newline="") as handle:
            saved = {(int(r["image_id"]), int(r["annotation_id"])): r["primary_class"] for r in csv.DictReader(handle)}
        counts, transitions = defaultdict(Counter), defaultdict(Counter)
        checks = Counter()
        for image in coco["images"]:
            anns = anns_by_image[image["id"]]
            preds = preds_by_image[image["id"]]
            gt_rles = [fixed.normalize_gt(a, image["height"], image["width"]) for a in anns]
            pred_rles = [fixed.rle(p["segmentation"]) for p in preds]
            iou, cov, pur = fixed.matrices(gt_rles, pred_rles)
            core = (iou >= .50) | (cov >= .50)
            rules = {"core_040": (iou >= .40) | (cov >= .40), "baseline_050": core,
                "core_060": (iou >= .60) | (cov >= .60), "core_050_purity_050": core & (pur >= .50),
                "legacy_core_or_weak": core | ((pur >= .50) & (cov >= .10))}
            classified = {rule: labels_for(fixed, edge, cov, pur) for rule, edge in rules.items()}
            baseline, comps = classified["baseline_050"]
            checks["iou_disjunct_only_edges"] += int((core & ~(cov >= .50)).sum())
            comp_for_gt = {g: (gs, ps) for gs, ps in comps for g in gs}
            for rule, (labels, _) in classified.items():
                counts[rule].update(labels)
                transitions[rule].update(zip(baseline, labels))
            for g, ann in enumerate(anns):
                if baseline[g] != saved[(image["id"], ann["id"])]:
                    raise RuntimeError("Baseline label does not reproduce")
                row = {"dataset": name, "image_id": image["id"], "annotation_id": ann["id"],
                    "file_name": image["file_name"], "source_split": image["source_split"], "baseline": baseline[g],
                    **{rule: labels[g] for rule, (labels, _) in classified.items()},
                    "core_degree": int(core[g].sum()), "o_min_core_purity": None, "o_max_pair_iou": None,
                    "o_strict_single_count": None, "subhalf_union_coverage": None, "subhalf_union_purity": None,
                    "subhalf_screen": False}
                if baseline[g] == "O":
                    _, ps = comp_for_gt[g]
                    selected = [pred_rles[p] for p in ps]
                    pair_iou = masks.iou(selected, selected, [0] * len(selected))
                    row["o_min_core_purity"] = float(pur[g, ps].min())
                    row["o_max_pair_iou"] = float(pair_iou[np.triu_indices(len(ps), 1)].max())
                    row["o_strict_single_count"] = int(((cov[g, ps] >= .75) & (pur[g, ps] >= .75)).sum())
                    checks["baseline_O"] += 1
                    checks["O_with_purity_below_050_edge"] += row["o_min_core_purity"] < .50
                    checks["O_with_strict_single"] += row["o_strict_single_count"] > 0
                    checks["O_with_pair_iou_ge_080"] += row["o_max_pair_iou"] >= .80
                    checks["O_lost_under_purity_guard"] += row["core_050_purity_050"] != "O"
                if baseline[g] == "X" and row["core_degree"] >= 2:
                    checks["X_with_local_one_to_many_degree"] += 1
                pieces = np.flatnonzero((pur[g] >= .75) & (cov[g] >= .10) & (cov[g] < .50))
                if len(pieces) >= 2:
                    union = masks.merge([pred_rles[p] for p in pieces])
                    _, uc, up = fixed.matrices([gt_rles[g]], [union])
                    row["subhalf_union_coverage"], row["subhalf_union_purity"] = float(uc[0, 0]), float(up[0, 0])
                    row["subhalf_screen"] = bool(uc[0, 0] >= .75 and up[0, 0] >= .75)
                    if row["subhalf_screen"]:
                        checks[f"subhalf_screen_current_{baseline[g]}"] += 1
                rows.append(row)
            if image["id"] % 500 == 0:
                print(f"review {name} image {image['id']}", flush=True)
        for rule, counter in counts.items():
            summaries.append({"dataset": name, "rule": rule, "gt": sum(counter.values()), **{k: counter[k] for k in task.PRIMARY},
                "O_delta": counter["O"] - counts["baseline_050"]["O"],
                "O_relative_delta": (counter["O"] - counts["baseline_050"]["O"]) / counts["baseline_050"]["O"]})
            for (before, after), count in transitions[rule].items():
                transition_rows.append({"dataset": name, "rule": rule, "before": before, "after": after, "gt": count})
        aggregate.append({"dataset": name, "images": len(coco["images"]), "gt": len(saved), **checks})
    if any(task.sha(p) != digest for p, digest in hashes.items()):
        raise RuntimeError("Review input/source changed")
    task.csv_write(output / "audit_rows.csv", rows)
    task.csv_write(output / "rule_summary.csv", summaries)
    task.csv_write(output / "transitions.csv", transition_rows)
    task.write(output / "input_sha256.json", hashes)
    task.write(output / "summary.json", {"status": "completed", "gt_rows_verified": len(rows), "input_hashes_verified": len(hashes),
        "results": aggregate, "scope": "Criteria review on frozen predictions; screens are not validated fragmentation labels; baseline unchanged"})
    print("O_CRITERION_REVIEW_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
