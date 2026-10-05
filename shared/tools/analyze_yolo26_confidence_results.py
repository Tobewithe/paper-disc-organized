"""Summarize frozen R005c results with paired image-cluster uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_yolo26_oracle_results import (
    DATASETS, PRIMARY, digest, gt_map, manifest_images, ratio_bootstrap, rows, write_csv,
)


CONDITIONS = ("BASELINE", "CONFIDENCE_ONLY", "ADDITION", "REMOVAL", "ADDITION_REMOVAL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")

    input_names = ("all_gt_analysis.csv", "condition_summary.csv", "oracle_candidate_mapping.csv", "coco_metrics.json", "run_summary.json")
    inputs = [args.run_dir / name for name in input_names]
    inputs += [spec["manifest"] for spec in DATASETS.values()]
    inputs += [Path(__file__), Path(__file__).with_name("audit_yolo26_oracle_results.py")]
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    all_rows = rows(args.run_dir / "all_gt_analysis.csv")
    if {row["condition"] for row in all_rows} != set(CONDITIONS):
        raise ValueError("Unexpected condition set")
    maps = {condition: gt_map([r for r in all_rows if r["condition"] == condition], condition) for condition in CONDITIONS}
    baseline = maps["BASELINE"]
    if {key[0] for key in baseline} != set(DATASETS):
        raise ValueError("Unexpected dataset set")
    if any(set(current) != set(baseline) for current in maps.values()):
        raise ValueError("Condition GT keys differ from baseline")
    mapping_rows = rows(args.run_dir / "oracle_candidate_mapping.csv")
    mapping = {(r["dataset"], int(r["image_id"]), int(r["annotation_id"])): r for r in mapping_rows}
    if len(mapping) != len(mapping_rows) or not set(mapping).issubset(baseline):
        raise ValueError("Duplicate or unknown target keys")
    if any(baseline[key] == "C" for key in mapping):
        raise ValueError("Target pool contains baseline-correct GT")
    selected = {key for key, row in mapping.items() if row["selected_source_candidate_id"]}
    recorded = {(r["condition"], r["dataset"]): r for r in rows(args.run_dir / "condition_summary.csv")}
    if len(recorded) != len(CONDITIONS) * (len(DATASETS) + 1):
        raise ValueError("Incomplete condition summary")
    metrics = json.loads((args.run_dir / "coco_metrics.json").read_text(encoding="utf-8"))
    statistics, intervals, transitions, outside_recovery = [], [], [], []
    rng = np.random.default_rng(args.seed)
    for dataset, spec in DATASETS.items():
        image_ids = manifest_images(dataset)
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        manifest = json.loads(spec["manifest"].read_text(encoding="utf-8"))
        keys = {key for key in baseline if key[0] == dataset}
        annotation_keys = {(dataset, int(a["image_id"]), int(a["id"])) for a in manifest["annotations"]}
        if keys != annotation_keys or len(keys) != spec["gt"]:
            raise ValueError(f"GT manifest mismatch: {dataset}")
        failed = {key for key in keys if baseline[key] != "C"}
        correct = keys - failed
        if len(failed) != spec["failed"]:
            raise ValueError(f"Baseline failure count mismatch: {dataset}")
        target_keys, selected_keys = keys & set(mapping), keys & selected
        draws = rng.integers(0, len(image_ids), size=(args.replicates, len(image_ids)))

        def vector(subset: set) -> np.ndarray:
            result = np.zeros(len(image_ids), dtype=np.int32)
            for key in subset:
                result[image_index[key[1]]] += 1
            return result

        for condition, current in maps.items():
            recovered = {key for key in failed if current[key] == "C"}
            regressed = {key for key in correct if current[key] != "C"}
            after_failed = {key for key in keys if current[key] != "C"}
            counts = Counter(current[key] for key in keys)
            summary = recorded[(condition, dataset)]
            if int(summary["n_gt"]) != len(keys) or int(summary["recovered_to_C"]) != len(recovered) or int(summary["C_to_failure"]) != len(regressed):
                raise ValueError(f"Summary mismatch: {condition}/{dataset}")
            if any(int(summary[f"new_{label}"]) != counts[label] for label in PRIMARY):
                raise ValueError(f"Summary class mismatch: {condition}/{dataset}")
            ap = metrics[f"{condition}/{dataset}"]
            base_ap = metrics[f"BASELINE/{dataset}"]
            ap_delta = ap["mask_ap"] - base_ap["mask_ap"]
            record = {
                "dataset": dataset, "condition": condition, "images": len(image_ids), "n_gt": len(keys),
                "baseline_failed": len(failed), "baseline_correct": len(correct),
                "target_count": len(target_keys), "selected_count": len(selected_keys),
                "recovered": len(recovered), "regressed": len(regressed),
                "target_recovered": len(recovered & target_keys),
                "selected_recovered": len(recovered & selected_keys),
                "outside_target_recovered": len(recovered - target_keys),
                "outside_selected_recovered": len(recovered - selected_keys),
                "current_failed": len(after_failed), "net_failure_reduction": len(recovered) - len(regressed),
                "failure_rate": len(after_failed) / len(keys),
                "failure_rate_delta_pp": 100 * (len(regressed) - len(recovered)) / len(keys),
                "relative_failure_reduction": (len(recovered) - len(regressed)) / len(failed),
                "mask_ap": ap["mask_ap"], "mask_ap50": ap["mask_ap50"], "mask_ap75": ap["mask_ap75"],
                "mask_ap_delta_points": 100 * ap_delta,
                "mask_ap_relative_delta": ap_delta / base_ap["mask_ap"],
                "predictions": ap["prediction_count"],
                "added_prediction_count_delta": ap["prediction_count"] - base_ap["prediction_count"],
            }
            statistics.append(record)
            for label in PRIMARY:
                transitions.append({"dataset": dataset, "condition": condition, "original_class": "C", "new_class": label, "n_gt": sum(current[key] == label for key in correct)})
            for key in sorted(recovered - selected_keys):
                outside_recovery.append({"dataset": dataset, "condition": condition, "image_id": key[1], "annotation_id": key[2], "baseline_class": baseline[key], "target_member": int(key in target_keys), "selected_target_member": 0})
            # The same image draws preserve pairing across conditions and denominators.
            vectors = {
                "recovery_over_baseline_failed": (vector(recovered), vector(failed)),
                "regression_over_baseline_correct": (vector(regressed), vector(correct)),
                "failure_rate": (vector(after_failed), vector(keys)),
                "failure_rate_delta": (vector(regressed) - vector(recovered), vector(keys)),
                "selected_target_recovery": (vector(recovered & selected_keys), vector(selected_keys)),
            }
            for metric, (numerator, denominator) in vectors.items():
                intervals.append({"dataset": dataset, "condition": condition, "metric": metric, **ratio_bootstrap(numerator, denominator, draws)})

    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise ValueError("Input changed during analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_statistics.csv", statistics)
    write_csv(args.output_dir / "image_cluster_bootstrap.csv", intervals)
    write_csv(args.output_dir / "correct_gt_transitions.csv", transitions)
    write_csv(args.output_dir / "outside_selected_recoveries.csv", outside_recovery)
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "run_dir": str(args.run_dir.resolve()),
        "input_sha256": hashes, "conditions": list(CONDITIONS), "gt_per_condition": len(baseline),
        "bootstrap": {"unit": "image", "replicates": args.replicates, "seed": args.seed, "paired_conditions": True, "includes_all_manifest_images": True},
        "scope": "Descriptive reanalysis of frozen results, not an independent prediction audit or method acceptance.",
        "limitations": ["No AP uncertainty estimated.", "Image resampling does not address correlation between images from the same sequence or farm.", "Zero observed regressions gives a degenerate empirical bootstrap interval, not proof of zero population risk.", "GT-guided controls are not deployable methods or proven optimal bounds."],
        "output_sha256": {p.name: digest(p) for p in args.output_dir.glob("*.csv")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "statistics": statistics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
