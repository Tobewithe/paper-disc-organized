"""Validate and summarize completed R006 GT-guided output-set recovery results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
CONDITIONS = {
    "BASELINE": "baseline_gt_analysis.csv",
    "ADDITION": "addition_gt_analysis.csv",
    "REMOVAL": "removal_gt_analysis.csv",
    "ADDITION_REMOVAL": "addition_removal_gt_analysis.csv",
}
DATASETS = {
    "PigLife_public_test": {
        "manifest": Path(r"C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json"),
        "images": 426,
        "gt": 4474,
        "failed": 394,
    },
    "FaroPigSeg_test": {
        "manifest": Path(r"C:\Users\Public\pigcv-task05\external-inference\faropigseg_v1\test\manifest.json"),
        "images": 160,
        "gt": 1752,
        "failed": 797,
    },
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for value in values for field in value))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gt_map(values: list[dict[str, str]], label: str) -> dict[tuple[str, int, int], str]:
    result = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["primary_class"] for row in values}
    if len(result) != len(values) or any(value not in PRIMARY for value in result.values()):
        raise ValueError(f"Invalid per-GT classes or duplicate keys in {label}")
    return result


def manifest_images(dataset: str) -> list[int]:
    spec = DATASETS[dataset]
    payload = json.loads(spec["manifest"].read_text(encoding="utf-8"))
    image_ids = [int(image["id"]) for image in payload["images"]]
    if len(image_ids) != spec["images"] or len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Unexpected manifest image IDs for {dataset}")
    return sorted(image_ids)


def ratio_bootstrap(numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    sampled_numerator = numerator[draws].sum(axis=1)
    sampled_denominator = denominator[draws].sum(axis=1)
    valid = sampled_denominator > 0
    values = sampled_numerator[valid] / sampled_denominator[valid]
    return {
        "estimate": float(numerator.sum() / denominator.sum()) if denominator.sum() else None,
        "ci95_low": float(np.quantile(values, 0.025)) if len(values) else None,
        "ci95_high": float(np.quantile(values, 0.975)) if len(values) else None,
        "valid_draws": int(valid.sum()),
        "null_denominator_draws": int((~valid).sum()),
    }


def validate(run_dir: Path) -> tuple[dict[str, dict[tuple[str, int, int], str]], dict[str, Any], dict[str, list[dict[str, str]]]]:
    required = [*CONDITIONS.values(), "counterfactual_summary.csv", "counterfactual_transitions.csv", "oracle_candidate_mapping.csv", "run_summary.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing R006 result files: {missing}")
    per_gt_rows = {condition: rows(run_dir / name) for condition, name in CONDITIONS.items()}
    maps = {condition: gt_map(values, condition) for condition, values in per_gt_rows.items()}
    baseline = maps["BASELINE"]
    expected_datasets = set(DATASETS)
    if {key[0] for key in baseline} != expected_datasets:
        raise ValueError("Baseline datasets differ from the R006 public-test contract")
    for condition, current in maps.items():
        if set(current) != set(baseline):
            raise ValueError(f"GT coverage differs for {condition}")

    baseline_checks: dict[str, Any] = {}
    for dataset, spec in DATASETS.items():
        keys = [key for key in baseline if key[0] == dataset]
        failed = sum(baseline[key] != "C" for key in keys)
        if len(keys) != spec["gt"] or failed != spec["failed"]:
            raise ValueError(f"Baseline count mismatch for {dataset}: gt={len(keys)}, failed={failed}")
        image_ids = manifest_images(dataset)
        if not {key[1] for key in keys}.issubset(image_ids):
            raise ValueError(f"GT image outside public manifest for {dataset}")
        baseline_checks[dataset] = {"n_gt": len(keys), "n_failed": failed, "manifest_images": len(image_ids)}

    transition_rows = rows(run_dir / "counterfactual_transitions.csv")
    transition = {(row["condition"], row["dataset"], row["original_class"], row["new_class"]): int(row["n_gt"]) for row in transition_rows}
    expected_transition_rows = len(CONDITIONS) * (len(DATASETS) + 1) * len(PRIMARY) ** 2
    if len(transition) != expected_transition_rows or len(transition_rows) != expected_transition_rows:
        raise ValueError("Transition CSV does not contain every 8x8 condition/dataset matrix exactly once")
    for condition, current in maps.items():
        for dataset in [*DATASETS, "ALL"]:
            keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
            expected = Counter((baseline[key], current[key]) for key in keys)
            for old in PRIMARY:
                for new in PRIMARY:
                    if transition.get((condition, dataset, old, new)) != expected[(old, new)]:
                        raise ValueError(f"Transition mismatch for {condition}/{dataset}/{old}->{new}")

    mapping_rows = rows(run_dir / "oracle_candidate_mapping.csv")
    mapping = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row for row in mapping_rows}
    failed_keys = {key for key, value in baseline.items() if value != "C"}
    if len(mapping) != len(mapping_rows) or set(mapping) != failed_keys:
        raise ValueError("Oracle mapping does not exactly cover baseline-failed GT")
    selected = {key: row for key, row in mapping.items() if row["selected_source_candidate_id"] not in {"", None}}
    sources = [(key[0], key[1], int(row["selected_source_candidate_id"])) for key, row in selected.items()]
    if len(sources) != len(set(sources)):
        raise ValueError("Oracle mapping reuses a selected source within dataset/image")

    summary_rows = rows(run_dir / "counterfactual_summary.csv")
    summary = {(row["condition"], row["dataset"]): row for row in summary_rows}
    if len(summary) != len(summary_rows) or len(summary) != len(CONDITIONS) * (len(DATASETS) + 1):
        raise ValueError("Summary CSV is missing condition/dataset rows")
    for condition, current in maps.items():
        for dataset in [*DATASETS, "ALL"]:
            keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
            before, after = Counter(baseline[key] for key in keys), Counter(current[key] for key in keys)
            moves = Counter((baseline[key], current[key]) for key in keys)
            row = summary[(condition, dataset)]
            if int(row["n_gt"]) != len(keys) or any(int(row[f"baseline_{label}"]) != before[label] or int(row[f"new_{label}"]) != after[label] for label in PRIMARY):
                raise ValueError(f"Summary class counts mismatch for {condition}/{dataset}")
            if int(row["recovered_to_C"]) != sum(count for (old, new), count in moves.items() if old != "C" and new == "C") or int(row["C_to_failure"]) != sum(count for (old, new), count in moves.items() if old == "C" and new != "C"):
                raise ValueError(f"Summary transition counts mismatch for {condition}/{dataset}")
    for condition in CONDITIONS:
        row = summary[(condition, "ALL")]
        original, removed, added, new = (int(row[field]) for field in ("original_predictions", "removed_predictions", "added_predictions", "new_predictions"))
        if new != original - removed + added:
            raise ValueError(f"Prediction accounting mismatch for {condition}")

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if run_summary["baseline_gt"] != len(baseline) or run_summary["target_failed_gt"] != len(failed_keys) or run_summary["oracle_assigned"] != len(selected):
        raise ValueError("run_summary counts differ from CSV inputs")
    checks = {"baseline": baseline_checks, "oracle_assigned": len(selected), "target_failed_gt": len(failed_keys), "transition_rows": expected_transition_rows}
    return maps, checks, {"mapping": mapping_rows, "summary": summary_rows}


def statistics(maps: dict[str, dict[tuple[str, int, int], str]], auxiliary: dict[str, list[dict[str, str]]], replicates: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = maps["BASELINE"]
    mapping = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row for row in auxiliary["mapping"]}
    selected_keys = {key for key, row in mapping.items() if row["selected_source_candidate_id"] not in {"", None}}
    records: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for dataset in DATASETS:
        image_ids = manifest_images(dataset)
        index = {image_id: position for position, image_id in enumerate(image_ids)}
        draws = rng.integers(0, len(image_ids), size=(replicates, len(image_ids)))
        keys = [key for key in baseline if key[0] == dataset]
        baseline_fail = {key for key in keys if baseline[key] != "C"}
        baseline_correct = {key for key in keys if baseline[key] == "C"}
        assigned = selected_keys & set(keys)
        for condition, current in maps.items():
            recovered = {key for key in baseline_fail if current[key] == "C"}
            matched_recovered = recovered & assigned
            c_to_failure = {key for key in baseline_correct if current[key] != "C"}
            current_fail = {key for key in keys if current[key] != "C"}
            record = {
                "dataset": dataset,
                "condition": condition,
                "N_all": len(keys),
                "N_fail": len(baseline_fail),
                "N_baseline_C": len(baseline_correct),
                "N_assigned": len(assigned),
                "recovered_to_C": len(recovered),
                "recovered_to_C_over_N_fail": len(recovered) / len(baseline_fail),
                "matched_target_recovered": len(matched_recovered),
                "matched_target_recovered_over_N_assigned": len(matched_recovered) / len(assigned) if assigned else None,
                "C_to_failure": len(c_to_failure),
                "C_to_failure_over_N_baseline_C": len(c_to_failure) / len(baseline_correct),
                "current_failures": len(current_fail),
                "net_failure_reduction": len(baseline_fail) - len(current_fail),
            }
            records.append(record)
            vectors = {
                "recovered_to_C_over_N_fail": (recovered, baseline_fail),
                "matched_target_recovered_over_N_assigned": (matched_recovered, assigned),
                "C_to_failure_over_N_baseline_C": (c_to_failure, baseline_correct),
            }
            for metric, (numerator_keys, denominator_keys) in vectors.items():
                numerator = np.zeros(len(image_ids), dtype=np.int32)
                denominator = np.zeros(len(image_ids), dtype=np.int32)
                for key in numerator_keys:
                    numerator[index[key[1]]] += 1
                for key in denominator_keys:
                    denominator[index[key[1]]] += 1
                bootstrap.append({"dataset": dataset, "condition": condition, "metric": metric, "bootstrap_replicates": replicates, "seed": seed, **ratio_bootstrap(numerator, denominator, draws)})
    return records, bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    maps, checks, auxiliary = validate(args.run_dir)
    records, bootstrap = statistics(maps, auxiliary, args.bootstrap_replicates, args.seed)
    summary = {(row["condition"], row["dataset"]): row for row in auxiliary["summary"]}
    accounting = [
        {"condition": condition, **{field: int(summary[(condition, "ALL")][field]) for field in ("original_predictions", "removed_predictions", "added_predictions", "new_predictions")}}
        for condition in CONDITIONS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "r006_condition_statistics.csv", records)
    write_csv(args.output_dir / "r006_image_cluster_bootstrap.csv", bootstrap)
    write_csv(args.output_dir / "r006_prediction_accounting.csv", accounting)
    report = ["# R006 Full-Trace Statistics", "", "This is realized GT-guided output-set recovery for one specified selection policy. It is not a proven optimal upper bound, causal ranking contribution, deployed AP gain, or candidate-scorer Gate.", "", "| Dataset | Condition | N_all | N_fail | N_assigned | Recovered/N_fail | Matched recovered/N_assigned | C-to-failure/N_baseline_C | Net failure reduction |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    report.extend(f"| {row['dataset']} | {row['condition']} | {row['N_all']} | {row['N_fail']} | {row['N_assigned']} | {row['recovered_to_C']}/{row['N_fail']} | {row['matched_target_recovered']}/{row['N_assigned']} | {row['C_to_failure']}/{row['N_baseline_C']} | {row['net_failure_reduction']} |" for row in records)
    report += ["", "## Output-Set Accounting", "", "| Condition | Original predictions | Removed | Added | New predictions |", "|---|---:|---:|---:|---:|"]
    report.extend(f"| {row['condition']} | {row['original_predictions']} | {row['removed_predictions']} | {row['added_predictions']} | {row['new_predictions']} |" for row in accounting)
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    input_files = [*CONDITIONS.values(), "counterfactual_summary.csv", "counterfactual_transitions.csv", "oracle_candidate_mapping.csv", "run_summary.json"]
    validation = {
        "run_dir": str(args.run_dir),
        "bootstrap": {"unit": "image", "replicates": args.bootstrap_replicates, "seed": args.seed, "manifest_images_included": True, "paired_numerator_denominator_draws": True, "null_denominators_recorded": True},
        "checks": checks,
        "input_sha256": {name: digest(args.run_dir / name) for name in input_files},
        "output_sha256": {name: digest(args.output_dir / name) for name in ("r006_condition_statistics.csv", "r006_image_cluster_bootstrap.csv", "r006_prediction_accounting.csv", "report.md")},
        "script_sha256": digest(Path(__file__)),
    }
    (args.output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "validated", "output_dir": str(args.output_dir), "checks": checks}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
