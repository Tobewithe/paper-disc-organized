"""CPU audit of whether relaxing only YOLO26 Top-K changes strict-conf candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TRACES = {
    "PigLife_public_test": Path(r"C:\Dpan\codexproject\pigcv_research\artifacts\inference_cache\yolo26seg_trace_20260905_1930_piglife_full"),
    "FaroPigSeg_test": Path(r"C:\Dpan\codexproject\pigcv_research\artifacts\inference_cache\yolo26seg_trace_20260905_1945_faro_full"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit_trace(dataset: str, trace: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    config_path, cache_path = trace / "inference_config.json", trace / "diagnostic_cache.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    threshold, top_k, expected_images = float(config["config"]["conf"]), int(config["config"]["max_det"]), int(config["manifest_images"])
    if threshold != 0.05 or top_k != 300:
        raise ValueError(f"Unexpected requested gate settings in {trace}: conf={threshold}, top_k={top_k}")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(cache_path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        image_id = int(row["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image {dataset}/{image_id}")
        seen.add(image_id)
        with np.load(trace / row["raw_cache"], allow_pickle=False) as raw:
            source = raw["source_candidate_id"].astype(np.int64)
            scores = raw["scores"].astype(float)
            top_indices = raw["top_indices"].astype(np.int64)
            framework = raw["framework_source_candidate_id"].astype(np.int64)
            if not np.array_equal(source, np.arange(len(source))) or len(scores) != len(source):
                raise ValueError(f"Invalid raw source IDs or scores in {dataset}/{image_id}")
            if len(top_indices) != top_k or len(set(top_indices.tolist())) != top_k or np.any((top_indices < 0) | (top_indices >= len(source))):
                raise ValueError(f"Invalid Top-K IDs in {dataset}/{image_id}")
            if np.any((framework < 0) | (framework >= len(source))):
                raise ValueError(f"Invalid framework source IDs in {dataset}/{image_id}")
            top_set = set(top_indices.tolist())
            strict_pass = scores > threshold
            strict_ids = source[strict_pass]
            outside = [int(candidate) for candidate in strict_ids if int(candidate) not in top_set]
            cached_conf = raw["conf_pass"].astype(bool)
            cached_top = raw["top_k_member"].astype(bool)
            if not np.array_equal(cached_conf, strict_pass):
                raise ValueError(f"Cached conf_pass differs from strict score > conf in {dataset}/{image_id}")
            if not np.array_equal(np.flatnonzero(cached_top), np.sort(top_indices)):
                raise ValueError(f"Cached top_k_member differs from top_indices in {dataset}/{image_id}")
            if outside:
                raise ValueError(f"Strict-conf candidates outside Top-K in {dataset}/{image_id}: {len(outside)}")
            records.append({
                "dataset": dataset,
                "image_id": image_id,
                "raw_candidates": len(source),
                "strict_conf_pass_count": int(strict_pass.sum()),
                "strict_conf_pass_outside_topk_count": len(outside),
                "framework_source_count": len(framework),
                "framework_sources_outside_topk_count": sum(int(candidate) not in top_set for candidate in framework),
                "framework_sources_fail_strict_conf_count": int((scores[framework] <= threshold).sum()),
            })
    if len(seen) != expected_images:
        raise ValueError(f"Trace image count differs from config for {dataset}: {len(seen)} != {expected_images}")
    total = Counter()
    for row in records:
        total.update({key: int(value) for key, value in row.items() if key.endswith("_count") or key == "raw_candidates"})
    summary = {"dataset": dataset, "images": len(records), "conf_strictly_greater_than": threshold, "top_k": top_k, **dict(total), "topk_relaxation_only_is_noop": total["strict_conf_pass_outside_topk_count"] == 0}
    hashes = {"inference_config.json": sha256(config_path), "diagnostic_cache.jsonl": sha256(cache_path), "artifact_hashes.json": sha256(trace / "artifact_hashes.json")}
    return records, summary, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    per_image: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for dataset, trace in TRACES.items():
        records, totals, hashes = audit_trace(dataset, trace)
        per_image.extend(records)
        summary.append(totals)
        input_hashes[dataset] = hashes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_image_gate_counts.csv", per_image)
    write_csv(args.output_dir / "summary.csv", summary)
    report = ["# YOLO26 Strict-Confidence / Top-K Binding Audit", "", "Both traces satisfy: every raw source with `score > 0.05` is already in Top-300. Relaxing only Top-K therefore has no observable effect on the strict-confidence candidate set in these frozen traces.", "", "At the Top-K stage, the first-loss count relative to strict confidence is zero for both traces. Any later output difference is outside the effect of changing only this gate. This audit does not establish which upstream score or ranking mechanism caused any mask-match outcome, and it does not treat mask matching as causal ranking evidence.", "", "| Dataset | Images | Strict-conf raw sources | Strict-conf sources outside Top-300 | Only-Top-K relaxation no-op |", "|---|---:|---:|---:|---|"]
    report.extend(f"| {row['dataset']} | {row['images']} | {row['strict_conf_pass_count']} | {row['strict_conf_pass_outside_topk_count']} | {row['topk_relaxation_only_is_noop']} |" for row in summary)
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_hashes = {name: sha256(args.output_dir / name) for name in ("per_image_gate_counts.csv", "summary.csv", "report.md")}
    validation = {"strict_condition": "score > 0.05", "input_hashes": input_hashes, "output_hashes": output_hashes, "script_sha256": sha256(Path(__file__)), "validation": {"all_images_checked": True, "all_strict_conf_sources_in_top300": all(row["strict_conf_pass_outside_topk_count"] == 0 for row in summary), "framework_sources_checked": True}}
    (args.output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "validated", "summary": summary}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
