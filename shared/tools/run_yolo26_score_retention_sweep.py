"""R007 frozen-trace sweep for YOLO26 end-to-end Top-K and confidence gates.

YOLO26's end-to-end head selects Top-K candidates before its confidence filter;
there is no NMS at this stage.  This script reconstructs that exact gate from
the cached one-forward tensors, validates the recorded baseline first, then
reclassifies every GT for each requested gate setting.  It never runs a model
forward or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research")
LEGACY_SCRIPT = LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py"
BASELINE_CSV = (
    LEGACY
    / "artifacts"
    / "analysis"
    / "yolo26seg_diagnostic_full_single_forward_20260901_final02"
    / "gt_analysis.csv"
)
TRACES = {
    "PigLife_public_test": PROJECT / "experiments" / "yolo26_raw_trace_piglife_full_20260905_v1",
    "FaroPigSeg_test": LEGACY
    / "artifacts"
    / "inference_cache"
    / "yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse"
    / "faropigseg_test",
}
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
RELATION = {"O", "M", "X"}


def load_legacy() -> Any:
    sys.path.insert(0, str(LEGACY / "scripts"))
    spec = importlib.util.spec_from_file_location("yolo26_fixed_classifier", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import fixed classifier: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_records(dataset_specs: tuple[dict[str, Any], ...]) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for spec in dataset_specs:
        rows = read_jsonl(TRACES[spec["name"]] / "diagnostic_cache.jsonl")
        by_image = {int(row["image_id"]): row for row in rows}
        if len(by_image) != len(rows) or len(by_image) != int(spec["images"]):
            raise RuntimeError(f"Trace coverage mismatch: {spec['name']} ({len(by_image)}/{spec['images']})")
        records[spec["name"]] = by_image
    return records


def class_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    result = {(str(row["dataset"]), int(row["image_id"]), int(row["annotation_id"])): str(row["primary_class"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Fixed classifier produced duplicate GT keys")
    return result


def load_saved_baseline(datasets: set[str]) -> dict[tuple[str, int, int], str]:
    with BASELINE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        result = {
            (row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["primary_class"]
            for row in csv.DictReader(handle)
            if row["dataset"] in datasets
        }
    if not result:
        raise RuntimeError(f"No baseline GT rows for {sorted(datasets)}")
    return result


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return {"size": [int(value) for value in rle["size"]], "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def top_ids(raw: Any, top_k: int) -> np.ndarray:
    scores = raw["scores"].astype(np.float32, copy=False)
    if not 1 <= top_k <= len(scores):
        raise ValueError(f"Top-K must be in [1, {len(scores)}], got {top_k}")
    if top_k == len(raw["top_indices"]):
        return raw["top_indices"].astype(np.int64, copy=False)
    import torch

    return torch.topk(torch.from_numpy(scores), top_k).indices.numpy().astype(np.int64, copy=False)


def reconstruct_record(record: dict[str, Any], trace: Path, top_k: int, conf: float, device: str) -> dict[str, Any]:
    """Rebuild one end-to-end output: Top-K first, then strict score filtering."""
    import torch
    from ultralytics.utils import ops

    rebuilt = dict(record)
    h, w = (int(value) for value in record["preprocess_meta"]["original_shape"])
    with np.load(trace / record["raw_cache"], allow_pickle=False) as raw:
        selected = top_ids(raw, top_k)
        selected = selected[raw["scores"][selected] > conf]
        predictions: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        proto = torch.from_numpy(raw["prototype"]).to(device)
        for start in range(0, len(selected), 2):
            ids = selected[start : start + 2]
            coeff = torch.from_numpy(raw["mask_coefficients"][ids]).to(device)
            boxes = torch.from_numpy(raw["boxes_xyxy"][ids]).to(device)
            masks = ops.process_mask_native(proto, coeff, boxes, (h, w)).cpu().numpy().astype(bool)
            for source, mask in zip(ids.tolist(), masks, strict=True):
                if not mask.any():
                    continue
                pred_id = len(predictions) + 1
                predictions.append(
                    {
                        "pred_id": pred_id,
                        "box_xyxy": raw["boxes_xyxy"][source].astype(float).tolist(),
                        "score": float(raw["scores"][source]),
                        "class": 0,
                        "mask_rle": encode_mask(mask),
                    }
                )
                mappings.append({"pred_id": pred_id, "source_candidate_id": int(source), "replayed": True})
    rebuilt["final_predictions"] = predictions
    rebuilt["final_pred_to_source_candidate"] = mappings
    return rebuilt


def reconstruct_all(
    records: dict[str, dict[int, dict[str, Any]]], top_k: int, conf: float, device: str
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset, by_image in records.items():
        output[dataset] = {
            image_id: reconstruct_record(record, TRACES[dataset], top_k, conf, device)
            for image_id, record in sorted(by_image.items())
        }
    return output


def assert_baseline_replay(
    fixed: Any, records: dict[str, dict[int, dict[str, Any]]], baseline: dict[tuple[str, int, int], str], device: str
) -> None:
    rebuilt = reconstruct_all(records, top_k=300, conf=0.05, device=device)
    mismatches = []
    for dataset, by_image in records.items():
        for image_id, record in by_image.items():
            expected, actual = record["final_predictions"], rebuilt[dataset][image_id]["final_predictions"]
            if expected != actual:
                mismatches.append((dataset, image_id, len(expected), len(actual)))
    if mismatches:
        raise RuntimeError(f"Baseline final-prediction replay mismatch in {len(mismatches)} images: {mismatches[:3]}")
    replay_rows, _, _ = fixed.classify_all(PROJECT, rebuilt)
    if class_map(replay_rows) != baseline:
        raise RuntimeError("Baseline Top-K/conf replay changes the fixed GT classification")


def condition_id(top_k: int, conf: float) -> str:
    return f"k{top_k}_conf{conf:.3f}".replace(".", "p")


def summary_rows(
    condition: str, top_k: int, conf: float, baseline: dict[tuple[str, int, int], str], current: dict[tuple[str, int, int], str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in [*sorted({key[0] for key in baseline}), "ALL"]:
        keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
        before, after = Counter(baseline[key] for key in keys), Counter(current[key] for key in keys)
        transitions = Counter((baseline[key], current[key]) for key in keys)
        output.append(
            {
                "condition": condition,
                "top_k": top_k,
                "confidence_threshold": conf,
                "dataset": dataset,
                "n_gt": len(keys),
                **{f"baseline_{label}": before[label] for label in PRIMARY},
                **{f"new_{label}": after[label] for label in PRIMARY},
                "total_failure_rate": sum(current[key] != "C" for key in keys) / len(keys),
                "relation_failure_rate": sum(current[key] in RELATION for key in keys) / len(keys),
                "recovered_to_C": sum(count for (old, new), count in transitions.items() if old != "C" and new == "C"),
                "C_to_failure": sum(count for (old, new), count in transitions.items() if old == "C" and new != "C"),
                "relation_to_C": sum(count for (old, new), count in transitions.items() if old in RELATION and new == "C"),
                "C_to_relation": sum(count for (old, new), count in transitions.items() if old == "C" and new in RELATION),
                "O_to_C": transitions[("O", "C")],
                "M_to_C": transitions[("M", "C")],
                "X_to_C": transitions[("X", "C")],
            }
        )
    return output


def transition_rows(
    condition: str, top_k: int, conf: float, baseline: dict[tuple[str, int, int], str], current: dict[tuple[str, int, int], str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in [*sorted({key[0] for key in baseline}), "ALL"]:
        keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
        counts = Counter((baseline[key], current[key]) for key in keys)
        output.extend(
            {
                "condition": condition,
                "top_k": top_k,
                "confidence_threshold": conf,
                "dataset": dataset,
                "original_class": old,
                "new_class": new,
                "n_gt": counts[(old, new)],
            }
            for old in PRIMARY
            for new in PRIMARY
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(TRACES), default=sorted(TRACES))
    parser.add_argument("--top-k", nargs="+", type=int, default=[100, 300, 500, 1000])
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.05, 0.03, 0.01])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    if 300 not in args.top_k or 0.05 not in args.confidence:
        raise ValueError("The grid must include the recorded baseline: --top-k 300 and --confidence 0.05")
    if any(value < 1 for value in args.top_k) or any(not 0.0 <= value <= 1.0 for value in args.confidence):
        raise ValueError("Top-K must be positive and confidence values must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = load_legacy()
    dataset_specs = tuple(spec for spec in fixed.DATASETS if spec["name"] in args.datasets)
    fixed.DATASETS = dataset_specs
    records = load_records(dataset_specs)
    saved_baseline = load_saved_baseline(set(args.datasets))

    print("[1/3] validating frozen baseline Top-K/conf replay", flush=True)
    assert_baseline_replay(fixed, records, saved_baseline, args.device)

    all_summaries: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    all_gt: list[dict[str, Any]] = []
    cases = [(top_k, conf) for top_k in sorted(set(args.top_k)) for conf in sorted(set(args.confidence), reverse=True)]
    for index, (top_k, conf) in enumerate(cases, start=1):
        condition = condition_id(top_k, conf)
        print(f"[2/3] {index}/{len(cases)} {condition}", flush=True)
        rebuilt = reconstruct_all(records, top_k=top_k, conf=conf, device=args.device)
        current_rows, _, _ = fixed.classify_all(PROJECT, rebuilt)
        current = class_map(current_rows)
        if set(current) != set(saved_baseline):
            raise RuntimeError(f"GT coverage changed in {condition}")
        all_summaries.extend(summary_rows(condition, top_k, conf, saved_baseline, current))
        all_transitions.extend(transition_rows(condition, top_k, conf, saved_baseline, current))
        all_gt.extend({"condition": condition, "top_k": top_k, "confidence_threshold": conf, **row} for row in current_rows)

    if any(
        sum(row["n_gt"] for row in all_transitions if row["condition"] == condition and row["dataset"] == "ALL")
        != len(saved_baseline)
        for condition, _, _ in [(condition_id(top_k, conf), top_k, conf) for top_k, conf in cases]
    ):
        raise RuntimeError("Transition matrix coverage check failed")
    write_csv(args.output_dir / "sweep_summary.csv", all_summaries)
    write_csv(args.output_dir / "sweep_transitions.csv", all_transitions)
    write_csv(args.output_dir / "sweep_gt_analysis.csv", all_gt)

    all_rows = [row for row in all_summaries if row["dataset"] == "ALL"]
    report = [
        "# R007 冻结 Trace 的 Top-K / 置信度保留扫描",
        "",
        "## 设计与边界",
        "",
        "- 重放 YOLO26 end-to-end 推理的两个原始门：先按分数取 Top-K，随后保留 `score > confidence` 的候选；该 head 在此路径不执行 NMS。",
        "- 所有候选框、分数、mask coefficient、prototype 与 GT 分类器均来自冻结 trace；不训练、不做前向推理、不改变模型权重。",
        "- 运行前已要求 `K=300, conf=0.05` 与缓存最终 source-candidate 映射及全 GT 分类完全一致；否则扫描终止。",
        "- 改变门控会同时增加候选和可能引入关系型回归，因此只以全 GT 重分类的收益与回归解释，不能把 oracle 可用性当作部署收益。",
        "",
        "## 全量汇总",
        "",
        "| 条件 | GT | 失败率 | O/M/X率 | 失败转C | C转失败 | O/M/X转C | C转O/M/X |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['condition']} | {row['n_gt']} | {row['total_failure_rate']:.2%} | {row['relation_failure_rate']:.2%} | {row['recovered_to_C']} | {row['C_to_failure']} | {row['relation_to_C']} | {row['C_to_relation']} |"
        for row in all_rows
    )
    report += [
        "",
        "逐数据集汇总见 `sweep_summary.csv`；全 8x8 GT 转移见 `sweep_transitions.csv`；逐 GT 分类见 `sweep_gt_analysis.csv`。",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "mode": "frozen-trace deployable inference-gate replay; no model forward or training",
        "datasets": args.datasets,
        "top_k": sorted(set(args.top_k)),
        "confidence": sorted(set(args.confidence), reverse=True),
        "baseline_replay": {"top_k": 300, "confidence": 0.05, "source_mapping_and_gt_taxonomy": "passed"},
        "outputs": {"summary": "sweep_summary.csv", "transitions": "sweep_transitions.csv", "gt_analysis": "sweep_gt_analysis.csv"},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[3/3] completed", flush=True)
    print(json.dumps({"output": str(args.output_dir), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
