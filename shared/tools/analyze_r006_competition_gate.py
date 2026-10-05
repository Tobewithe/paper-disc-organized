"""Post-hoc R006 competition-gate analysis from saved traces and oracle mapping."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research")
LEGACY_SCRIPT = LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py"
TRACES = {
    "PigLife_public_test": PROJECT / "experiments" / "yolo26_raw_trace_piglife_full_20260905_v1",
    "FaroPigSeg_test": LEGACY / "artifacts" / "inference_cache" / "yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse" / "faropigseg_test",
}
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


def image_cluster_interval(rows: list[dict[str, Any]], denominator: str, numerator: str, seed: int) -> tuple[float | None, float | None]:
    clusters: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row[denominator]:
            clusters[int(row["image_id"])][0] += 1
            clusters[int(row["image_id"])][1] += int(row[numerator])
    values = list(clusters.values())
    if not values:
        return None, None
    counts = np.asarray(values, dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        sample = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        if sample[0]:
            samples.append(sample[1] / sample[0])
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def decode_masks(raw: Any, candidate_ids: np.ndarray, shape: tuple[int, int]) -> dict[int, dict[str, Any]]:
    import torch
    from pycocotools import mask as mask_utils
    from ultralytics.utils import ops

    result: dict[int, dict[str, Any]] = {}
    proto = torch.from_numpy(raw["prototype"])
    for start in range(0, len(candidate_ids), 8):
        ids = candidate_ids[start : start + 8]
        coeff = torch.from_numpy(raw["mask_coefficients"][ids])
        boxes = torch.from_numpy(raw["boxes_xyxy"][ids])
        masks = ops.process_mask_native(proto, coeff, boxes, shape).cpu().numpy().astype(np.uint8)
        for candidate, mask in zip(ids.tolist(), masks, strict=True):
            rle = mask_utils.encode(np.asfortranarray(mask))
            result[int(candidate)] = {"size": [int(x) for x in rle["size"]], "counts": rle["counts"].decode("ascii")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r006-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.r006_dir / "competition_gate_analysis_v3.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    fixed = load_legacy()
    specs = {spec["name"]: spec for spec in fixed.DATASETS if spec["name"] in TRACES}
    choices = list(csv.DictReader((args.r006_dir / "oracle_candidate_mapping.csv").open(encoding="utf-8-sig", newline="")))
    post = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row for row in csv.DictReader((args.r006_dir / "addition_removal_gt_analysis.csv").open(encoding="utf-8-sig", newline=""))}
    by_image: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in choices:
        by_image[(row["dataset"], int(row["image_id"]))].append(row)

    records = {dataset: {int(row["image_id"]): row for row in read_jsonl(TRACES[dataset] / "diagnostic_cache.jsonl")} for dataset in specs}
    manifests = {dataset: fixed.read_json(spec["manifest"]) for dataset, spec in specs.items()}
    images = {dataset: {int(row["id"]): row for row in manifest["images"]} for dataset, manifest in manifests.items()}
    annotations = {dataset: defaultdict(list) for dataset in specs}
    for dataset, manifest in manifests.items():
        for row in manifest["annotations"]:
            if int(row.get("category_id", 1)) == 1 and not row.get("iscrowd", 0):
                annotations[dataset][int(row["image_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, image_id), group in by_image.items():
        record = records[dataset][image_id]
        image = images[dataset][image_id]
        h, w = int(image["height"]), int(image["width"])
        ann_by_id = {int(row["id"]): row for row in annotations[dataset][image_id]}
        target_masks = [fixed.normalize_gt(ann_by_id[int(row["annotation_id"])], h, w) for row in group]
        pred_rles = [fixed.rle(row["mask_rle"]) for row in record["final_predictions"]]
        iou, coverage, purity = fixed.matrices(target_masks, pred_rles)
        source_by_pred = {int(row["pred_id"]): int(row["source_candidate_id"]) for row in record["final_pred_to_source_candidate"]}
        with np.load(TRACES[dataset] / record["raw_cache"], allow_pickle=False) as raw:
            ranks = raw["global_rank"].astype(int)
            scores = raw["scores"].astype(float)
            top_indices = set(raw["top_indices"].astype(int).tolist())
            selected_ids = np.asarray(sorted({int(row["selected_source_candidate_id"]) for row in group if row["selected_source_candidate_id"]}), dtype=np.int64)
            selected_rles = decode_masks(raw, selected_ids, (h, w)) if len(selected_ids) else {}
        for index, choice in enumerate(group):
            core = (iou[index] >= 0.50) | (coverage[index] >= 0.50)
            core_indices = np.flatnonzero(core)
            core_ranks = [int(ranks[source_by_pred[int(record["final_predictions"][pred_index]["pred_id"])]]) for pred_index in core_indices]
            final_complete = (iou[index] >= 0.50) & (coverage[index] >= 0.70)
            final_local = (iou[index] >= 0.10) & (iou[index] < 0.50) & (coverage[index] >= 0.10) & (purity[index] >= 0.60)
            selected = choice["selected_source_candidate_id"]
            selected_rank = int(choice["selected_global_rank"]) if selected else None
            selected_complete = bool(selected) and float(choice["selected_raw_gt_coverage"]) >= 0.70
            selected_missing = bool(selected) and int(choice["selected_already_final"]) == 0
            selected_source = int(selected) if selected else None
            selected_topk = int(selected_source in top_indices) if selected_source is not None else 0
            selected_conf = int(selected_source is not None and scores[selected_source] >= 0.05)
            duplicate = np.zeros_like(core, dtype=bool)
            if selected_source is not None and pred_rles:
                from pycocotools import mask as mask_utils

                pair_iou = mask_utils.iou([selected_rles[selected_source]], pred_rles, [0] * len(pred_rles))[0]
                duplicate = pair_iou >= 0.80
            competitor = final_local | duplicate
            competitor_indices = np.flatnonzero(competitor)
            competitor_ranks = [int(ranks[source_by_pred[int(record["final_predictions"][pred_index]["pred_id"])]]) for pred_index in competitor_indices]
            rows.append({
                "dataset": dataset,
                "image_id": image_id,
                "annotation_id": int(choice["annotation_id"]),
                "baseline_class": choice["baseline_class"],
                "addition_removal_class": post[(dataset, image_id, int(choice["annotation_id"]))]["primary_class"],
                "recovered_to_C": int(choice["baseline_class"] != "C" and post[(dataset, image_id, int(choice["annotation_id"]))]["primary_class"] == "C"),
                "relation_failure": int(choice["baseline_class"] in RELATION),
                "oracle_assigned": int(bool(selected)),
                "selected_complete": int(selected_complete),
                "selected_missing_from_final": int(selected_missing),
                "selected_global_rank": selected_rank,
                "selected_in_top300": selected_topk,
                "selected_pass_conf005": selected_conf,
                "best_final_core_rank": min(core_ranks) if core_ranks else None,
                "worst_final_core_rank": max(core_ranks) if core_ranks else None,
                "final_duplicate_competitor_count": int(duplicate.sum()),
                "final_local_prediction_count": int(final_local.sum()),
                "final_competitor_count": len(competitor_ranks),
                "best_final_competitor_rank": min(competitor_ranks) if competitor_ranks else None,
                "pre_top300_score_loss": int(selected_missing and selected_complete and not selected_topk),
                "top300_confidence_loss": int(selected_missing and selected_complete and selected_topk and not selected_conf),
                "post_ranking_exclusion": int(selected_missing and selected_complete and selected_topk and selected_conf and bool(competitor_ranks) and selected_rank < min(competitor_ranks)),
                "final_set_competition": int(selected_missing and selected_complete and selected_topk and selected_conf and bool(competitor_ranks) and selected_rank > min(competitor_ranks)),
                "final_complete_prediction_count": int(final_complete.sum()),
                "final_complete_local_coexistence": int(final_complete.any() and final_local.any()),
            })

    for row in rows:
        row["gate_competition_case"] = int(row["pre_top300_score_loss"] or row["top300_confidence_loss"] or row["final_set_competition"] or row["final_complete_local_coexistence"])
    write_csv(output, rows)
    summary: list[dict[str, Any]] = []
    for dataset in sorted(specs):
        relation_rows = [row for row in rows if row["dataset"] == dataset and row["baseline_class"] in RELATION]
        recovered = [row for row in relation_rows if row["recovered_to_C"]]
        competition = [row for row in recovered if row["gate_competition_case"]]
        recovery_low, recovery_high = image_cluster_interval(relation_rows, "relation_failure", "recovered_to_C", 20260905)
        competition_low, competition_high = image_cluster_interval(relation_rows, "recovered_to_C", "gate_competition_case", 20260905)
        summary.append({
            "dataset": dataset,
            "relation_failed_gt": len(relation_rows),
            "oracle_recovered_gt": len(recovered),
            "oracle_recovery_rate": len(recovered) / len(relation_rows) if relation_rows else None,
            "oracle_recovery_ci95_low": recovery_low,
            "oracle_recovery_ci95_high": recovery_high,
            "pre_top300_score_loss_recovered": sum(row["pre_top300_score_loss"] for row in recovered),
            "top300_confidence_loss_recovered": sum(row["top300_confidence_loss"] for row in recovered),
            "final_set_competition_recovered": sum(row["final_set_competition"] for row in recovered),
            "post_ranking_exclusion_recovered": sum(row["post_ranking_exclusion"] for row in recovered),
            "final_complete_local_coexistence_recovered": sum(row["final_complete_local_coexistence"] for row in recovered),
            "competition_union_recovered": len(competition),
            "competition_share_of_recovered": len(competition) / len(recovered) if recovered else None,
            "competition_share_ci95_low": competition_low,
            "competition_share_ci95_high": competition_high,
        })
    write_csv(args.r006_dir / "competition_gate_summary_v3.csv", summary)
    print(json.dumps({"rows": len(rows), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
