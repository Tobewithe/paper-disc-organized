"""GT-guided candidate addition/removal diagnosis for frozen YOLO26-seg traces.

This is an oracle diagnostic, not deployable inference: GT masks select the
candidate.  The legacy fixed classifier is then rerun over every GT so gains
and regressions use exactly the established C/I/L/S/O/M/X/MISS taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research")
LEGACY_SCRIPT = LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py"
BASELINE_CSV = LEGACY / "artifacts" / "analysis" / "yolo26seg_diagnostic_full_single_forward_20260901_final02" / "gt_analysis.csv"
TRACES = {
    "PigLife_public_test": PROJECT / "experiments" / "yolo26_raw_trace_piglife_full_20260905_v1",
    "FaroPigSeg_test": LEGACY / "artifacts" / "inference_cache" / "yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse" / "faropigseg_test",
}
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
DECODE_BATCH_SIZE = 2


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


def rle_from_mask(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return {"size": [int(x) for x in rle["size"]], "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def load_records(dataset_specs: tuple[dict[str, Any], ...]) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for spec in dataset_specs:
        rows = read_jsonl(TRACES[spec["name"]] / "diagnostic_cache.jsonl")
        by_image = {int(row["image_id"]): row for row in rows}
        if len(by_image) != len(rows) or len(by_image) != int(spec["images"]):
            raise RuntimeError(f"Trace coverage mismatch: {spec['name']} ({len(by_image)}/{spec['images']})")
        records[spec["name"]] = by_image
    return records


def load_baseline_classes(datasets: set[str]) -> dict[tuple[str, int, int], str]:
    with BASELINE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] in datasets]
    result = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["primary_class"] for row in rows}
    if not result:
        raise RuntimeError(f"No saved baseline rows for {sorted(datasets)}")
    return result


def class_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    result = {(str(row["dataset"]), int(row["image_id"]), int(row["annotation_id"])): str(row["primary_class"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Fixed classifier produced duplicate GT keys")
    return result


def clone_records(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[int, dict[str, Any]]]:
    return {dataset: {image_id: copy(record) for image_id, record in by_image.items()} for dataset, by_image in records.items()}


def box_mask_coverage(boxes: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Exact GT-mask coverage upper bound for masks constrained to each box."""
    h, w = gt_mask.shape
    x0 = np.clip(np.floor(boxes[:, 0]).astype(np.int64), 0, w)
    y0 = np.clip(np.floor(boxes[:, 1]).astype(np.int64), 0, h)
    x1 = np.clip(np.ceil(boxes[:, 2]).astype(np.int64), 0, w)
    y1 = np.clip(np.ceil(boxes[:, 3]).astype(np.int64), 0, h)
    integral = np.pad(gt_mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    inside = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return inside / max(float(gt_mask.sum()), 1.0)


def decode_masks(raw: Any, candidate_ids: np.ndarray, shape: tuple[int, int], device: str) -> dict[int, np.ndarray]:
    import torch
    from ultralytics.utils import ops

    result: dict[int, np.ndarray] = {}
    proto = torch.from_numpy(raw["prototype"]).to(device)
    for start in range(0, len(candidate_ids), DECODE_BATCH_SIZE):
        ids = candidate_ids[start : start + DECODE_BATCH_SIZE]
        coeff = torch.from_numpy(raw["mask_coefficients"][ids]).to(device)
        boxes = torch.from_numpy(raw["boxes_xyxy"][ids]).to(device)
        masks = ops.process_mask_native(proto, coeff, boxes, shape).cpu().numpy().astype(bool)
        result.update({int(candidate): mask for candidate, mask in zip(ids.tolist(), masks, strict=True)})
    return result


def candidate_choice_key(choice: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (-float(choice["raw_mask_iou"]), -float(choice["raw_gt_coverage"]), -float(choice["score"]), int(choice["global_rank"]), int(choice["source_candidate_id"]))


def maximum_cardinality_assignment(options: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """Match the most GTs possible, using the declared candidate order for ties."""
    ordered = {annotation_id: sorted(choices, key=candidate_choice_key) for annotation_id, choices in options.items()}
    owner: dict[int, int] = {}
    assigned: dict[int, dict[str, Any]] = {}

    def augment(annotation_id: int, seen: set[int]) -> bool:
        for choice in ordered[annotation_id]:
            source = int(choice["source_candidate_id"])
            if source in seen:
                continue
            seen.add(source)
            previous = owner.get(source)
            if previous is None or augment(previous, seen):
                owner[source] = annotation_id
                assigned[annotation_id] = choice
                return True
        return False

    target_order = sorted(ordered, key=lambda annotation_id: (len(ordered[annotation_id]), candidate_choice_key(ordered[annotation_id][0]) if ordered[annotation_id] else (float("inf"),) * 5, annotation_id))
    for annotation_id in target_order:
        augment(annotation_id, set())
    return assigned


def final_source_parity(fixed: Any, record: dict[str, Any], decoded: dict[int, np.ndarray]) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    predictions = {int(prediction["pred_id"]): prediction for prediction in record["final_predictions"]}
    mappings = {int(item["pred_id"]): int(item["source_candidate_id"]) for item in record["final_pred_to_source_candidate"]}
    if set(predictions) != set(mappings) or len(mappings) != len(record["final_pred_to_source_candidate"]):
        raise RuntimeError(f"Final prediction/source mapping is not one-to-one: image_id={record['image_id']}")
    if len(set(mappings.values())) != len(mappings):
        raise RuntimeError(f"Final source candidates are not unique: image_id={record['image_id']}")
    xors: list[int] = []
    mismatched_pred_ids: list[int] = []
    for pred_id, source in mappings.items():
        actual = decoded.get(source)
        if actual is None:
            raise RuntimeError(f"Missing reconstructed final source: image_id={record['image_id']}, pred_id={pred_id}")
        expected = mask_utils.decode(fixed.rle(predictions[pred_id]["mask_rle"])).astype(bool)
        if actual.shape != expected.shape:
            raise RuntimeError(f"Final mask shape mismatch: image_id={record['image_id']}, pred_id={pred_id}")
        xor = int(np.count_nonzero(np.logical_xor(actual, expected)))
        xors.append(xor)
        if xor:
            mismatched_pred_ids.append(pred_id)
    return {
        "final_sources_compared": len(mappings),
        "all_exact": not mismatched_pred_ids,
        "xor_pixels_total": sum(xors),
        "xor_pixels_max": max(xors, default=0),
        "mismatched_pred_ids": mismatched_pred_ids[:20],
    }


def verify_final_source_parity(fixed: Any, records: dict[str, dict[int, dict[str, Any]]], dataset_specs: tuple[dict[str, Any], ...], device: str) -> dict[str, Any]:
    from ultralytics import __version__ as ultralytics_version

    summary: dict[str, Any] = {"decoder": "ops.process_mask_native", "decoder_batch_size": DECODE_BATCH_SIZE, "ultralytics_version": ultralytics_version, "datasets": {}}
    for spec in dataset_specs:
        dataset = spec["name"]
        manifest = fixed.read_json(spec["manifest"])
        images = {int(image["id"]): image for image in manifest["images"]}
        totals = Counter()
        for image_id, record in sorted(records[dataset].items()):
            image = images.get(image_id)
            if image is None:
                raise RuntimeError(f"Trace image is missing from manifest: {dataset}/{image_id}")
            shape = (int(image["height"]), int(image["width"]))
            with np.load(TRACES[dataset] / record["raw_cache"], allow_pickle=False) as raw:
                if "framework_shape" not in raw.files or tuple(int(value) for value in raw["framework_shape"].tolist()) != shape:
                    raise RuntimeError(f"Framework mask shape does not match manifest: {dataset}/{image_id}")
                source_ids = raw["source_candidate_id"].astype(np.int64)
                if not np.array_equal(source_ids, np.arange(len(source_ids))):
                    raise RuntimeError(f"Raw candidate IDs are not contiguous: {dataset}/{image_id}")
                final_sources = np.asarray(sorted(int(item["source_candidate_id"]) for item in record["final_pred_to_source_candidate"]), dtype=np.int64)
                if np.any((final_sources < 0) | (final_sources >= len(source_ids))):
                    raise RuntimeError(f"Final source ID is outside raw candidates: {dataset}/{image_id}")
                decoded = decode_masks(raw, final_sources, shape, device)
            audit = final_source_parity(fixed, record, decoded)
            if not audit["all_exact"]:
                raise RuntimeError(f"Final-source parity failed: {dataset}/{image_id}, pred_ids={audit['mismatched_pred_ids']}")
            totals.update(images=1, final_sources=audit["final_sources_compared"], xor_pixels_total=audit["xor_pixels_total"])
            totals["xor_pixels_max"] = max(int(totals.get("xor_pixels_max", 0)), audit["xor_pixels_max"])
        summary["datasets"][dataset] = {"all_exact": True, **dict(totals)}
    totals = Counter()
    for dataset_summary in summary["datasets"].values():
        totals.update(images=int(dataset_summary["images"]), final_sources=int(dataset_summary["final_sources"]), xor_pixels_total=int(dataset_summary["xor_pixels_total"]))
        totals["xor_pixels_max"] = max(int(totals.get("xor_pixels_max", 0)), int(dataset_summary.get("xor_pixels_max", 0)))
    summary["totals"] = {"all_exact": True, **dict(totals)}
    return summary


def per_image_choices(
    fixed: Any,
    dataset: str,
    spec: dict[str, Any],
    record: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], set[int]]]:
    """Select one unique raw-good source per failed GT for this image."""
    from pycocotools import mask as mask_utils

    image_id = int(record["image_id"])
    manifest = fixed.read_json(spec["manifest"])
    image = next(item for item in manifest["images"] if int(item["id"]) == image_id)
    anns = sorted((item for item in manifest["annotations"] if int(item["image_id"]) == image_id and int(item.get("category_id", 1)) == 1 and not item.get("iscrowd", 0)), key=lambda item: int(item["id"]))
    ann_by_id = {int(item["id"]): item for item in anns}
    targets = [row for row in baseline_rows if row["dataset"] == dataset and int(row["image_id"]) == image_id and row["primary_class"] != "C"]
    if not targets:
        return [], {}
    h, w = int(image["height"]), int(image["width"])
    gt_masks = {annotation_id: mask_utils.decode(fixed.normalize_gt(ann_by_id[annotation_id], h, w)).astype(bool) for annotation_id in [int(row["annotation_id"]) for row in targets]}
    with np.load(TRACES[dataset] / record["raw_cache"], allow_pickle=False) as raw:
        boxes = raw["boxes_xyxy"].astype(float)
        scores = raw["scores"].astype(float)
        ranks = raw["global_rank"].astype(int)
        final_sources = {int(item["source_candidate_id"]): int(item["pred_id"]) for item in record["final_pred_to_source_candidate"]}
        candidate_ids: set[int] = set()
        for annotation_id, gt_mask in gt_masks.items():
            coverage = box_mask_coverage(boxes, gt_mask)
            # A box-constrained mask with IoU >= .50 must cover >= .50 of the GT mask.
            candidate_ids.update(np.flatnonzero(coverage >= 0.50).tolist())
            candidate_ids.update(source for source in final_sources if coverage[source] >= 0.10)
        decoded = decode_masks(raw, np.asarray(sorted(candidate_ids), dtype=np.int64), (h, w), device)
        rows: list[dict[str, Any]] = []
        removable: dict[tuple[str, int, int], set[int]] = {}
        final_rles = [fixed.rle(pred["mask_rle"]) for pred in record["final_predictions"]]
        final_gt_rles = [fixed.normalize_gt(ann_by_id[int(row["annotation_id"])], h, w) for row in targets]
        final_iou, final_cov, _ = fixed.matrices(final_gt_rles, final_rles)
        options: dict[int, list[dict[str, Any]]] = {}
        for index, target in enumerate(targets):
            annotation_id = int(target["annotation_id"])
            gt = gt_masks[annotation_id]
            area = max(float(gt.sum()), 1.0)
            candidates: list[dict[str, Any]] = []
            for source, mask in decoded.items():
                inter = float(np.logical_and(mask, gt).sum())
                pred_area = float(mask.sum())
                iou = inter / max(area + pred_area - inter, 1.0)
                cov = inter / area
                purity = inter / max(pred_area, 1.0)
                if iou >= 0.50:
                    candidates.append({"source_candidate_id": source, "raw_mask_iou": iou, "raw_gt_coverage": cov, "raw_purity": purity, "score": float(scores[source]), "global_rank": int(ranks[source]), "already_final": int(source in final_sources)})
            candidates.sort(key=candidate_choice_key)
            options[annotation_id] = candidates
            core_pred_ids = {int(record["final_predictions"][pred_index]["pred_id"]) for pred_index in np.flatnonzero((final_iou[index] >= 0.50) | (final_cov[index] >= 0.50))}
            removable[(dataset, image_id, annotation_id)] = core_pred_ids
            local_final = 0
            for source, pred_id in final_sources.items():
                mask = decoded.get(source)
                if mask is None:
                    continue
                inter = float(np.logical_and(mask, gt).sum())
                pred_area = float(mask.sum())
                iou = inter / max(area + pred_area - inter, 1.0)
                cov = inter / area
                purity = inter / max(pred_area, 1.0)
                local_final += int(0.10 <= iou < 0.50 and cov >= 0.10 and purity >= 0.60)
            rows.append({
                "dataset": dataset, "image_id": image_id, "annotation_id": annotation_id, "image_height": h, "image_width": w, "baseline_class": target["primary_class"],
                "raw_good_count": len(candidates), "raw_complete_count": sum(item["raw_gt_coverage"] >= 0.70 for item in candidates),
                "final_core_prediction_count": len(core_pred_ids), "final_local_candidate_count": local_final,
                "complete_local_coexistence": int(any(item["raw_gt_coverage"] >= 0.70 for item in candidates) and local_final > 0),
            })
    assignments = maximum_cardinality_assignment(options)
    for row in rows:
        choices = options[int(row["annotation_id"])]
        selected = assignments.get(int(row["annotation_id"]))
        row["candidate_assignment_conflict"] = int(bool(choices) and selected is None)
        row["candidate_assignment_method"] = "maximum_cardinality; tie=raw_mask_iou,raw_gt_coverage,score,global_rank,source_candidate_id"
        if selected is None:
            row.update({"selected_source_candidate_id": None, "selected_raw_mask_iou": None, "selected_raw_gt_coverage": None, "selected_raw_purity": None, "selected_score": None, "selected_global_rank": None, "selected_already_final": None, "post_ranking_exclusion": 0})
            continue
        row.update({f"selected_{key}": value for key, value in selected.items()})
        row["selected_source_candidate_id"] = row.pop("selected_source_candidate_id")
        core_ranks = []
        for pred_id in removable[(dataset, image_id, int(row["annotation_id"]))]:
            source = next(int(item["source_candidate_id"]) for item in record["final_pred_to_source_candidate"] if int(item["pred_id"]) == pred_id)
            core_ranks.append(int(ranks[source]))
        # Lower rank is better: the candidate outranks every core error but is later excluded.
        row["post_ranking_exclusion"] = int(not selected["already_final"] and bool(core_ranks) and int(selected["global_rank"]) < min(core_ranks))
    return rows, removable


def select_oracle(fixed: Any, records: dict[str, dict[int, dict[str, Any]]], baseline_rows: list[dict[str, Any]], dataset_specs: tuple[dict[str, Any], ...], device: str, limit: int | None) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], set[int]]]:
    rows: list[dict[str, Any]] = []
    removable: dict[tuple[str, int, int], set[int]] = {}
    selected = 0
    specs = {spec["name"]: spec for spec in dataset_specs}
    for dataset, image_records in records.items():
        for image_id, record in sorted(image_records.items()):
            if limit is not None and selected >= limit:
                break
            image_rows, image_removable = per_image_choices(fixed, dataset, specs[dataset], record, baseline_rows, device)
            if limit is not None:
                remaining = limit - selected
                image_rows = image_rows[:remaining]
                image_removable = {key: value for key, value in image_removable.items() if key[2] in {int(row["annotation_id"]) for row in image_rows}}
            rows.extend(image_rows)
            removable.update(image_removable)
            selected += len(image_rows)
    selected_sources = [(row["dataset"], int(row["image_id"]), int(row["selected_source_candidate_id"])) for row in rows if row["selected_source_candidate_id"] is not None]
    if len(selected_sources) != len(set(selected_sources)):
        raise RuntimeError("Oracle candidate mapping reused a source candidate within one image")
    return rows, removable


def perturb(records: dict[str, dict[int, dict[str, Any]]], choices: list[dict[str, Any]], removable: dict[tuple[str, int, int], set[int]], mode: str, device: str) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, int]]:
    selected = clone_records(records)
    by_image: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in choices:
        if row["selected_source_candidate_id"] is not None:
            by_image[(row["dataset"], int(row["image_id"]))].append(row)
    totals = Counter()
    for (dataset, image_id), rows in by_image.items():
        record = selected[dataset][image_id]
        original = list(record["final_predictions"])
        mapping = list(record["final_pred_to_source_candidate"])
        source_to_pred = {int(item["source_candidate_id"]): int(item["pred_id"]) for item in mapping}
        keep_sources = {int(row["selected_source_candidate_id"]) for row in rows if int(row["selected_source_candidate_id"]) in source_to_pred}
        remove_pred_ids: set[int] = set()
        if mode in {"REMOVAL", "ADDITION_REMOVAL"}:
            for row in rows:
                selected_source = int(row["selected_source_candidate_id"])
                for pred_id in removable[(dataset, image_id, int(row["annotation_id"]))]:
                    source = next(int(item["source_candidate_id"]) for item in mapping if int(item["pred_id"]) == pred_id)
                    if source != selected_source and source not in keep_sources:
                        remove_pred_ids.add(pred_id)
        predictions = [item for item in original if int(item["pred_id"]) not in remove_pred_ids]
        mapping = [item for item in mapping if int(item["pred_id"]) not in remove_pred_ids]
        additions = 0
        if mode in {"ADDITION", "ADDITION_REMOVAL"}:
            needed = [row for row in rows if int(row["selected_source_candidate_id"]) not in source_to_pred]
            if needed:
                h, w = int(needed[0]["image_height"]), int(needed[0]["image_width"])
                with np.load(TRACES[dataset] / record["raw_cache"], allow_pickle=False) as raw:
                    ids = np.asarray([int(row["selected_source_candidate_id"]) for row in needed], dtype=np.int64)
                    decoded = decode_masks(raw, ids, (int(h), int(w)), device)
                    next_pred_id = max((int(item["pred_id"]) for item in original), default=0) + 1
                    for row in needed:
                        source = int(row["selected_source_candidate_id"])
                        predictions.append({"pred_id": next_pred_id, "box_xyxy": raw["boxes_xyxy"][source].astype(float).tolist(), "score": float(raw["scores"][source]), "class": 0, "mask_rle": rle_from_mask(decoded[source])})
                        mapping.append({"pred_id": next_pred_id, "source_candidate_id": source, "oracle_added": True})
                        next_pred_id += 1
                        additions += 1
        record["final_predictions"] = predictions
        record["final_pred_to_source_candidate"] = mapping
        totals.update(affected_images=1, removed_predictions=len(remove_pred_ids), added_predictions=additions)
    totals["original_predictions"] = sum(len(record["final_predictions"]) for group in records.values() for record in group.values())
    totals["new_predictions"] = sum(len(record["final_predictions"]) for group in selected.values() for record in group.values())
    return selected, dict(totals)


def transition_rows(condition: str, baseline: dict[tuple[str, int, int], str], current: dict[tuple[str, int, int], str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({key[0] for key in baseline}):
        keys = [key for key in baseline if key[0] == dataset]
        counts = Counter((baseline[key], current[key]) for key in keys)
        output.extend({"condition": condition, "dataset": dataset, "original_class": old, "new_class": new, "n_gt": counts[(old, new)]} for old in PRIMARY for new in PRIMARY)
    counts = Counter((baseline[key], current[key]) for key in baseline)
    output.extend({"condition": condition, "dataset": "ALL", "original_class": old, "new_class": new, "n_gt": counts[(old, new)]} for old in PRIMARY for new in PRIMARY)
    return output


def summary_rows(condition: str, baseline: dict[tuple[str, int, int], str], current: dict[tuple[str, int, int], str], accounting: dict[str, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dataset in [*sorted({key[0] for key in baseline}), "ALL"]:
        keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
        before, after = Counter(baseline[key] for key in keys), Counter(current[key] for key in keys)
        moves = Counter((baseline[key], current[key]) for key in keys)
        result.append({"condition": condition, "dataset": dataset, "n_gt": len(keys), **{f"baseline_{label}": before[label] for label in PRIMARY}, **{f"new_{label}": after[label] for label in PRIMARY}, "total_failure_rate": sum(current[key] != "C" for key in keys) / len(keys), "relation_failure_rate": sum(current[key] in {"O", "M", "X"} for key in keys) / len(keys), "recovered_to_C": sum(count for (old, new), count in moves.items() if old != "C" and new == "C"), "C_to_failure": sum(count for (old, new), count in moves.items() if old == "C" and new != "C"), "O_to_C": moves[("O", "C")], "M_to_C": moves[("M", "C")], "X_to_C": moves[("X", "C")], "MISS_to_C": moves[("MISS", "C")], **(accounting if dataset == "ALL" else {})})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(TRACES), default=sorted(TRACES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-failures", type=int, help="Sanity-only cap; full classifier still reruns on all GT.")
    parser.add_argument("--piglife-trace", type=Path)
    parser.add_argument("--faro-trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.piglife_trace is not None:
        TRACES["PigLife_public_test"] = args.piglife_trace
    if args.faro_trace is not None:
        TRACES["FaroPigSeg_test"] = args.faro_trace
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = load_legacy()
    dataset_specs = tuple(spec for spec in fixed.DATASETS if spec["name"] in args.datasets)
    fixed.DATASETS = dataset_specs
    records = load_records(dataset_specs)
    print("[1/5] verifying final-source batch-2 parity", flush=True)
    parity_summary = verify_final_source_parity(fixed, records, dataset_specs, args.device)
    print("[2/5] rerunning the fixed baseline classifier", flush=True)
    baseline_gt, _, _ = fixed.classify_all(PROJECT, records)
    baseline = class_map(baseline_gt)
    saved = load_baseline_classes(set(args.datasets))
    if baseline != saved:
        mismatch = sum(baseline.get(key) != value for key, value in saved.items()) + len(set(baseline) - set(saved))
        raise RuntimeError(f"Baseline classifier drift against saved GT analysis: {mismatch} GT")
    print("[3/5] selecting GT-guided raw-mask oracle candidates", flush=True)
    choices, removable = select_oracle(fixed, records, baseline_gt, dataset_specs, args.device, args.limit_failures)
    selected = [row for row in choices if row["selected_source_candidate_id"] is not None]
    if not selected:
        raise RuntimeError("No raw-good oracle candidates selected")
    write_csv(args.output_dir / "oracle_candidate_mapping.csv", choices)
    all_summaries: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "datasets": args.datasets, "mode": "Achieved GT-guided output-set replacement recovery under the specified policy; not an optimal bound, deployable inference, or causal attribution", "candidate_rule": "raw mask IoU >= 0.50; each raw source candidate assigned to at most one failed GT per image", "candidate_assignment": "maximum_cardinality; tie=raw_mask_iou,raw_gt_coverage,score,global_rank,source_candidate_id", "complete_rule": "raw IoU >= 0.50 and GT coverage >= 0.70", "local_rule": "0.10 <= raw IoU < 0.50, GT coverage >= 0.10, purity >= 0.60", "final_source_parity": parity_summary, "baseline_gt": len(baseline), "target_failed_gt": len(choices), "oracle_assigned": len(selected), "post_ranking_exclusions": sum(int(row["post_ranking_exclusion"]) for row in selected), "complete_local_coexistence": sum(int(row["complete_local_coexistence"]) for row in choices), "sanity_limit_failures": args.limit_failures}
    for condition in ("BASELINE", "ADDITION", "REMOVAL", "ADDITION_REMOVAL"):
        if condition == "BASELINE":
            current_rows, accounting = baseline_gt, {"original_predictions": sum(len(record["final_predictions"]) for group in records.values() for record in group.values()), "removed_predictions": 0, "added_predictions": 0, "new_predictions": sum(len(record["final_predictions"]) for group in records.values() for record in group.values())}
        else:
            print(f"[4/5] full reclassification: {condition}", flush=True)
            changed, accounting = perturb(records, choices, removable, condition, args.device)
            current_rows, _, _ = fixed.classify_all(PROJECT, changed)
        current = class_map(current_rows)
        if set(current) != set(baseline):
            raise RuntimeError(f"GT coverage drift after {condition}")
        write_csv(args.output_dir / f"{condition.lower()}_gt_analysis.csv", current_rows)
        all_summaries.extend(summary_rows(condition, baseline, current, accounting))
        all_transitions.extend(transition_rows(condition, baseline, current))
    if any(sum(row["n_gt"] for row in all_transitions if row["condition"] == condition and row["dataset"] == "ALL") != len(baseline) for condition in ("BASELINE", "ADDITION", "REMOVAL", "ADDITION_REMOVAL")):
        raise RuntimeError("Transition matrix coverage check failed")
    write_csv(args.output_dir / "counterfactual_summary.csv", all_summaries)
    write_csv(args.output_dir / "counterfactual_transitions.csv", all_transitions)
    payload["outputs"] = {"summary": "counterfactual_summary.csv", "transitions": "counterfactual_transitions.csv", "candidate_mapping": "oracle_candidate_mapping.csv"}
    (args.output_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_rows = [row for row in all_summaries if row["dataset"] == "ALL"]
    report = ["# R006 YOLO26-seg 候选 addition/removal 反事实诊断", "", "## 边界", "", "- 本实验以 GT 选择 raw-mask IoU >= 0.50 的候选，测量指定策略下实际达到的输出集合替换恢复；未证明最优上界，不能作为可部署推理性能或 ranking/NMS 因果归因。", "- 固定 YOLO26 原始 trace、最终预测和 C/I/L/S/O/M/X/MISS 分类器；不训练、不开新模型。", "- addition/removal 后均对全部 GT 重跑固定分类，故能暴露相邻实例的回归。", "", "## 全量汇总", "", "| 条件 | GT | 失败率 | O/M/X率 | 失败转C | C转失败 | O转C | M转C | X转C | MISS转C |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    report.extend(f"| {row['condition']} | {row['n_gt']} | {row['total_failure_rate']:.2%} | {row['relation_failure_rate']:.2%} | {row['recovered_to_C']} | {row['C_to_failure']} | {row['O_to_C']} | {row['M_to_C']} | {row['X_to_C']} | {row['MISS_to_C']} |" for row in all_rows)
    report += ["", "## Oracle 覆盖", "", f"- 失败 GT：{len(choices)}；唯一候选分配：{len(selected)}；候选分数更高却被后续排除：{payload['post_ranking_exclusions']}；complete/local 原始共现：{payload['complete_local_coexistence']}。", "- 逐 GT 候选、可用性、rank 和冲突见 `oracle_candidate_mapping.csv`；完整 8x8 转移矩阵见 `counterfactual_transitions.csv`。", ""]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output_dir), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
