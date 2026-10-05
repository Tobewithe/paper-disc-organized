"""Frozen YOLO26 confidence-retention diagnostic (R005c).

The fixed `.01` gate is replayed from cache.  GT masks select the three
oracle output-set conditions, so those conditions are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research")
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
RELATION = {"O", "M", "X"}
DATASET_NAMES = ("PigLife_public_test", "FaroPigSeg_test")


def load_tool(name: str) -> Any:
    path = PROJECT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(expected: dict[str, str], path: Path, label: str) -> None:
    wanted = expected.get(str(path))
    if wanted is None:
        wanted = next((value for key, value in expected.items() if Path(key).name == path.name), None)
    if wanted is None or sha256(path) != wanted:
        raise RuntimeError(f"Frozen provenance hash mismatch: {label}: {path}")


def frozen_oracle_tuple(choices: list[dict[str, Any]], removable: dict[tuple[str, int, int], set[int]]) -> tuple[str, str, str]:
    return (*digest_choices(choices), digest_removals(removable))


def assert_frozen_oracle(expected: tuple[str, str, str], choices: list[dict[str, Any]], removable: dict[tuple[str, int, int], set[int]], condition: str) -> tuple[str, str, str]:
    actual = frozen_oracle_tuple(choices, removable)
    if actual != expected:
        raise RuntimeError(f"Oracle target/source/removal mapping drift: {condition}")
    return actual


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def class_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    result = {(str(row["dataset"]), int(row["image_id"]), int(row["annotation_id"])): str(row["primary_class"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Classifier produced duplicate GT/image keys")
    return result


def strict_baseline(per_gt: Path) -> dict[tuple[str, int, int], str]:
    rows = read_csv(per_gt)
    result = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["baseline_class"] for row in rows}
    if len(result) != len(rows) or set(key[0] for key in result) != set(DATASET_NAMES):
        raise RuntimeError("Strict per_gt has duplicate or unexpected baseline GT keys")
    return result


def target_rows(per_gt: Path, edges: Path, image_shapes: dict[tuple[str, int], tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    """Return every confidence-lost target, including targets without an edge."""
    targets: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in read_csv(per_gt):
        if row["baseline_class"] == "C":
            continue
        if row["coverage75_purity75_topk_available"] != "1" or row["coverage75_purity75_conf_available"] != "0":
            continue
        if row["coverage75_purity75_final_available"] != "0":
            raise RuntimeError("Strict confidence-lost target is unexpectedly final-available")
        key = (row["dataset"], int(row["image_id"]), int(row["annotation_id"]))
        target = {
            "dataset": key[0], "image_id": key[1], "annotation_id": key[2], "baseline_class": row["baseline_class"],
            "target_stratum": "baseline_non_C__strict_topk_available__strict_conf_unavailable",
            "eligible_edge_count": 0,
        }
        if image_shapes is not None:
            try:
                target["image_height"], target["image_width"] = image_shapes[(key[0], key[1])]
            except KeyError as error:
                raise RuntimeError(f"Target image is missing from manifest geometry: {key[0]}/{key[1]}") from error
        targets[key] = target
    if not targets:
        raise RuntimeError("No strict confidence-lost targets in per_gt")
    for row in read_csv(edges):
        key = (row["dataset"], int(row["image_id"]), int(row["annotation_id"]))
        if key not in targets or row["quality_coverage75_purity75"] != "1" or row["topk_member"] != "1" or row["conf_member"] != "0" or row["final_member"] != "0":
            continue
        score = float(row["source_score"])
        if not 0.01 < score <= 0.05 or float(row["iou"]) < 0.50 or float(row["coverage"]) < 0.75 or float(row["purity"]) < 0.75:
            continue
        target = targets[key]
        target.setdefault("options", []).append({
            "source_candidate_id": int(row["source_candidate_id"]), "raw_mask_iou": float(row["iou"]),
            "raw_gt_coverage": float(row["coverage"]), "raw_purity": float(row["purity"]),
            "score": score, "global_rank": int(row["global_rank"]), "already_final": 0,
        })
        target["eligible_edge_count"] += 1
    return [targets[key] for key in sorted(targets)]


def choose_oracle(targets: list[dict[str, Any]], assign: Any) -> list[dict[str, Any]]:
    """Use one maximum-cardinality assignment for each dataset/image graph."""
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        grouped[(target["dataset"], int(target["image_id"]))].append(target)
    for _, group in sorted(grouped.items()):
        options = {int(row["annotation_id"]): list(row.get("options", [])) for row in group}
        selected = assign(options)
        for row in group:
            row = dict(row)
            choice = selected.get(int(row["annotation_id"]))
            row["candidate_assignment_method"] = "maximum_cardinality; tie=raw_mask_iou,raw_gt_coverage,score,global_rank,source_candidate_id"
            row["candidate_assignment_conflict"] = int(bool(row.get("options")) and choice is None)
            if choice is None:
                row.update({"selected_source_candidate_id": None, "selected_raw_mask_iou": None, "selected_raw_gt_coverage": None, "selected_raw_purity": None, "selected_score": None, "selected_global_rank": None, "selected_already_final": None})
            else:
                row.update({f"selected_{key}": value for key, value in choice.items()})
            row.pop("options", None)
            result.append(row)
    selected_sources = [(row["dataset"], row["image_id"], row["selected_source_candidate_id"]) for row in result if row["selected_source_candidate_id"] is not None]
    if len(selected_sources) != len(set(selected_sources)):
        raise RuntimeError("Assignment reuses a source candidate within an image")
    return result


def clone_records(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[int, dict[str, Any]]]:
    return {dataset: {image_id: copy(record) for image_id, record in rows.items()} for dataset, rows in records.items()}


def removable_predictions(fixed: Any, records: dict[str, dict[int, dict[str, Any]]], specs: tuple[dict[str, Any], ...], choices: list[dict[str, Any]]) -> dict[tuple[str, int, int], set[int]]:
    """Only original final core predictions associated with selected target GTs."""
    from pycocotools import mask as mask_utils

    wanted = {(row["dataset"], int(row["image_id"])): [] for row in choices if row["selected_source_candidate_id"] is not None}
    for row in choices:
        if row["selected_source_candidate_id"] is not None:
            wanted[(row["dataset"], int(row["image_id"]))].append(row)
    manifests = {spec["name"]: fixed.read_json(spec["manifest"]) for spec in specs}
    output: dict[tuple[str, int, int], set[int]] = {}
    for (dataset, image_id), rows in wanted.items():
        manifest = manifests[dataset]
        image = next(item for item in manifest["images"] if int(item["id"]) == image_id)
        h, w = int(image["height"]), int(image["width"])
        annotations = {int(item["id"]): item for item in manifest["annotations"] if int(item["image_id"]) == image_id}
        record = records[dataset][image_id]
        pred_rles = [fixed.rle(pred["mask_rle"]) for pred in record["final_predictions"]]
        for row in rows:
            gt = fixed.normalize_gt(annotations[int(row["annotation_id"])], h, w)
            iou, coverage, _ = fixed.matrices([gt], pred_rles)
            output[(dataset, image_id, int(row["annotation_id"]))] = {
                int(record["final_predictions"][index]["pred_id"])
                for index in np.flatnonzero((iou[0] >= 0.50) | (coverage[0] >= 0.50))
            }
    return output


def attach_manifest_geometry(fixed: Any, specs: tuple[dict[str, Any], ...], choices: list[dict[str, Any]]) -> None:
    manifests = {spec["name"]: fixed.read_json(spec["manifest"]) for spec in specs}
    for row in choices:
        image = next((item for item in manifests[row["dataset"]]["images"] if int(item["id"]) == int(row["image_id"])), None)
        if image is None:
            raise RuntimeError(f"Target image is missing from manifest: {row['dataset']}/{row['image_id']}")
        shape = (int(image["height"]), int(image["width"]))
        if "image_height" in row and (int(row["image_height"]), int(row["image_width"])) != shape:
            raise RuntimeError(f"Target geometry differs between manifests: {row['dataset']}/{row['image_id']}")
        row["image_height"], row["image_width"] = shape


def assert_selected_geometry(records: dict[str, dict[int, dict[str, Any]]], choices: list[dict[str, Any]], traces: dict[str, Path]) -> None:
    for row in choices:
        if row["selected_source_candidate_id"] is None:
            continue
        record = records[row["dataset"]][int(row["image_id"])]
        expected = (int(row["image_height"]), int(row["image_width"]))
        if tuple(int(value) for value in record["preprocess_meta"]["original_shape"]) != expected:
            raise RuntimeError(f"Selected target geometry differs from trace record: {row['dataset']}/{row['image_id']}")
        with np.load(traces[row["dataset"]] / record["raw_cache"], allow_pickle=False) as raw:
            if tuple(int(value) for value in raw["framework_shape"].tolist()) != expected:
                raise RuntimeError(f"Selected target geometry differs from raw framework shape: {row['dataset']}/{row['image_id']}")


def validate_selected_edges(fixed: Any, r006: Any, records: dict[str, dict[int, dict[str, Any]]], specs: tuple[dict[str, Any], ...], choices: list[dict[str, Any]], traces: dict[str, Path], device: str) -> None:
    """Recompute every selected edge from raw scores, membership and decoded masks."""
    from pycocotools import mask as mask_utils

    manifests = {spec["name"]: fixed.read_json(spec["manifest"]) for spec in specs}
    for row in choices:
        source = row.get("selected_source_candidate_id")
        if source is None:
            continue
        dataset, image_id = row["dataset"], int(row["image_id"])
        anns = {int(item["id"]): item for item in manifests[dataset]["annotations"] if int(item["image_id"]) == image_id}
        gt = fixed.normalize_gt(anns[int(row["annotation_id"])], int(row["image_height"]), int(row["image_width"]))
        record = records[dataset][image_id]
        final_sources = {int(item["source_candidate_id"]) for item in record["final_pred_to_source_candidate"]}
        with np.load(traces[dataset] / record["raw_cache"], allow_pickle=False) as raw:
            source = int(source)
            if source < 0 or source >= len(raw["scores"]):
                raise RuntimeError(f"Selected source outside raw candidates: {dataset}/{image_id}/{source}")
            score, rank = float(raw["scores"][source]), int(raw["global_rank"][source])
            if int(raw["source_candidate_id"][source]) != source or score != float(row["selected_score"]) or not 0.01 < score <= 0.05 or rank != int(row["selected_global_rank"]):
                raise RuntimeError(f"Selected edge score/rank mismatch: {dataset}/{image_id}/{source}")
            if source not in set(raw["top_indices"].astype(np.int64).tolist()) or source in final_sources:
                raise RuntimeError(f"Selected edge Top-K/final membership mismatch: {dataset}/{image_id}/{source}")
            decoded = r006.decode_masks(raw, np.asarray([source], dtype=np.int64), (int(row["image_height"]), int(row["image_width"])), device)[source]
        inter = float(np.logical_and(decoded, mask_utils.decode(gt).astype(bool)).sum())
        area, pred_area = max(float(mask_utils.area(gt)), 1.0), max(float(decoded.sum()), 1.0)
        iou, coverage, purity = inter / max(area + pred_area - inter, 1.0), inter / area, inter / pred_area
        expected = (float(row["selected_raw_mask_iou"]), float(row["selected_raw_gt_coverage"]), float(row["selected_raw_purity"]))
        if min(iou, coverage, purity) < 0 or not (iou >= .50 and coverage >= .75 and purity >= .75):
            raise RuntimeError(f"Selected edge fails raw mask quality: {dataset}/{image_id}/{source}")
        if max(abs(iou - expected[0]), abs(coverage - expected[1]), abs(purity - expected[2])) > 1e-5:
            raise RuntimeError(f"Selected edge quality differs from edge_metrics: {dataset}/{image_id}/{source}")


def digest_choices(choices: list[dict[str, Any]]) -> tuple[str, str]:
    targets = [(row["dataset"], int(row["image_id"]), int(row["annotation_id"])) for row in choices]
    mapping = [(row["dataset"], int(row["image_id"]), int(row["annotation_id"]), row.get("selected_source_candidate_id")) for row in choices]
    encode = lambda value: json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encode(targets)).hexdigest(), hashlib.sha256(encode(mapping)).hexdigest()


def digest_removals(removable: dict[tuple[str, int, int], set[int]]) -> str:
    rows = [(key[0], key[1], key[2], sorted(values)) for key, values in sorted(removable.items())]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def smoke_specs(fixed: Any, specs: tuple[dict[str, Any], ...], choices: list[dict[str, Any]], output: Path) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    selected_images = {}
    for dataset in DATASET_NAMES:
        selected_images[dataset] = min(int(row["image_id"]) for row in choices if row["dataset"] == dataset and row["selected_source_candidate_id"] is not None)
    chosen = [row for row in choices if int(row["image_id"]) == selected_images[row["dataset"]]]
    result = []
    for spec in specs:
        image_id = selected_images[spec["name"]]
        manifest = fixed.read_json(spec["manifest"])
        subset = {**manifest, "images": [image for image in manifest["images"] if int(image["id"]) == image_id], "annotations": [ann for ann in manifest["annotations"] if int(ann["image_id"]) == image_id]}
        path = output / "smoke_manifests" / f"{spec['name']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(subset, ensure_ascii=True) + "\n", encoding="utf-8")
        result.append({**spec, "manifest": path, "images": 1, "gt": len(subset["annotations"])})
    return tuple(result), chosen


def load_records_subset(r007: Any, specs: tuple[dict[str, Any], ...]) -> dict[str, dict[int, dict[str, Any]]]:
    records = {}
    for spec in specs:
        manifest = json.loads(Path(spec["manifest"]).read_text(encoding="utf-8"))
        ids = {int(image["id"]) for image in manifest["images"]}
        rows = [row for row in r007.read_jsonl(r007.TRACES[spec["name"]] / "diagnostic_cache.jsonl") if int(row["image_id"]) in ids]
        if len(rows) != len(ids) or len({int(row["image_id"]) for row in rows}) != len(rows):
            raise RuntimeError(f"Trace subset coverage mismatch: {spec['name']}")
        records[spec["name"]] = {int(row["image_id"]): row for row in rows}
    return records


def prediction_pairs(record: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = [(int(item["pred_id"]), int(item["source_candidate_id"])) for item in record["final_pred_to_source_candidate"]]
    if len(pairs) != len(set(pairs)) or len({pair[0] for pair in pairs}) != len(pairs) or len({pair[1] for pair in pairs}) != len(pairs):
        raise RuntimeError(f"Invalid prediction mapping: image {record['image_id']}")
    if set(pred_id for pred_id, _ in pairs) != {int(pred["pred_id"]) for pred in record["final_predictions"]}:
        raise RuntimeError(f"Prediction mapping does not cover predictions: image {record['image_id']}")
    return sorted(pairs)


def validate_baseline_replay(native: dict[str, dict[int, dict[str, Any]]], replay: dict[str, dict[int, dict[str, Any]]]) -> dict[str, int]:
    """R007 replay must reproduce all native fields; exact RLE gives XOR=0."""
    counts = Counter(images=0, predictions=0, native_final_xor_pixels=0)
    for dataset, rows in native.items():
        for image_id, original in rows.items():
            rebuilt = replay[dataset][image_id]
            if original["final_predictions"] != rebuilt["final_predictions"]:
                raise RuntimeError(f"Baseline final prediction attributes/RLE mismatch: {dataset}/{image_id}")
            if prediction_pairs(original) != prediction_pairs(rebuilt):
                raise RuntimeError(f"Baseline pred_id/source mapping mismatch: {dataset}/{image_id}")
            counts.update(images=1, predictions=len(original["final_predictions"]))
    return dict(counts)


def reconstruct_with_progress(r007: Any, records: dict[str, dict[int, dict[str, Any]]], traces: dict[str, Path], confidence: float, device: str) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    total = sum(len(rows) for rows in records.values())
    done = 0
    for dataset, rows in records.items():
        output[dataset] = {}
        for image_id, record in sorted(rows.items()):
            output[dataset][image_id] = r007.reconstruct_record(record, traces[dataset], 300, confidence, device)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  replay score>{confidence:.2f}: {done}/{total} images", flush=True)
    return output


def validate_trace_invariants(records: dict[str, dict[int, dict[str, Any]]], traces: dict[str, Path]) -> dict[str, int]:
    totals = Counter(images=0, final_predictions=0)
    total = sum(len(rows) for rows in records.values())
    done = 0
    for dataset, rows in records.items():
        artifact_hashes = json.loads((traces[dataset] / "artifact_hashes.json").read_text(encoding="utf-8"))["files"]
        for image_id, record in sorted(rows.items()):
            prediction_pairs(record)
            h, w = (int(value) for value in record["preprocess_meta"]["original_shape"])
            with np.load(traces[dataset] / record["raw_cache"], allow_pickle=False) as raw:
                raw_path = traces[dataset] / record["raw_cache"]
                expected_raw_hash = next((value for key, value in artifact_hashes.items() if key.replace("\\", "/") == str(Path(record["raw_cache"])).replace("\\", "/")), None)
                if expected_raw_hash is None or sha256(raw_path) != expected_raw_hash:
                    raise RuntimeError(f"Raw NPZ artifact hash mismatch: {dataset}/{image_id}")
                source = raw["source_candidate_id"].astype(np.int64, copy=False)
                top = raw["top_indices"].astype(np.int64, copy=False)
                if not np.array_equal(source, np.arange(len(source))) or len(top) != 300 or len(np.unique(top)) != 300:
                    raise RuntimeError(f"Noncontiguous source IDs or nonunique Top-300: {dataset}/{image_id}")
                if "framework_shape" not in raw.files or tuple(int(value) for value in raw["framework_shape"].tolist()) != (h, w):
                    raise RuntimeError(f"Framework mask shape mismatch: {dataset}/{image_id}")
                sources = [source_id for _, source_id in prediction_pairs(record)]
                if any(source_id < 0 or source_id >= len(source) for source_id in sources):
                    raise RuntimeError(f"Final source outside raw candidates: {dataset}/{image_id}")
            totals.update(images=1, final_predictions=len(record["final_predictions"]))
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  trace invariants: {done}/{total} images", flush=True)
    return dict(totals)


def compare_historical_r007(current: dict[tuple[str, int, int], str], path: Path, subset: bool = False) -> list[dict[str, Any]]:
    historical = {
        (row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["primary_class"]
        for row in read_csv(path)
        if row.get("condition") == "k300_conf0p010"
    }
    if (not subset and set(historical) != set(current)) or (subset and not set(current).issubset(historical)):
        raise RuntimeError("Historical R007 .01 GT coverage differs from the current diagnostic scope")
    return [{"dataset": key[0], "image_id": key[1], "annotation_id": key[2], "current_class": current[key], "r007_class": historical[key], "same": int(current[key] == historical[key])} for key in sorted(current)]


def export_coco(records: dict[int, dict[str, Any]], spec: dict[str, Any], output: Path) -> Path:
    from pycocotools import mask as mask_utils

    config = json.loads((Path(spec["trace"]) / "inference_config.json").read_text(encoding="utf-8"))
    manifest = Path(spec["manifest"])
    coco = json.loads(manifest.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in coco["images"]}
    image_ids = set(images)
    if len(image_ids) != len(coco["images"]) or set(records) != image_ids:
        raise RuntimeError(f"Manifest/image coverage mismatch: {spec['name']}")
    category = config["config"]["output_category_id"]
    predictions: list[dict[str, Any]] = []
    for image_id, record in sorted(records.items()):
        source_by_pred = dict(prediction_pairs(record))
        with np.load(Path(spec["trace"]) / record["raw_cache"], allow_pickle=False) as raw:
            for prediction in record["final_predictions"]:
                pred_id = int(prediction["pred_id"])
                source = source_by_pred[pred_id]
                if int(raw["source_candidate_id"][source]) != source or float(prediction["score"]) != float(raw["scores"][source]):
                    raise RuntimeError(f"Prediction/raw source score mismatch: {spec['name']}/{image_id}/{pred_id}")
                rle = prediction.get("mask_rle")
                if prediction.get("class") != config["config"]["source_class_id"] or not isinstance(rle, dict) or not rle.get("counts") or not rle.get("size"):
                    raise RuntimeError(f"Invalid nonempty mask prediction: {spec['name']}/{image_id}")
                if tuple(int(value) for value in rle["size"]) != (int(images[image_id]["height"]), int(images[image_id]["width"])) or float(mask_utils.area(rle)) <= 0:
                    raise RuntimeError(f"Invalid prediction mask shape/area: {spec['name']}/{image_id}/{pred_id}")
                score = float(prediction["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise RuntimeError(f"Invalid prediction score: {spec['name']}/{image_id}")
                predictions.append({"image_id": image_id, "category_id": category, "score": score, "segmentation": rle})
    path = output / f"{spec['name']}_coco_predictions.json"
    path.write_text(json.dumps(predictions, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def transitions(condition: str, baseline: dict[tuple[str, int, int], str], current: dict[tuple[str, int, int], str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, matrix = [], []
    for dataset in [*DATASET_NAMES, "ALL"]:
        keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
        counts = Counter((baseline[key], current[key]) for key in keys)
        matrix.extend({"condition": condition, "dataset": dataset, "original_class": old, "new_class": new, "n_gt": counts[(old, new)]} for old in PRIMARY for new in PRIMARY)
        summary.append({"condition": condition, "dataset": dataset, "n_gt": len(keys), "recovered_to_C": sum(n for (old, new), n in counts.items() if old != "C" and new == "C"), "C_to_failure": sum(n for (old, new), n in counts.items() if old == "C" and new != "C"), "relation_to_C": sum(n for (old, new), n in counts.items() if old in RELATION and new == "C"), "C_to_relation": sum(n for (old, new), n in counts.items() if old == "C" and new in RELATION), "failure_rate": sum(current[key] != "C" for key in keys) / len(keys), **{f"new_{label}": sum(current[key] == label for key in keys) for label in PRIMARY}})
    return summary, matrix


def validate_frozen_preflight(quality_dir: Path, traces: tuple[Path, Path], historical: Path) -> list[Path]:
    from ultralytics.utils import ops

    quality = json.loads((quality_dir / "provenance.json").read_text(encoding="utf-8"))
    matching = json.loads((PROJECT / "experiments" / "yolo26_strict_matching_fulltrace_20260905_v1" / "validation.json").read_text(encoding="utf-8"))
    per_gt, edges = quality_dir / "per_gt.csv", quality_dir / "edge_metrics.csv"
    require_hash(matching["inputs_sha256"], per_gt, "strict-matching per_gt")
    require_hash(matching["inputs_sha256"], edges, "strict-matching edges")
    for path in (PROJECT / "tools" / "run_yolo26_candidate_perturbation.py", LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py", Path(ops.__file__)):
        require_hash(quality["files_sha256"] | quality["replay_source_sha256"], path, "strict-quality helper")
    inputs = [per_gt, edges, quality_dir / "provenance.json", historical, Path(__file__), Path(ops.__file__), PROJECT / "tools" / "run_yolo26_candidate_perturbation.py", PROJECT / "tools" / "run_yolo26_score_retention_sweep.py", LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py", LEGACY / "scripts" / "evaluate_task05_public.py"]
    for trace in traces:
        config_path, cache_path, hashes_path = trace / "inference_config.json", trace / "diagnostic_cache.jsonl", trace / "artifact_hashes.json"
        config, artifacts = json.loads(config_path.read_text(encoding="utf-8")), json.loads(hashes_path.read_text(encoding="utf-8"))["files"]
        require_hash(quality["files_sha256"], config_path, "strict-quality trace config")
        require_hash(quality["files_sha256"], cache_path, "strict-quality trace cache")
        require_hash(quality["files_sha256"], hashes_path, "strict-quality artifact manifest")
        if sha256(config_path) != artifacts.get("inference_config.json") or sha256(cache_path) != artifacts.get("diagnostic_cache.jsonl"):
            raise RuntimeError(f"Trace artifact manifest disagrees with trace files: {trace}")
        manifest = Path(config["manifest"])
        if sha256(manifest) != config["provenance_hashes"]["manifest_sha256"]:
            raise RuntimeError(f"Trace manifest provenance hash mismatch: {manifest}")
        inputs.extend([config_path, cache_path, hashes_path, trace / "run_status.json", trace / "summary.json", manifest])
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piglife-trace", type=Path, required=True)
    parser.add_argument("--faro-trace", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--smoke", action="store_true", help="Run one eligible image per dataset; outputs are explicitly non-full.")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    for trace in (args.piglife_trace, args.faro_trace):
        for name in ("diagnostic_cache.jsonl", "inference_config.json", "run_status.json", "summary.json"):
            if not (trace / name).is_file():
                raise FileNotFoundError(trace / name)
        for name in ("run_status.json", "summary.json"):
            if json.loads((trace / name).read_text(encoding="utf-8")).get("status") != "completed":
                raise RuntimeError(f"Trace is not completed: {trace}")
    per_gt, edges = args.quality_dir / "per_gt.csv", args.quality_dir / "edge_metrics.csv"
    if not per_gt.is_file() or not edges.is_file():
        raise FileNotFoundError("quality-dir must contain per_gt.csv and edge_metrics.csv")
    quality_provenance = json.loads((args.quality_dir / "provenance.json").read_text(encoding="utf-8"))
    if {str(args.piglife_trace), str(args.faro_trace)} != set(quality_provenance.get("traces", [])):
        raise RuntimeError("Strict-quality provenance does not name the requested canonical traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical_r007 = PROJECT / "experiments" / "r007_score_retention_20260905_v1" / "sweep_gt_analysis.csv"
    provenance_files = validate_frozen_preflight(args.quality_dir, (args.piglife_trace, args.faro_trace), historical_r007)
    input_hash_before = {str(path): sha256(path) for path in provenance_files}
    (args.output_dir / "start_receipt.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "scope": "smoke_non_full" if args.smoke else "full_586_images_6226_gt", "input_sha256": input_hash_before}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r006, r007 = load_tool("run_yolo26_candidate_perturbation.py"), load_tool("run_yolo26_score_retention_sweep.py")
    traces = {"PigLife_public_test": args.piglife_trace, "FaroPigSeg_test": args.faro_trace}
    r006.TRACES = traces
    r007.TRACES = traces
    fixed = r007.load_legacy()
    specs = tuple(spec for spec in fixed.DATASETS if spec["name"] in DATASET_NAMES)
    if len(specs) != 2:
        raise RuntimeError("Fixed classifier does not expose both required datasets")
    specs = tuple({**spec, "trace": traces[spec["name"]]} for spec in specs)
    image_shapes = {(spec["name"], int(image["id"])): (int(image["height"]), int(image["width"])) for spec in specs for image in fixed.read_json(spec["manifest"])["images"]}
    choices = choose_oracle(target_rows(per_gt, edges, image_shapes), r006.maximum_cardinality_assignment)
    if args.smoke:
        specs, choices = smoke_specs(fixed, specs, choices, args.output_dir)
    fixed.DATASETS = tuple({key: value for key, value in spec.items() if key != "trace"} for spec in specs)
    print("[1/6] loading completed canonical traces", flush=True)
    records = load_records_subset(r007, fixed.DATASETS) if args.smoke else r007.load_records(fixed.DATASETS)
    invariants = validate_trace_invariants(records, traces)
    print("[2/6] replaying K300 score>.05 baseline", flush=True)
    replay = reconstruct_with_progress(r007, records, traces, 0.05, args.device)
    replay_audit = validate_baseline_replay(records, replay)
    baseline_rows, _, _ = fixed.classify_all(PROJECT, records)
    baseline = class_map(baseline_rows)
    strict = strict_baseline(per_gt)
    expected_strict = strict if not args.smoke else {key: value for key, value in strict.items() if key in baseline}
    if baseline != expected_strict or (not args.smoke and len(baseline) != 6226) or {key[0] for key in baseline} != set(DATASET_NAMES):
        raise RuntimeError("New baseline classes do not exactly match the required strict per_gt scope")
    print("[3/6] selecting frozen strict confidence-lost targets", flush=True)
    attach_manifest_geometry(fixed, specs, choices)
    assert_selected_geometry(records, choices, traces)
    selected = [row for row in choices if row["selected_source_candidate_id"] is not None]
    validate_selected_edges(fixed, r006, records, specs, choices, traces, args.device)
    removable = removable_predictions(fixed, records, specs, choices)
    frozen_oracle = frozen_oracle_tuple(choices, removable)
    condition_digests: dict[str, dict[str, tuple[str, str, str]]] = {}
    write_csv(args.output_dir / "oracle_candidate_mapping.csv", choices)
    print("[4/6] replaying fixed K300 score>.01 confidence-only condition", flush=True)
    confidence = reconstruct_with_progress(r007, records, traces, 0.01, args.device)
    conditions: dict[str, dict[str, dict[int, dict[str, Any]]]] = {"BASELINE": records, "CONFIDENCE_ONLY": confidence}
    for name, mode in (("ADDITION", "ADDITION"), ("REMOVAL", "REMOVAL"), ("ADDITION_REMOVAL", "ADDITION_REMOVAL")):
        print(f"[5/6] building oracle {name}", flush=True)
        before_digest = assert_frozen_oracle(frozen_oracle, choices, removable, f"before_{name}")
        conditions[name], _ = r006.perturb(records, choices, removable, mode, args.device)
        after_digest = assert_frozen_oracle(frozen_oracle, choices, removable, f"after_{name}")
        condition_digests[name] = {"before": before_digest, "after": after_digest}
    evaluator = load_tool_from_path(LEGACY / "scripts" / "evaluate_task05_public.py")
    all_gt, all_summary, all_transitions, metrics, current_maps, parity = [], [], [], {}, {}, {}
    for name, current_records in conditions.items():
        print(f"[6/6] classifying and exporting {name}", flush=True)
        parity[name] = r006.verify_final_source_parity(fixed, current_records, specs, args.device)
        if not parity[name]["totals"]["all_exact"]:
            raise RuntimeError(f"Final-source XOR guard failed: {name}")
        rows, _, _ = fixed.classify_all(PROJECT, current_records)
        current = class_map(rows)
        current_maps[name] = current
        if set(current) != set(baseline):
            raise RuntimeError(f"All-GT coverage drift: {name}")
        all_gt.extend({"condition": name, **row} for row in rows)
        summary, matrix = transitions(name, baseline, current)
        all_summary.extend(summary)
        all_transitions.extend(matrix)
        for spec in specs:
            condition_dir = args.output_dir / name.lower()
            condition_dir.mkdir(exist_ok=True)
            path = export_coco(current_records[spec["name"]], spec, condition_dir)
            metrics[f"{name}/{spec['name']}"] = evaluator.evaluate(Path(spec["manifest"]), path)
    if any(sum(row["n_gt"] for row in all_transitions if row["condition"] == name and row["dataset"] == "ALL") != len(baseline) for name in conditions):
        raise RuntimeError("8x8 transition accounting failed")
    write_csv(args.output_dir / "all_gt_analysis.csv", all_gt)
    write_csv(args.output_dir / "condition_summary.csv", all_summary)
    write_csv(args.output_dir / "condition_transitions_8x8.csv", all_transitions)
    r007_comparison = compare_historical_r007(current_maps["CONFIDENCE_ONLY"], historical_r007, args.smoke)
    write_csv(args.output_dir / "r007_confidence_only_comparison.csv", r007_comparison)
    (args.output_dir / "coco_metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    input_hash_after = {str(path): sha256(path) for path in provenance_files}
    if input_hash_after != input_hash_before:
        raise RuntimeError("Frozen inputs changed during diagnostic")
    target_hash, mapping_hash, removal_hash = frozen_oracle
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "completed", "scope": "smoke_non_full" if args.smoke else "full_586_images_6226_gt", "mode": "fixed K300 confidence replay plus GT-guided oracle output-set conditions; no forward/training/tuning", "conditions": list(conditions), "baseline": {"top_k": 300, "score_rule": "score > .05", "strict_gt_coverage": len(baseline), "strict_baseline_match": True, "trace_invariants": invariants, "replay": replay_audit}, "confidence_only": {"top_k": 300, "score_rule": "score > .01", "r007_same_gt_classes": sum(row["same"] for row in r007_comparison), "r007_changed_gt_classes": sum(not row["same"] for row in r007_comparison)}, "target_rule": "baseline non-C AND strict coverage/purity Top-K available AND strict confidence unavailable", "eligible_edge_rule": "Top-300, .01 < score <= .05, IoU >= .50, coverage >= .75, purity >= .75, not final", "targets": len(choices), "selected": len(selected), "target_key_hash": target_hash, "selected_source_mapping_hash": mapping_hash, "removal_mapping_hash": removal_hash, "oracle_condition_digests": condition_digests, "prediction_accounting": {name: sum(len(record["final_predictions"]) for dataset in rows.values() for record in dataset.values()) for name, rows in conditions.items()}, "final_source_parity": parity, "coco_evaluation": {"iou_type": "segm", "bbox_exported": False, "max_dets": 100}, "input_sha256_before": input_hash_before, "input_sha256_after": input_hash_after, "script_sha256": {"runner": sha256(Path(__file__)), "r006": sha256(PROJECT / "tools" / "run_yolo26_candidate_perturbation.py"), "r007": sha256(PROJECT / "tools" / "run_yolo26_score_retention_sweep.py"), "coco_evaluator": sha256(LEGACY / "scripts" / "evaluate_task05_public.py")}, "runtime": {"python": sys.version, "platform": platform.platform()}, "outputs": {"start_receipt": "start_receipt.json", "all_gt": "all_gt_analysis.csv", "summary": "condition_summary.csv", "transitions": "condition_transitions_8x8.csv", "metrics": "coco_metrics.json", "mapping": "oracle_candidate_mapping.csv", "r007_comparison": "r007_confidence_only_comparison.csv"}}
    (args.output_dir / "run_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output_dir), "targets": len(choices), "selected": len(selected)}, ensure_ascii=False))
    return 0


def load_tool_from_path(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
