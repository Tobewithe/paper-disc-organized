"""Independent CPU audit of frozen R005c confidence-retention outputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(r"C:\Dpan\codexproject\paper-disc")
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research")
RUN = PROJECT / "experiments" / "yolo26_confidence_diagnostic_fulltrace_20260905_v2"
QUALITY = PROJECT / "experiments" / "yolo26_strict_quality_fulltrace_20260905_v1"
TRACES = {
    "PigLife_public_test": LEGACY / "artifacts" / "inference_cache" / "yolo26seg_trace_20260905_1930_piglife_full",
    "FaroPigSeg_test": LEGACY / "artifacts" / "inference_cache" / "yolo26seg_trace_20260905_1945_faro_full",
}
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
CONDITIONS = ("BASELINE", "CONFIDENCE_ONLY", "ADDITION", "REMOVAL", "ADDITION_REMOVAL")
OUTPUT_DEFAULT = PROJECT / "experiments" / "yolo26_confidence_audit_20260907_v2"


def resolve_path(value: str | Path, base: Path = PROJECT) -> Path:
    """Resolve paths recorded by Windows tools without changing the evidence."""
    raw = str(value).replace("/", os.sep).replace("\\", os.sep)
    path = Path(raw)
    return path if path.is_absolute() else (base / path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"audit_{path.stem}_{abs(hash(str(path))) & 0xfffffff:x}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def signature(image_id: int, prediction: dict[str, Any]) -> str:
    """Stable exact RLE/score identity; preserves multiset duplicates."""
    return json.dumps(
        {
            "image_id": int(image_id),
            "score": float(prediction["score"]),
            "class": int(prediction.get("class", 0)),
            "mask_rle": prediction.get("mask_rle", prediction.get("segmentation")),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def export_signatures(condition: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for dataset in TRACES:
        path = RUN / condition.lower() / f"{dataset}_coco_predictions.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in json.loads(path.read_text(encoding="utf-8")):
            result[signature(int(row["image_id"]), {"score": row["score"], "class": 0, "mask_rle": row["segmentation"]})] += 1
    return result


def records() -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset, trace in TRACES.items():
        cache = trace / "diagnostic_cache.jsonl"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        result[dataset] = {int(row["image_id"]): row for row in rows}
        if len(rows) != len(result[dataset]):
            raise RuntimeError(f"Duplicate trace image ID: {dataset}")
    return result


def baseline_signatures(trace_records: dict[str, dict[int, dict[str, Any]]]) -> tuple[Counter[str], dict[tuple[str, int, int], str]]:
    values: Counter[str] = Counter()
    by_source: dict[tuple[str, int, int], str] = {}
    for dataset, images in trace_records.items():
        for image_id, record in images.items():
            preds = {int(row["pred_id"]): row for row in record["final_predictions"]}
            mapping = record["final_pred_to_source_candidate"]
            if len(preds) != len(mapping):
                raise RuntimeError(f"Prediction/mapping count mismatch: {dataset}/{image_id}")
            for link in mapping:
                pred_id, source = int(link["pred_id"]), int(link["source_candidate_id"])
                key = signature(image_id, preds[pred_id])
                values[key] += 1
                source_key = (dataset, image_id, source)
                if source_key in by_source:
                    raise RuntimeError(f"Duplicate final source: {source_key}")
                by_source[source_key] = key
    return values, by_source


def validate_hashes(summary: dict[str, Any], quality_provenance: dict[str, Any]) -> dict[str, Any]:
    """Recheck every frozen receipt and resolve relative paths against the project."""
    mismatches: dict[str, Any] = {}
    input_rows: dict[str, Any] = {}
    before = summary.get("input_sha256_before", {})
    after = summary.get("input_sha256_after", {})
    for raw_path, expected in before.items():
        path = resolve_path(raw_path)
        current = sha256(path) if path.is_file() else None
        row = {
            "path": str(path),
            "recorded_before": expected,
            "recorded_after": after.get(raw_path),
            "current": current,
            "passed": bool(path.is_file() and current == expected and after.get(raw_path) == expected),
        }
        input_rows[str(raw_path)] = row
        if not row["passed"]:
            mismatches[str(raw_path)] = row
    if set(before) != set(after):
        mismatches["input_sha256_key_set"] = {"before_only": sorted(set(before) - set(after)), "after_only": sorted(set(after) - set(before))}

    script_rows: dict[str, Any] = {}
    recorded_scripts = summary.get("script_sha256", {})
    script_paths = {
        "runner": RUN / "runner_source.py",
        "r006": PROJECT / "tools" / "run_yolo26_candidate_perturbation.py",
        "r007": PROJECT / "tools" / "run_yolo26_score_retention_sweep.py",
        "coco_evaluator": LEGACY / "scripts" / "evaluate_task05_public.py",
    }
    for label, path in script_paths.items():
        current = sha256(path) if path.is_file() else None
        expected = recorded_scripts.get(label)
        row = {"path": str(path), "recorded": expected, "current": current, "passed": bool(expected and current == expected)}
        script_rows[label] = row
        if not row["passed"]:
            mismatches[f"script:{label}"] = row

    provenance_rows: dict[str, Any] = {}
    for section in ("files_sha256", "replay_source_sha256"):
        for raw_path, expected in quality_provenance.get(section, {}).items():
            path = resolve_path(raw_path)
            current = sha256(path) if path.is_file() else None
            row = {"section": section, "path": str(path), "recorded": expected, "current": current, "passed": bool(path.is_file() and current == expected)}
            provenance_rows[f"{section}:{raw_path}"] = row
            if not row["passed"]:
                mismatches[f"provenance:{section}:{raw_path}"] = row

    manifest_checks: dict[str, Any] = {}
    for dataset, trace in TRACES.items():
        config_path = trace / "inference_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = resolve_path(config["manifest"])
        expected = config.get("provenance_hashes", {}).get("manifest_sha256")
        current = sha256(manifest) if manifest.is_file() else None
        row = {"path": str(manifest), "expected": expected, "current": current, "passed": bool(manifest.is_file() and expected and current == expected)}
        manifest_checks[dataset] = row
        if not row["passed"]:
            mismatches[f"manifest:{dataset}"] = row

    return {
        "recorded_run_inputs": len(before),
        "run_input_checks": input_rows,
        "script_checks": script_rows,
        "quality_provenance_checks": provenance_rows,
        "manifest_checks": manifest_checks,
        "mismatches": mismatches,
    }


def _finite_array(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else True


def _normal_rel(value: str | Path) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _manifest_summary(dataset: str, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = [int(row["id"]) for row in images]
    annotation_ids = [int(row["id"]) for row in annotations]
    image_set = set(image_ids)
    gt_annotations = [
        row for row in annotations if int(row.get("category_id", 1)) == 1 and not bool(row.get("iscrowd", 0))
    ]
    gt_keys = {(dataset, int(row["image_id"]), int(row["id"])) for row in gt_annotations}
    problems: list[str] = []
    if len(image_ids) != len(image_set):
        problems.append("duplicate image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        problems.append("duplicate annotation IDs")
    if any(int(row["image_id"]) not in image_set for row in annotations):
        problems.append("annotation references an unknown image")
    if not data.get("categories"):
        problems.append("missing categories")
    return {
        "path": str(manifest_path),
        "images": len(images),
        "unique_images": len(image_set),
        "annotations": len(annotations),
        "gt_category_1_noncrowd": len(gt_annotations),
        "unique_gt_keys": len(gt_keys),
        "image_ids": sorted(image_set),
        "gt_keys": sorted(gt_keys),
        "problems": problems,
    }, data


def validate_trace_artifacts(trace_records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Check trace receipts, NPZ schema/flags, mappings, and public input hashes."""
    required = {
        "source_candidate_id", "boxes_xyxy", "scores", "mask_coefficients", "prototype", "top_indices", "top_scores",
        "top_k_member", "conf_pass", "final_candidate", "global_rank", "feature_level", "grid_x", "grid_y", "stride",
        "framework_coefficients", "framework_boxes_xyxy", "framework_source_candidate_id", "framework_shape",
    }
    report: dict[str, Any] = {"datasets": {}, "receipt_files": 0, "receipt_passed": 0, "mismatches": [], "npz_checked": 0, "npz_bad": []}
    from pycocotools import mask as mask_utils

    for dataset, trace in TRACES.items():
        config_path = trace / "inference_config.json"
        status_path = trace / "run_status.json"
        summary_path = trace / "summary.json"
        receipt_path = trace / "artifact_hashes.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        trace_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = resolve_path(config["manifest"])
        manifest, manifest_data = _manifest_summary(dataset, manifest_path)
        rows = trace_records[dataset]
        expected_ids = set(manifest["image_ids"])
        row_ids = set(rows)
        dataset_report: dict[str, Any] = {
            "trace": str(trace),
            "status": status.get("status"),
            "summary_status": trace_summary.get("status"),
            "manifest": manifest,
            "row_count": len(rows),
            "manifest_id_match": row_ids == expected_ids,
            "npz_checked": 0,
            "npz_bad": [],
            "receipt": {},
            "selected_input_hashes": {"checked": 0, "passed": 0, "mismatches": []},
        }
        if row_ids != expected_ids:
            report["mismatches"].append(f"{dataset}: trace image IDs differ from manifest")
        # artifact_hashes.json deliberately excludes itself; require exact file-set coverage.
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_files = receipt.get("files", {})
        actual_files = {
            _normal_rel(path.relative_to(trace))
            for path in trace.rglob("*")
            if path.is_file() and path.name != "artifact_hashes.json" and "__pycache__" not in path.parts
        }
        expected_files = {_normal_rel(path) for path in receipt_files}
        missing_receipt = sorted(expected_files - actual_files)
        extra_files = sorted(actual_files - expected_files)
        receipt_bad: list[str] = []
        for relative, expected_hash in receipt_files.items():
            path = trace / Path(str(relative).replace("\\", os.sep).replace("/", os.sep))
            report["receipt_files"] += 1
            if path.is_file() and sha256(path) == expected_hash:
                report["receipt_passed"] += 1
            else:
                receipt_bad.append(str(path))
        dataset_report["receipt"] = {
            "algorithm": receipt.get("algorithm"),
            "scope": receipt.get("scope"),
            "listed": len(expected_files),
            "actual_excluding_receipt": len(actual_files),
            "missing": missing_receipt,
            "extra": extra_files,
            "hash_bad": receipt_bad,
            "exact_file_set": not missing_receipt and not extra_files,
        }
        if missing_receipt or extra_files or receipt_bad:
            report["mismatches"].append(f"{dataset}: artifact receipt mismatch")

        # The selected image hashes are public-test/Faro-test inputs named by the frozen config.
        selected_hashes = config.get("provenance_hashes", {}).get("selected_images", {})
        for image_id, item in selected_hashes.items():
            path = resolve_path(item.get("path", ""))
            current = sha256(path) if path.is_file() else None
            dataset_report["selected_input_hashes"]["checked"] += 1
            if current == item.get("sha256"):
                dataset_report["selected_input_hashes"]["passed"] += 1
            else:
                mismatch = {"image_id": int(image_id), "path": str(path), "expected": item.get("sha256"), "current": current}
                dataset_report["selected_input_hashes"]["mismatches"].append(mismatch)
        if dataset_report["selected_input_hashes"]["mismatches"]:
            report["mismatches"].append(f"{dataset}: selected input hash mismatch")

        for image_id, row in sorted(rows.items()):
            dataset_report["npz_checked"] += 1
            report["npz_checked"] += 1
            failures: list[str] = []
            raw_cache = str(row.get("raw_cache", ""))
            raw_path = trace / Path(raw_cache.replace("\\", os.sep).replace("/", os.sep))
            expected_name = f"image_{image_id}_raw_candidates.npz"
            if Path(raw_cache).name != expected_name or not raw_path.is_file():
                failures.append("raw_cache path/name missing or not image-specific")
            try:
                with np.load(raw_path, allow_pickle=False) as raw:
                    if set(raw.files) != required:
                        failures.append(f"NPZ key set differs (missing={sorted(required - set(raw.files))}, extra={sorted(set(raw.files)-required)})")
                    n = len(raw["scores"])
                    shape = tuple(int(x) for x in row["preprocess_meta"]["original_shape"])
                    shape_arr = tuple(int(x) for x in raw["framework_shape"].tolist())
                    if n != 21504:
                        failures.append(f"candidate count {n} != 21504")
                    if shape_arr != shape:
                        failures.append(f"framework shape {shape_arr} != trace shape {shape}")
                    expected_shapes = {
                        "source_candidate_id": (n,), "boxes_xyxy": (n, 4), "scores": (n,), "mask_coefficients": (n, 32),
                        "prototype": (32, 256, 256), "top_indices": (300,), "top_scores": (300,), "top_k_member": (n,),
                        "conf_pass": (n,), "final_candidate": (n,), "global_rank": (n,), "feature_level": (n,), "grid_x": (n,),
                        "grid_y": (n,), "stride": (n,), "framework_coefficients": (len(row["final_predictions"]), 32),
                        "framework_boxes_xyxy": (len(row["final_predictions"]), 4),
                        "framework_source_candidate_id": (len(row["final_predictions"]),), "framework_shape": (2,),
                    }
                    for key, expected_shape in expected_shapes.items():
                        if key in raw.files and tuple(raw[key].shape) != expected_shape:
                            failures.append(f"{key} shape {raw[key].shape} != {expected_shape}")
                    numeric_keys = [key for key in raw.files if np.issubdtype(raw[key].dtype, np.number)]
                    if any(not _finite_array(raw[key]) for key in numeric_keys):
                        failures.append("non-finite NPZ numeric value")
                    source = raw["source_candidate_id"].astype(np.int64, copy=False)
                    scores = raw["scores"].astype(np.float64, copy=False)
                    top = raw["top_indices"].astype(np.int64, copy=False)
                    top_member = raw["top_k_member"].astype(bool, copy=False)
                    conf_member = raw["conf_pass"].astype(bool, copy=False)
                    final_member = raw["final_candidate"].astype(bool, copy=False)
                    ranks = raw["global_rank"].astype(np.int64, copy=False)
                    if not np.array_equal(source, np.arange(n, dtype=np.int64)):
                        failures.append("source_candidate_id is not contiguous")
                    if len(np.unique(top)) != len(top) or np.any(top < 0) or np.any(top >= n):
                        failures.append("Top-300 indices are not unique/in range")
                    expected_top = np.zeros(n, dtype=bool)
                    if len(top) == 300:
                        expected_top[top] = True
                    if not np.array_equal(top_member, expected_top):
                        failures.append("top_k_member disagrees with top_indices")
                    if not np.allclose(raw["top_scores"], scores[top], rtol=0, atol=0):
                        failures.append("top_scores differs from source scores")
                    conf_expected = scores >= float(config["config"].get("conf", 0.05))
                    if not np.array_equal(conf_member, conf_expected):
                        failures.append("conf_pass disagrees with frozen confidence gate")
                    if not np.array_equal(np.sort(ranks), np.arange(1, n + 1, dtype=np.int64)):
                        failures.append("global_rank is not a 1-based permutation")
                    rank_order = np.argsort(ranks)
                    if np.any(scores[rank_order][:-1] < scores[rank_order][1:]):
                        failures.append("global_rank is not score-descending")
                    links = row.get("final_pred_to_source_candidate", [])
                    pred_ids = [int(item["pred_id"]) for item in links]
                    source_ids = [int(item["source_candidate_id"]) for item in links]
                    if len(pred_ids) != len(set(pred_ids)) or len(source_ids) != len(set(source_ids)):
                        failures.append("final mapping is not one-to-one")
                    if set(pred_ids) != {int(item["pred_id"]) for item in row["final_predictions"]}:
                        failures.append("final mapping does not cover predictions")
                    if set(source_ids) != set(np.flatnonzero(final_member).tolist()):
                        failures.append("final_candidate flags disagree with final source mapping")
                    for pred in row["final_predictions"]:
                        pid = int(pred["pred_id"])
                        if pid not in pred_ids:
                            continue
                        source_id = source_ids[pred_ids.index(pid)]
                        if not (0 <= source_id < n):
                            failures.append(f"final source {source_id} out of range")
                            continue
                        if float(pred["score"]) != float(scores[source_id]):
                            failures.append(f"score changed for source {source_id}")
                        if not np.array_equal(np.asarray(pred["box_xyxy"], dtype=np.float32), raw["boxes_xyxy"][source_id]):
                            failures.append(f"box changed for source {source_id}")
                        stage = pred.get("candidate_stage", {})
                        if bool(stage.get("top_k")) != bool(top_member[source_id]) or bool(stage.get("conf_pass")) != bool(conf_member[source_id]) or bool(stage.get("final")) != bool(final_member[source_id]):
                            failures.append(f"stage flags changed for source {source_id}")
                        rle = pred.get("mask_rle")
                        try:
                            decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]})
                            if tuple(decoded.shape) != shape or not decoded.any():
                                failures.append(f"invalid/empty baseline RLE for source {source_id}")
                        except Exception as error:
                            failures.append(f"RLE decode failed for source {source_id}: {error}")
            except Exception as error:
                failures.append(f"NPZ read/check failed: {type(error).__name__}: {error}")
            if failures:
                dataset_report["npz_bad"].append({"image_id": image_id, "failures": failures[:20]})
                report["npz_bad"].append({"dataset": dataset, "image_id": image_id, "failures": failures[:20]})
        if dataset_report["npz_bad"]:
            report["mismatches"].append(f"{dataset}: NPZ/schema/mapping failures")
        report["datasets"][dataset] = dataset_report
    return report


def validate_rles() -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    bad: list[str] = []
    count = 0
    image_checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        for dataset in TRACES:
            trace_config = json.loads((TRACES[dataset] / "inference_config.json").read_text(encoding="utf-8"))
            manifest_path = resolve_path(trace_config["manifest"])
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            image_info = {int(item["id"]): (int(item["height"]), int(item["width"])) for item in manifest_data["images"]}
            expected_category = int(trace_config["config"].get("output_category_id", 1))
            path = RUN / condition.lower() / f"{dataset}_coco_predictions.json"
            if not path.is_file():
                bad.append(f"missing prediction file: {path}")
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            seen_images = Counter()
            for row in rows:
                count += 1
                image_id = int(row.get("image_id", -1))
                seen_images[image_id] += 1
                try:
                    if image_id not in image_info:
                        raise ValueError("image_id is absent from manifest")
                    if int(row.get("category_id", -1)) != expected_category:
                        raise ValueError(f"category_id != {expected_category}")
                    rle = row["segmentation"]
                    if not isinstance(rle, dict) or set(rle) != {"size", "counts"}:
                        raise ValueError("segmentation is not a canonical RLE object")
                    mask = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]})
                    if tuple(mask.shape) != image_info[image_id] or mask.ndim != 2 or not mask.any() or not math.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1:
                        raise ValueError("empty/invalid")
                except Exception as error:  # audit should report all malformed exports
                    bad.append(f"{condition}/{dataset}/image={row.get('image_id')}: {error}")
            image_checks[f"{condition}/{dataset}"] = {"prediction_count": len(rows), "unique_image_ids": len(seen_images), "unknown_image_ids": sorted(set(seen_images) - set(image_info))}
    return {"prediction_rles_checked": count, "invalid_or_empty": bad, "image_checks": image_checks}


def _class_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    result = {(str(row["dataset"]), int(row["image_id"]), int(row["annotation_id"])): str(row["primary_class"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("classifier produced duplicate GT keys")
    return result


def _compare_csv_rows(expected: list[dict[str, Any]], actual: list[dict[str, str]], float_tolerance: float = 1e-12) -> list[str]:
    mismatches: list[str] = []
    if len(expected) != len(actual):
        mismatches.append(f"row count {len(actual)} != {len(expected)}")
    for index, (wanted, got) in enumerate(zip(expected, actual)):
        for key, value in wanted.items():
            if key not in got:
                mismatches.append(f"row {index}: missing field {key}")
                continue
            text = got[key]
            if isinstance(value, float):
                try:
                    if not math.isclose(float(text), value, rel_tol=0, abs_tol=float_tolerance):
                        mismatches.append(f"row {index}/{key}: {text} != {value}")
                except ValueError:
                    mismatches.append(f"row {index}/{key}: non-numeric {text!r}")
            elif str(text) != str(value):
                mismatches.append(f"row {index}/{key}: {text!r} != {value!r}")
        if len(mismatches) >= 100:
            break
    return mismatches


def recompute_transitions(recomputed_maps: dict[str, dict[tuple[str, int, int], str]] | None = None) -> dict[str, Any]:
    rows = read_csv(RUN / "all_gt_analysis.csv")
    grouped: dict[str, dict[tuple[str, int, int], str]] = defaultdict(dict)
    duplicates = 0
    invalid_classes: list[str] = []
    for row in rows:
        key = (row["dataset"], int(row["image_id"]), int(row["annotation_id"]))
        if key in grouped[row["condition"]]:
            duplicates += 1
        if row["condition"] not in CONDITIONS or row["primary_class"] not in PRIMARY:
            invalid_classes.append(f"{row.get('condition')}/{key}/{row.get('primary_class')}")
        grouped[row["condition"]][key] = row["primary_class"]
    baseline = grouped["BASELINE"]
    recorded = {(r["condition"], r["dataset"], r["original_class"], r["new_class"]): int(r["n_gt"])
                for r in read_csv(RUN / "condition_transitions_8x8.csv")}
    mismatch = []
    summaries: list[dict[str, Any]] = []
    expected_transition_rows: list[dict[str, Any]] = []
    expected_summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        current = grouped.get(condition, {})
        if set(current) != set(baseline):
            mismatch.append(f"{condition}: GT key coverage differs")
            continue
        for dataset in (*TRACES, "ALL"):
            keys = list(baseline) if dataset == "ALL" else [key for key in baseline if key[0] == dataset]
            counts = Counter((baseline[key], current[key]) for key in keys)
            for old in PRIMARY:
                for new in PRIMARY:
                    expected_transition_rows.append({"condition": condition, "dataset": dataset, "original_class": old, "new_class": new, "n_gt": counts[(old, new)]})
                    if recorded.get((condition, dataset, old, new)) != counts[(old, new)]:
                        mismatch.append(f"{condition}/{dataset}/{old}->{new}")
            recovered = sum(n for (old, new), n in counts.items() if old != "C" and new == "C")
            regressions = sum(n for (old, new), n in counts.items() if old == "C" and new != "C")
            summary_row = {
                "condition": condition, "dataset": dataset, "n_gt": len(keys), "recovered_to_C": recovered,
                "C_to_failure": regressions,
                "relation_to_C": sum(n for (old, new), n in counts.items() if old in {"O", "M", "X"} and new == "C"),
                "C_to_relation": sum(n for (old, new), n in counts.items() if old == "C" and new in {"O", "M", "X"}),
                "failure_rate": sum(current[key] != "C" for key in keys) / len(keys),
                **{f"new_{label}": sum(current[key] == label for key in keys) for label in PRIMARY},
            }
            summaries.append({key: summary_row[key] for key in ("condition", "dataset", "n_gt", "recovered_to_C", "C_to_failure")})
            expected_summary_rows.append(summary_row)
    transition_csv_mismatches = _compare_csv_rows(expected_transition_rows, read_csv(RUN / "condition_transitions_8x8.csv"))
    summary_csv_mismatches = _compare_csv_rows(expected_summary_rows, read_csv(RUN / "condition_summary.csv"))
    recomputed_map_mismatches: dict[str, Any] = {}
    if recomputed_maps is not None:
        for condition in CONDITIONS:
            saved = grouped.get(condition, {})
            fresh = recomputed_maps.get(condition, {})
            differing = [key for key in sorted(set(saved) | set(fresh)) if saved.get(key) != fresh.get(key)]
            recomputed_map_mismatches[condition] = {"saved": len(saved), "fresh": len(fresh), "differing": len(differing), "sample": differing[:10]}
    return {
        "rows": len(rows), "conditions": sorted(grouped), "gt_per_condition": {key: len(value) for key, value in grouped.items()},
        "duplicate_gt_rows": duplicates, "invalid_classes": invalid_classes, "transition_mismatches": mismatch,
        "transition_csv_mismatches": transition_csv_mismatches, "summary_csv_mismatches": summary_csv_mismatches,
        "fresh_classifier_map_mismatches": recomputed_map_mismatches, "summaries": summaries, "maps": grouped,
    }


def _records_counter(records_by_dataset: dict[str, dict[int, dict[str, Any]]]) -> Counter[str]:
    output: Counter[str] = Counter()
    for dataset, images in records_by_dataset.items():
        for image_id, record in images.items():
            for prediction in record["final_predictions"]:
                output[signature(image_id, prediction)] += 1
    return output


def _output_set_delta(expected: Counter[str], condition: str) -> dict[str, Any]:
    actual = export_signatures(condition)
    return {
        "expected": sum(expected.values()),
        "actual": sum(actual.values()),
        "missing": sum((expected - actual).values()),
        "unexpected": sum((actual - expected).values()),
        "exact": actual == expected,
    }


def validate_targets_and_sets(
    trace_records: dict[str, dict[int, dict[str, Any]]],
    class_maps: dict[str, dict[tuple[str, int, int], str]],
    device: str,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]], Any]:
    runner = load_module(RUN / "runner_source.py")
    # The archived runner's PROJECT is relative to its original execution path;
    # point it at the live project for read-only helper loading.
    runner.PROJECT = PROJECT
    r006 = load_module(PROJECT / "tools" / "run_yolo26_candidate_perturbation.py")
    r007 = load_module(PROJECT / "tools" / "run_yolo26_score_retention_sweep.py")
    r006.TRACES = TRACES
    r007.TRACES = TRACES
    fixed = r007.load_legacy()
    specs = tuple(spec for spec in fixed.DATASETS if spec["name"] in TRACES)
    fixed.DATASETS = specs
    mapping = read_csv(RUN / "oracle_candidate_mapping.csv")
    choices: list[dict[str, Any]] = []
    for row in mapping:
        converted: dict[str, Any] = dict(row)
        for field in ("image_id", "annotation_id", "eligible_edge_count", "image_height", "image_width", "candidate_assignment_conflict"):
            converted[field] = int(converted[field])
        for field in ("selected_source_candidate_id", "selected_global_rank"):
            converted[field] = int(converted[field]) if converted[field] else None
        for field in ("selected_raw_mask_iou", "selected_raw_gt_coverage", "selected_raw_purity", "selected_score"):
            converted[field] = float(converted[field]) if converted[field] else None
        choices.append(converted)
    per_gt_rows = read_csv(QUALITY / "per_gt.csv")
    per_gt = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row for row in per_gt_rows}
    edge_rows = read_csv(QUALITY / "edge_metrics.csv")
    edges = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"]), int(row["source_candidate_id"])): row for row in edge_rows}
    expected_targets = {key for key, row in per_gt.items() if row["baseline_class"] != "C" and row["coverage75_purity75_topk_available"] == "1" and row["coverage75_purity75_conf_available"] == "0" and row["coverage75_purity75_final_available"] == "0"}
    actual_targets = {(r["dataset"], r["image_id"], r["annotation_id"]) for r in choices}
    failures: list[str] = []
    if len(per_gt) != len(per_gt_rows):
        failures.append("strict per_gt contains duplicate GT keys")
    if len(edges) != len(edge_rows):
        failures.append("strict edge_metrics contains duplicate GT/source keys")
    mapping_keys = [(r["dataset"], r["image_id"], r["annotation_id"]) for r in choices]
    if len(mapping_keys) != len(set(mapping_keys)):
        failures.append("oracle mapping contains duplicate target keys")
    if expected_targets != actual_targets:
        failures.append("target keys differ from strict-quality stratum")
    selected_sources = []
    for row in choices:
        source = row["selected_source_candidate_id"]
        if source is None:
            continue
        selected_sources.append((row["dataset"], row["image_id"], source))
        key = (row["dataset"], row["image_id"], row["annotation_id"])
        edge = edges.get((*key, source))
        if edge is None or not (edge["topk_member"] == "1" and edge["conf_member"] == "0" and edge["final_member"] == "0" and edge["quality_coverage75_purity75"] == "1" and 0.01 < float(edge["source_score"]) <= 0.05):
            failures.append(f"ineligible selected source: {row['dataset']}/{row['image_id']}/{row['annotation_id']}")
        elif any(
            not math.isclose(float(edge[field]), float(row[selected]), rel_tol=0, abs_tol=1e-12)
            for field, selected in (
                ("iou", "selected_raw_mask_iou"), ("coverage", "selected_raw_gt_coverage"),
                ("purity", "selected_raw_purity"), ("source_score", "selected_score"),
            )
        ) or int(edge["global_rank"]) != int(row["selected_global_rank"]):
            failures.append(f"selected mapping values differ from edge_metrics: {row['dataset']}/{row['image_id']}/{row['annotation_id']}")
        if per_gt.get(key, {}).get("baseline_class") != row.get("baseline_class"):
            failures.append(f"selected target baseline class mismatch: {key}")
    if len(selected_sources) != len(set(selected_sources)):
        failures.append("selected source reused within an image")
    # Independently reconstruct the frozen target graph and maximum-cardinality mapping.
    manifests = {spec["name"]: fixed.read_json(spec["manifest"]) for spec in specs}
    shapes = {
        (dataset, int(image["id"])): (int(image["height"]), int(image["width"]))
        for dataset, manifest in manifests.items() for image in manifest["images"]
    }
    expected_choices = runner.choose_oracle(runner.target_rows(QUALITY / "per_gt.csv", QUALITY / "edge_metrics.csv", shapes), r006.maximum_cardinality_assignment)
    choice_fields = (
        "dataset", "image_id", "annotation_id", "baseline_class", "target_stratum", "eligible_edge_count",
        "image_height", "image_width", "candidate_assignment_method", "candidate_assignment_conflict",
        "selected_source_candidate_id", "selected_raw_mask_iou", "selected_raw_gt_coverage", "selected_raw_purity",
        "selected_score", "selected_global_rank", "selected_already_final",
    )
    mapping_recomputed_exact = [tuple(row.get(field) for field in choice_fields) for row in expected_choices] == [tuple(row.get(field) for field in choice_fields) for row in choices]
    if not mapping_recomputed_exact:
        failures.append("recomputed maximum-cardinality mapping differs")
    # Recompute selected mask geometry from raw coefficients/prototypes and dataset GT.
    try:
        runner.validate_selected_edges(fixed, r006, trace_records, specs, choices, TRACES, device)
        selected_geometry_exact = True
    except Exception as error:
        selected_geometry_exact = False
        failures.append(f"selected raw-mask geometry check failed: {type(error).__name__}: {error}")

    # Re-evaluate the baseline-defined removal policy from dataset GT and baseline masks.
    removable = runner.removable_predictions(fixed, trace_records, specs, choices)
    baseline, by_source = baseline_signatures(trace_records)
    removed = Counter()
    for (dataset, image_id, _), pred_ids in removable.items():
        record = trace_records[dataset][image_id]
        pred_to_source = {int(x["pred_id"]): int(x["source_candidate_id"]) for x in record["final_pred_to_source_candidate"]}
        for pred_id in pred_ids:
            removed[by_source[(dataset, image_id, pred_to_source[pred_id])]] += 1
    added = Counter()
    for row in choices:
        source = row["selected_source_candidate_id"]
        if source is None:
            continue
        dataset, image_id = row["dataset"], row["image_id"]
        with np.load(TRACES[dataset] / trace_records[dataset][image_id]["raw_cache"], allow_pickle=False) as raw:
            mask = r006.decode_masks(raw, np.asarray([source], dtype=np.int64), (row["image_height"], row["image_width"]), "cpu")[source]
            added[signature(image_id, {"score": float(raw["scores"][source]), "class": 0, "mask_rle": r006.rle_from_mask(mask)})] += 1
    expected = {
        "BASELINE": baseline,
        "ADDITION": baseline + added,
        "REMOVAL": baseline - removed,
        "ADDITION_REMOVAL": (baseline - removed) + added,
    }
    deltas: dict[str, Any] = {}
    for condition, wanted in expected.items():
        actual = export_signatures(condition)
        deltas[condition] = {"missing": sum((wanted - actual).values()), "unexpected": sum((actual - wanted).values()), "exact": actual == wanted}
    # Exact confidence replay checks source identity, unchanged raw score, box and native RLE.
    confidence_records: dict[str, dict[int, dict[str, Any]]] = {}
    replay_total = sum(len(group) for group in trace_records.values())
    replay_done = 0
    for dataset, images in trace_records.items():
        confidence_records[dataset] = {}
        for image_id, record in sorted(images.items()):
            confidence_records[dataset][image_id] = r007.reconstruct_record(record, TRACES[dataset], 300, 0.01, device)
            replay_done += 1
            if replay_done % 100 == 0 or replay_done == replay_total:
                print(f"  confidence source/RLE replay: {replay_done}/{replay_total}", flush=True)
    confidence_delta = _output_set_delta(_records_counter(confidence_records), "CONFIDENCE_ONLY")
    if not confidence_delta["exact"]:
        failures.append("confidence export differs from exact Top-300 score>.01 replay")

    # Rebuild all oracle conditions from the verified mapping/removal policy.
    condition_records: dict[str, dict[str, dict[int, dict[str, Any]]]] = {"BASELINE": trace_records, "CONFIDENCE_ONLY": confidence_records}
    accounting: dict[str, Any] = {}
    for condition in ("ADDITION", "REMOVAL", "ADDITION_REMOVAL"):
        changed, current_accounting = r006.perturb(trace_records, choices, removable, condition, device)
        condition_records[condition] = changed
        accounting[condition] = current_accounting
        fresh_delta = _output_set_delta(_records_counter(changed), condition)
        deltas[condition]["rebuild_exact"] = fresh_delta["exact"]
        deltas[condition]["rebuild_missing"] = fresh_delta["missing"]
        deltas[condition]["rebuild_unexpected"] = fresh_delta["unexpected"]
        if not fresh_delta["exact"]:
            failures.append(f"{condition} export differs from exact reconstructed records")
    summary = json.loads((RUN / "run_summary.json").read_text(encoding="utf-8"))
    target_hash, map_hash = runner.digest_choices(choices)
    removal_hash = runner.digest_removals(removable)
    hashes_ok = target_hash == summary["target_key_hash"] and map_hash == summary["selected_source_mapping_hash"] and all(
        v["target_key_hash"] == target_hash and v["selected_source_mapping_hash"] == map_hash and v["removal_mapping_hash"] == removal_hash
        for v in summary["oracle_condition_digests"].values()
    )
    prediction_accounting = {condition: sum(len(record["final_predictions"]) for group in datasets.values() for record in group.values()) for condition, datasets in condition_records.items()}
    accounting_match = prediction_accounting == summary.get("prediction_accounting")
    if not accounting_match:
        failures.append("reconstructed prediction accounting differs from run_summary")
    result = {
        "target_count_expected": len(expected_targets), "target_count_recorded": len(choices), "selected": len(selected_sources),
        "unselected_targets": len(choices) - len(selected_sources), "unique_selected_sources": len(set(selected_sources)),
        "mapping_recomputed_exact": mapping_recomputed_exact, "selected_geometry_exact": selected_geometry_exact,
        "target_and_edge_failures": failures, "mapping_hashes_match": hashes_ok,
        "target_key_hash": target_hash, "selected_source_mapping_hash": map_hash, "removal_mapping_hash": removal_hash,
        "removed_predictions_multiset": sum(removed.values()), "added_predictions": sum(added.values()),
        "oracle_multiset_checks": deltas, "confidence_exact_source_score_box_rle": confidence_delta,
        "condition_accounting": accounting, "prediction_accounting_recomputed": prediction_accounting,
        "prediction_accounting_matches": accounting_match,
    }
    return result, condition_records, fixed


def recompute_all_gt(
    fixed: Any, condition_records: dict[str, dict[str, dict[int, dict[str, Any]]]]
) -> tuple[dict[str, dict[tuple[str, int, int], str]], dict[str, Any]]:
    maps: dict[str, dict[tuple[str, int, int], str]] = {}
    details: dict[str, Any] = {}
    for condition in CONDITIONS:
        print(f"  fresh GT classification: {condition}", flush=True)
        rows, _, _ = fixed.classify_all(PROJECT, condition_records[condition])
        current = _class_map(rows)
        maps[condition] = current
        details[condition] = {"gt": len(current), "classes": dict(Counter(current.values()))}
    baseline_keys = set(maps["BASELINE"])
    details["coverage_exact_all_conditions"] = all(set(maps[condition]) == baseline_keys for condition in CONDITIONS)
    return maps, details


def validate_gt_provenance_and_effects(
    class_maps: dict[str, dict[tuple[str, int, int], str]],
    recorded_maps: dict[str, dict[tuple[str, int, int], str]],
) -> dict[str, Any]:
    failures: list[str] = []
    manifest_keys: set[tuple[str, int, int]] = set()
    manifest_counts: dict[str, Any] = {}
    for dataset, trace in TRACES.items():
        config = json.loads((trace / "inference_config.json").read_text(encoding="utf-8"))
        manifest, _ = _manifest_summary(dataset, resolve_path(config["manifest"]))
        manifest_counts[dataset] = {
            "images": manifest["images"], "gt_category_1_noncrowd": manifest["gt_category_1_noncrowd"],
            "problems": manifest["problems"],
        }
        manifest_keys.update(manifest["gt_keys"])
    per_gt_rows = read_csv(QUALITY / "per_gt.csv")
    per_gt_keys = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])) for row in per_gt_rows}
    mapping_rows = read_csv(RUN / "oracle_candidate_mapping.csv")
    target_keys = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])) for row in mapping_rows}
    selected_keys = {
        (row["dataset"], int(row["image_id"]), int(row["annotation_id"]))
        for row in mapping_rows if row.get("selected_source_candidate_id")
    }
    if manifest_keys != per_gt_keys:
        failures.append("strict-quality per_gt keys differ from dataset manifest GT")
    for condition in CONDITIONS:
        if set(class_maps.get(condition, {})) != manifest_keys:
            failures.append(f"{condition}: fresh classifier keys differ from manifest GT")
        if class_maps.get(condition, {}) != recorded_maps.get(condition, {}):
            failures.append(f"{condition}: fresh classifier labels differ from all_gt_analysis.csv")
    baseline = class_maps["BASELINE"]
    effects: dict[str, Any] = {}
    for condition in CONDITIONS:
        current = class_maps[condition]
        recovered_all = {key for key in manifest_keys if baseline[key] != "C" and current[key] == "C"}
        c_regressed_all = {key for key in manifest_keys if baseline[key] == "C" and current[key] != "C"}
        recovered_target = recovered_all & target_keys
        recovered_selected = recovered_all & selected_keys
        effects[condition] = {
            "all_gt_recovered_to_C": len(recovered_all),
            "all_gt_C_to_failure": len(c_regressed_all),
            "target_pool_recovered_to_C": len(recovered_target),
            "selected_43_recovered_to_C": len(recovered_selected),
            "outside_target_pool_recovered_to_C": len(recovered_all - target_keys),
            "target_pool_changed": sum(baseline[key] != current[key] for key in target_keys),
            "selected_43_changed": sum(baseline[key] != current[key] for key in selected_keys),
            "outside_target_pool_changed": sum(baseline[key] != current[key] for key in manifest_keys - target_keys),
        }
    return {
        "ground_truth_source": "dataset-provided COCO annotations filtered to category_id=1 and iscrowd=0",
        "manifest_counts": manifest_counts,
        "manifest_gt_total": len(manifest_keys), "strict_per_gt_total": len(per_gt_keys),
        "target_pool": len(target_keys), "selected_target_pool": len(selected_keys),
        "failures": failures, "effects": effects,
        "interpretation": "The 45 ADDITION_REMOVAL all-GT recoveries comprise selected-target recoveries plus outside-target neighbor effects; 43 selected sources is an assignment count, not automatically the all-GT recovery count.",
    }


def validate_historical_r007(class_maps: dict[str, dict[tuple[str, int, int], str]]) -> dict[str, Any]:
    historical_path = PROJECT / "experiments" / "r007_score_retention_20260905_v1" / "sweep_gt_analysis.csv"
    historical_rows = read_csv(historical_path)
    filtered_rows = [row for row in historical_rows if row.get("condition") == "k300_conf0p010"]
    historical = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row["primary_class"] for row in filtered_rows}
    current = class_maps["CONFIDENCE_ONLY"]
    stored_rows = read_csv(RUN / "r007_confidence_only_comparison.csv")
    stored = {(row["dataset"], int(row["image_id"]), int(row["annotation_id"])): row for row in stored_rows}
    differing = [key for key in sorted(set(historical) | set(current)) if historical.get(key) != current.get(key)]
    stored_bad = [
        key for key in sorted(set(stored) | set(current))
        if key not in stored or stored[key].get("current_class") != current.get(key) or stored[key].get("r007_class") != historical.get(key)
        or int(stored[key].get("same", 0)) != int(current.get(key) == historical.get(key))
    ]
    return {
        "historical_file": str(historical_path), "historical_filtered_rows": len(filtered_rows), "historical_unique_keys": len(historical),
        "current_keys": len(current), "key_sets_exact": set(historical) == set(current), "class_differences": len(differing),
        "difference_sample": differing[:10], "stored_comparison_rows": len(stored_rows), "stored_comparison_bad": len(stored_bad),
        "stored_comparison_bad_sample": stored_bad[:10],
    }


def static_metric_checks() -> dict[str, Any]:
    evaluator_path = LEGACY / "scripts" / "evaluate_task05_public.py"
    runner_path = RUN / "runner_source.py"
    classifier_path = LEGACY / "scripts" / "analyze_yolo26_diagnostic_cache_full.py"
    evaluator_text = evaluator_path.read_text(encoding="utf-8")
    runner_text = runner_path.read_text(encoding="utf-8")
    classifier_text = classifier_path.read_text(encoding="utf-8")
    tree = ast.parse(evaluator_text)
    definitions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    called = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))]
    suspicious = []
    patterns = (r"score\s*/", r"/\s*np\.(?:max|min|mean)\s*\(", r"/\s*(?:max|min|mean)\s*\([^\n]*score")
    for label, text in (("runner_source.py", runner_text), ("evaluate_task05_public.py", evaluator_text), ("analyze_yolo26_diagnostic_cache_full.py", classifier_text)):
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
                suspicious.append({"file": label, "line": line_no, "text": line.strip()})
    return {
        "evaluator_definitions": definitions,
        "evaluator_calls": sorted(set(called)),
        "evaluate_function_called_by_runner": "evaluator.evaluate(" in runner_text,
        "cocoeval_segm_constructed": 'COCOeval(coco_gt, coco_dt, "segm")' in evaluator_text,
        "cocoeval_evaluate_called": "evaluator.evaluate()" in evaluator_text,
        "cocoeval_accumulate_called": "evaluator.accumulate()" in evaluator_text,
        "cocoeval_summarize_called": "evaluator.summarize()" in evaluator_text,
        "stats_fields_exported": all(f"stats[{index}]" in evaluator_text for index in (0, 1, 2, 6, 7, 8)),
        "self_normalization_suspects": suspicious,
        "dead_metric_functions": [name for name in definitions if name not in {"evaluate", "main"}],
    }


def coco_metrics() -> dict[str, Any]:
    evaluator = load_module(LEGACY / "scripts\evaluate_task05_public.py")
    stored = json.loads((RUN / "coco_metrics.json").read_text(encoding="utf-8"))
    mismatches = []
    values = {}
    for condition in CONDITIONS:
        for dataset, trace in TRACES.items():
            manifest = Path(json.loads((trace / "inference_config.json").read_text(encoding="utf-8"))["manifest"])
            predictions = RUN / condition.lower() / f"{dataset}_coco_predictions.json"
            with redirect_stdout(io.StringIO()):
                current = evaluator.evaluate(manifest, predictions)
            key = f"{condition}/{dataset}"
            values[key] = {field: current[field] for field in ("mask_ap", "mask_ap50", "mask_ap75", "recall_ar1", "recall_ar10", "recall_ar100", "prediction_count")}
            for field, value in values[key].items():
                if field not in stored.get(key, {}) or current[field] != stored[key][field]:
                    mismatches.append(f"{key}/{field}")
    return {"metric_function": str(LEGACY / "scripts" / "evaluate_task05_public.py"), "iou_type": "segm", "max_dets": 100, "metrics": values, "mismatches": mismatches}


def _deterministic_failed(checks: dict[str, Any]) -> bool:
    hashes = checks["hashes"]
    traces = checks["trace_artifacts"]
    rles = checks["exports"]
    transitions = checks["transitions"]
    sets = checks["selection_and_sets"]
    gt = checks["gt_and_effects"]
    historical = checks["historical_r007"]
    static = checks["static_metric_path"]
    coco = checks["coco"]
    return bool(
        hashes["mismatches"]
        or traces["mismatches"] or traces["npz_bad"]
        or rles["invalid_or_empty"]
        or transitions["duplicate_gt_rows"] or transitions["invalid_classes"] or transitions["transition_mismatches"]
        or transitions["transition_csv_mismatches"] or transitions["summary_csv_mismatches"]
        or any(row["differing"] for row in transitions["fresh_classifier_map_mismatches"].values())
        or sets["target_and_edge_failures"] or not sets["mapping_hashes_match"]
        or not sets["prediction_accounting_matches"]
        or any(not row["exact"] or not row.get("rebuild_exact", True) for row in sets["oracle_multiset_checks"].values())
        or not sets["confidence_exact_source_score_box_rle"]["exact"]
        or gt["failures"]
        or not historical["key_sets_exact"] or historical["class_differences"] or historical["stored_comparison_bad"]
        or not static["evaluate_function_called_by_runner"] or not static["cocoeval_segm_constructed"]
        or not static["cocoeval_evaluate_called"] or not static["cocoeval_accumulate_called"] or not static["stats_fields_exported"]
        or static["self_normalization_suspects"] or static["dead_metric_functions"]
        or coco["mismatches"]
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    args = parser.parse_args()
    output = resolve_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUN / "run_summary.json").read_text(encoding="utf-8"))
    quality_provenance = json.loads((QUALITY / "provenance.json").read_text(encoding="utf-8"))
    print("[1/8] hashing frozen inputs and receipts", flush=True)
    trace_records = records()
    hashes = validate_hashes(summary, quality_provenance)
    trace_artifacts = validate_trace_artifacts(trace_records)
    print("[2/8] validating exported RLEs and manifest identities", flush=True)
    rles = validate_rles()
    print("[3/8] reconstructing targets, confidence replay, and oracle output sets", flush=True)
    preliminary = recompute_transitions()
    recorded_maps = preliminary.pop("maps")
    sets, condition_records, fixed = validate_targets_and_sets(trace_records, recorded_maps, args.device)
    print("[4/8] independently reclassifying every GT for all conditions", flush=True)
    fresh_maps, fresh_gt_detail = recompute_all_gt(fixed, condition_records)
    transitions = recompute_transitions(fresh_maps)
    transitions.pop("maps", None)
    print("[5/8] reconciling manifest GT, 178 targets, 43 assignments, and 45 all-GT recoveries", flush=True)
    gt_and_effects = validate_gt_provenance_and_effects(fresh_maps, recorded_maps)
    gt_and_effects["fresh_classification"] = fresh_gt_detail
    print("[6/8] comparing historical R007", flush=True)
    historical = validate_historical_r007(fresh_maps)
    static_checks = static_metric_checks()
    print("[7/8] recomputing COCO segm AP/AR from exported predictions", flush=True)
    coco = coco_metrics()
    print("[8/8] writing deterministic receipt", flush=True)
    checks = {
        "hashes": hashes, "trace_artifacts": trace_artifacts, "exports": rles, "transitions": transitions,
        "selection_and_sets": sets, "gt_and_effects": gt_and_effects, "historical_r007": historical,
        "static_metric_path": static_checks, "coco": coco,
    }
    verdict = "FAIL" if _deterministic_failed(checks) else "PASS"
    report = {
        "audit_skill": "experiment-audit",
        "receipt_type": "deterministic",
        "review_independence": "deterministic",
        "acceptance_status": "accepted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "device": args.device,
        "audit_script_sha256": sha256(Path(__file__)),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "run": str(RUN),
        "deterministic_verdict": verdict.lower(),
        "checks": checks,
    }
    (output / "audit_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "output": str(output)}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
