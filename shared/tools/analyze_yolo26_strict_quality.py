"""Compute frozen-trace per-GT strict mask-quality edges without matching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audit_yolo26_trace_replay import DEFAULT_TRACES
from run_yolo26_candidate_perturbation import BASELINE_CSV, box_mask_coverage, decode_masks, load_legacy


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("PigLife_public_test", "FaroPigSeg_test")
STAGES = ("raw", "topk", "conf", "final")
QUALITY = ("iou50", "coverage75_purity75", "iou75")
DECODE_WINDOW = 16  # The shared decoder itself remains fixed at batches of two.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in rows:
                raise ValueError(f"Duplicate trace image ID {image_id} in {path} at line {line_number}")
            rows[image_id] = row
    if not rows:
        raise ValueError(f"Empty trace cache: {path}")
    return rows


def quality_flags(iou: float, coverage: float, purity: float) -> dict[str, int]:
    return {
        "iou50": int(iou >= 0.50),
        "coverage75_purity75": int(coverage >= 0.75 and purity >= 0.75),
        "iou75": int(iou >= 0.75),
    }


def source_stage(source: int, top: set[int], conf: set[int], final: set[int]) -> str:
    if source in final:
        return "final"
    if source in conf:
        return "conf"
    if source in top:
        return "topk"
    return "raw"


def final_source_mapping(record: dict[str, Any], dataset: str, image_id: int) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    predictions = record["final_predictions"]
    mappings = record["final_pred_to_source_candidate"]
    prediction_ids = [int(prediction["pred_id"]) for prediction in predictions]
    mapping_prediction_ids = [int(mapping["pred_id"]) for mapping in mappings]
    mapping_source_ids = [int(mapping["source_candidate_id"]) for mapping in mappings]
    location = f"{dataset}/{image_id}"
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError(f"Duplicate final prediction ID: {location}")
    if len(mapping_prediction_ids) != len(set(mapping_prediction_ids)):
        raise ValueError(f"Duplicate final mapping prediction ID: {location}")
    if len(mapping_source_ids) != len(set(mapping_source_ids)):
        raise ValueError(f"Duplicate final mapping source ID: {location}")
    if set(mapping_prediction_ids) != set(prediction_ids):
        raise ValueError(f"Final mapping prediction IDs differ from final predictions: {location}")
    predictions_by_id = {int(prediction["pred_id"]): prediction for prediction in predictions}
    final_by_source = {
        int(mapping["source_candidate_id"]): predictions_by_id[int(mapping["pred_id"])]
        for mapping in mappings
    }
    prediction_index = {prediction_id: index for index, prediction_id in enumerate(prediction_ids)}
    final_index = {
        int(mapping["source_candidate_id"]): prediction_index[int(mapping["pred_id"])]
        for mapping in mappings
    }
    return final_by_source, final_index


def assert_count_invariants(counts: list[dict[str, dict[str, int]]], dataset: str, image_id: int) -> None:
    for gt_index, values in enumerate(counts):
        for quality in QUALITY:
            stages = values[quality]
            if not (stages["raw"] >= stages["topk"] >= stages["conf"] >= stages["final"]):
                raise ValueError(f"Stage nesting failed: {dataset}/{image_id}/gt={gt_index}/{quality}")
        for stage in STAGES:
            if values["iou50"][stage] < values["coverage75_purity75"][stage]:
                raise ValueError(f"Coverage/purity subset failed: {dataset}/{image_id}/gt={gt_index}/{stage}")
            if values["iou50"][stage] < values["iou75"][stage]:
                raise ValueError(f"IoU-75 subset failed: {dataset}/{image_id}/gt={gt_index}/{stage}")


def manifest_target_keys(manifest: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], set[tuple[int, int]]]:
    images: dict[int, dict[str, Any]] = {}
    for image in manifest["images"]:
        image_id = int(image["id"])
        if image_id in images:
            raise ValueError(f"Duplicate manifest image ID: {image_id}")
        images[image_id] = image
    annotations: dict[int, dict[str, Any]] = {}
    keys: set[tuple[int, int]] = set()
    for annotation in manifest["annotations"]:
        annotation_id = int(annotation["id"])
        if annotation_id in annotations:
            raise ValueError(f"Duplicate manifest annotation ID: {annotation_id}")
        annotations[annotation_id] = annotation
        image_id = int(annotation["image_id"])
        if int(annotation.get("category_id", 1)) == 1 and not annotation.get("iscrowd", 0) and image_id in images:
            keys.add((image_id, annotation_id))
    return images, annotations, keys


def _write_header(handle: Any, fields: list[str]) -> csv.DictWriter:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return writer


def _raw_trace(trace: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(trace / record["raw_cache"], allow_pickle=False) as loaded:
        return {key: loaded[key] for key in (
            "source_candidate_id", "boxes_xyxy", "scores", "mask_coefficients", "prototype",
            "top_indices", "global_rank", "framework_source_candidate_id", "framework_shape",
        )}


def _rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(value) for value in encoded["size"]], "counts": encoded["counts"]}


def analyze_image(
    *,
    fixed: Any,
    dataset: str,
    trace: Path,
    config: dict[str, Any],
    record: dict[str, Any],
    targets: list[dict[str, str]],
    annotations: dict[int, dict[str, Any]],
    image_dimensions: tuple[int, int],
    edge_writer: csv.DictWriter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from pycocotools import mask as mask_utils

    image_id = int(record["image_id"])
    framework_shape = tuple(int(value) for value in record["framework_mask_call"]["shape"])
    if len(framework_shape) != 2 or framework_shape != image_dimensions:
        raise ValueError(f"Trace/manifest dimensions differ: {dataset}/{image_id}")
    height, width = framework_shape
    gt_rles = [fixed.normalize_gt(annotations[int(row["annotation_id"])], height, width) for row in targets]
    gt_masks = [np.asarray(mask_utils.decode(rle), dtype=bool) for rle in gt_rles]
    raw = _raw_trace(trace, record)
    raw_shape = tuple(int(value) for value in raw["framework_shape"].tolist())
    if raw_shape != framework_shape:
        raise ValueError(f"Raw/trace framework dimensions differ: {dataset}/{image_id}")
    candidate_ids = raw["source_candidate_id"].astype(np.int64)
    if not np.array_equal(candidate_ids, np.arange(len(candidate_ids))):
        raise ValueError(f"Raw candidate IDs are not contiguous: {dataset}/{image_id}")
    top = set(raw["top_indices"].astype(int).tolist())
    conf = set(raw["top_indices"][raw["scores"][raw["top_indices"].astype(int)] > float(config["config"]["conf"])].astype(int).tolist())
    framework = set(raw["framework_source_candidate_id"].astype(int).tolist())
    if conf != framework:
        raise ValueError(f"Strict confidence sources differ from framework inputs: {dataset}/{image_id}")
    final_by_source, final_index = final_source_mapping(record, dataset, image_id)
    final = set(final_by_source)
    if not final <= conf or len(final) != len(final_by_source):
        raise ValueError(f"Final-source nesting failed: {dataset}/{image_id}")

    screened_by_source: dict[int, list[int]] = defaultdict(list)
    for gt_index, gt_mask in enumerate(gt_masks):
        sources = np.flatnonzero(box_mask_coverage(raw["boxes_xyxy"], gt_mask) >= 0.50)
        for source in sources.tolist():
            screened_by_source[int(source)].append(gt_index)
    decode_ids = np.asarray(sorted(set(screened_by_source) | final), dtype=np.int64)
    final_rles = [fixed.rle(prediction["mask_rle"]) for prediction in record["final_predictions"]]
    final_iou, final_coverage, final_purity = fixed.matrices(gt_rles, final_rles)
    counts = [{quality: {stage: 0 for stage in STAGES} for quality in QUALITY} for _ in targets]
    final_xor_total = 0
    final_xor_max = 0

    for start in range(0, len(decode_ids), DECODE_WINDOW):
        window = decode_ids[start : start + DECODE_WINDOW]
        decoded = decode_masks(raw, window, (height, width), config["config"]["device"])
        for source, mask in decoded.items():
            source = int(source)
            if source in final:
                expected = np.asarray(mask_utils.decode(fixed.rle(final_by_source[source]["mask_rle"])), dtype=bool)
                xor = int(np.count_nonzero(np.logical_xor(mask, expected)))
                final_xor_total += xor
                final_xor_max = max(final_xor_max, xor)
                if xor:
                    raise ValueError(f"Final-source XOR mismatch: {dataset}/{image_id}/{source}")
            for gt_index in screened_by_source.get(source, []):
                gt = gt_masks[gt_index]
                intersection = int(np.count_nonzero(np.logical_and(mask, gt)))
                mask_area = int(np.count_nonzero(mask))
                gt_area = int(np.count_nonzero(gt))
                union = mask_area + gt_area - intersection
                iou = intersection / max(union, 1)
                coverage = intersection / max(gt_area, 1)
                purity = intersection / max(mask_area, 1)
                flags = quality_flags(iou, coverage, purity)
                if not flags["iou50"]:
                    continue
                memberships = {"raw": 1, "topk": int(source in top), "conf": int(source in conf)}
                if source in final and flags["iou50"] and not all((
                    np.isclose(iou, final_iou[gt_index, final_index[source]]),
                    np.isclose(coverage, final_coverage[gt_index, final_index[source]]),
                    np.isclose(purity, final_purity[gt_index, final_index[source]]),
                )):
                    raise ValueError(f"Final RLE quality mismatch: {dataset}/{image_id}/{source}")
                for quality, passes in flags.items():
                    if passes:
                        for stage, member in memberships.items():
                            counts[gt_index][quality][stage] += member
                edge_writer.writerow({
                    "dataset": dataset, "image_id": image_id, "annotation_id": int(targets[gt_index]["annotation_id"]),
                    "baseline_class": targets[gt_index]["primary_class"], "scene_label": targets[gt_index]["scene_label"],
                    "source_candidate_id": source, "source_score": float(raw["scores"][source]), "global_rank": int(raw["global_rank"][source]),
                    "highest_stage": source_stage(source, top, conf, final), "topk_member": memberships["topk"], "conf_member": memberships["conf"], "final_member": int(source in final),
                    "iou": iou, "coverage": coverage, "purity": purity, **{f"quality_{key}": value for key, value in flags.items()},
                })

    for gt_index in range(len(targets)):
        for final_column in range(len(final_rles)):
            flags = quality_flags(
                float(final_iou[gt_index, final_column]),
                float(final_coverage[gt_index, final_column]),
                float(final_purity[gt_index, final_column]),
            )
            for quality, passes in flags.items():
                counts[gt_index][quality]["final"] += passes
    assert_count_invariants(counts, dataset, image_id)

    rows = []
    for target, values in zip(targets, counts, strict=True):
        row = {"dataset": dataset, "image_id": image_id, "annotation_id": int(target["annotation_id"]), "baseline_class": target["primary_class"], "scene_label": target["scene_label"]}
        for quality in QUALITY:
            for stage in STAGES:
                row[f"{quality}_{stage}_count"] = values[quality][stage]
                row[f"{quality}_{stage}_available"] = int(values[quality][stage] > 0)
        rows.append(row)
    return rows, {
        "dataset": dataset, "image_id": image_id, "gt_count": len(targets), "raw_candidate_count": len(candidate_ids),
        "topk_candidate_count": len(top), "conf_source_count": len(conf), "final_source_count": len(final),
        "box_screened_candidate_count": len(screened_by_source), "decoded_candidate_count": len(decode_ids),
        "final_source_xor_total": final_xor_total, "final_source_xor_max": final_xor_max,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        scoped = [row for row in rows if row["dataset"] == dataset]
        groups = [("ALL_GT", scoped)] + [(f"class:{label}", [row for row in scoped if row["baseline_class"] == label]) for label in sorted({row["baseline_class"] for row in scoped})]
        for scope, subset in groups:
            for quality in QUALITY:
                for stage in STAGES:
                    available = sum(row[f"{quality}_{stage}_available"] for row in subset)
                    output.append({"dataset": dataset, "scope": scope, "quality": quality, "stage": stage, "gt_count": len(subset), "available_gt": available, "rate": available / len(subset) if subset else 0.0})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, default=PROJECT / "experiments" / "yolo26_trace_replay_full_20260905_v1")
    parser.add_argument("--max-images-per-dataset", type=int, help="Smoke-only cap; omit for the full 586-image run.")
    args = parser.parse_args()
    if args.max_images_per_dataset is not None and args.max_images_per_dataset < 1:
        parser.error("--max-images-per-dataset must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    replay = read_json(args.replay_audit / "summary.json")
    if replay["status"] != "completed" or replay["violations"] or len(replay["traces"]) != 2 or not all(row["full_coverage"] for row in replay["traces"]):
        raise ValueError("A successful full two-trace replay audit is required")
    replay_metadata = read_json(args.replay_audit / "run_metadata.json")
    if replay_metadata["traces"] != [str(trace) for trace in DEFAULT_TRACES]:
        raise ValueError("Replay audit refers to different frozen traces")
    from ultralytics import __version__ as ultralytics_version
    from ultralytics.utils import ops

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fixed = load_legacy()
    decoder_source = Path(decode_masks.__code__.co_filename).resolve()
    legacy_source = Path(fixed.__file__).resolve()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "traces": [str(trace) for trace in DEFAULT_TRACES],
        "definition": "Per-GT quality availability only; no matching, recovery, causal attribution, or deployable selection.",
        "quality": {"iou50": "IoU >= 0.50", "coverage75_purity75": "coverage >= 0.75 AND purity >= 0.75", "iou75": "IoU >= 0.75"},
        "raw_screen": "Lossless inclusion for IoU >= 0.50: box-envelope GT coverage >= 0.50; no quota, downsampling, or purity screen.",
        "decoder": {"function": "run_yolo26_candidate_perturbation.decode_masks", "batch_size": 2, "window_size": DECODE_WINDOW, "ultralytics": ultralytics_version},
        "replay_source_sha256": replay_metadata["source_sha256"],
        "files_sha256": {str(Path(__file__)): sha256(Path(__file__)), str(Path(ops.__file__)): sha256(Path(ops.__file__)), str(decoder_source): sha256(decoder_source), str(legacy_source): sha256(legacy_source), str(BASELINE_CSV): sha256(BASELINE_CSV), str(args.replay_audit / "summary.json"): sha256(args.replay_audit / "summary.json"), str(args.replay_audit / "run_metadata.json"): sha256(args.replay_audit / "run_metadata.json")},
        "max_images_per_dataset": args.max_images_per_dataset,
    }
    for trace in DEFAULT_TRACES:
        for name in ("inference_config.json", "diagnostic_cache.jsonl", "artifact_hashes.json"):
            path = trace / name
            metadata["files_sha256"][str(path)] = sha256(path)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_status.json").write_text(json.dumps({"status": "running", "started_at": metadata["created_at"]}) + "\n", encoding="utf-8")
    try:
        with BASELINE_CSV.open(encoding="utf-8-sig", newline="") as handle:
            baseline = [row for row in csv.DictReader(handle) if row["dataset"] in DATASETS]
        baseline_by_dataset: dict[str, dict[int, list[dict[str, str]]]] = {dataset: defaultdict(list) for dataset in DATASETS}
        baseline_keys_by_dataset: dict[str, set[tuple[int, int]]] = {dataset: set() for dataset in DATASETS}
        for row in baseline:
            dataset = row["dataset"]
            key = (int(row["image_id"]), int(row["annotation_id"]))
            if key in baseline_keys_by_dataset[dataset]:
                raise ValueError(f"Duplicate baseline GT key: {dataset}/{key[0]}/{key[1]}")
            baseline_keys_by_dataset[dataset].add(key)
            baseline_by_dataset[dataset][key[0]].append(row)
        per_gt_rows: list[dict[str, Any]] = []
        image_rows: list[dict[str, Any]] = []
        edge_fields = ["dataset", "image_id", "annotation_id", "baseline_class", "scene_label", "source_candidate_id", "source_score", "global_rank", "highest_stage", "topk_member", "conf_member", "final_member", "iou", "coverage", "purity", "quality_iou50", "quality_coverage75_purity75", "quality_iou75"]
        with (args.output_dir / "edge_metrics.csv").open("w", encoding="utf-8-sig", newline="") as edge_handle:
            edge_writer = _write_header(edge_handle, edge_fields)
            for dataset, trace in zip(DATASETS, DEFAULT_TRACES, strict=True):
                config = read_json(trace / "inference_config.json")
                manifest = read_json(Path(config["manifest"]))
                if sha256(Path(config["manifest"])) != config["provenance_hashes"]["manifest_sha256"]:
                    raise ValueError(f"Manifest provenance mismatch: {dataset}")
                images, annotations, manifest_keys = manifest_target_keys(manifest)
                if baseline_keys_by_dataset[dataset] != manifest_keys:
                    raise ValueError(f"Baseline/manifest GT key mismatch: {dataset}")
                records = read_jsonl(trace / "diagnostic_cache.jsonl")
                image_ids = sorted(baseline_by_dataset[dataset])
                if set(image_ids) != set(records):
                    raise ValueError(f"Baseline/trace image coverage mismatch: {dataset}")
                if args.max_images_per_dataset is not None:
                    image_ids = image_ids[:args.max_images_per_dataset]
                for ordinal, image_id in enumerate(image_ids, start=1):
                    image_dimensions = (int(images[image_id]["height"]), int(images[image_id]["width"]))
                    gt_rows, image_row = analyze_image(fixed=fixed, dataset=dataset, trace=trace, config=config, record=records[image_id], targets=baseline_by_dataset[dataset][image_id], annotations=annotations, image_dimensions=image_dimensions, edge_writer=edge_writer)
                    per_gt_rows.extend(gt_rows)
                    image_rows.append(image_row)
                    if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(image_ids):
                        print(json.dumps({"event": "progress", "dataset": dataset, "processed": ordinal, "total": len(image_ids), "decoded": image_row["decoded_candidate_count"]}), flush=True)
        per_gt_fields = list(per_gt_rows[0]) if per_gt_rows else []
        for name, rows, fields in (("per_gt.csv", per_gt_rows, per_gt_fields), ("image_audit.csv", image_rows, list(image_rows[0]) if image_rows else []), ("stage_summary.csv", summarize(per_gt_rows), ["dataset", "scope", "quality", "stage", "gt_count", "available_gt", "rate"])):
            with (args.output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = _write_header(handle, fields)
                writer.writerows(rows)
        expected_gt = len(baseline) if args.max_images_per_dataset is None else len(per_gt_rows)
        if len(per_gt_rows) != expected_gt:
            raise ValueError(f"GT coverage mismatch: {len(per_gt_rows)} != {expected_gt}")
        summary = {"status": "completed", "full_coverage": args.max_images_per_dataset is None, "images": len(image_rows), "gt": len(per_gt_rows), "edges": sum(1 for _ in (args.output_dir / "edge_metrics.csv").open(encoding="utf-8-sig")) - 1, "elapsed_seconds": round(time.monotonic() - started, 3), "final_source_xor_total": sum(row["final_source_xor_total"] for row in image_rows), "final_source_xor_max": max((row["final_source_xor_max"] for row in image_rows), default=0)}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "run_status.json").write_text(json.dumps({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat()}) + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as error:
        (args.output_dir / "run_status.json").write_text(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
