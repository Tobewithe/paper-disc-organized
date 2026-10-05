"""Per-GT mask-IoU availability from an audited raw oracle and frozen traces.

Raw counts reuse the completed, coverage-bound exhaustive R006 search.
Only Top-K masks need decoding here. Availability is not simultaneous matching,
taxonomy recovery, or a causal attribution to ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_yolo26_trace_replay import DEFAULT_TRACES
from bootstrap_yolo26_gate import bootstrap_rate
from run_yolo26_candidate_perturbation import BASELINE_CSV, box_mask_coverage, decode_masks, load_legacy, write_csv


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("PigLife_public_test", "FaroPigSeg_test")
STAGES = ("raw", "topk", "conf", "final")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_chunks(handle)


def _hash_chunks(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def first_loss(counts):
    if any(counts[a] < counts[b] for a, b in zip(STAGES, STAGES[1:])):
        raise ValueError(f"Non-nested candidate availability: {counts}")
    return next((stage for stage in STAGES if counts[stage] == 0), "survives_final")


def analyze_image(fixed, trace, config, record, targets, annotations):
    from pycocotools import mask as mask_utils

    height, width = record["framework_mask_call"]["shape"]
    gt_rles = [fixed.normalize_gt(annotations[int(row["annotation_id"])], height, width) for row in targets]
    with np.load(trace / record["raw_cache"], allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in ("source_candidate_id", "boxes_xyxy", "scores", "mask_coefficients", "prototype", "top_indices", "framework_source_candidate_id")}
    ids = raw["source_candidate_id"]
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Raw candidate IDs must equal array rows")
    top = raw["top_indices"].astype(int)
    conf = set(top[raw["scores"][top] > float(config["config"]["conf"])].tolist())
    framework = set(raw["framework_source_candidate_id"].astype(int).tolist())
    if conf != framework:
        raise ValueError(f"Top-K/strict-confidence sources differ from framework inputs: {record['image_id']}")
    final_sources = {int(item["source_candidate_id"]) for item in record["final_pred_to_source_candidate"]}
    if not final_sources <= conf:
        raise ValueError("Final sources are not a subset of confidence-passing sources")
    relevant = np.zeros(len(top), dtype=bool)
    for gt in gt_rles:
        relevant |= box_mask_coverage(raw["boxes_xyxy"][top], mask_utils.decode(gt).astype(bool)) >= 0.50
    candidate_ids = np.sort(top[relevant])
    decoded = decode_masks(raw, candidate_ids, (height, width), config["config"]["device"])
    predicted = [mask_utils.encode(np.asfortranarray(decoded[int(source)].astype(np.uint8))) for source in candidate_ids]
    ious = np.asarray(mask_utils.iou(predicted, gt_rles, [0] * len(gt_rles)), dtype=float).reshape(len(predicted), len(gt_rles)).T
    good = ious >= 0.50
    final_rles = [fixed.rle(pred["mask_rle"]) for pred in record["final_predictions"]]
    final_iou = np.asarray(mask_utils.iou(final_rles, gt_rles, [0] * len(gt_rles)), dtype=float).reshape(len(final_rles), len(gt_rles)).T
    conf_mask = np.isin(candidate_ids, list(conf))
    final_mask = np.isin(candidate_ids, list(final_sources))
    output = []
    for index, target in enumerate(targets):
        counts = {"raw": int(target["raw_good_count"]), "topk": int(good[index].sum()), "conf": int(good[index, conf_mask].sum()), "final": int((final_iou[index] >= 0.50).sum())}
        if counts["final"] != int(good[index, final_mask].sum()):
            raise ValueError(f"Reconstructed versus final-RLE IoU eligibility differs: {record['image_id']}/{target['annotation_id']}")
        loss = first_loss(counts)
        output.append({"dataset": target["dataset"], "image_id": record["image_id"], "annotation_id": int(target["annotation_id"]), "baseline_class": target["baseline_class"], "scene_label": target["scene_label"], **{f"{stage}_good_count": counts[stage] for stage in STAGES}, **{f"{stage}_available": int(counts[stage] > 0) for stage in STAGES}, "first_unavailable_stage": loss})
    return output, {"image_id": record["image_id"], "failed_gt": len(targets), "raw_positions": len(ids), "topk_positions": len(top), "topk_decoded": len(candidate_ids), "framework_conf_count": len(conf), "final_count": len(final_sources), "strict_conf_source_parity": True}


def summarize(rows):
    output = []
    for dataset in DATASETS:
        scoped = [row for row in rows if row["dataset"] == dataset]
        if not scoped:
            continue
        groups = [("ALL_FAILED", scoped)]
        groups += [(f"class:{label}", [row for row in scoped if row["baseline_class"] == label]) for label in sorted({row["baseline_class"] for row in scoped})]
        groups += [(f"scene:{label}", [row for row in scoped if row["scene_label"] == label]) for label in sorted({row["scene_label"] for row in scoped})]
        for name, subset in groups:
            for stage in STAGES:
                rate, low, high = bootstrap_rate(subset, f"{stage}_available", 3000, 20260905)
                output.append({"dataset": dataset, "scope": name, "stage": stage, "n_failed_gt": len(subset), "n_images_with_failed_gt": len({row["image_id"] for row in subset}), "available_gt": sum(row[f"{stage}_available"] for row in subset), "rate": rate, "ci95_low": low, "ci95_high": high})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, help="Sanity-only cap on failed images per dataset")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    audit = read_json(args.replay_audit / "summary.json")
    if audit["status"] != "completed" or audit["violations"] or len(audit["traces"]) != 2 or not all(row["full_coverage"] for row in audit["traces"]):
        raise ValueError("A successful full two-dataset replay audit is required")
    metadata = read_json(args.replay_audit / "run_metadata.json")
    if [str(path) for path in DEFAULT_TRACES] != metadata["traces"]:
        raise ValueError("Replay audit refers to different input traces")
    from ultralytics.utils import ops
    for source in (str(Path(ops.__file__)),):
        if metadata["source_sha256"][source] != sha256(Path(source)):
            raise ValueError("Native mask runtime changed after replay audit")
    oracle = read_json(args.oracle_dir / "run_summary.json")
    if oracle["sanity_limit_failures"] is not None or set(oracle["datasets"]) != set(DATASETS):
        raise ValueError("A full two-dataset R006 run is required")
    targets = read_csv(args.oracle_dir / "oracle_candidate_mapping.csv")
    baseline = read_csv(args.oracle_dir / "baseline_gt_analysis.csv")
    key = lambda row: (row["dataset"], int(row["image_id"]), int(row["annotation_id"]))
    by_key = {key(row): row for row in baseline if row["primary_class"] != "C"}
    geometry = {key(row): row for row in read_csv(BASELINE_CSV)}
    if len({key(row) for row in targets}) != len(targets) or {key(row) for row in targets} != set(by_key) or len(targets) != oracle["target_failed_gt"]:
        raise ValueError("Raw oracle does not cover each baseline failed GT exactly once")
    for row in targets:
        if row["baseline_class"] != by_key[key(row)]["primary_class"]:
            raise ValueError("Baseline class drift in raw oracle")
        if geometry[key(row)]["primary_class"] != row["baseline_class"]:
            raise ValueError("Legacy geometry labels refer to different baseline classes")
        row["scene_label"] = geometry[key(row)]["scene_label"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = load_legacy()
    provenance = {"definition": "Per-failed-GT mask IoU >= .50 availability; no simultaneous one-to-one allocation or causal interpretation.", "raw_source": str(args.oracle_dir), "raw_screen_proof": "IoU(M,G) <= coverage(M,G) <= coverage(box envelope,G). R006 screens only envelope coverage >= .50, with no box purity or candidate quota.", "confidence": "Top-K AND score > conf; exact source-set equality with framework mask inputs is required.", "scene_labels": "Legacy GT-geometry labels joined by dataset/image/annotation, with baseline class equality checked. Touching is not occlusion.", "bootstrap": "3000 image-cluster resamples within each dataset/scope; seed 20260905; conditional on images with eligible failed GT.", "sanity_max_images": args.max_images, "files_sha256": {str(path): sha256(path) for path in (Path(__file__), Path(ops.__file__), BASELINE_CSV, args.oracle_dir / "oracle_candidate_mapping.csv", args.oracle_dir / "baseline_gt_analysis.csv", args.replay_audit / "summary.json")}}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    all_rows, image_rows = [], []
    for dataset, trace in zip(DATASETS, DEFAULT_TRACES):
        config = read_json(trace / "inference_config.json")
        manifest = read_json(Path(config["manifest"]))
        annotations = {int(ann["id"]): ann for ann in manifest["annotations"]}
        if sha256(Path(config["manifest"])) != config["provenance_hashes"]["manifest_sha256"]:
            raise ValueError("GT manifest changed since the trace")
        grouped = {}
        for target in targets:
            if target["dataset"] == dataset:
                grouped.setdefault(int(target["image_id"]), []).append(target)
        with (trace / "diagnostic_cache.jsonl").open(encoding="utf-8") as handle:
            records = {int(row["image_id"]): row for row in map(json.loads, handle)}
        image_ids = sorted(grouped)
        if args.max_images is not None:
            image_ids = image_ids[:args.max_images]
        for ordinal, image_id in enumerate(image_ids, 1):
            rows, image = analyze_image(fixed, trace, config, records[image_id], grouped[image_id], annotations)
            all_rows.extend(rows)
            image_rows.append({"dataset": dataset, **image})
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(image_ids):
                print(json.dumps({"dataset": dataset, "processed_failed_images": ordinal, "total_failed_images": len(image_ids)}), flush=True)
    if args.max_images is None and {key(row) for row in all_rows} != set(by_key):
        raise ValueError("Final stage table does not cover every failed GT")
    write_csv(args.output_dir / "stage_gt.csv", all_rows)
    write_csv(args.output_dir / "stage_summary.csv", summarize(all_rows))
    write_csv(args.output_dir / "image_audit.csv", image_rows)
    counts = {dataset: dict(Counter(row["first_unavailable_stage"] for row in all_rows if row["dataset"] == dataset)) for dataset in DATASETS}
    summary = {"status": "completed", "failed_gt": len(all_rows), "full_failure_coverage": args.max_images is None, "first_unavailable_stage": counts}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
