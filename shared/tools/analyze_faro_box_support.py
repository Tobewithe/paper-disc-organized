"""Audit Faro labels and box-constrained support against frozen mask quality."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from analyze_yolo26_strict_quality import read_jsonl, read_json, sha256
from audit_yolo26_trace_replay import DEFAULT_TRACES
from audit_yolo26_oracle_results import rows, write_csv
from run_yolo26_candidate_perturbation import load_legacy


PROJECT = Path(__file__).resolve().parents[1]
QUALITY = PROJECT / "experiments/yolo26_strict_quality_fulltrace_20260905_v1"
SOURCE = Path(r"C:\Dpan\document\model_datasets\datasets\FaroPigSeg\test")
DATASET = "FaroPigSeg_test"
STAGES = ("raw", "topk", "conf", "final")


def bounds(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    result = np.clip(np.ceil(boxes).astype(np.int64), 0, [w, h, w, h])
    # Inverted raw boxes have empty support under the runtime inequalities.
    result[:, 2:] = np.maximum(result[:, 2:], result[:, :2])
    return result


def support_coverage(boxes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = bounds(boxes, mask.shape).T
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    inside = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return inside / int(mask.sum())


def check_runtime() -> dict:
    import torch
    from ultralytics.utils import ops

    body = ast.parse(inspect.getsource(ops.process_mask_native)).body[0]
    last = body.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call) or not isinstance(last.value.func, ast.Name) or last.value.func.id != "crop_mask":
        raise ValueError("Native decoder no longer ends with crop_mask")
    boxes = np.asarray([[-2.2, -1, 3.3, 4.4], [.1, .9, 8.1, 7.2], [2, 3, 9, 11], [12.2, 1, 16, 8], [0, 0, 14, 12], [8, 9, 3, 4], [2, 9, 8, 4]], dtype=np.float32)
    mask = np.random.default_rng(20260907).integers(0, 2, (12, 14), dtype=np.uint8)
    cropped = ops.crop_mask(torch.ones((len(boxes), 12, 14), dtype=torch.uint8), torch.from_numpy(boxes)).numpy()
    direct = (cropped * mask).sum(axis=(1, 2)) / mask.sum()
    if not np.array_equal(direct, support_coverage(boxes, mask)):
        raise ValueError("Runtime crop differs from integral support")
    return {"source": inspect.getfile(ops), "source_sha256": sha256(Path(inspect.getfile(ops))), "fractional_edge_cases": len(boxes), "crop_parity": True}


def rle_signature(rle: dict) -> tuple:
    value = rle["counts"]
    return (tuple(rle["size"]), value.encode("ascii") if isinstance(value, str) else value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    runtime = check_runtime()
    trace = DEFAULT_TRACES[1]
    config = read_json(trace / "inference_config.json")
    manifest_path = Path(config["manifest"])
    manifest = read_json(manifest_path)
    if sha256(manifest_path) != config["provenance_hashes"]["manifest_sha256"]:
        raise ValueError("Manifest hash differs from trace")
    receipt = read_json(QUALITY / "provenance.json")
    frozen = read_json(PROJECT / "experiments/yolo26_confidence_diagnostic_fulltrace_20260905_v2/run_summary.json")
    for name in ("per_gt.csv", "edge_metrics.csv", "provenance.json"):
        file = QUALITY / name
        expected = next(value for path, value in frozen["input_sha256_before"].items() if Path(path).name == name)
        if sha256(file) != expected:
            raise ValueError(f"Strict input hash mismatch: {file}")
    historical_verified = []
    for path, expected in receipt["files_sha256"].items():
        file = Path(path)
        if file in {manifest_path, Path(runtime["source"]), trace / "diagnostic_cache.jsonl"}:
            if sha256(file) != expected:
                raise ValueError(f"Historical hash mismatch: {file}")
            historical_verified.append(path)
    tracked_paths = [manifest_path, trace / "inference_config.json", trace / "diagnostic_cache.jsonl", QUALITY / "per_gt.csv", QUALITY / "edge_metrics.csv", QUALITY / "provenance.json", Path(__file__), Path(runtime["source"]), PROJECT / "experiments/faro_mask_quality_20260907_protocol.md"]
    hashes = {str(p): sha256(p) for p in tracked_paths}
    records = read_jsonl(trace / "diagnostic_cache.jsonl")
    images = {int(x["id"]): x for x in manifest["images"]}
    annotations = defaultdict(list)
    for ann in manifest["annotations"]:
        if ann["category_id"] != 1 or ann.get("iscrowd", 0):
            raise ValueError("Unexpected category/crowd contract")
        annotations[int(ann["image_id"])].append(ann)
    quality = {(int(r["image_id"]), int(r["annotation_id"])): r for r in rows(QUALITY / "per_gt.csv") if r["dataset"] == DATASET}
    edges = defaultdict(list)
    for row in rows(QUALITY / "edge_metrics.csv"):
        if row["dataset"] == DATASET:
            edges[(int(row["image_id"]), int(row["annotation_id"]))].append(row)
    if set(images) != set(records) or len(images) != 160 or len(quality) != 1752:
        raise ValueError("Faro coverage mismatch")
    fixed = load_legacy()
    output, conversions = [], []
    source_hashes, raw_hashes = {}, {}
    total_final, outside_final = 0, 0
    for image_id, image in sorted(images.items()):
        h, w = int(image["height"]), int(image["width"])
        record = records[image_id]
        raw_path = trace / record["raw_cache"]
        raw_hashes[str(raw_path)] = sha256(raw_path)
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("source_candidate_id", "boxes_xyxy", "scores", "top_indices", "framework_shape")}
        boxes = raw["boxes_xyxy"]
        if tuple(raw["framework_shape"]) != (h, w) or not np.array_equal(raw["source_candidate_id"], np.arange(len(boxes))) or not np.isfinite(boxes).all():
            raise ValueError(f"Raw geometry/source mismatch: {image_id}")
        source_image = SOURCE / "images" / Path(image["file_name"]).name
        source_label = SOURCE / "labels" / (source_image.stem + ".txt")
        for file in (source_image, source_label):
            source_hashes[str(file)] = sha256(file)
        expected_image_hash = config["provenance_hashes"]["selected_images"][str(image_id)]["sha256"]
        with Image.open(source_image) as opened:
            image_shape = opened.size
        if source_hashes[str(source_image)] != expected_image_hash or image_shape != (w, h):
            raise ValueError(f"Original source image mismatch: {image_id}")
        source_rles = []
        out_of_bounds = 0
        for line in source_label.read_text(encoding="utf-8-sig").splitlines():
            tokens = [float(v) for v in line.split()]
            if not tokens:
                continue
            if tokens[0] != 0 or len(tokens) < 7 or (len(tokens) - 1) % 2:
                raise ValueError(f"Invalid source polygon: {source_label}")
            polygon = np.asarray(tokens[1:]).reshape(-1, 2)
            out_of_bounds += int(((polygon < 0) | (polygon > 1)).any())
            polygon *= [w, h]
            source_rles.append(mask_utils.merge(mask_utils.frPyObjects([polygon.ravel().tolist()], h, w)))
        anns = annotations[image_id]
        gt_rles = [fixed.normalize_gt(ann, h, w) for ann in anns]
        if Counter(map(rle_signature, source_rles)) != Counter(map(rle_signature, gt_rles)):
            raise ValueError(f"Source label/COCO mask multiset mismatch: {image_id}")
        masks = [mask_utils.decode(rle).astype(bool) for rle in gt_rles]
        occupied = np.sum(masks, axis=0)
        conversions.append({"image_id": image_id, "source_labels": len(source_rles), "manifest_gt": len(anns), "raster_multiset_equal": True, "out_of_bounds_source_polygons": out_of_bounds, "inverted_raw_boxes": int((boxes[:, 2:] < boxes[:, :2]).any(axis=1).sum())})
        final_ids = np.asarray([int(m["source_candidate_id"]) for m in record["final_pred_to_source_candidate"]], dtype=np.int64)
        by_pred = {int(m["pred_id"]): int(m["source_candidate_id"]) for m in record["final_pred_to_source_candidate"]}
        for pred in record["final_predictions"]:
            source = by_pred[int(pred["pred_id"])]
            pmask = mask_utils.decode(fixed.rle(pred["mask_rle"])).astype(bool)
            x0, y0, x1, y1 = bounds(boxes[[source]], (h, w))[0]
            outside = int(pmask.sum() - pmask[y0:y1, x0:x1].sum())
            outside_final += outside
            total_final += 1
            if outside:
                raise ValueError(f"Final pixels outside support: {image_id}/{source}")
        top = raw["top_indices"].astype(np.int64)
        stage_ids = {"raw": np.arange(len(boxes)), "topk": top, "conf": top[raw["scores"][top] > .05], "final": final_ids}
        for ann, gt in zip(anns, masks):
            key = (image_id, int(ann["id"]))
            q = quality[key]
            if not gt.any():
                raise ValueError(f"Empty GT: {key}")
            support = support_coverage(boxes, gt)
            for edge in edges[key]:
                if float(edge["coverage"]) > support[int(edge["source_candidate_id"])] + 1e-12:
                    raise ValueError(f"Stored mask exceeds support: {key}")
            best_edge = max(edges[key], key=lambda e: float(e["iou"]), default=None)
            for stage, ids in stage_ids.items():
                best = int(ids[np.argmax(support[ids])]) if len(ids) else None
                count = int(np.count_nonzero(support[ids] >= .75))
                mask_count = int(q[f"coverage75_purity75_{stage}_count"])
                if mask_count > count:
                    raise ValueError(f"Mask/support count inconsistency: {key}/{stage}")
                output.append({"image_id": image_id, "annotation_id": ann["id"], "baseline_class": q["baseline_class"], "scene_label": q["scene_label"], "stage": stage, "raw_strict_mask_absent": int(q["coverage75_purity75_raw_available"] == "0"), "candidate_count": len(ids), "support75_count": count, "strict_mask_count": mask_count, "support_available": int(count > 0), "strict_mask_available": int(mask_count > 0), "best_support": float(support[best]) if best is not None else None, "best_support_source": best, "best_support_score": float(raw["scores"][best]) if best is not None else None, "gt_area": int(gt.sum()), "gt_overlap_other_fraction": float(np.count_nonzero(gt & (occupied > 1)) / gt.sum()), "gt_touches_image_edge": int(gt[0].any() or gt[-1].any() or gt[:, 0].any() or gt[:, -1].any()), "best_raw_iou50_edge_iou": float(best_edge["iou"]) if best_edge else None, "best_raw_iou50_edge_coverage": float(best_edge["coverage"]) if best_edge else None, "best_raw_iou50_edge_purity": float(best_edge["purity"]) if best_edge else None})
        if image_id % 40 == 0:
            print(json.dumps({"images_processed": image_id, "gt_stage_rows": len(output)}), flush=True)
    summaries = []
    for stage in STAGES:
        for group in ("all", "baseline_C", "baseline_failed", "failed_raw_strict_absent"):
            scoped = [r for r in output if r["stage"] == stage and (group == "all" or group == "baseline_C" and r["baseline_class"] == "C" or group == "baseline_failed" and r["baseline_class"] != "C" or group == "failed_raw_strict_absent" and r["baseline_class"] != "C" and r["raw_strict_mask_absent"])]
            summaries.append({"stage": stage, "group": group, "gt": len(scoped), "support_unavailable": sum(not r["support_available"] for r in scoped), "support_available_mask_absent": sum(r["support_available"] and not r["strict_mask_available"] for r in scoped), "strict_mask_available": sum(r["strict_mask_available"] for r in scoped)})
    if len(output) != 1752 * 4 or total_final != 3484:
        raise ValueError("Incomplete GT/final accounting")
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during analysis: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_gt_stage.csv", output)
    write_csv(args.output_dir / "source_conversion.csv", conversions)
    write_csv(args.output_dir / "support_summary.csv", summaries)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "completed", "images": 160, "gt": 1752, "gt_stage_rows": len(output), "source_label_raster_parity": True, "final_masks_checked": total_final, "final_pixels_outside_box_support": outside_final, "runtime": runtime, "historical_hashes_verified": historical_verified, "input_sha256": hashes, "source_sha256": source_hashes, "current_raw_sha256": raw_hashes, "summary": summaries, "interpretation": "Support feasibility is necessary only; feasible boxes do not establish learned-mask capacity or correct box localization. No modality or causal method claim.", "independent_review": "pending"}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
