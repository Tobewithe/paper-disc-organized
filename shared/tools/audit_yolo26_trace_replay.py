"""Replay immutable YOLO26 full traces and audit same-forward provenance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PIGCV_ROOT = PROJECT.parent / "pigcv_research"
DEFAULT_CONFIG = PIGCV_ROOT / "configs" / "yolo26seg" / "diagnostic.yaml"
DEFAULT_TRACES = (
    PIGCV_ROOT / "artifacts" / "inference_cache" / "yolo26seg_trace_20260905_1930_piglife_full",
    PIGCV_ROOT / "artifacts" / "inference_cache" / "yolo26seg_trace_20260905_1945_faro_full",
)
RAW_KEYS = (
    "source_candidate_id",
    "boxes_xyxy",
    "scores",
    "mask_coefficients",
    "prototype",
    "top_indices",
    "top_scores",
    "global_rank",
    "feature_level",
    "grid_x",
    "grid_y",
    "stride",
    "framework_coefficients",
    "framework_boxes_xyxy",
    "framework_source_candidate_id",
    "framework_shape",
)
MAPPING_KEYS = ("pred_id", "source_candidate_id", "top_k_position", "box_iou", "score_delta")
IOU_THRESHOLDS = (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def _array_audit(old: np.ndarray, replay: np.ndarray) -> dict[str, Any]:
    old = np.asarray(old)
    replay = np.asarray(replay)
    same_shape = old.shape == replay.shape
    same_dtype = old.dtype == replay.dtype
    exact = bool(same_shape and same_dtype and np.array_equal(old, replay))
    if same_shape and old.size and np.issubdtype(old.dtype, np.number) and np.issubdtype(replay.dtype, np.number):
        max_abs_delta: float | None = float(np.max(np.abs(old.astype(np.float64) - replay.astype(np.float64))))
    else:
        max_abs_delta = 0.0 if same_shape else None
    return {
        "match": exact,
        "old_shape": list(old.shape),
        "replay_shape": list(replay.shape),
        "old_dtype": str(old.dtype),
        "replay_dtype": str(replay.dtype),
        "max_abs_delta": max_abs_delta,
    }


def _json_exact(old: Any, replay: Any) -> bool:
    return json.dumps(old, sort_keys=True, separators=(",", ":")) == json.dumps(replay, sort_keys=True, separators=(",", ":"))


def _mapping_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in MAPPING_KEYS} for row in rows]


def _mask_replay_audit(replayed: list[np.ndarray], expected: list[np.ndarray], pred_ids: list[int]) -> dict[str, Any]:
    if len(replayed) != len(expected) or len(expected) != len(pred_ids):
        raise ValueError("Mask replay rows and final predictions must have the same length")
    xors: list[int] = []
    ious: list[float] = []
    mismatch_ids: list[int] = []
    for mask, reference, pred_id in zip(replayed, expected, pred_ids, strict=True):
        mask = np.asarray(mask, dtype=bool)
        reference = np.asarray(reference, dtype=bool)
        if mask.shape != reference.shape:
            raise ValueError(f"Mask shape mismatch for pred_id={pred_id}: {mask.shape} != {reference.shape}")
        xor = int(np.count_nonzero(np.logical_xor(mask, reference)))
        intersection = int(np.count_nonzero(np.logical_and(mask, reference)))
        union = int(np.count_nonzero(np.logical_or(mask, reference)))
        iou = 1.0 if union == 0 else intersection / union
        xors.append(xor)
        ious.append(iou)
        if xor:
            mismatch_ids.append(int(pred_id))
    return {
        "final_masks_compared": len(expected),
        "all_exact": not mismatch_ids,
        "xor_pixels_total": sum(xors),
        "xor_pixels_max": max(xors, default=0),
        "min_iou": min(ious, default=1.0),
        "iou_threshold_pass_count": {f"{threshold:.2f}": sum(iou >= threshold for iou in ious) for threshold in IOU_THRESHOLDS},
        "mismatched_pred_ids": mismatch_ids[:20],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _trace_config(trace: Path) -> dict[str, Any]:
    with (trace / "inference_config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _reconstruct_old_final_masks(record: dict[str, Any], raw: Any, device: str) -> dict[str, Any]:
    import torch
    from pycocotools import mask as mask_utils
    from ultralytics.utils import ops

    predictions = record["final_predictions"]
    sources = {int(row["pred_id"]): int(row["source_candidate_id"]) for row in record["final_pred_to_source_candidate"]}
    if len(sources) != len(predictions):
        raise ValueError(f"Final source mapping is not one-to-one for image_id={record['image_id']}")
    candidate_rows = {int(candidate): row for row, candidate in enumerate(np.asarray(raw["source_candidate_id"]).tolist())}
    pred_ids = [int(prediction["pred_id"]) for prediction in predictions]
    source_rows = np.asarray([candidate_rows[sources[pred_id]] for pred_id in pred_ids], dtype=np.int64)
    shape = tuple(int(value) for value in np.asarray(raw["framework_shape"]).tolist())
    proto = torch.from_numpy(np.asarray(raw["prototype"])).to(device)
    replayed: list[np.ndarray] = []
    for start in range(0, len(source_rows), 2):
        rows = source_rows[start : start + 2]
        masks = ops.process_mask_native(
            proto,
            torch.from_numpy(np.asarray(raw["mask_coefficients"])[rows]).to(device),
            torch.from_numpy(np.asarray(raw["boxes_xyxy"])[rows]).to(device),
            shape,
        )
        replayed.extend(masks.detach().cpu().numpy().astype(bool))
    expected = []
    for prediction in predictions:
        rle = dict(prediction["mask_rle"])
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        expected.append(np.asarray(mask_utils.decode(rle), dtype=bool))
    return _mask_replay_audit(replayed, expected, pred_ids)


def _audit_trace(trace: Path, model: Any, config: Any, output: Path, max_images: int | None = None) -> dict[str, Any]:
    from models.yolo26seg.diagnostic_inferencer import _collect_raw_candidates

    trace_config = _trace_config(trace)
    old_config = trace_config["config"]
    if old_config != config.as_dict():
        raise ValueError(f"Formal config mismatch for trace: {trace}")
    manifest = Path(trace_config["manifest"])
    image_root = Path(trace_config["image_root"])
    with manifest.open(encoding="utf-8") as handle:
        images = {int(image["id"]): image for image in json.load(handle)["images"]}
    rows = _read_jsonl(trace / "diagnostic_cache.jsonl")
    if len(rows) != int(trace_config["selected_images"]) or len({int(row["image_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Trace JSONL coverage is invalid: {trace}")
    full_count = len(rows)
    rows = rows[:max_images] if max_images is not None else rows
    result_path = output / f"{trace.name}.jsonl"
    violations = 0
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        for ordinal, old_record in enumerate(rows, start=1):
            image_id = int(old_record["image_id"])
            image = images.get(image_id)
            if image is None:
                raise ValueError(f"Trace image_id={image_id} is absent from its manifest")
            raw_path = trace / str(old_record["raw_cache"])
            with np.load(raw_path, allow_pickle=False) as old_raw:
                replay = _collect_raw_candidates(model, config, image_root / str(image["file_name"]), image_id)
                replay["framework_shape"] = np.asarray(replay["framework_shape"], dtype=np.int32)
                arrays = {key: _array_audit(old_raw[key], replay[key]) for key in RAW_KEYS}
                raw_exact = all(item["match"] for item in arrays.values())
                final_exact = _json_exact(old_record["final_predictions"], replay["final_predictions"])
                mapping_exact = _json_exact(
                    _mapping_projection(old_record["final_pred_to_source_candidate"]),
                    _mapping_projection(replay["final_pred_to_source_candidate"]),
                )
                old_framework_prototype_available = "framework_prototype" in old_raw.files
                mask_replay = _reconstruct_old_final_masks(old_record, old_raw, config.device)
            passed = bool(
                replay["framework_prototype_match"]["match"]
                and raw_exact
                and final_exact
                and mapping_exact
                and mask_replay["all_exact"]
            )
            violations += int(not passed)
            row = {
                "trace": trace.name,
                "image_id": image_id,
                "ordinal": ordinal,
                "passed": passed,
                "framework_prototype_raw_exact_match": replay["framework_prototype_match"],
                "legacy_framework_prototype": {
                    "available": old_framework_prototype_available,
                    "note": None if old_framework_prototype_available else "Legacy NPZ has no framework_prototype; this replay validates the new forward only.",
                },
                "raw_tensor_exact": raw_exact,
                "raw_tensor_fields": arrays,
                "top_ids_exact": arrays["top_indices"]["match"],
                "final_predictions_exact": final_exact,
                "final_mapping_exact": mapping_exact,
                "old_raw_process_mask_native_batch_2": mask_replay,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(rows):
                print(json.dumps({"event": "progress", "trace": trace.name, "processed": ordinal, "total": len(rows), "violations": violations}), flush=True)
    return {"trace": trace.name, "images": len(rows), "full_trace_images": full_count, "full_coverage": len(rows) == full_count, "violations": violations, "result": result_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--diagnostic-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-images", type=int, help="Sanity-only cap per trace; omitted for full audit.")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    traces = tuple(args.trace) if args.trace else DEFAULT_TRACES
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(PIGCV_ROOT) not in sys.path:
        sys.path.insert(0, str(PIGCV_ROOT))
    from models.yolo26seg.diagnostic_inferencer import _sha256, load_diagnostic_config
    from models.yolo26seg.inferencer import load_inference_config, set_seed
    from models.yolo26seg.loader import load_model
    from ultralytics.utils import ops
    import ultralytics
    import torch

    config_path, _ = load_diagnostic_config(args.diagnostic_config)
    config, _ = load_inference_config(config_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "traces": [str(trace) for trace in traces],
        "diagnostic_config": str(args.diagnostic_config),
        "formal_config": config.as_dict(),
        "max_images_per_trace": args.max_images,
        "runtime": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "source_sha256": {str(path): _sha256(path) for path in (Path(__file__), Path(ops.__file__), PIGCV_ROOT / "models/yolo26seg/diagnostic_inferencer.py")},
        "mask_reconstruction": "legacy raw prototype/coefficient/box with ultralytics.ops.process_mask_native in batches of 2",
        "legacy_limit": "The 20260905 legacy traces do not contain framework_prototype, so historical framework-prototype equality cannot be reconstructed.",
    }
    _write_json(args.output_dir / "run_metadata.json", metadata)
    _write_json(args.output_dir / "run_status.json", {"status": "running", "started_at": metadata["created_at"]})
    try:
        set_seed(config.seed)
        model = load_model(config.checkpoint, config.device)
        summaries = [_audit_trace(trace, model, config, args.output_dir, args.max_images) for trace in traces]
        violations = sum(int(summary["violations"]) for summary in summaries)
        summary = {"status": "completed" if not violations else "failed_validation", "traces": summaries, "violations": violations}
        _write_json(args.output_dir / "summary.json", summary)
        _write_json(args.output_dir / "run_status.json", {"status": summary["status"], "finished_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, ensure_ascii=False))
        return int(bool(violations))
    except Exception as error:
        _write_json(args.output_dir / "run_status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
