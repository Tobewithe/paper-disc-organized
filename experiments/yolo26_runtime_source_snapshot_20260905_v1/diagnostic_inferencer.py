"""Optional raw-candidate export that reuses the formal YOLO26-seg configuration."""

from __future__ import annotations

import json
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .inferencer import (
    _write_json,
    final_outputs_from_result,
    letterbox_metadata,
    load_manifest,
    load_inference_config,
    predict_image,
    set_seed,
    validate_images,
)
from .loader import load_model, model_metadata
from .types import InferenceConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def _exact_tensor_match(actual: torch.Tensor | np.ndarray, expected: torch.Tensor | np.ndarray) -> dict[str, Any]:
    """Describe an exact same-forward tensor equality check."""
    actual_array = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    expected_array = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    same_shape = actual_array.shape == expected_array.shape
    same_dtype = actual_array.dtype == expected_array.dtype
    elementwise_exact = bool(same_shape and np.array_equal(actual_array, expected_array))
    max_abs_delta = (
        float(np.max(np.abs(actual_array.astype(np.float64) - expected_array.astype(np.float64))))
        if same_shape and actual_array.size
        else 0.0 if same_shape else None
    )
    return {
        "match": bool(same_shape and same_dtype and elementwise_exact),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "elementwise_exact": elementwise_exact,
        "actual_shape": list(actual_array.shape),
        "expected_shape": list(expected_array.shape),
        "actual_dtype": str(actual_array.dtype),
        "expected_dtype": str(expected_array.dtype),
        "element_count": int(actual_array.size),
        "max_abs_delta": max_abs_delta,
    }


def _mask_parity_from_forward_output(
    *,
    framework_mask: torch.Tensor | np.ndarray,
    final_rle: dict[str, Any],
    method: str,
    reconstruction_device: str,
) -> dict[str, Any]:
    """Compare the actual forward mask with the mask exported to COCO RLE."""
    from pycocotools import mask as mask_utils

    shape = tuple(int(value) for value in final_rle["size"])
    if torch.is_tensor(framework_mask):
        actual = framework_mask.detach().cpu().numpy()
    else:
        actual = np.asarray(framework_mask)
    if actual.ndim == 3 and actual.shape[0] == 1:
        actual = actual[0]
    if actual.ndim != 2 or tuple(actual.shape) != shape:
        raise RuntimeError(
            f"Captured framework mask shape {tuple(actual.shape)} does not match RLE shape {shape}"
        )
    actual = np.ascontiguousarray(actual, dtype=np.uint8)
    decoded = np.asarray(mask_utils.decode({"size": list(shape), "counts": final_rle["counts"]}), dtype=np.uint8)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    xor_pixels = int(np.count_nonzero(np.bitwise_xor(actual, decoded)))
    return {
        "method": method,
        "encoding": "COCO_RLE_ascii_counts",
        "shape": list(shape),
        "framework_output_mask_sha256": _mask_sha256(actual),
        "final_rle_mask_sha256": _mask_sha256(decoded),
        "xor_pixels": xor_pixels,
        "parity": xor_pixels == 0,
        "reconstruction_device": reconstruction_device,
    }


def load_diagnostic_config(path: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a diagnostic YAML that may add export settings, never inference settings."""
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or "base_infer_config" not in payload:
        raise ValueError("diagnostic.yaml must contain base_infer_config")
    locked = set(InferenceConfig.__dataclass_fields__)
    conflict = sorted(locked & set(payload))
    if conflict:
        raise ValueError(f"diagnostic.yaml cannot override formal inference settings: {', '.join(conflict)}")
    return (path.parent / str(payload["base_infer_config"])).resolve(), payload


def _box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lt = np.maximum(left[:, None, :2], right[None, :, :2])
    rb = np.minimum(left[:, None, 2:], right[None, :, 2:])
    inter = np.prod(np.maximum(0.0, rb - lt), axis=-1)
    left_area = np.prod(np.maximum(0.0, left[:, 2:] - left[:, :2]), axis=1)
    right_area = np.prod(np.maximum(0.0, right[:, 2:] - right[:, :2]), axis=1)
    return inter / np.maximum(left_area[:, None] + right_area[None, :] - inter, 1e-12)


def map_final_predictions_to_candidates(
    final_predictions: list[dict[str, Any]], raw_boxes: np.ndarray, raw_scores: np.ndarray, top_indices: np.ndarray
) -> list[dict[str, Any]]:
    """Map final predictions to the exact same-forward Top-K source candidates."""
    if not final_predictions:
        return []
    final_boxes = [prediction["box_xyxy"] for prediction in final_predictions]
    top_indices = top_indices.astype(np.int64)
    ious = _box_iou(np.asarray(final_boxes, dtype=np.float32), raw_boxes[top_indices].astype(np.float32))
    score_delta = np.abs(
        np.asarray([prediction["score"] for prediction in final_predictions], dtype=np.float32)[:, None]
        - raw_scores[top_indices][None]
    )
    masked_iou = np.where(score_delta <= 1e-5, ious, -1.0)
    matches = masked_iou.argmax(axis=1)
    result = []
    for index, top_position in enumerate(matches.tolist()):
        candidate_id = int(top_indices[top_position])
        mapping = {
            "pred_id": int(final_predictions[index]["pred_id"]),
            "source_candidate_id": candidate_id,
            "top_k_position": int(top_position + 1),
            "box_iou": float(ious[index, top_position]),
            "score_delta": float(score_delta[index, top_position]),
        }
        if mapping["box_iou"] < 0.99 or mapping["score_delta"] > 1e-5:
            raise RuntimeError(f"Same-forward final/source mapping failed: {mapping}")
        result.append(mapping)
    if len({row["source_candidate_id"] for row in result}) != len(result):
        raise RuntimeError("Same-forward final/source mapping is not one-to-one")
    return result


def _collect_raw_candidates(model: Any, config: InferenceConfig, image_path: Path, image_id: int) -> dict[str, Any]:
    """Collect one image while reusing an already loaded model."""
    import cv2
    from ultralytics.utils import ops

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV cannot read image: {image_path}")
    height, width = image.shape[:2]
    head = model.model.model[-1]
    captured: list[Any] = []
    framework_call: dict[str, Any] = {}
    original_process_mask_native = ops.process_mask_native

    def capture_process_mask_native(
        protos: torch.Tensor,
        masks_in: torch.Tensor,
        bboxes: torch.Tensor,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        framework_prototype = protos.detach().cpu().numpy()
        output = original_process_mask_native(protos, masks_in, bboxes, shape)
        framework_call.update(
            {
                "prototype": framework_prototype,
                "coefficients": masks_in.detach().cpu().numpy(),
                "boxes": bboxes.detach().cpu().numpy(),
                "shape": [int(shape[0]), int(shape[1])],
                "mask": output.detach().cpu(),
                "device": str(output.device),
            }
        )
        return output

    hook = head.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    ops.process_mask_native = capture_process_mask_native
    try:
        result = predict_image(model, image_path, config)
    finally:
        ops.process_mask_native = original_process_mask_native
        hook.remove()
    if not captured or not isinstance(captured[-1], tuple) or not isinstance(captured[-1][1], dict):
        raise RuntimeError("Segment26 head hook did not capture same-forward raw outputs")
    final_predictions, _ = final_outputs_from_result(
        result, image_id=image_id, width=width, height=height, config=config
    )
    if not framework_call:
        raise RuntimeError("Framework did not call ultralytics.ops.process_mask_native")
    raw = captured[-1][1]
    one = raw["one2one"]
    raw_prototype = one["proto"][0].detach().cpu().numpy()
    framework_prototype_match = _exact_tensor_match(framework_call["prototype"], raw_prototype)
    if not framework_prototype_match["match"]:
        raise RuntimeError(f"Framework prototype is not the same-forward raw prototype: {framework_prototype_match}")
    boxes_input = head._get_decode_boxes(one)[0].T
    scores = one["scores"][0].sigmoid().T[:, 0]
    top_scores, _, top_indices = head.get_topk_index(scores[None, :, None], config.max_det)
    top_indices = top_indices[0, :, 0].detach().cpu().numpy().astype(np.int64)
    ranks = torch.argsort(scores, descending=True, stable=True).detach().cpu().numpy()
    global_rank = np.empty(len(ranks), dtype=np.int32)
    global_rank[ranks] = np.arange(1, len(ranks) + 1)
    levels: list[str] = []
    grid_x: list[int] = []
    grid_y: list[int] = []
    strides: list[float] = []
    for level, feature in enumerate(one["feats"]):
        feature_height, feature_width = feature.shape[-2:]
        y, x = np.divmod(np.arange(int(feature_height * feature_width)), int(feature_width))
        levels.extend([f"P{level + 3}"] * len(x))
        grid_x.extend(x.tolist())
        grid_y.extend(y.tolist())
        strides.extend([float(head.stride[level])] * len(x))
    input_shape = [int(one["feats"][0].shape[-2] * head.stride[0]), int(one["feats"][0].shape[-1] * head.stride[0])]
    raw_boxes = ops.scale_boxes(tuple(input_shape), boxes_input.clone(), image.shape[:2]).cpu().numpy()
    raw_scores = scores.detach().cpu().numpy()
    raw_coefficients = one["mask_coefficient"][0].T.detach().cpu().numpy()
    framework_boxes = np.asarray(framework_call["boxes"], dtype=np.float32)
    framework_coefficients = np.asarray(framework_call["coefficients"], dtype=np.float32)
    if len(framework_boxes):
        # Coefficients are copied from the same forward tensor and remain exact even
        # for degenerate boxes whose IoU is necessarily zero at an image boundary.
        framework_source = np.asarray(
            [
                int(np.max(np.abs(raw_coefficients - coefficients[None, :]), axis=1).argmin())
                for coefficients in framework_coefficients
            ],
            dtype=np.int32,
        )
        coefficient_match = np.max(
            np.abs(framework_coefficients - raw_coefficients[framework_source]), axis=1
        )
        if np.any(coefficient_match > 1e-6) or len(set(framework_source.tolist())) != len(framework_source):
            raise RuntimeError("Framework mask inputs could not be mapped one-to-one to raw candidates")
    else:
        framework_source = np.empty(0, dtype=np.int32)
    coefficient_delta = (
        float(np.max(np.abs(np.asarray(framework_call["coefficients"], dtype=np.float32) - raw_coefficients[framework_source])))
        if len(framework_source)
        else 0.0
    )
    box_delta = (
        float(np.max(np.abs(framework_boxes - raw_boxes[framework_source])))
        if len(framework_source)
        else 0.0
    )
    raw_input_match = bool(coefficient_delta == 0.0 and box_delta <= 5e-4)
    if not raw_input_match:
        raise RuntimeError(
            "Framework mask inputs do not match same-forward raw candidates: "
            f"coefficient_max_abs_delta={coefficient_delta}, box_max_abs_delta={box_delta}"
        )
    preprocess_meta = letterbox_metadata(height, width, config)
    return {
        "source_candidate_id": np.arange(len(raw_boxes), dtype=np.int32),
        "boxes_xyxy": raw_boxes,
        "scores": raw_scores,
        "mask_coefficients": raw_coefficients,
        "prototype": raw_prototype,
        "framework_prototype": framework_call["prototype"],
        "top_indices": top_indices,
        "top_scores": top_scores[0, :, 0].detach().cpu().numpy(),
        "global_rank": global_rank,
        "feature_level": np.asarray(levels),
        "grid_x": np.asarray(grid_x, dtype=np.int32),
        "grid_y": np.asarray(grid_y, dtype=np.int32),
        "stride": np.asarray(strides, dtype=np.float32),
        "input_shape": [1, 3, *input_shape],
        "p3_p4_p5_shapes": [[int(feature.shape[-2]), int(feature.shape[-1])] for feature in one["feats"]],
        "preprocess_meta": preprocess_meta,
        "framework_masks": framework_call["mask"],
        "framework_coefficients": np.asarray(framework_call["coefficients"], dtype=np.float32),
        "framework_boxes_xyxy": framework_boxes,
        "framework_source_candidate_id": framework_source,
        "framework_shape": framework_call["shape"],
        "framework_reconstruction_device": framework_call["device"],
        "framework_prototype_match": framework_prototype_match,
        "framework_raw_input_match": {
            "match": raw_input_match,
            "coefficient_max_abs_delta": coefficient_delta,
            "box_max_abs_delta": box_delta,
            "box_tolerance": 5e-4,
        },
        "final_predictions": final_predictions,
        "final_pred_to_source_candidate": map_final_predictions_to_candidates(
            final_predictions, raw_boxes, raw_scores, top_indices
        ),
    }


def collect_raw_candidates(diagnostic_config_path: Path, image_path: Path) -> dict[str, Any]:
    """Compatibility wrapper for a one-image diagnostic."""
    config_path, _ = load_diagnostic_config(diagnostic_config_path)
    config, _ = load_inference_config(config_path)
    return _collect_raw_candidates(load_model(config.checkpoint, config.device), config, image_path, 0)


def run_diagnostic_inference(
    *,
    diagnostic_config_path: Path,
    manifest: Path,
    image_root: Path,
    output_dir: Path,
    max_images: int | None = None,
    start_index: int = 0,
) -> dict[str, Any]:
    """Cache every raw candidate for a selected COCO image manifest."""
    config_path, diagnostic_payload = load_diagnostic_config(diagnostic_config_path)
    config, config_payload = load_inference_config(config_path)
    all_images = load_manifest(manifest)
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected = all_images[start_index : start_index + max_images if max_images else None]
    if not selected:
        raise ValueError("No images selected")
    validate_images(selected, image_root)
    allowed_launcher_files = {"run.log", "run.stderr.log"}
    existing = set() if not output_dir.exists() else {path.name for path in output_dir.iterdir()}
    if existing - allowed_launcher_files:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_candidates"
    raw_dir.mkdir()
    _write_json(output_dir / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        set_seed(config.seed)
        model = load_model(config.checkpoint, config.device)
        metadata = model_metadata(model, config.checkpoint, config.device)
        resolved = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "diagnostic_config_file": str(diagnostic_config_path),
            "diagnostic_config": diagnostic_payload,
            "base_infer_config_file": str(config_path),
            "config": config.as_dict(),
            "config_source": config_payload,
            "manifest": str(manifest),
            "image_root": str(image_root),
            "manifest_images": len(all_images),
            "selected_images": len(selected),
            "start_index": start_index,
            "provenance_hashes": {
                "checkpoint_sha256": _sha256(config.checkpoint),
                "manifest_sha256": _sha256(manifest),
                "selected_images": {
                    str(image["id"]): {
                        "path": str(image_root / str(image["file_name"])),
                        "sha256": _sha256(image_root / str(image["file_name"])),
                    }
                    for image in selected
                },
            },
            "runtime": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "deterministic_seed": config.seed,
                "seed_applied": True,
                "note": "Backend flags are recorded from the loaded runtime; no determinism flags are changed by diagnostic inference.",
            },
            "model_meta": metadata,
            "cache_format": "one uncompressed NPZ per image plus diagnostic_cache.jsonl",
        }
        _write_json(output_dir / "inference_config.json", resolved)
        print(json.dumps({"event": "start", "images": len(selected), **config.as_dict()}), flush=True)
        processed_ids: list[int] = []
        final_prediction_count = 0
        started = time.monotonic()
        cache_path = output_dir / "diagnostic_cache.jsonl"
        with cache_path.open("w", encoding="utf-8", newline="\n") as cache:
            for index, image_record in enumerate(selected, start=1):
                image_id = int(image_record["id"])
                image_path = image_root / str(image_record["file_name"])
                record = _collect_raw_candidates(model, config, image_path, image_id)
                mapping = []
                top_set = set(int(value) for value in record["top_indices"])
                framework_rows = {
                    int(candidate_id): row
                    for row, candidate_id in enumerate(record["framework_source_candidate_id"].tolist())
                }
                final_candidate = np.zeros(len(record["source_candidate_id"]), dtype=np.uint8)
                for prediction, source in zip(record["final_predictions"], record["final_pred_to_source_candidate"]):
                    candidate_id = int(source["source_candidate_id"])
                    final_candidate[candidate_id] = 1
                    if candidate_id not in framework_rows:
                        raise RuntimeError(f"Final candidate is absent from framework mask call: {candidate_id}")
                    framework_row = framework_rows[candidate_id]
                    parity = _mask_parity_from_forward_output(
                        framework_mask=record["framework_masks"][framework_row],
                        final_rle=prediction["mask_rle"],
                        method="same_forward_process_mask_native_output",
                        reconstruction_device=record["framework_reconstruction_device"],
                    )
                    if not parity["parity"]:
                        raise RuntimeError(f"Final/source mask parity failed: {source}, xor={parity['xor_pixels']}")
                    mapping.append({
                        **source,
                        "candidate_stage": {
                            "top_k": candidate_id in top_set,
                            "conf_pass": bool(record["scores"][candidate_id] >= config.conf),
                            "final": True,
                            "nms_suppressed": False,
                        },
                        "mask_parity": parity,
                        "framework_mask_row": int(framework_row),
                    })
                if len(mapping) != len(record["final_predictions"]):
                    raise RuntimeError("Final/source trace count mismatch")
                raw_path = raw_dir / f"image_{image_id}_raw_candidates.npz"
                np.savez(
                    raw_path,
                    source_candidate_id=record["source_candidate_id"],
                    boxes_xyxy=record["boxes_xyxy"],
                    scores=record["scores"],
                    mask_coefficients=record["mask_coefficients"],
                    prototype=record["prototype"],
                    framework_prototype=record["framework_prototype"],
                    top_indices=record["top_indices"],
                    top_scores=record["top_scores"],
                    top_k_member=np.isin(record["source_candidate_id"], record["top_indices"]).astype(np.uint8),
                    conf_pass=(record["scores"] >= config.conf).astype(np.uint8),
                    final_candidate=final_candidate,
                    global_rank=record["global_rank"],
                    feature_level=record["feature_level"],
                    grid_x=record["grid_x"],
                    grid_y=record["grid_y"],
                    stride=record["stride"],
                    framework_coefficients=record["framework_coefficients"],
                    framework_boxes_xyxy=record["framework_boxes_xyxy"],
                    framework_source_candidate_id=record["framework_source_candidate_id"],
                    framework_shape=np.asarray(record["framework_shape"], dtype=np.int32),
                )
                final_prediction_count += len(record["final_predictions"])
                cache.write(json.dumps({
                    "image_id": image_id,
                    "file_name": str(image_record["file_name"]),
                    "raw_cache": str(raw_path.relative_to(output_dir)),
                    "raw_candidate_count": int(len(record["source_candidate_id"])),
                    "top_k_count": int(len(record["top_indices"])),
                    "input_shape": record["input_shape"],
                    "p3_p4_p5_shapes": record["p3_p4_p5_shapes"],
                    "preprocess_meta": record["preprocess_meta"],
                    "final_predictions": record["final_predictions"],
                    "final_pred_to_source_candidate": mapping,
                    "framework_mask_call": {
                        "source_candidate_ids": record["framework_source_candidate_id"].tolist(),
                        "shape": record["framework_shape"],
                        "prototype_raw_exact_match": record["framework_prototype_match"],
                        "raw_input_match": record["framework_raw_input_match"],
                        "reconstruction_device": record["framework_reconstruction_device"],
                    },
                    "candidate_stage_semantics": {
                        "top_k": "head.get_topk_index(scores, max_det)",
                        "conf_pass": f"score >= {config.conf}",
                        "final": "present in final_outputs_from_result after framework post-processing and non-empty mask filtering",
                        "nms_suppressed": "unavailable; framework does not expose a per-candidate suppression reason in this hook",
                    },
                }, ensure_ascii=False) + "\n")
                cache.flush()
                processed_ids.append(image_id)
                if index == 1 or index % 10 == 0 or index == len(selected):
                    print(json.dumps({"event": "progress", "processed": index, "total": len(selected)}), flush=True)

        expected_ids = {int(image["id"]) for image in selected}
        if len(processed_ids) != len(selected) or set(processed_ids) != expected_ids:
            raise RuntimeError("Processed image ids do not cover the selected manifest exactly")
        summary = {
            "status": "completed",
            "images_processed": len(processed_ids),
            "manifest_images": len(all_images),
            "selected_image_ids_cover_manifest": len(selected) == len(all_images),
            "final_prediction_records": final_prediction_count,
            "raw_cache_files": len(processed_ids),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output_dir": str(output_dir),
            "validation": {
                "processed_ids_exact": True,
                "cache_rows": len(processed_ids),
                "checkpoint_missing_keys": len(metadata["checkpoint_key_audit"]["missing_keys"]),
                "checkpoint_unexpected_keys": len(metadata["checkpoint_key_audit"]["unexpected_keys"]),
            },
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "run_status.json", {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat()})
        artifact_hashes = {}
        for artifact in sorted(output_dir.rglob("*")):
            if artifact.is_file() and artifact.name != "artifact_hashes.json":
                artifact_hashes[str(artifact.relative_to(output_dir))] = _sha256(artifact)
        _write_json(output_dir / "artifact_hashes.json", {
            "algorithm": "sha256",
            "scope": "all completed run files except this manifest",
            "files": artifact_hashes,
        })
        print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
        return summary
    except Exception as error:
        _write_json(output_dir / "run_status.json", {
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        })
        raise
