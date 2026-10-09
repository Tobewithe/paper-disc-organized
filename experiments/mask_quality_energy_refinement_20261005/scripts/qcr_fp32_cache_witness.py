"""Replay the fixed first ten DEV images as an FP32 cache-precision witness.

This is a local sample of historical-cache versus original-forward precision,
not a regeneration or validation of the full FIT/DEV population. Numeric and
decoded-mask differences are observations, not execution failures. Permanent
candidate/annotation identity changes are fatal and are preserved in the report.
No added head is instantiated, trained, or tuned by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from online_runtime import dump, load_asset, load_json, resolve_runtime_config, sha256
from qcr_streaming_data import StreamingFinal


IDENTITY_FIELDS = ("image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
NUMERIC_FIELDS = ("proto", "c0", "boxes", "operator.h0", "target_boxes")
METRIC_FIELDS = (
    "iou_A", "iou_B", "iou_D1", "iou_D", "delta_norm_B", "delta_norm_D1", "delta_norm_D",
    "q_c0", "q_c1", "q_c2", "heldout_q_values",
)
ATOL = RTOL = 3e-5


def identity(row):
    return tuple(str(row[name]) if name == "branch" else int(row[name]) for name in IDENTITY_FIELDS)


def nested_value(x, name):
    for part in name.split("."):
        x = x[part]
    return x


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def numeric_comparison(cached, replayed, source_dtype=None):
    a, b = cached.detach().cpu(), replayed.detach().cpu()
    result = {
        "cached_shape": list(a.shape), "replayed_shape": list(b.shape),
        "cached_loaded_dtype": str(a.dtype), "cached_source_dtype": source_dtype or str(a.dtype),
        "replayed_dtype": str(b.dtype), "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype, "torch_equal": bool(torch.equal(a, b)),
        "atol": ATOL, "rtol": RTOL,
    }
    if a.shape != b.shape:
        result.update(max_abs=None, mean_abs=None, close=False, compared_values=0)
        return result
    a, b = a.double(), b.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    result["nonfinite_values"] = int((~finite).sum())
    error = (a - b).abs()
    result.update(
        max_abs=float(error.max()) if error.numel() and bool(finite.all()) else (0. if not error.numel() else None),
        mean_abs=float(error.mean()) if error.numel() and bool(finite.all()) else (0. if not error.numel() else None),
        close=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)), compared_values=a.numel(),
    )
    return result


@torch.no_grad()
def decode_baseline(x, lo, hi, device):
    from ultralytics.utils import ops

    padded = ops.process_mask(
        x["proto"].to(device=device, dtype=torch.float32),
        x["c0"][lo:hi].to(device=device, dtype=torch.float32),
        x["boxes"][lo:hi].to(device=device, dtype=torch.float32), (640, 640), upsample=True,
    )
    original = ops.scale_masks(
        padded[None].float(), tuple(x["original_shape"]), ratio_pad=x["ratio_pad"]
    )[0] > .5
    return padded.bool().cpu(), original.cpu()


def baseline_mask_comparison(cached, replayed, coco, device):
    if (len(cached["rows"]) != len(replayed["rows"]) or
            tuple(cached["original_shape"]) != tuple(replayed["original_shape"]) or
            tuple(cached["proto"].shape) != (32, 160, 160) or
            tuple(replayed["proto"].shape) != (32, 160, 160) or
            tuple(cached["c0"].shape) != tuple(replayed["c0"].shape) or
            tuple(cached["boxes"].shape) != tuple(replayed["boxes"].shape)):
        return {"available": False, "reason": "Shape/geometry differs; pixel correspondence is unavailable"}
    records = []
    for lo in range(0, len(replayed["rows"]), 4):
        hi = min(lo + 4, len(replayed["rows"]))
        padded_a, original_a = decode_baseline(cached, lo, hi, device)
        padded_b, original_b = decode_baseline(replayed, lo, hi, device)
        for j, k in enumerate(range(lo, hi)):
            source = replayed["rows"][k]
            gt = torch.as_tensor(coco.annToMask(coco.anns[int(source["annotation_id"])]).astype(bool))
            if tuple(gt.shape) != tuple(original_a[j].shape):
                raise AssertionError("Original COCO mask geometry differs from the decoded image")
            iou_a = int((original_a[j] & gt).sum()) / max(int((original_a[j] | gt).sum()), 1)
            iou_b = int((original_b[j] & gt).sum()) / max(int((original_b[j] | gt).sum()), 1)
            record = {name: source[name] for name in IDENTITY_FIELDS}
            record.update(
                letterbox_binary_pixel_differences=int((padded_a[j] != padded_b[j]).sum()),
                original_binary_pixel_differences=int((original_a[j] != original_b[j]).sum()),
                iou_cached=iou_a, iou_replayed=iou_b, delta_iou=iou_b - iou_a,
                mask75_cached=int(iou_a >= .75), mask75_replayed=int(iou_b >= .75),
            )
            records.append(record)
    return {
        "available": True, "candidates": len(records), "per_candidate": records,
        "letterbox_binary_pixel_differences": sum(r["letterbox_binary_pixel_differences"] for r in records),
        "original_binary_pixel_differences": sum(r["original_binary_pixel_differences"] for r in records),
        "max_abs_iou_difference": max((abs(r["delta_iou"]) for r in records), default=0.),
        "mean_abs_iou_difference": float(np.mean([abs(r["delta_iou"]) for r in records])) if records else 0.,
        "mask75_changed": sum(r["mask75_cached"] != r["mask75_replayed"] for r in records),
        "decoder": "official process_mask(upscale=True) -> inverse-letterbox binary scale_masks -> >0.5",
    }


def image_comparison(iid, cached, replayed, coco, device):
    cache_dtype = cached.get("_asset_integrity", {}).get("proto_storage_dtype", str(cached["proto"].dtype))
    cached_keys = [identity(row) for row in cached["rows"]]
    replayed_keys = [identity(row) for row in replayed["rows"]]
    result = {
        "image_id": iid, "identity_fields": list(IDENTITY_FIELDS),
        "candidate_identity_and_order_exact": cached_keys == replayed_keys,
        "cached_candidate_identities": [list(key) for key in cached_keys],
        "replayed_candidate_identities": [list(key) for key in replayed_keys],
        "raw_ids_exact": bool(torch.equal(cached["raw_ids"], replayed["raw_ids"])),
        "owners_exact": bool(torch.equal(cached["owners"], replayed["owners"])),
        "all_annotation_ids_exact": cached["all_annotation_ids"] == replayed["all_annotation_ids"],
        "cached_annotation_ids": cached["all_annotation_ids"],
        "replayed_annotation_ids": replayed["all_annotation_ids"],
        "masks_exact": bool(torch.equal(cached["masks"], replayed["masks"])),
        "cached_masks_shape": list(cached["masks"].shape),
        "replayed_masks_shape": list(replayed["masks"].shape),
        "original_shape_exact": plain(cached["original_shape"]) == plain(replayed["original_shape"]),
        "ratio_pad_exact": plain(cached["ratio_pad"]) == plain(replayed["ratio_pad"]),
        "cached_original_shape": plain(cached["original_shape"]),
        "replayed_original_shape": plain(replayed["original_shape"]),
        "cached_ratio_pad": plain(cached["ratio_pad"]), "replayed_ratio_pad": plain(replayed["ratio_pad"]),
        "cached_integrity": cached.get("_asset_integrity", {}),
        "input_uint8": {"available": "input_uint8" in cached},
        "numeric": {},
    }
    if "input_uint8" in cached:
        result["input_uint8"].update(
            exact=bool(torch.equal(cached["input_uint8"], replayed["input_uint8"])),
            cached_shape=list(cached["input_uint8"].shape), replayed_shape=list(replayed["input_uint8"].shape),
        )
    else:
        result["input_uint8"]["reason"] = "Historical compact cache does not retain uint8 input"
    for name in NUMERIC_FIELDS:
        result["numeric"][name] = numeric_comparison(
            nested_value(cached, name), nested_value(replayed, name), cache_dtype if name == "proto" else None
        )
    result["identity_passed"] = all(result[name] for name in (
        "candidate_identity_and_order_exact", "raw_ids_exact", "owners_exact", "all_annotation_ids_exact"
    ))
    if result["identity_passed"]:
        result["baseline_masks"] = baseline_mask_comparison(cached, replayed, coco, device)
    else:
        result["baseline_masks"] = {"available": False, "reason": "Permanent identity/order differs"}
    return result


def read_candidate_rows(path, image_ids):
    path = Path(path)
    if not path.is_file():
        return {}, {"available": False, "path": str(path), "reason": "Existing metric rows not yet available"}
    before = path.stat()
    wanted = set(image_ids)
    rows, truncated_final_line = {}, False
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    truncated_final_line = True
                    break
                raise
            if int(row["image_id"]) not in wanted:
                continue
            key = identity(row)
            if key in rows:
                raise AssertionError(f"Duplicate permanent candidate identity in existing rows: {path}, {key}")
            rows[key] = row
    after = path.stat()
    return rows, {
        "available": True, "path": str(path), "selected_candidates": len(rows),
        "size_bytes": after.st_size, "sha256": sha256(path),
        "stable_while_reading": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "truncated_final_line_skipped": truncated_final_line,
        "parent_run_complete_present": (path.parent / "COMPLETE.json").is_file(),
    }


def checkpoint_row_comparison(cached_path, replayed_path, image_ids):
    a, ar = read_candidate_rows(cached_path, image_ids)
    b, br = read_candidate_rows(replayed_path, image_ids)
    joined = sorted(set(a) & set(b))
    result = {
        "cached_rows": ar, "replayed_rows": br, "joined_candidates": len(joined),
        "identity_fields": list(IDENTITY_FIELDS),
        "cached_only_identities": [list(key) for key in sorted(set(a) - set(b))],
        "replayed_only_identities": [list(key) for key in sorted(set(b) - set(a))],
        "complete_identity_match_for_available_rows": bool(ar["available"] and br["available"] and set(a) == set(b)),
        "numeric": {}, "mask75": {},
        "heads_instantiated": False,
        "scope": "Read-only join of existing checkpoint metrics for the same first-ten DEV images",
    }
    for name in METRIC_FIELDS:
        errors, exact, close, present, unavailable = [], 0, 0, 0, 0
        for key in joined:
            av, bv = a[key].get(name), b[key].get(name)
            if av is None or bv is None:
                unavailable += 1
                continue
            av, bv = np.asarray(av, dtype=np.float64), np.asarray(bv, dtype=np.float64)
            if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
                unavailable += 1
                continue
            error = np.abs(av - bv).reshape(-1)
            errors.extend(error.tolist())
            present += 1
            exact += int(np.array_equal(av, bv))
            close += int(np.allclose(av, bv, atol=ATOL, rtol=RTOL))
        result["numeric"][name] = {
            "compared_candidates": present, "unavailable_candidates": unavailable,
            "compared_values": len(errors), "exact_candidates": exact, "close_candidates": close,
            "max_abs": max(errors) if errors else None,
            "mean_abs": float(np.mean(errors)) if errors else None, "atol": ATOL, "rtol": RTOL,
        }
    for arm in ("A", "B", "D1", "D"):
        name = f"mask75_{arm}"
        keys = [key for key in joined if a[key].get(name) is not None and b[key].get(name) is not None]
        changed = [key for key in keys if int(a[key][name]) != int(b[key][name])]
        result["mask75"][arm] = {
            "compared_candidates": len(keys), "changed": len(changed),
            "cached_positive": sum(int(a[key][name]) for key in keys),
            "replayed_positive": sum(int(b[key][name]) for key in keys),
            "changed_identities": [list(key) for key in changed],
        }
    return result


def main(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Original model replay is permitted only on the authorized Linux CUDA server")
    cfg = resolve_runtime_config(load_json(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "NUMERIC_WITNESS.json").exists() or (out / "COMPLETE.json").exists():
        raise RuntimeError("Preserve the existing witness Run; use another Run ID")
    frozen_dev = [int(iid) for iid in load_json(cfg["split"])["dev_images"]]
    if len(frozen_dev) < 10 or len(set(frozen_dev)) != len(frozen_dev):
        raise AssertionError("Require the frozen unique DEV list with at least ten images")
    image_ids = frozen_dev[:10]
    started = time.monotonic()
    report = {
        "schema": "qcr-fp32-cache-witness-v1", "image_ids": image_ids,
        "selection": "First ten IDs in the frozen IMAGE_SPLIT.dev_images order; no outcome selection",
        "split_sha256": sha256(cfg["split"]), "code_sha256": sha256(__file__),
        "planned_images": 10, "images": [], "atol": ATOL, "rtol": RTOL,
        "scope": "Partial ten-image DEV precision witness; cannot generalize to full FIT/DEV",
        "parameter_updates": False, "added_heads_instantiated": False,
        "frozen_original_buffers_unchanged": None,
    }
    dump(out / "NUMERIC_WITNESS.json", report)
    with StreamingFinal(cfg, image_ids, out, split="dev") as stream:
        for iid, replayed in stream.iter_images():
            cached = load_asset(cfg, iid, verify=True)
            item = image_comparison(iid, cached, replayed, stream.coco, stream.replay.device)
            report["images"].append(item)
            report["elapsed_s"] = time.monotonic() - started
            dump(out / "NUMERIC_WITNESS.json", report)
            if not item["identity_passed"]:
                raise AssertionError(f"Permanent annotation/candidate identity/order differs for DEV image {iid}")
            print(json.dumps({"image_id": iid, "images": len(report["images"]),
                              "baseline_original_pixel_differences": item["baseline_masks"].get("original_binary_pixel_differences"),
                              "prototype_max_abs": item["numeric"]["proto"]["max_abs"]}), flush=True)
            del cached, replayed, item
    report["frozen_original_buffers_unchanged"] = True
    root = Path(cfg["server_root"]) / "runs"
    cached_path = args.cached_rows or str(root / "RUN_stage1_dev_original_metrics_seed0" / "PER_CANDIDATE.jsonl")
    replayed_path = args.replayed_rows or str(root / "RUN_stage1_dev_fp32_witness_seed0" / "PER_CANDIDATE.jsonl")
    report["existing_checkpoint_outputs"] = checkpoint_row_comparison(cached_path, replayed_path, image_ids)
    report["all_numeric_close"] = all(
        value["close"] for image in report["images"] for value in image["numeric"].values()
    )
    report["all_baseline_binary_masks_exact"] = all(
        image["baseline_masks"].get("available") and
        image["baseline_masks"]["original_binary_pixel_differences"] == 0 for image in report["images"]
    )
    report["elapsed_s"] = time.monotonic() - started
    dump(out / "NUMERIC_WITNESS.json", report)
    dump(out / "COMPLETE.json", {
        "completed": True, "passed": True, "validation_scope": "Execution, fixed identity/order, and frozen-model integrity",
        "images": len(report["images"]), "partial_sample": True,
        "frozen_original_buffers_unchanged": True,
        "all_numeric_close": report["all_numeric_close"],
        "all_baseline_binary_masks_exact": report["all_baseline_binary_masks_exact"],
        "numeric_differences_are_scientific_observations": True,
        "full_fit_or_dev_equivalence_claimed": False, "elapsed_s": report["elapsed_s"],
        "numeric_witness_sha256": sha256(out / "NUMERIC_WITNESS.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cached-rows")
    parser.add_argument("--replayed-rows")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except BaseException as exc:
        dump(Path(arguments.out) / "FAILURE.json", {
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise
