"""Four-image, no-GT native decoder replay before a TriFlow full evaluation.

Calls the actual evaluate_triflow.detection_records adapter using unchanged
official coefficients, and compares all records against the completed official
reference. This verifies a fixed sample contract; it does not verify 5000
images, calculate AP, or manufacture a formal epoch-8 checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import evaluate_triflow
import frozen_io
import readout_support
import triflow_model
from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, sha256

VERIFICATION_VERSION = "triflow_four_image_native_decoder_verification_v1"
SOURCE_FILES = ("verify_evaluator.py", "evaluate_triflow.py", "readout_support.py",
                "frozen_io.py", "triflow_model.py")
VENDOR_FILES = {"segment_validator": "ultralytics/models/yolo/segment/val.py",
                "ops": "ultralytics/utils/ops.py", "nms": "ultralytics/utils/nms.py",
                "letterbox": "ultralytics/data/augment.py"}
INFERENCE_CONFIG = {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False,
                    "conf": 0.001, "max_det": 300, "coco_maxDets": [1, 10, 100],
                    "rect": False, "scaleup": False, "augment": False,
                    "native_one2one": True, "decode": evaluate_triflow.DECODER, "tf32": False}
IDENTITY_FIELDS = ("image_id", "category_id", "bbox", "score", "detection_index",
                   "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class", "segmentation")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def tensor_digest(value):
    import torch
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(str((str(value.dtype), tuple(value.shape))).encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def inspect_sources(run):
    executing_directory = Path(__file__).resolve().parent
    for module in (evaluate_triflow, frozen_io, readout_support, triflow_model):
        require(Path(module.__file__).resolve().parent == executing_directory,
                f"Runtime imported a source outside the executing snapshot: {module.__name__}")
    snapshot = run / "source"
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in SOURCE_FILES:
        executing = executing_directory / name
        require(executing.is_file(), f"Executing source missing: {name}")
        copied = snapshot / name
        current_sha = sha256(executing)
        if copied.exists():
            require(sha256(copied) == current_sha, f"Existing Run snapshot differs from executing source: {name}")
        elif copied.resolve() != executing.resolve():
            shutil.copy2(executing, copied)
        sources[name] = {"executed_path": str(executing), "executed_sha256": current_sha,
                         "captured_path": str(copied.resolve()), "captured_sha256": sha256(copied),
                         "executed_in_captured_snapshot": executing.resolve() == copied.resolve()}
    return sources


def selected_images(images_list, fingerprint):
    paths, identities = [], []
    for line in images_list.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = images_list.parent / path
        path = path.resolve()
        require(path.stem.isdigit(), f"Cannot get saved val image identity from filename: {path}")
        identities.append(int(path.stem))
        paths.append(path)
    require(len(paths) >= 4, "Decoder verification requires at least four image entries")
    require(len(identities) == len(set(identities)), "Duplicate image identity in declared image list")
    require(identities == fingerprint.get("image_ids"), "Declared full image-list identities differ from original cache")
    for path in paths[:4]:
        require(path.is_file(), f"Selected actual val image is absent: {path}")
    return paths[:4], identities[:4], len(identities)


def compare_records(expected, actual):
    """Canonical JSON equality plus explicit FP32 bytes, class/order and RLE."""
    import numpy as np
    result = {"expected_detection_count": len(expected), "actual_detection_count": len(actual),
              "count_exact": len(expected) == len(actual), "all_record_keys_exact": True,
              "all_fields_canonical_json_exact": True, "raw_boxes_fp32_bitwise_exact": True,
              "scaled_boxes_fp32_bitwise_exact": True, "raw_confidence_fp32_bitwise_exact": True,
              "classes_categories_indices_and_order_exact": True, "rle_size_and_ascii_counts_exact": True,
              "expected_records_sha256": canonical_sha256(expected),
              "actual_records_sha256": canonical_sha256(actual), "differences": []}
    if not result["count_exact"]:
        result["differences"].append({"kind": "detection_count", "expected": len(expected), "actual": len(actual)})
    for index, (old, fresh) in enumerate(zip(expected, actual)):
        keys_equal = set(old) == set(fresh) == set(IDENTITY_FIELDS)
        result["all_record_keys_exact"] &= keys_equal
        if not keys_equal and len(result["differences"]) < 20:
            result["differences"].append({"row": index, "kind": "record_keys",
                                          "expected": sorted(old), "actual": sorted(fresh)})
        for field in IDENTITY_FIELDS:
            if field not in old or field not in fresh:
                result["all_fields_canonical_json_exact"] = False
                continue
            exact = canonical(old[field]) == canonical(fresh[field])
            result["all_fields_canonical_json_exact"] &= exact
            if not exact and len(result["differences"]) < 20:
                result["differences"].append({"row": index, "field": field,
                                              "expected_value_sha256": canonical_sha256(old[field]),
                                              "actual_value_sha256": canonical_sha256(fresh[field])})
        for field, result_key in (("raw_input_box_xyxy", "raw_boxes_fp32_bitwise_exact"),
                                  ("box_xyxy", "scaled_boxes_fp32_bitwise_exact"),
                                  ("raw_confidence", "raw_confidence_fp32_bitwise_exact")):
            if field not in old or field not in fresh:
                result[result_key] = False
                continue
            result[result_key] &= (np.asarray(old[field], dtype="<f4").tobytes() ==
                                   np.asarray(fresh[field], dtype="<f4").tobytes())
        for field in ("image_id", "category_id", "model_class", "detection_index"):
            result["classes_categories_indices_and_order_exact"] &= (
                field in old and field in fresh and type(old[field]) is int and
                type(fresh[field]) is int and old[field] == fresh[field])
        result["classes_categories_indices_and_order_exact"] &= (old.get("detection_index") == index == fresh.get("detection_index"))
        old_rle, new_rle = old.get("segmentation", {}), fresh.get("segmentation", {})
        rle_exact = (set(old_rle) == set(new_rle) == {"size", "counts"} and
                     old_rle.get("size") == new_rle.get("size") and
                     isinstance(old_rle.get("counts"), str) and isinstance(new_rle.get("counts"), str) and
                     old_rle["counts"].encode("ascii") == new_rle["counts"].encode("ascii"))
        result["rle_size_and_ascii_counts_exact"] &= rle_exact
    keys = ("count_exact", "all_record_keys_exact", "all_fields_canonical_json_exact",
            "raw_boxes_fp32_bitwise_exact", "scaled_boxes_fp32_bitwise_exact", "raw_confidence_fp32_bitwise_exact",
            "classes_categories_indices_and_order_exact", "rle_size_and_ascii_counts_exact")
    result["passed"] = all(result[key] for key in keys) and expected == actual
    return result


def verify_smoke_rejection(path, device):
    import torch
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "Supplied smoke artifact is not an actual checkpoint payload")
    require(payload.get("kind") == "triflow_module_final", "Smoke payload kind does not exercise the formal loader epoch guard")
    require(type(payload.get("epoch")) is int and payload["epoch"] == 1 and payload.get("smoke_only") is True,
            "Strict-loader negative fixture must be the actual epoch-1 smoke payload")
    original_sha = sha256(path)
    try:
        evaluate_triflow.load_triflow_checkpoint(path, device)
    except ValueError as error:
        message = str(error)
        require("Only fixed formal epoch8" in message, f"Smoke rejected for an unrelated reason, not the epoch guard: {message}")
    else:
        raise AssertionError("Final evaluator incorrectly accepted the real epoch-1 smoke artifact")
    require(sha256(path) == original_sha, "Loader negative check mutated the actual smoke checkpoint")
    return {"measured": True, "passed": True, "head_path": str(path), "head_sha256": original_sha,
            "head_bytes": path.stat().st_size, "actual_epoch": payload["epoch"], "actual_smoke_only": True,
            "kind": payload["kind"], "rejection_guard": "formal_epoch8_and_smoke_only_false",
            "rejection_message": message, "formal_positive_checkpoint_fabricated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "weights", "vendor", "baseline-cache", "images-list"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-head")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to replace historical evaluator verification: {out}")
    began = time.monotonic()
    receipt = {"verification_version": VERIFICATION_VERSION, "started_at": now(), "completed_at": None,
               "status": "running", "passed": False, "gt_opened": False,
               "sample_selection": "first four entries of the exact original full-val image list, before observing outputs",
               "sample_count_required": 4, "images": [], "successful_images": 0,
               "frozen_whole_state_unchanged": None,
               "strict_loader_smoke_rejection": {"measured": False, "passed": None, "status": "not_requested"},
               "limitations": ["A four-image adapter/decoder contract check, not an all-5000 replay proof.",
                               "No AP or learned-method gain is computed.",
                               "Official c0 is used unchanged; a formal epoch-8 positive loader payload is not fabricated.",
                               "Legacy frozen digest omitted mutable scopes, so only the new full-state before/after digests are compared."]}
    extractor = None
    try:
        receipt["sources"] = inspect_sources(out.parent)
        cache = Path(args.baseline_cache).resolve()
        complete_path = cache / "COMPLETE.json"
        require(complete_path.is_file(), "Original official cache completion receipt missing")
        complete_sha = sha256(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        require(complete.get("status") == "prediction_complete" and complete.get("image_count") == 5000,
                "Reference is not a completed full-val original official cache")
        fingerprint = complete.get("fingerprint", {})
        require(fingerprint.get("weights", {}).get("checkpoint", {}).get("sha256") == OFFICIAL_SHA256,
                "Original reference checkpoint is not the locked official initialization")
        images_list = Path(args.images_list).resolve()
        list_sha = sha256(images_list)
        require(list_sha == fingerprint.get("images_list", {}).get("sha256"),
                "Image list bytes differ from the original official evaluation")
        for key, value in INFERENCE_CONFIG.items():
            require(fingerprint.get("config", {}).get(key) == value, f"Original inference configuration differs: {key}")
        vendor = Path(args.vendor).resolve()
        expected_vendor = fingerprint.get("vendor_source_sha256", {})
        require(set(expected_vendor) == set(VENDOR_FILES), "Original decoder source fingerprint lacks expected exact vendor scope")
        vendor_sources = {}
        for key, relative in VENDOR_FILES.items():
            path = vendor / relative
            observed = sha256(path)
            require(observed == expected_vendor[key], f"Official decoder source differs from reference: {key}")
            vendor_sources[key] = {"path": str(path), "observed_sha256": observed, "expected_sha256": expected_vendor[key], "exact": True}
        paths, identities, image_list_count = selected_images(images_list, fingerprint)
        receipt["inputs"] = {"weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights), "expected_sha256": OFFICIAL_SHA256},
                             "images_list": {"path": str(images_list), "sha256": list_sha, "image_count": image_list_count},
                             "baseline_cache": str(cache), "baseline_complete_sha256": complete_sha,
                             "baseline_evaluator_sha256_from_receipt": fingerprint.get("evaluator_sha256"),
                             "baseline_predictions_sha256_from_receipt_not_recomputed": complete.get("predictions_sha256"),
                             "inference_config": INFERENCE_CONFIG | {"device": args.device},
                             "vendor_sources": vendor_sources, "expected_source_hashes": expected_vendor,
                             "selected_image_ids": identities}
        extractor = FrozenYOLO(args.weights, args.vendor, device=args.device, image_size=640)
        import cv2
        import torch
        import ultralytics
        from pycocotools import mask as mask_utils
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        cv2.setNumThreads(1)
        require(ultralytics.__version__ == fingerprint.get("ultralytics") == "8.4.100", "Official vendor version differs from cached evaluator")
        require(torch.__version__ == fingerprint.get("torch"), "Torch runtime differs from the original reference")
        receipt["environment"] = {"python": sys.version, "platform": platform.platform(),
                                  "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                  "pycocotools": importlib.metadata.version("pycocotools"),
                                  "opencv": cv2.__version__, "requested_device": args.device}
        validator = SegmentationValidator(args={"conf": 0.001, "iou": 0.7, "max_det": 300,
                                                "save_json": True, "plots": False, "imgsz": 640,
                                                "half": False, "rect": False})
        validator.nc = len(extractor.model.names)
        require(validator.nc == 80, "Expected official COCO-80 classes")
        validator.end2end = True
        validator.process = ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        receipt["category_mapping"] = {"coco80_to91": categories, "canonical_sha256": canonical_sha256(categories)}
        receipt["decoder"] = {"adapter": "evaluate_triflow.detection_records", "version": evaluate_triflow.EVALUATOR_VERSION,
                              "route": evaluate_triflow.DECODER, "save_json": True,
                              "process": "ultralytics.utils.ops.process_mask_native", "scale": "SegmentationValidator.scale_preds.byte()",
                              "mask_encoding": "pycocotools.mask.encode(fortran uint8), ASCII counts"}
        receipt["frozen_before"] = extractor.verify_frozen()
        for position, (path, iid) in enumerate(zip(paths, identities), 1):
            record = {"image_id": iid, "position": position, "image_path": str(path), "passed": False}
            cache_path = cache / "images" / f"{iid:012d}.json"
            image_began = time.monotonic()
            try:
                require(cache_path.is_file(), f"Original per-image cache missing: {cache_path}")
                record["original_cache_path"] = str(cache_path)
                record["original_cache_sha256"] = sha256(cache_path)
                original = json.loads(cache_path.read_text(encoding="utf-8"))
                require(original.get("image_id") == iid, "Original per-image cache identity mismatch")
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                forward_began = time.monotonic()
                extracted = extractor.extract(path)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                record["actual_forward_seconds"] = time.monotonic() - forward_began
                require(extracted["native_replay_exact"] is True, "Native raw top-k reconstruction not exact")
                require(original.get("original_shape") == extracted["original_shape"], "Original image shape differs from saved reference")
                tensors_before = {key: tensor_digest(value) for key, value in extracted.items() if isinstance(value, torch.Tensor)}
                decode_began = time.monotonic()
                with torch.inference_mode():
                    fresh = evaluate_triflow.detection_records(extracted, iid, extracted["c0"], validator,
                                                               args.device, categories, mask_utils)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                record["actual_decode_seconds"] = time.monotonic() - decode_began
                tensors_after = {key: tensor_digest(value) for key, value in extracted.items() if isinstance(value, torch.Tensor)}
                require(tensors_before == tensors_after, "Native decode mutated frozen inputs")
                parity = compare_records(original["detections"], fresh)
                record.update({"original_shape": extracted["original_shape"], "original_shape_exact": True,
                               "native_replay_exact": True, "image_sha256": extracted["image_sha256"],
                               "input_sha256": extracted["input_sha256"], "unchanged_c0_sha256": tensors_before["c0"],
                               "all_frozen_input_tensors_unchanged_by_decoder": True, "parity": parity})
                require(parity["passed"], f"Official native decoder differs at image {iid}: {parity['differences'][:3]}")
                require(sha256(cache_path) == record["original_cache_sha256"], "Verification mutated original image cache")
                record["passed"] = True
                receipt["successful_images"] += 1
            except Exception as error:
                record.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            record["elapsed_seconds"] = time.monotonic() - image_began
            receipt["images"].append(record)
            print(json.dumps({"image_id": iid, "passed": record["passed"], "elapsed_seconds": record["elapsed_seconds"],
                              "error": record.get("error")}), flush=True)
        receipt["frozen_after"] = extractor.verify_frozen()
        receipt["frozen_whole_state_unchanged"] = (
            receipt["frozen_before"]["initial_state_sha256"] == receipt["frozen_after"]["final_state_sha256"] and
            receipt["frozen_after"]["passed"])
        require(receipt["frozen_whole_state_unchanged"], "Whole frozen model state changed across the four-image loop")
        require(sha256(complete_path) == complete_sha, "Verification mutated the reference completion receipt")
        for source in receipt["sources"].values():
            require(sha256(source["executed_path"]) == source["executed_sha256"], "Executing source changed during decoder verification")
            require(sha256(source["captured_path"]) == source["captured_sha256"], "Captured source changed during decoder verification")
        if args.smoke_head:
            receipt["strict_loader_smoke_rejection"] = verify_smoke_rejection(args.smoke_head, args.device)
        receipt["passed"] = (receipt["successful_images"] == 4 and receipt["frozen_whole_state_unchanged"] is True and
                             (not args.smoke_head or receipt["strict_loader_smoke_rejection"]["passed"] is True))
        receipt["status"] = "passed" if receipt["passed"] else "failed"
    except Exception as error:
        receipt.update({"status": "failed", "passed": False, "error_type": type(error).__name__,
                        "error": str(error), "traceback": traceback.format_exc()})
    finally:
        if extractor is not None:
            extractor.close()
    receipt["completed_at"] = now()
    receipt["elapsed_seconds"] = time.monotonic() - began
    receipt["all_5000_images_replayed"] = None
    receipt["ap_measured"] = False
    receipt["formal_epoch8_positive_loader_verified"] = False
    dump_json(out, receipt)
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"],
                      "successful_images": receipt["successful_images"], "receipt": str(out)}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
