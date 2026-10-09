"""One frozen forward per image, seven native mask readouts, no GT inference.

Engineering uses exactly the first64 images of the locked5000 source list.
Formal mode exports all5000; independent CPU scoring supplies AP and the fixed
final8 paired associations. This script does not parse GT or score candidates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, sha256
from native_mask_adapter import ARMS, ACTIONS, ADAPTER_VERSION, MULTI_SHA256, RESPONSE_SHA256, NativeMaskAdapter, Timer, prepare_native, timed_native_forward

EVALUATOR_VERSION = "frozen_native_comparison_eval_v1"
SOURCE_FILES = ("evaluate_native_comparison.py", "native_mask_adapter.py", "frozen_io.py", "portable_risk.py", "risk_calibration.py", "mask_calibration.py", "local_features.py")
IMAGES_LIST_SHA256 = "b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db"
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
DECODER = "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def assemble_predictions(arm, ids):
    temporary = arm / "predictions.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for iid in ids:
            records = json.loads((arm / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
            for record in records:
                if not first:
                    handle.write(",\n")
                handle.write(json.dumps({key: record[key] for key in ("image_id", "category_id", "bbox", "score", "segmentation")}, separators=(",", ":"), allow_nan=False))
                first = False
        handle.write("\n]\n")
    temporary.replace(arm / "predictions.json")


def peak_memory(torch):
    cpu_peak = current_rss = None
    try:
        import psutil
        memory = psutil.Process().memory_info()
        cpu_peak, current_rss = getattr(memory, "peak_wset", None), memory.rss
    except ImportError:
        pass
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            "cpu_peak_working_set_bytes": cpu_peak, "cpu_current_rss_bytes": current_rss,
            "scope": "this process; unavailable CPU peak remains null"}


def check_reference(cache, original_ids, actual_vendor):
    receipt = json.loads((cache / "COMPLETE.json").read_text(encoding="utf-8"))
    fingerprint = receipt.get("fingerprint", {})
    if receipt.get("status") != "prediction_complete" or receipt.get("image_count") != 5000 or fingerprint.get("image_ids") != original_ids:
        raise ValueError("Reference must be the original complete matching5000 native baseline")
    if fingerprint.get("weights", {}).get("checkpoint", {}).get("sha256") != OFFICIAL_SHA256:
        raise ValueError("Reference official checkpoint identity differs")
    if fingerprint.get("images_list", {}).get("sha256") != IMAGES_LIST_SHA256 or fingerprint.get("annotations", {}).get("sha256") != ANNOTATIONS_SHA256:
        raise ValueError("Reference source list/annotations differ")
    if fingerprint.get("vendor_source_sha256") != actual_vendor:
        raise ValueError("Reference/runtime actual vendor source bytes differ")
    expected = {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False, "conf": .001, "max_det": 300,
                "rect": False, "scaleup": False, "augment": False, "native_one2one": True, "decode": DECODER, "tf32": False}
    if any(fingerprint.get("config", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Reference native inference settings differ")
    if sha256(cache / "predictions.json") != receipt.get("predictions_sha256"):
        raise ValueError("Reference original prediction bytes changed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "weights", "vendor", "images-list", "response-model", "multi-model", "baseline-cache"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--multi-model-sha256", default=MULTI_SHA256)
    parser.add_argument("--annotations", help="Optional original JSON identity verification only; GT is not parsed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--max-images", type=int, help="Engineering must explicitly use64; formal cannot truncate")
    args = parser.parse_args()
    if (args.engineering and args.max_images != 64) or (not args.engineering and args.max_images is not None):
        parser.error("Engineering requires --max-images64; formal5000 must not set --max-images")
    run = Path(args.root).resolve() / "runs" / args.run_id
    if (run / "SUMMARY.json").exists() or (run / "BASELINE_PARITY_IMAGES.jsonl").exists() or (run / "EVALUATION_FAILURE.json").exists():
        raise FileExistsError("Inference history exists; retry needs a distinct Run ID")
    run.mkdir(parents=True, exist_ok=True)
    source, assets = run / "source", run / "assets"
    source.mkdir(exist_ok=True)
    assets.mkdir(exist_ok=True)
    source_hashes = {}
    for filename in SOURCE_FILES:
        original, archived = Path(__file__).with_name(filename), source / filename
        source_hashes[filename] = sha256(original)
        if archived.exists() and sha256(archived) != source_hashes[filename]:
            raise ValueError("Archived inference source differs: " + filename)
        if not archived.exists():
            shutil.copy2(original, archived)
    for original, basename, expected in ((args.response_model, "response.json", RESPONSE_SHA256), (args.multi_model, "multi_local.json", MULTI_SHA256)):
        if sha256(original) != expected:
            raise ValueError("Frozen asset bytes differ: " + basename)
        target = assets / basename
        if target.exists() and sha256(target) != expected:
            raise ValueError("Archived frozen asset differs: " + basename)
        if not target.exists():
            shutil.copy2(original, target)
    started, began = now(), time.perf_counter()
    extractor = None
    try:
        import torch
        if sha256(args.images_list) != IMAGES_LIST_SHA256:
            raise ValueError("Actual source must be the locked5000 native image list")
        if args.annotations and sha256(args.annotations) != ANNOTATIONS_SHA256:
            raise ValueError("Optional original annotation bytes differ")
        paths = [Path(line.strip()).resolve() for line in Path(args.images_list).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        original_ids = [int(path.stem) for path in paths]
        if len(paths) != 5000 or len(paths) != len(set(paths)) or len(original_ids) != len(set(original_ids)) or not all(path.is_file() for path in paths):
            raise ValueError("Actual original5000 source image identities are incomplete or duplicated")
        ids = original_ids[:64] if args.engineering else original_ids
        paths = paths[:len(ids)]
        extractor = FrozenYOLO(args.weights, args.vendor, args.device)
        import ultralytics
        from pycocotools import mask as mask_utils
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        vendor_paths = {"segment_validator": "ultralytics/models/yolo/segment/val.py", "ops": "ultralytics/utils/ops.py",
                        "nms": "ultralytics/utils/nms.py", "letterbox": "ultralytics/data/augment.py"}
        vendor_sha = {key: sha256(Path(args.vendor) / value) for key, value in vendor_paths.items()}
        cache = Path(args.baseline_cache).resolve()
        original_receipt = check_reference(cache, original_ids, vendor_sha)
        shutil.copy2(cache / "COMPLETE.json", run / "ORIGINAL_BASELINE_RECEIPT.json")
        adapter = NativeMaskAdapter(assets / "response.json", assets / "multi_local.json", args.device, args.multi_model_sha256)
        validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True, "plots": False,
                                               "imgsz": 640, "half": False, "rect": False})
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        configuration = {"evaluator_version": EVALUATOR_VERSION, "adapter_version": ADAPTER_VERSION, "source_sha256": source_hashes,
                         "protocol_sha256": sha256(Path(args.root) / "PROTOCOL.md"),
                         "vendor_source_sha256": vendor_sha, "models": adapter.assets, "image_ids": ids,
                         "source_image_ids": original_ids, "images_list_sha256": IMAGES_LIST_SHA256,
                         "annotations_sha256": ANNOTATIONS_SHA256, "annotation_file_hash_verified": bool(args.annotations),
                         "weights_sha256": OFFICIAL_SHA256, "engineering": args.engineering, "arms": list(ARMS), "actions": list(ACTIONS),
                         "config": {"imgsz": 640, "batch": 1, "fp32": True, "tf32": False, "conf": .001, "max_det": 300,
                                    "coco_maxDets": [1, 10, 100], "native_one2one": True, "decode": DECODER, "device": args.device},
                         "feature_geometry": "native input640 threshold/crop; explicit actual ratio_pad; original-grid binary area/width/height; input640 compactness/local12",
                         "historical4500_byte_reproduction_claimed": False, "gt_used_in_inference": False}
        dump_json(run / "EVALUATION_INPUTS.json", {"configuration": configuration, "configuration_sha256": canonical_sha(configuration),
                                                  "interpreter": sys.executable, "started_at": started, "baseline_cache": str(cache),
                                                  "baseline_cache_receipt_sha256": sha256(cache / "COMPLETE.json")})
        for arm in ARMS:
            (run / arm / "images").mkdir(parents=True, exist_ok=True)
        total_rows = supported = selected = unsupported = 0
        total_extract = total_model = total_decode = total_prepare = total_io = 0.0
        copy_chain, image_timings = [], []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for position, (path, iid) in enumerate(zip(paths, ids), 1):
                extracted, forward_time = timed_native_forward(extractor, path)
                with Timer(args.device) as timer:
                    prepared = prepare_native(extracted, validator, args.device)
                prepare_seconds = timer.seconds
                with Timer(args.device) as timer:
                    outputs, diagnostics = adapter.decode_arms(extracted, prepared, iid, categories, mask_utils)
                decode_seconds = timer.seconds
                cached_path = cache / "images" / f"{iid:012d}.json"
                cached = json.loads(cached_path.read_text(encoding="utf-8"))
                if cached.get("image_id") != iid or cached.get("detections") != outputs["baseline"]:
                    raise RuntimeError("Reference baseline row/identity/RLE byte replay failed for image " + str(iid))
                cache_sha = sha256(cached_path)
                copy_chain.append(cache_sha)
                coefficient_sha = hashlib.sha256(prepared.coefficients.cpu().contiguous().numpy().tobytes()).hexdigest()
                native_identity = {"eligible_output_rows": prepared.eligible_output_rows,
                                   "raw_indices": extracted["raw_indices"][prepared.eligible_output_rows].tolist(), "coefficients_sha256": coefficient_sha,
                                   "boxes": prepared.raw_boxes.tolist(), "scores": prepared.scores.tolist(), "classes": prepared.classes.tolist()}
                parity = {"image_id": iid, "native_and_tau0_exact": diagnostics["tau0_native_exact"],
                          "reference_baseline_all_rows_rle_exact": True, "all_arm_native_identity_exact": diagnostics["all_arm_native_identity_exact"],
                          "empty_and_first64_invariants": diagnostics["empty_and_first64_invariants"], "native_rows": diagnostics["native_rows"],
                          "source_image_cache_sha256": cache_sha, "native_identity_sha256": canonical_sha(native_identity),
                          "coefficients_sha256": coefficient_sha, "input_sha256": extracted["input_sha256"], "image_sha256": extracted["image_sha256"]}
                io_start = time.perf_counter()
                for arm in ARMS:
                    dump_json(run / arm / "images" / f"{iid:012d}.json", {"image_id": iid, "image_path": str(path),
                              "original_shape": extracted["original_shape"], "letterbox": extracted["letterbox"],
                              "source_baseline_identity_sha256": parity["native_identity_sha256"], "detections": outputs[arm]})
                diagnostics.update(image_id=iid, native_identity_sha256=parity["native_identity_sha256"],
                                   coefficients_sha256=coefficient_sha, frozen_forward=forward_time, native_reference_decode_seconds=prepare_seconds,
                                   all_action_decode_seconds=decode_seconds)
                with (run / "NATIVE_DECISIONS.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostics, separators=(",", ":"), allow_nan=False) + "\n")
                with (run / "BASELINE_PARITY_IMAGES.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(parity, separators=(",", ":"), allow_nan=False) + "\n")
                io_seconds = time.perf_counter() - io_start
                record_time = {"image_id": iid, **forward_time, "native_reference_decode_seconds": prepare_seconds,
                               "all_action_decode_seconds": decode_seconds, "json_write_seconds": io_seconds}
                image_timings.append(record_time)
                with (run / "IMAGE_TIMINGS.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record_time, allow_nan=False) + "\n")
                total_rows += diagnostics["native_rows"]
                selected += diagnostics["selected_first64"]
                supported += diagnostics["supported_first64"]
                unsupported += diagnostics["unsupported_first64"]
                total_extract += forward_time["frozen_extract_wall_seconds"]
                total_model += forward_time["native_model_seconds"]
                total_decode += decode_seconds
                total_prepare += prepare_seconds
                total_io += io_seconds
                print(f"NATIVE_READOUT {position}/{len(ids)} rows={total_rows} first64supported={supported} elapsed_s={time.perf_counter()-began:.1f}", flush=True)
                del extracted, prepared, outputs, diagnostics, cached
        frozen = extractor.verify_frozen()
        parity_summary = {"status": "passed", "passed": True, "images": len(ids), "native_rows": total_rows,
                          "all_requested_images_actual_native_forward": True, "all_tau0_native_exact": True,
                          "all_reference_baseline_rows_identity_rle_exact": True, "all_arm_native_identity_exact": True,
                          "all_empty_and_first64_invariants": True, "covers_full5000": not args.engineering,
                          "engineering": args.engineering, "original_baseline_receipt_sha256": sha256(cache / "COMPLETE.json"),
                          "source_image_cache_sha256_chain": hashlib.sha256("".join(copy_chain).encode()).hexdigest(),
                          "original_frozen_state_sha256": original_receipt["frozen_state_sha256"],
                          "original_digest_scope": "legacy partial hash; not asserted equal to current whole-state hash",
                          "current_frozen_integrity": frozen, "gt_used_in_inference": False}
        dump_json(run / "BASELINE_PARITY.json", parity_summary)
        assemblies_start = time.perf_counter()
        receipts = {}
        for arm in ARMS:
            assemble_predictions(run / arm, ids)
            receipt = {"status": "prediction_complete", "image_count": len(ids), "engineering": args.engineering,
                       "fingerprint": configuration, "frozen_state_sha256": frozen["final_state_sha256"], "frozen_integrity": frozen,
                       "predictions_sha256": sha256(run / arm / "predictions.json"), "model": {"official_base_sha256": OFFICIAL_SHA256,
                       "official_weights_unchanged": True, "no_gt_forward": True, "mask_action": arm},
                       "baseline_parity_receipt_sha256": sha256(run / "BASELINE_PARITY.json"),
                       "empty_masks_preserved": True, "readout_view": "all actual native post-conf rows; not full raw candidate set"}
            receipts[arm] = receipt
            dump_json(run / arm / "COMPLETE.json", receipt)
        assembly_seconds = time.perf_counter() - assemblies_start
        source_after = {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
        if source_after != source_hashes:
            raise RuntimeError("Actual inference source changed during execution")
        summary = {"status": "prediction_complete", "passed": True,
                   "engineering": args.engineering, "engineering_passed": True if args.engineering and len(ids) == 64 else None,
                   "image_count": len(ids), "arm_names": list(ARMS), "native_rows": total_rows,
                   "selected_first64": selected, "supported_first64": supported, "unsupported_first64": unsupported,
                   "baseline_parity_passed": True, "all_arm_identity_passed": True, "tau0_native_exact": True,
                   "empty_and_first64_invariants_passed": True, "frozen_integrity_passed": frozen["passed"],
                   "source_sha256": source_hashes, "source_unchanged": True,
                   "model_sha256": {"response": RESPONSE_SHA256, "multi_local": MULTI_SHA256},
                   "configuration_sha256": canonical_sha(configuration), "prediction_receipts": {arm: {"predictions_sha256": receipts[arm]["predictions_sha256"], "complete_sha256": sha256(run / arm / "COMPLETE.json")} for arm in ARMS},
                   "baseline_parity_sha256": sha256(run / "BASELINE_PARITY.json"),
                   "metrics": None, "metrics_status": "unmeasured; separate independent CPU scoring required",
                   "gt_parsed": False, "gt_used_in_inference": False,
                   "timing": {"native_model_seconds": total_model, "frozen_extract_wall_seconds": total_extract,
                              "native_reference_decode_seconds": total_prepare, "all_action_decode_seconds": total_decode,
                              "json_write_seconds": total_io, "prediction_assembly_seconds": assembly_seconds,
                              "total_elapsed_seconds": time.perf_counter() - began, "raw_image_timings_file": "IMAGE_TIMINGS.jsonl",
                              "raw_action_timings_file": "NATIVE_DECISIONS.jsonl", "shared_work_not_repeated_per_arm": True},
                   "peak_memory": peak_memory(torch), "started_at": started, "completed_at": now(),
                   "limitations": ["Frozen old gates evaluated in current explicit-ratio_pad native geometry; not historical4500 byte reproduction",
                                   "Full-vs-first64 comparisons preserve candidate identity; they do not evaluate missing raw candidates",
                                   "No AP, paired GT diagnostic, training, tuning or new-method claim in this inference stage"]}
        dump_json(run / "SUMMARY.json", summary)
        dump_json(run / "EVALUATION_COMPLETE.json", {"status": "prediction_complete", "summary_sha256": sha256(run / "SUMMARY.json"),
                                                    "image_count": len(ids), "engineering_passed": summary["engineering_passed"], "completed_at": now()})
        print(f"PREDICTION_COMPLETE {run/'SUMMARY.json'} engineering_passed={summary['engineering_passed']}", flush=True)
        return 0
    except Exception as exc:
        dump_json(run / "EVALUATION_FAILURE.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "time": now(), "engineering_passed": False})
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    raise SystemExit(main())
