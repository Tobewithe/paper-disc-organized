"""Smoke-only first64 native inference parity of full and minimal diagnostics.

Use an actual TriFlow checkpoint as engineering weights, without presenting it
as a formally completed epoch/head. Each real image is extracted exactly once;
both arms reuse those native tensors and the original deployment/readout code.
No GT annotations, AP, paired matching, or bootstrap are opened or evaluated.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import fields
import hashlib
import json
import math
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

VERIFIER_VERSION = "triflow_diagnostics_eval_equivalence_v1"
EXPECTED_RUNTIME_SHA256 = "c24bc853ff17c7f4a0be3bc847a8b6733283b2aa06bc16388ab1b84fca3380b3"
SOURCE_FILES = ("verify_diagnostics_eval_equivalence.py", "diagnostics_runtime.py", "evaluate_epoch.py",
                "readout_support.py", "frozen_io.py", "triflow_model.py")
CHECK_NAMES = ("all_refined_coefficients_exact", "all_detection_records_exact", "all_rle_exact",
               "box_class_score_order_exact", "deployment_selection_exact", "minimal_diagnostics_contract",
               "same_actual_input_tensors", "no_gt_inference", "head_state_unchanged", "frozen_integrity", "sources_unchanged")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def snapshot_engineering_models(head_path, device, torch):
    """Read actual tensors only; deliberately do not claim formal provenance."""
    from diagnostics_runtime import TriFlowRuntimeModel, guard_original_source
    from frozen_io import OFFICIAL_SHA256, state_digest
    from triflow_model import TriFlowConfig, TriFlowModel
    guard_original_source()
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") not in ("triflow_20k_training_snapshot", "triflow_20k_module_final"):
        raise ValueError("Use an actual20k snapshot or module-final as engineering input weights")
    contract = payload.get("contract", {})
    configuration = payload.get("configuration", contract.get("configuration"))
    expected_names = {field.name for field in fields(TriFlowConfig)} | {"feature_channels", "instance_hidden_channels"}
    if not isinstance(configuration, dict) or set(configuration) != expected_names:
        raise ValueError("Actual input checkpoint configuration keys differ")
    config = TriFlowConfig(**{field.name: configuration[field.name] for field in fields(TriFlowConfig)})
    if config != TriFlowConfig():
        raise ValueError("Engineering comparison requires the unchanged locked method defaults")
    declared_configuration_sha = payload.get("configuration_sha256", contract.get("configuration_sha256"))
    if canonical_sha(configuration) != declared_configuration_sha:
        raise ValueError("Actual checkpoint configuration digest differs")
    declared_weights = payload.get("base_weights_sha256", contract.get("base_weights_sha256"))
    if declared_weights != OFFICIAL_SHA256:
        raise ValueError("Actual checkpoint declared frozen official base differs")
    if contract and canonical_sha(contract) != payload.get("contract_sha256"):
        raise ValueError("Actual checkpoint contract content differs from its own digest")
    tensors = payload.get("head_state_dict")
    model_args = (configuration["feature_channels"], configuration["instance_hidden_channels"], config)
    full = TriFlowModel(*model_args).float()
    minimal = TriFlowRuntimeModel(*model_args, diagnostic_scope="minimal").float()
    expected = full.state_dict()
    if not isinstance(tensors, dict) or set(tensors) != set(expected):
        raise ValueError("Actual checkpoint head tensor keys differ")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Actual checkpoint tensor is not finite expected-shape FP32: " + name)
    full.load_state_dict(tensors, strict=True)
    minimal.load_state_dict(tensors, strict=True)
    digest = state_digest(full)
    if state_digest(minimal) != digest or digest != payload.get("head_state_sha256"):
        raise ValueError("Actual loaded engineering states differ from checkpoint head digest")
    record = {"path": str(Path(head_path).resolve()), "sha256": file_sha(head_path), "bytes": Path(head_path).stat().st_size,
              "kind": payload["kind"], "run_id": payload.get("run_id"), "configuration": configuration,
              "configuration_sha256": declared_configuration_sha, "loaded_state_sha256": digest,
              "declared_head_state_sha256": payload["head_state_sha256"], "engineering_weights_only": True,
              "formal_epoch_or_final_audit_verified": False, "training_coverage_verified": False,
              "input_frozen_integrity": payload.get("frozen_integrity")}
    full, minimal = full.to(device).eval(), minimal.to(device).eval()
    full.requires_grad_(False)
    minimal.requires_grad_(False)
    del payload, tensors, expected
    return full, minimal, record


def validate_minimal_counts(diagnostic, full_diagnostic, minimal_keys):
    chunks, full_chunks = diagnostic.get("chunks", []), full_diagnostic.get("chunks", [])
    if len(chunks) != len(full_chunks):
        return False
    for chunk, full_chunk in zip(chunks, full_chunks):
        count = len(chunk.get("native_output_rows", []))
        values = chunk.get("diagnostics", {})
        if chunk.get("native_output_rows") != full_chunk.get("native_output_rows") or set(values) != set(minimal_keys):
            return False
        if not all(isinstance(vector, list) and len(vector) == count for vector in values.values()):
            return False
        if any(values[key] != full_chunk["diagnostics"].get(key) for key in minimal_keys):
            return False
    return True


def input_tensor_sha(extracted, torch):
    """Bind every native tensor passed to the original refine/readout helpers."""
    digest = hashlib.sha256()
    for name in sorted(key for key, value in extracted.items() if isinstance(value, torch.Tensor)):
        value = extracted[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "head", "weights", "vendor", "images-list"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_images <= 8:
        parser.error("This is an explicit real-image engineering check; use max-images1..8")
    run = Path(args.root).resolve() / "runs" / args.run_id
    receipt_path = run / "DIAGNOSTICS_EVAL_PARITY.json"
    if receipt_path.exists() or (run / "EVAL_PARITY_FAILURE.json").exists():
        raise FileExistsError("Engineering history exists; choose a distinct Run ID")
    run.mkdir(parents=True, exist_ok=True)
    archive = run / "source"
    archive.mkdir(exist_ok=True)
    sources = {}
    for name in SOURCE_FILES:
        current, target = Path(__file__).with_name(name), archive / name
        sources[name] = file_sha(current)
        if target.exists() and file_sha(target) != sources[name]:
            raise ValueError("Archived actual engineering source differs: " + name)
        if not target.exists():
            shutil.copy2(current, target)
    receipt = {"verifier_version": VERIFIER_VERSION, "status": "running", "passed": False,
               "smoke_only": True, "engineering_scope": "first requested real val images; full native first64 deployment budget",
               "scientific_scale_claimed": False, "formal_snapshot_provenance_claimed": False, "ap_measured": False,
               "gt_opened": False, "image_count": 0, "selected_instances": 0, "refined_instances": 0,
               "unsupported_roi_instances": 0, "started_at": now(), "source_sha256": sources,
               "runtime_sha256": sources["diagnostics_runtime.py"], "verifier_sha256": sources["verify_diagnostics_eval_equivalence.py"],
               "checks": {name: None for name in CHECK_NAMES}, "images": [],
               "inputs": {"images_list": str(Path(args.images_list).resolve()), "images_list_sha256": file_sha(args.images_list),
                          "weights": str(Path(args.weights).resolve()), "weights_sha256": file_sha(args.weights),
                          "vendor": str(Path(args.vendor).resolve()), "head": str(Path(args.head).resolve()),
                          "head_sha256": file_sha(args.head), "device": args.device, "requested_images": args.max_images},
               "limitations": ["Only the requested1..8 real engineering images; no AP or complete5000 inference claim",
                               "Snapshot is used as actual engineering weights; no formal epoch or final training audit claim",
                               "Measured refine time includes original Python/adapter transfers; not end-to-end deployment throughput"]}
    extractor = None
    try:
        import torch
        from diagnostics_runtime import MINIMAL_KEYS, ORIGINAL_MODEL_SOURCE_SHA256
        from frozen_io import FrozenYOLO, OFFICIAL_SHA256, state_digest
        from evaluate_epoch import DECODER, detection_records, refine_coefficients
        if sources["diagnostics_runtime.py"] != EXPECTED_RUNTIME_SHA256 or sources["triflow_model.py"] != ORIGINAL_MODEL_SOURCE_SHA256:
            raise ValueError("Engineering runtime/core bytes differ from the locked actual comparison target")
        if receipt["inputs"]["weights_sha256"] != OFFICIAL_SHA256:
            raise ValueError("Engineering frozen input checkpoint differs from official SHA")
        paths = [Path(line.strip()).resolve() for line in Path(args.images_list).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(paths) < args.max_images or len(paths) != len(set(paths)):
            raise ValueError("Actual image list is insufficient or contains repeated paths")
        selected_paths = paths[:args.max_images]
        if any(not path.is_file() or not path.stem.isdigit() for path in selected_paths):
            raise ValueError("Requested actual COCO image paths are missing or lack numeric IDs")
        ids = [int(path.stem) for path in selected_paths]
        if len(ids) != len(set(ids)):
            raise ValueError("Requested actual image identities repeat")
        receipt["inputs"]["listed_images"] = len(paths)
        receipt["inputs"]["selected_image_ids"] = ids
        extractor = FrozenYOLO(args.weights, args.vendor, args.device)
        import cv2
        import ultralytics
        from pycocotools import mask as mask_utils
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        full, minimal, head = snapshot_engineering_models(Path(args.head).resolve(), args.device, torch)
        receipt["head"] = head
        initial_head = {"original_full": state_digest(full), "minimal": state_digest(minimal)}
        receipt["environment"] = {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                  "cuda": torch.version.cuda, "tf32": False, "native_one2one": True, "decoder": DECODER}
        validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True,
                                               "plots": False, "imgsz": 640, "half": False, "rect": False})
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        per_image_gates = []
        with torch.inference_mode():
            for position, (path, iid) in enumerate(zip(selected_paths, ids), 1):
                began = time.perf_counter()
                extracted = extractor.extract(path)
                frozen_forward_seconds = time.perf_counter() - began
                if extracted["F"].shape[0] != head["configuration"]["feature_channels"] or extracted["h"].shape[-1] != head["configuration"]["instance_hidden_channels"]:
                    raise ValueError("Actual frozen native channels differ from engineering head input dimensions")
                before_input = input_tensor_sha(extracted, torch)
                full_coefficients, full_diagnostic = refine_coefficients(extracted, full, args.device)
                after_full_input = input_tensor_sha(extracted, torch)
                minimal_coefficients, minimal_diagnostic = refine_coefficients(extracted, minimal, args.device)
                after_minimal_input = input_tensor_sha(extracted, torch)
                full_rows = detection_records(extracted, iid, full_coefficients, validator, args.device, categories, mask_utils)
                minimal_rows = detection_records(extracted, iid, minimal_coefficients, validator, args.device, categories, mask_utils)
                coefficients_equal = bool(torch.equal(full_coefficients, minimal_coefficients))
                difference = (full_coefficients.double() - minimal_coefficients.double()).abs()
                identity_keys = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class")
                identity_equal = len(full_rows) == len(minimal_rows) and all(all(a[key] == b[key] for key in identity_keys) for a, b in zip(full_rows, minimal_rows))
                rle_equal = len(full_rows) == len(minimal_rows) and all(a["segmentation"] == b["segmentation"] for a, b in zip(full_rows, minimal_rows))
                selection_keys = ("selected_rows", "eligible_output_rows", "selected_count", "executable_rows", "refined_count", "unsupported_roi_count", "unsupported_roi", "unselected_coefficients_exact", "mappings")
                selections_equal = all(full_diagnostic[key] == minimal_diagnostic[key] for key in selection_keys)
                gates = {"all_refined_coefficients_exact": coefficients_equal, "all_detection_records_exact": full_rows == minimal_rows,
                         "all_rle_exact": rle_equal, "box_class_score_order_exact": identity_equal,
                         "deployment_selection_exact": selections_equal,
                         "minimal_diagnostics_contract": validate_minimal_counts(minimal_diagnostic, full_diagnostic, MINIMAL_KEYS),
                         "same_actual_input_tensors": before_input == after_full_input == after_minimal_input,
                         "no_gt_inference": full_diagnostic["gt_used_in_forward"] is False and minimal_diagnostic["gt_used_in_forward"] is False}
                per_image_gates.append(gates)
                image_receipt = {"image_id": iid, "image_path": str(path), "image_sha256": extracted["image_sha256"],
                                 "input_sha256": extracted["input_sha256"], "native_tensors_sha256": before_input,
                                 "native_rows": len(extracted["c0"]), "detections": len(full_rows),
                                 "selected_instances": full_diagnostic["selected_count"], "refined_instances": full_diagnostic["refined_count"],
                                 "unsupported_roi_instances": full_diagnostic["unsupported_roi_count"],
                                 "coefficient_max_abs_difference": float(difference.max()) if difference.numel() else 0.0,
                                 "canonical_full_rows_sha256": canonical_sha(full_rows), "canonical_minimal_rows_sha256": canonical_sha(minimal_rows),
                                 "checks": gates, "original_full_diagnostic": full_diagnostic, "minimal_diagnostic": minimal_diagnostic,
                                 "timing": {"frozen_forward_seconds": frozen_forward_seconds,
                                            "original_full_refine_seconds": full_diagnostic["module_seconds"], "minimal_refine_seconds": minimal_diagnostic["module_seconds"]}}
                write_json(run / f"IMAGE_{iid:012d}.json", image_receipt)
                write_json(run / f"ORIGINAL_FULL_{iid:012d}.json", {"image_id": iid, "detections": full_rows})
                write_json(run / f"MINIMAL_{iid:012d}.json", {"image_id": iid, "detections": minimal_rows})
                receipt["images"].append({key: image_receipt[key] for key in ("image_id", "image_sha256", "input_sha256", "selected_instances", "refined_instances", "unsupported_roi_instances", "coefficient_max_abs_difference", "checks", "timing")})
                receipt["image_count"] += 1
                for name in ("selected_instances", "refined_instances", "unsupported_roi_instances"):
                    receipt[name] += image_receipt[name]
                print(f"EVAL_PARITY image={iid} {position}/{len(ids)} first64={full_diagnostic['selected_count']} passed={all(gates.values())}", flush=True)
                del extracted, full_coefficients, minimal_coefficients, full_rows, minimal_rows, full_diagnostic, minimal_diagnostic
        receipt["checks"] = {key: bool(per_image_gates) and all(gates[key] for gates in per_image_gates) for key in per_image_gates[0]}
        receipt["final_head_state_sha256"] = {"original_full": state_digest(full), "minimal": state_digest(minimal)}
        receipt["checks"]["head_state_unchanged"] = receipt["final_head_state_sha256"] == initial_head
        receipt["frozen_integrity"] = extractor.verify_frozen()
        receipt["checks"]["frozen_integrity"] = receipt["frozen_integrity"].get("passed") is True
        receipt["source_after_sha256"] = {name: file_sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
        receipt["checks"]["sources_unchanged"] = receipt["source_after_sha256"] == sources
        receipt["initial_head_state_sha256"] = initial_head
        receipt["passed"] = (set(receipt["checks"]) == set(CHECK_NAMES) and all(receipt["checks"].values())
                             and receipt["image_count"] == args.max_images and receipt["refined_instances"] > 0)
        receipt["status"] = "passed" if receipt["passed"] else "failed"
        receipt["completed_at"] = now()
        write_json(receipt_path, receipt)
        print(f"DIAGNOSTICS_EVAL_PARITY {receipt_path} passed={receipt['passed']}", flush=True)
        return 0 if receipt["passed"] else 1
    except Exception as exc:
        receipt.update(status="failed", passed=False, completed_at=now(), error=repr(exc), traceback=traceback.format_exc())
        write_json(receipt_path, receipt)
        write_json(run / "EVAL_PARITY_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    raise SystemExit(main())
