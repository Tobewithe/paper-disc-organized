"""Bounded desktop diagnosis/reference via original seven-arm and final8 paths.

Never imports or executes an isolated cost executor. No GT, AP or training.
"""
import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, sha256, state_digest
from native_mask_adapter import ARMS, NativeMaskAdapter, prepare_native, _records

PROTOCOL_SHA = "f67d773459da9786f312d49acb77a8745ce83ded405e3ac0a6994e83f95b79a0"
CORE_SHA = "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"
HEAD_SHA = "5c47c39dd5912999af8d8a11e0052a9bcb291c55413eb63b3c4bdc3c054707c8"
STATE_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
RUNTIME_SHA = "c24bc853ff17c7f4a0be3bc847a8b6733283b2aa06bc16388ab1b84fca3380b3"


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    import numpy as np
    import torch
    if torch.__version__ != "2.9.1+cu128" or not torch.cuda.is_available() or sha256(args.protocol) != PROTOCOL_SHA:
        raise ValueError("Actual desktop environment or revision protocol differs")
    root, old = Path(args.root), Path(args.old_native)
    output = root / "runs" / args.run_id
    if (output / "SUMMARY.json").exists() or (output / "INPUTS.json").exists():
        raise FileExistsError("Reference/diagnostic history exists; use a new Run")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    base = None
    original_configuration = json.loads((old / "EVALUATION_INPUTS.json").read_text(encoding="utf-8"))["configuration"]
    ids = original_configuration["image_ids"][:1 if args.diagnostic_only else 32]
    paths = [Path(args.images_directory) / f"{iid:012d}.jpg" for iid in ids]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Actual original panel JPEGs unavailable")
    expected_inputs = {}
    for line in (old / "BASELINE_PARITY_IMAGES.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["image_id"] in ids:
            expected_inputs[row["image_id"]] = row
    for iid, path in zip(ids, paths):
        if sha256(path) != expected_inputs[iid]["image_sha256"]:
            raise ValueError("Desktop panel JPEG bytes changed")
    sources = {}
    for name in (Path(__file__).name, "native_mask_adapter.py", "frozen_io.py", "portable_risk.py", "risk_calibration.py", "local_features.py", "mask_calibration.py"):
        original = Path(__file__).with_name(name)
        sources[name] = sha256(original)
        shutil.copy2(original, source / name)
    shutil.copy2(args.protocol, source / "COST_PANEL_DESKTOP_LOCAL_REFERENCE_PROTOCOL.md")
    dump_json(output / "INPUTS.json", {"protocol_sha256": PROTOCOL_SHA, "source_sha256": sources,
                                      "image_ids": ids, "image_paths": [str(path) for path in paths],
                                      "old_native_reference": str(old), "diagnostic_only": args.diagnostic_only,
                                      "actual_torch": torch.__version__, "actual_gpu": torch.cuda.get_device_name(),
                                      "gt_used": False, "ap_measured": False, "isolated_cost_executor_used": False})
    try:
        base = FrozenYOLO(args.weights, args.vendor, "cuda")
        from pycocotools import mask as mask_utils
        from ultralytics.data.converter import coco80_to_coco91_class
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True,
                                               "plots": False, "imgsz": 640, "half": False, "rect": False})
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = coco80_to_coco91_class()
        adapter = evaluator = head = None
        if not args.diagnostic_only:
            origin = Path(args.triflow_origin)
            core_file, runtime_file = origin / "source/triflow_model.py", origin / "diagnostics_provenance/diagnostics_runtime.py"
            if sha256(core_file) != CORE_SHA or sha256(runtime_file) != RUNTIME_SHA or sha256(origin / "head_epoch_08.pt") != HEAD_SHA:
                raise ValueError("Actual final8 head/core/runtime bytes differ")
            sys.path.insert(0, str(origin / "source"))
            core = imported("triflow_model", core_file)
            runtime = imported("desktop_reference_full_runtime", runtime_file)
            evaluator = imported("desktop_original_epoch_evaluator", origin / "source/evaluate_epoch_r3.py")
            payload = torch.load(origin / "head_epoch_08.pt", map_location="cpu", weights_only=False)
            configuration = payload["contract"]["configuration"]
            head = runtime.install(core.TriFlowModel, "full")(
                configuration["feature_channels"], configuration["instance_hidden_channels"],
                core.TriFlowConfig(**{field.name: configuration[field.name] for field in fields(core.TriFlowConfig)})).float()
            head.load_state_dict(payload["head_state_dict"], strict=True)
            if state_digest(head) != STATE_SHA:
                raise ValueError("Loaded final8 head state differs")
            head.eval().requires_grad_(False).cuda()
            adapter = NativeMaskAdapter(args.response_model, args.multi_model, "cuda")
            for name, path in (("triflow_model.py", core_file), ("diagnostics_runtime.py", runtime_file),
                               ("evaluate_epoch_r3.py", origin / "source/evaluate_epoch_r3.py")):
                (source / "triflow").mkdir(exist_ok=True)
                shutil.copy2(path, source / "triflow" / name)
            dump_json(output / "HEAD_PROVENANCE.json", {"head_sha256": HEAD_SHA, "loaded_state_sha256": STATE_SHA,
                         "head_source_run": str(origin), "diagnostic_scope": "full", "core_sha256": CORE_SHA,
                         "runtime_sha256": RUNTIME_SHA, "evaluator_sha256": sha256(origin / "source/evaluate_epoch_r3.py"),
                         "kind": "actual_original_final8_state_for_desktop32_reference", "no_gt_forward": True})
        count, parity_rows = 0, []
        with torch.inference_mode():
            for iid, path in zip(ids, paths):
                extracted = base.extract(path)
                if extracted["input_sha256"] != expected_inputs[iid]["input_sha256"]:
                    raise RuntimeError("Desktop input tensor bytes differ from original source")
                prepared = prepare_native(extracted, validator, "cuda")
                baseline = _records(prepared.baseline_original, prepared, iid, 0, categories, mask_utils)
                if args.diagnostic_only:
                    reference_path = old / "baseline/images" / f"{iid:012d}.json"
                    reference = json.loads(reference_path.read_text(encoding="utf-8"))["detections"]
                    if len(reference) != len(baseline):
                        raise RuntimeError("Cross-environment candidate count changed")
                    differences = []
                    numeric_keys = {"bbox", "score", "raw_input_box_xyxy", "raw_confidence", "box_xyxy"}
                    for index, (actual, prior) in enumerate(zip(baseline, reference)):
                        for key in actual:
                            if actual[key] != prior[key]:
                                item = {"detection_index": index, "field": key, "actual": actual[key], "reference": prior[key]}
                                if key in numeric_keys:
                                    item["max_abs_numeric_delta"] = float(np.abs(np.asarray(actual[key]) - np.asarray(prior[key])).max())
                                differences.append(item)
                    original = base.cv2.imread(str(path))
                    resized = base.letterbox(image=original)
                    tensor = torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1)))[None].cuda().float() / 255
                    actual_input = tensor.cpu().contiguous().numpy()
                    if hashlib.sha256(actual_input.tobytes()).hexdigest() != extracted["input_sha256"]:
                        raise RuntimeError("Saved actual tensor does not reproduce source tensor bytes")
                    np.save(output / "INPUT_TENSOR_FP32.npy", actual_input, allow_pickle=False)
                    dump_json(output / "ACTUAL_RECORDS.json", {"image_id": iid, "detections": baseline})
                    shutil.copy2(reference_path, output / "REFERENCE_RECORDS.json")
                    dump_json(output / "CROSS_ENVIRONMENT_DIFFERENCES.json", {"image_id": iid, "differences": differences,
                              "different_records": len({row["detection_index"] for row in differences}),
                              "field_names": sorted({row["field"] for row in differences}),
                              "all_input_bytes_exact": True, "all_candidate_ordinal_class_exact": all(row["field"] in numeric_keys for row in differences),
                              "all_binary_rle_exact": all(actual["segmentation"] == prior["segmentation"] for actual, prior in zip(baseline, reference)),
                              "all_differences_numeric_identity_only": bool(differences) and all(row["field"] in numeric_keys for row in differences),
                              "numeric_differences_are_tolerance_pass": False})
                else:
                    # Original seven-arm producer and original epoch evaluator,
                    # both consuming this same actual native replay/coefficients.
                    outputs, diagnostics = adapter.decode_arms(extracted, prepared, iid, categories, mask_utils)
                    original_baseline = evaluator.detection_records(extracted, iid, extracted["c0"], validator, "cuda", categories, mask_utils)
                    if outputs["baseline"] != original_baseline or outputs["baseline"] != baseline:
                        raise RuntimeError("Seven-arm/epoch original paths disagree on common complete baseline")
                    refined, _ = evaluator.refine_coefficients(extracted, head, "cuda")
                    triflow_rows = evaluator.detection_records(extracted, iid, refined, validator, "cuda", categories, mask_utils)
                    for arm, records in {**outputs, "TriFlow_final8": triflow_rows}.items():
                        dump_json(output / arm / "images" / f"{iid:012d}.json", {"image_id": iid, "detections": records,
                                  "original_shape": extracted["original_shape"], "image_path": str(path), "letterbox": extracted["letterbox"]})
                    parity_rows.append({"image_id": iid, "input_sha256": extracted["input_sha256"], "image_sha256": extracted["image_sha256"],
                                        "native_rows": len(baseline), "coefficients_sha256": hashlib.sha256(extracted["c0"].numpy().tobytes()).hexdigest(),
                                        "native_replay_exact": extracted["native_replay_exact"], "complete_common_baseline_identity_rle_exact": True})
                count += len(baseline)
        frozen = base.verify_frozen()
        if head is not None and state_digest(head) != STATE_SHA:
            raise RuntimeError("Reference final8 state changed")
        if not args.diagnostic_only:
            with (output / "BASELINE_PARITY_IMAGES.jsonl").open("w", encoding="utf-8") as handle:
                for row in parity_rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            dump_json(output / "BASELINE_PARITY.json", {"passed": True, "images": 32, "common_original_paths_exact": True,
                                                        "frozen_integrity": frozen, "cross_device_identity_claimed": False})
        for name, before in sources.items():
            if sha256(Path(__file__).with_name(name)) != before:
                raise RuntimeError("Original reference source changed while executing")
        summary = {"status": "completed", "passed": True, "image_count": len(ids), "native_rows": count,
                   "diagnostic_only": args.diagnostic_only, "source_sha256": sources, "protocol_sha256": PROTOCOL_SHA,
                   "original_paths_used": ["FrozenYOLO.extract", "prepare_native", "NativeMaskAdapter.decode_arms", "original evaluate_epoch_r3.refine_coefficients/detection_records"] if not args.diagnostic_only else ["FrozenYOLO.extract", "prepare_native", "native_mask_adapter._records"],
                   "reference_arm_names": [*ARMS, "TriFlow_final8"] if not args.diagnostic_only else None,
                   "isolated_cost_executor_used": False, "complete_common_baseline_exact": True if not args.diagnostic_only else None,
                   "actual_torch": torch.__version__, "actual_gpu": torch.cuda.get_device_name(), "frozen_integrity": frozen,
                   "gt_used": False, "ap_measured": False, "cross_device_quality_equivalence_claimed": False}
        dump_json(output / "SUMMARY.json", summary)
        dump_json(output / "COMPLETE.json", {"status": "completed", "summary_sha256": sha256(output / "SUMMARY.json")})
        print(json.dumps({"completed": True, "diagnostic_only": args.diagnostic_only, "images": len(ids)}), flush=True)
        return 0
    finally:
        if base is not None:
            base.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "weights", "vendor", "images-directory", "old-native", "triflow-origin", "response-model", "multi-model", "protocol"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))
