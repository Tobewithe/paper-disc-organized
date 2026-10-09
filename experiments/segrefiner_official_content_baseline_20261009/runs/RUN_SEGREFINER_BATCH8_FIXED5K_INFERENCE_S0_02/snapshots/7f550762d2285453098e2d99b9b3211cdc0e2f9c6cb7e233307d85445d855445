"""GT-free frozen native-RLE/RGB -> official LR first64 refinement, no YOLO."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from segrefiner_runtime import OfficialSegRefiner, WEIGHT_SHA, sha256

ARM = "SegRefiner_LR_first64"
PROTOCOL_SHA = "f4703167b0fbb77fc324f89aa5d7d5b97ad6afe020371dc2ab3d2a2db11c1fde"
SUPPLEMENT_SHA = "4d4918a7a2844bd4bbb637915f9173dd9c163af01894a3a8267a41604ebf876a"


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_predictions(run, ids):
    destination = run / ARM / "predictions.json"
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for iid in ids:
            for record in read(run / ARM / "images" / f"{iid:012d}.json")["detections"]:
                if not first:
                    handle.write(",\n")
                handle.write(json.dumps({key: record[key] for key in ("image_id", "category_id", "bbox", "score", "segmentation")}, separators=(",", ":")))
                first = False
        handle.write("\n]\n")


def main(args):
    root = Path(args.root).resolve()
    run = root / "runs" / args.run_id
    if (run / "INFERENCE_INPUTS.json").exists() or (run / "SUMMARY.json").exists():
        raise FileExistsError("History exists; new retry Run required")
    run.mkdir(parents=True, exist_ok=True)
    if sha256(root / "PROTOCOL.md") != PROTOCOL_SHA or sha256(root / "EXECUTION_SUPPLEMENT.md") != SUPPLEMENT_SHA:
        raise ValueError("Locked SegRefiner protocol/sampling supplement differs")
    verification = read(args.runtime_verification)
    runtime_sha = sha256(Path(__file__).with_name("segrefiner_runtime.py"))
    if (verification.get("passed") is not True or verification.get("official_weight_gpu_forward_exercised") is not True
            or verification.get("runtime_source_sha256") != runtime_sha
            or not all(verification.get("checks", {}).values()) or len(verification.get("checks", {})) != 13):
        raise ValueError("Actual complete same-state GPU faithfulness gate not passed this source")
    if not args.engineering:
        if not args.engineering_receipt:
            raise ValueError("Full5000 requires actual engineering receipt")
        engineer = read(args.engineering_receipt)
        if engineer.get("engineering_passed") is not True or engineer.get("runtime_source_sha256") != runtime_sha:
            raise ValueError("Actual engineering gate did not pass this locked runtime")
    metadata = read(args.image_meta)
    paths = [Path(line.strip()) for line in Path(args.images_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [int(path.stem) for path in paths]
    if (len(paths) != 5000 or metadata.get("gt_read") is not False or metadata.get("image_count") != 5000
            or ids != [value["image_id"] for value in metadata["images"]]
            or sha256(args.images_list) != metadata["bindings"]["local_images_list_sha256"]):
        raise ValueError("Actual verified standalone original5000 RGB metadata differs")
    original_ids = ids
    if args.engineering:
        paths, ids = paths[:32], ids[:32]
    native = Path(args.native_run).resolve()
    baseline = native / "baseline"
    summary = read(native / "SUMMARY.json")
    if summary.get("image_count") != 5000 or summary.get("passed") is not True or summary.get("baseline_parity_passed") is not True:
        raise ValueError("Complete sealed original native baseline required")
    if sha256(baseline / "COMPLETE.json") != metadata["bindings"]["native_baseline_complete_sha256"]:
        raise ValueError("Baseline receipt differs from standalone input binding")
    selected = {}
    wanted = set(ids)
    with (native / "NATIVE_DECISIONS.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["image_id"] in wanted:
                decisions = row["decisions"]
                if len(decisions) != row["native_rows"] or [value["detection_index"] for value in decisions] != list(range(len(decisions))):
                    raise ValueError("Actual sealed native ordinal eligibility is incomplete")
                positions = [value["detection_index"] for value in decisions if value["first64_proto_supported"] is True]
                if len(positions) > 64 or any(position >= 64 for position in positions):
                    raise ValueError("Stored native first64 eligibility differs")
                selected[row["image_id"]] = {"positions": positions, "rows": row["native_rows"],
                                             "native_identity_sha256": row["native_identity_sha256"]}
    if set(selected) != wanted:
        raise ValueError("Stored deployment eligibility is missing requested images")
    source_hashes = {name: sha256(Path(__file__).with_name(name)) for name in ("run_segrefiner_inference.py", "segrefiner_runtime.py")}
    source = run / "source"
    source.mkdir(exist_ok=True)
    for name in source_hashes:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    for name in ("PROTOCOL.md", "EXECUTION_SUPPLEMENT.md", "TRAINING_PROVENANCE.md"):
        shutil.copy2(root / name, source / name)
    inputs = {"configuration": {"arm": ARM, "engineering": args.engineering, "image_ids": ids, "original_source_image_ids": original_ids,
                               "protocol_sha256": PROTOCOL_SHA, "supplement_sha256": SUPPLEMENT_SHA, "source_sha256": source_hashes,
                               "official_weight_sha256": WEIGHT_SHA, "native_run": str(native),
                               "native_summary_sha256": sha256(native / "SUMMARY.json"),
                               "native_decisions_sha256": sha256(native / "NATIVE_DECISIONS.jsonl"),
                               "image_meta_sha256": sha256(args.image_meta), "images_list_sha256": sha256(args.images_list),
                               "runtime_verification_sha256": sha256(args.runtime_verification), "base_seed": 20261009,
                               "seed_formula": "20261009 + image_id", "gt_used": False, "new_yolo_forward": False,
                               "full_class_box_score_order_preserved": True}, "interpreter": sys.executable,
              "endpoint": "preloaded original RGB and frozen normal binary CPU masks -> all original binary masks CPU",
              "deployment_boundary_differs_from_old_yolo_forward_inclusive_cost": True,
              "initialization_and_io_rle_audit_outside_deployment": True}
    write(run / "INFERENCE_INPUTS.json", inputs)
    started = time.perf_counter()
    initialize = time.perf_counter()
    runtime = OfficialSegRefiner(args.upstream, args.checkpoint, "cuda", 20261009)
    initialize_seconds = time.perf_counter() - initialize
    write(run / "MODEL_PROVENANCE.json", runtime.provenance)
    records_count = selected_count = valid_count = tiny_count = calls = 0
    timings = []
    try:
        for offset, (iid, path) in enumerate(zip(ids, paths)):
            meta = metadata["images"][offset]
            outside_begin = time.perf_counter()
            if sha256(path) != meta["image_sha256"]:
                raise ValueError("Actual JPEG bytes differ from verified native source")
            bgr = cv2.imread(str(path))
            if bgr is None or list(bgr.shape[:2]) != meta["original_shape"]:
                raise ValueError("Actual original JPEG dimensions differ")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            original_path = baseline / "images" / f"{iid:012d}.json"
            if sha256(original_path) != meta["baseline_image_json_sha256"]:
                raise ValueError("Original native RLE/identity file bytes changed")
            original = read(original_path)
            native_records = original["detections"]
            if len(native_records) != selected[iid]["rows"] or selected[iid]["native_identity_sha256"] != meta["native_identity_sha256"]:
                raise ValueError("Frozen candidate/eligibility identity changed")
            if native_records:
                coarse = mask_utils.decode([record["segmentation"] for record in native_records]).transpose(2, 0, 1).copy()
            else:
                coarse = np.zeros((0, *rgb.shape[:2]), np.uint8)
            identities = np.asarray([record["box_xyxy"] + [record["raw_confidence"], record["model_class"]] for record in native_records], np.float32).reshape(-1, 6)
            prepare_seconds = time.perf_counter() - outside_begin
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            refined, diagnostics = runtime.refine(rgb, coarse, selected[iid]["positions"], iid, identities)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            after = time.perf_counter()
            rng_arrays = {name: value.cpu().numpy() for name, value in runtime.last_rng_states.items()}
            rng_path = run / "rng" / f"{iid:012d}.npz"
            rng_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(rng_path, **rng_arrays)
            records = copy.deepcopy(native_records)
            for index in selected[iid]["positions"]:
                encoded = mask_utils.encode(np.asfortranarray(refined[index].astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                records[index]["segmentation"] = encoded
            for index, (prior, actual) in enumerate(zip(native_records, records)):
                if any(prior[key] != actual[key] for key in prior if key != "segmentation"):
                    raise RuntimeError("Refinement changed frozen native metadata")
                if index not in selected[iid]["positions"] and actual != prior:
                    raise RuntimeError("Out-of-scope ordinal changed")
                if index in selected[iid]["positions"] and coarse[index].sum() < 512 and actual["segmentation"] != prior["segmentation"]:
                    raise RuntimeError("Official tiny/empty identity fallback failed")
            if len(refined) != len(records) or not np.isin(refined, [0, 1]).all():
                raise RuntimeError("All original binary ordinal outputs were not preserved")
            audit_rle_rng_seconds = time.perf_counter() - after
            payload = {"image_id": iid, "image_path": str(path), "original_shape": meta["original_shape"],
                       "source_baseline_identity_sha256": meta["native_identity_sha256"], "detections": records}
            write_begin = time.perf_counter()
            write(run / ARM / "images" / f"{iid:012d}.json", payload)
            timing = {"image_id": iid, "deployment_seconds": seconds, "prepare_rgb_rle_source_seconds": prepare_seconds,
                      "audit_rle_rng_save_seconds": audit_rle_rng_seconds, "write_seconds": time.perf_counter() - write_begin,
                      "gpu_peak_allocated_bytes": peak_allocated, "gpu_peak_reserved_bytes": peak_reserved,
                      "rng_state_file": str(rng_path.relative_to(run)), "rng_state_sha256": sha256(rng_path),
                      "original_image_json_sha256": meta["baseline_image_json_sha256"], "image_sha256": meta["image_sha256"],
                      "output_mask_binary_sha256": hashlib.sha256(refined.tobytes()).hexdigest(),
                      "native_rows": len(records), "actual": diagnostics, "all_identity_fallback_guards_passed": True}
            with (run / "IMAGE_DIAGNOSTICS.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(timing, separators=(",", ":"), allow_nan=False) + "\n")
            timings.append(timing)
            records_count += len(records)
            selected_count += len(selected[iid]["positions"])
            valid_count += diagnostics["model_valid_area512"]
            tiny_count += diagnostics["tiny_or_empty_fallback"]
            calls += diagnostics["actual_denoiser_forward_calls"]
            print(f"SEGREFINER {offset+1}/{len(ids)} rows={records_count} model_instances={valid_count} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
            del original, native_records, coarse, refined, records, rgb, bgr
        integrity = runtime.verify_unchanged()
        build_begin = time.perf_counter()
        build_predictions(run, ids)
        assembly_seconds = time.perf_counter() - build_begin
        if any(sha256(Path(__file__).with_name(name)) != before for name, before in source_hashes.items()):
            raise RuntimeError("Locked inference source changed while running")
        receipt = {"status": "prediction_complete", "image_count": len(ids), "arm": ARM,
                   "predictions_sha256": sha256(run / ARM / "predictions.json"), "fingerprint": inputs["configuration"],
                   "official_model_provenance_sha256": sha256(run / "MODEL_PROVENANCE.json"), "frozen_integrity": integrity,
                   "empty_ordinal_preserved": True, "all_class_box_score_order_exact": True, "gt_used": False}
        write(run / ARM / "COMPLETE.json", receipt)
        total_deployment = sum(value["deployment_seconds"] for value in timings)
        summary = {"status": "prediction_complete", "passed": True, "engineering": args.engineering,
                   "engineering_passed": True if args.engineering and len(ids) == 32 else None,
                   "image_count": len(ids), "arm": ARM, "native_rows": records_count,
                   "selected_supported_first64": selected_count, "model_valid_area512": valid_count, "tiny_or_empty_fallback": tiny_count,
                   "actual_denoiser_forward_calls": calls, "all_identity_fallback_guards_passed": True,
                   "runtime_source_sha256": runtime_sha, "source_sha256": source_hashes,
                   "input_configuration_sha256": canonical(inputs["configuration"]), "model_integrity": integrity,
                   "predictions_sha256": receipt["predictions_sha256"], "image_diagnostics_sha256": sha256(run / "IMAGE_DIAGNOSTICS.jsonl"),
                   "initialization_seconds": initialize_seconds, "total_deployment_seconds": total_deployment,
                   "mean_deployment_seconds": total_deployment / len(ids), "prediction_assembly_seconds": assembly_seconds,
                   "elapsed_seconds": time.perf_counter() - started, "full5000_deployment_seconds_extrapolation": total_deployment / len(ids) * 5000 if args.engineering else None,
                   "gt_parsed": False, "new_yolo_forward": False, "ap_measured": False,
                   "training_manifest_verified": False, "declared_lvis_train_vs_val_overlap": 0,
                   "single_preregistered_stochastic_inference": True,
                   "limitations": ["official pretrained budget differs from project20k; actual training manifest unknown",
                                   "engineering32 does not establish method ability or generalization",
                                   "endpoint uses preloaded RGB+frozen normal masks, excludes YOLO forward; oldcost end-to-end times cannot be mixed directly",
                                   "deployment wrapper includes required RNG isolation and small diagnostic state capture; full audit/RLE/saving separately measured"]}
        write(run / "SUMMARY.json", summary)
        write(run / "INFERENCE_COMPLETE.json", {"status": "completed", "summary_sha256": sha256(run / "SUMMARY.json")})
        return 0
    except Exception as exc:
        write(run / "INFERENCE_FAILURE.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "upstream", "checkpoint", "native-run", "images-list", "image-meta", "runtime-verification"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--engineering-receipt")
    raise SystemExit(main(parser.parse_args()))
