"""Verify actual online samples against immutable authenticated pilot caches.

This is an engineering input-equivalence/resume check, not a capability screen.
The historical tensors and metadata are opened read-only and never rewritten.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import gc
import json
import sys
import time
import traceback
from pathlib import Path

from frozen_io import OFFICIAL_SHA256, dump_json, sha256
from stream_data import (OnlineCOCOProvider, PILOT_SELECTION_SOURCE_SHA256,
                         select_training_instances, build_training_targets)

VERIFIER_VERSION = "triflow_online_cache_equivalence_v1"
# Confirmed from the actual local RUN_TRIFLOW_TRAIN_S0_R1/TRAINING_INPUTS.json.
BASELINE_RUN_ID = "RUN_TRIFLOW_FIT_CACHE_S0"
BASELINE_RECEIPT_SHA256 = "bec4e547a668e6fe9433ccc4caa4d432ee275310407f41c6299f48a588cd2fb2"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalized_function(source, name):
    parsed = ast.parse(Path(source).read_text(encoding="utf-8-sig"))
    found = [item for item in parsed.body if isinstance(item, ast.FunctionDef) and item.name == name]
    if len(found) != 1:
        raise ValueError(f"Expected one top-level function {name}: {source}")
    function = found[0]
    if function.body and isinstance(function.body[0], ast.Expr):
        value = function.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            function.body = function.body[1:]
    return ast.dump(function, include_attributes=False)


def assert_exact(expected, actual, label, checks):
    """Compare every nested tensor and metadata value, including dtype/shape."""
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"Expected tensor: {label}")
        if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
            raise AssertionError(f"Tensor dtype/shape differs: {label}")
        if actual.device.type != "cpu" or actual.requires_grad or torch.is_inference(actual):
            raise AssertionError(f"Online payload is not an ordinary detached CPU tensor: {label}")
        first_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        second_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(first_bytes, second_bytes):
            different = int(torch.count_nonzero(first_bytes != second_bytes))
            raise AssertionError(f"Tensor differs bitwise: {label}; differing_bytes={different}")
        checks["tensors"] += 1
        checks["tensor_elements"] += actual.numel()
        checks["tensor_bytes"] += actual.numel() * actual.element_size()
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(f"Array dtype/shape differs: {label}")
        if not np.array_equal(expected, actual):
            raise AssertionError(f"Array differs: {label}")
        checks["arrays"] += 1
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"Dictionary keys differ: {label}")
        for key in sorted(expected):
            assert_exact(expected[key], actual[key], f"{label}.{key}", checks)
        return
    if isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"Sequence type/length differs: {label}")
        for position, (first, second) in enumerate(zip(expected, actual)):
            assert_exact(first, second, f"{label}[{position}]", checks)
        return
    if type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"Metadata differs: {label}; expected={expected!r}; actual={actual!r}")
    checks["metadata_values"] += 1


def assert_payload_exact(expected, actual, checks):
    # The only additional field labels the online transport. Byte-identical
    # images may have another path after full-dataset installation.
    extras = set(actual) - set(expected)
    if extras != {"stream_version"} or set(expected) - set(actual):
        raise AssertionError(f"Online/historical payload fields differ: extra={extras}")
    if sha256(actual["image_path"]) != expected["image_sha256"]:
        raise AssertionError("Current image bytes do not match the historical image SHA256")
    for key in sorted(expected):
        if key == "image_path":
            continue
        assert_exact(expected[key], actual[key], key, checks)


def authenticate_baseline(path, current_sources):
    baseline = Path(path).resolve()
    if baseline.name != BASELINE_RUN_ID:
        raise ValueError(f"Expected confirmed cache Run {BASELINE_RUN_ID}, received {baseline.name}")
    receipt_path = baseline / "CACHE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker = json.loads((baseline / "CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    receipt_sha = sha256(receipt_path)
    if receipt_sha != BASELINE_RECEIPT_SHA256 or marker.get("receipt_sha256") != receipt_sha:
        raise ValueError("Historical cache receipt differs from the authenticated training input")
    if receipt.get("status") != "completed" or marker.get("status") != "completed":
        raise ValueError("Historical cache is not completed")
    if receipt.get("base_weights_sha256") != OFFICIAL_SHA256:
        raise ValueError("Historical cache uses another official checkpoint")
    if receipt.get("frozen_integrity", {}).get("passed") is not True or receipt.get("all_native_replay_exact") is not True:
        raise ValueError("Historical cache frozen/native replay proof is not passed")
    origin = baseline / "source"
    for name in ("prepare_cache.py", "frozen_io.py", "triflow_model.py"):
        if sha256(origin / name) != receipt["sources"][name]:
            raise ValueError(f"Historical cache source bytes differ: {name}")
    if receipt["sources"]["prepare_cache.py"] != PILOT_SELECTION_SOURCE_SHA256:
        raise ValueError("Historical selection source is not the locked pilot version")
    for name in ("frozen_io.py", "triflow_model.py"):
        if receipt["sources"][name] != current_sources[name]:
            raise ValueError(f"Online extraction/ownership source differs from historical bytes: {name}")
    ast_results = {}
    for name in ("select_training_instances", "build_training_targets"):
        same = normalized_function(origin / "prepare_cache.py", name) == normalized_function(Path(__file__).with_name("stream_data.py"), name)
        ast_results[name] = {"executable_ast_exact": same, "docstrings_and_source_locations_ignored": True}
        if not same:
            raise AssertionError(f"Online selection or GT construction changed: {name}")
    return baseline, receipt, ast_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--images-list", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--baseline-cache-run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("Verification needs at least one real image")
    run = Path(args.root).resolve() / "runs" / args.run_id
    if (run / "VERIFY_STREAM.json").exists() or (run / "VERIFY_STREAM_FAILURE.json").exists():
        raise FileExistsError("Verification history is immutable; retries need another Run ID")
    run.mkdir(parents=True, exist_ok=True)
    started, began = now(), time.monotonic()
    provider = None
    resumed = None
    try:
        import torch
        current_sources = {name: sha256(Path(__file__).with_name(name))
                           for name in ("verify_stream.py", "stream_data.py", "frozen_io.py", "triflow_model.py")}
        baseline, receipt, ast_results = authenticate_baseline(args.baseline_cache_run, current_sources)
        cache_items = {Path(str(item["image_path"]).replace("\\", "/")).name: item for item in receipt["cache_files"]}
        if len(cache_items) != len(receipt["cache_files"]):
            raise ValueError("Historical cache contains duplicate image filenames")
        requested_list = Path(args.images_list).resolve()
        requested = [Path(line.strip()).resolve() for line in requested_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("Verification input list is empty or duplicates paths")
        candidates = [path for path in requested if path.name in cache_items]
        if len(candidates) < args.limit:
            raise ValueError(f"Only {len(candidates)} requested images have authenticated historical caches; need {args.limit}")
        selected = candidates[:args.limit]
        effective_list = run / "verification_images.txt"
        if effective_list.exists():
            raise FileExistsError("Verification image list already exists; use a new Run ID")
        effective_list.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
        provider = OnlineCOCOProvider(effective_list, args.annotations, args.weights, args.vendor,
                                      args.device, run_dir=run / "fresh_stream", require_full_split=False)
        if provider.annotation_sha256 != receipt["annotation_sha256"]:
            raise ValueError("Original annotations differ from the historical cache")
        if provider.extractor.initial_digest != receipt["frozen_integrity"]["initial_state_sha256"]:
            raise ValueError("Actual frozen-model initial state differs from pilot cache extraction")
        checks = {"tensors": 0, "tensor_elements": 0, "tensor_bytes": 0, "arrays": 0, "metadata_values": 0}
        images = []
        for index, path in enumerate(selected):
            item = cache_items[path.name]
            historical = (baseline / item["cache_file"]).resolve()
            if not historical.is_relative_to(baseline) or not historical.is_file():
                raise ValueError(f"Historical cache payload is missing or escapes its Run: {historical}")
            if historical.stat().st_size != item["bytes"] or sha256(historical) != item["sha256"]:
                raise ValueError(f"Historical cache bytes changed: {historical}")
            old = torch.load(historical, map_location="cpu", weights_only=False)
            actual = provider.get(index)
            if int(actual["image_id"]) != int(item["image_id"]):
                raise AssertionError("Online COCO identity differs from historical receipt")
            before = dict(checks)
            assert_payload_exact(old, actual, checks)
            images.append({"image_id": int(actual["image_id"]), "image_path": str(path),
                           "historical_cache": str(historical), "cache_sha256": item["sha256"],
                           "image_sha256": actual["image_sha256"], "input_sha256": actual["input_sha256"],
                           "instances": len(actual["training_targets"]), "payload_exact": True,
                           "image_path_changed": str(path) != old["image_path"],
                           "checks": {key: checks[key] - before[key] for key in checks}})
            del old, actual
        fresh_receipt = provider.receipt(verify_frozen=True)
        state = copy.deepcopy(provider.state_dict())
        # JSON changes int dictionary keys to strings, matching compact
        # serialization in addition to the trainer's torch checkpoint format.
        state = json.loads(json.dumps(state, allow_nan=False))
        before_counts = dict(provider.visits)
        before_unique = dict(provider.unique_images)
        with (run / "fresh_stream" / "STREAM_IMAGES.jsonl").open(encoding="utf-8") as handle:
            first_visit_counts = json.loads(next(handle))["counts"]
        before_dimensions = copy.deepcopy(provider.dimensions)
        expected_first_identity = copy.deepcopy(provider.observed_inputs[provider.ids[0]])
        provider.close()
        del provider
        provider = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        resumed = OnlineCOCOProvider(effective_list, args.annotations, args.weights, args.vendor,
                                     args.device, run_dir=run / "resumed_stream", require_full_split=False)
        resumed.load_state_dict(state)
        restored_before = resumed.receipt(verify_frozen=True)
        if resumed.visits != before_counts or resumed.unique_images != before_unique or resumed.dimensions != before_dimensions:
            raise AssertionError("Provider resume did not restore exact visit/unique counters or dimensions")
        if restored_before["observed_inputs_identity_sha256"] != fresh_receipt["observed_inputs_identity_sha256"]:
            raise AssertionError("Provider resume did not restore exact observed image/input identities")
        revisited = resumed.get(0)
        if {key: revisited[key] for key in ("image_sha256", "input_sha256")} != expected_first_identity:
            raise AssertionError("Resumed actual first-image extraction changed frozen input bytes")
        expected_after_visit = {key: before_counts[key] + first_visit_counts[key] for key in before_counts}
        if resumed.visits != expected_after_visit or resumed.unique_images != before_unique:
            raise AssertionError("Resumed revisit was not counted once without duplicating unique-image evidence")
        resumed_receipt = resumed.receipt(verify_frozen=True)
        if resumed_receipt["all_listed_images_observed"] is not True:
            raise AssertionError("Real verification subset was not fully observed")
        if fresh_receipt["frozen_integrity"]["passed"] is not True or resumed_receipt["frozen_integrity"]["passed"] is not True:
            raise AssertionError("Actual frozen-model audit did not pass")
        if list((run / "fresh_stream").glob("*.pt")) or list((run / "resumed_stream").glob("*.pt")):
            raise AssertionError("Online provider unexpectedly materialized dense feature tensors")
        result = {"verifier_version": VERIFIER_VERSION, "status": "completed", "passed": True,
                  "started_at": started, "completed_at": now(), "seconds": time.monotonic() - began,
                  "execution_directory": str(run), "interpreter": sys.executable, "device": args.device,
                  "sources": current_sources, "baseline_cache_run": str(baseline),
                  "baseline_cache_receipt_sha256": BASELINE_RECEIPT_SHA256,
                  "confirmed_baseline_identity_source": "RUN_TRIFLOW_TRAIN_S0_R1/TRAINING_INPUTS.json",
                  "requested_images_list": str(requested_list), "requested_images_list_sha256": sha256(requested_list),
                  "effective_images_list": str(effective_list), "effective_images_list_sha256": sha256(effective_list),
                  "selection": "first requested images also present in authenticated historical cache",
                  "images": images, "checks": checks, "selection_and_target_ast": ast_results,
                  "all_historical_payloads_hash_authenticated": True, "all_payload_fields_bitwise_exact": True,
                  "path_exception": "image_path may change; original image SHA, input SHA, all tensor and other metadata values remain exact",
                  "fresh_stream_receipt": fresh_receipt, "restored_stream_before_visit": restored_before,
                  "resumed_stream_receipt": resumed_receipt,
                  "resume": {"json_state_roundtrip": True, "before_visit_counts_exact": True,
                             "observed_image_and_input_identities_exact": True, "dimensions_exact": True,
                             "actual_revisited_input_exact": True, "revisit_unique_counts_unchanged": True,
                             "revisit_visit_count_incremented_once": True, "all_visit_counter_deltas_exact": True},
                  "dense_feature_cache_created": False,
                  "scientific_scope": "engineering equivalence on declared subset; method capability unmeasured"}
        dump_json(run / "VERIFY_STREAM.json", result)
        print(json.dumps({"passed": True, "images": len(images), "tensors": checks["tensors"],
                          "tensor_elements": checks["tensor_elements"], "resume_passed": True,
                          "result": str(run / "VERIFY_STREAM.json")}), flush=True)
    except Exception as exc:
        dump_json(run / "VERIFY_STREAM_FAILURE.json", {"verifier_version": VERIFIER_VERSION, "passed": False,
                  "started_at": started, "failed_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if provider is not None:
            provider.close()
        if resumed is not None:
            resumed.close()


if __name__ == "__main__":
    main()

