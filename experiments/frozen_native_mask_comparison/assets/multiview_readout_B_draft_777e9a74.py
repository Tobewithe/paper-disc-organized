"""Independent CPU multiview readout of sealed seven-arm native predictions.

Requires a completed ordinary scorer, its ORIGINAL final8 fixed-pair source,
and unchanged seven-arm inference files. Does not rematch GT, run a model,
train, choose thresholds, or infer raw/TAL/AUC quantities. Every native output
is retained, including empty masks and candidates absent from the fixed pairs.
Boundary AP is optional and requires a separately verified upstream manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np

from comparison_io import (ARMS, ArmReader, InputLock, check, encoded_rle,
                           json_array, jsonl, read_json, sha, write_json)
from score_and_verify_comparison import (ANNOTATIONS_SHA, METRIC_NAMES,
                                        load_fixed_source)

VERSION = "frozen_native_multiview_v1"
THRESHOLD = .75
DILATION_RATIO = .02
PIXEL_FIELDS = ("tp", "fp", "fn", "gt_pixel_area", "prediction_pixel_area",
                "mask_iou", "target_coverage", "prediction_purity",
                "fp_per_gt_area", "fn_per_gt_area")
QUALITY_FIELDS = PIXEL_FIELDS[5:]
FIELDS = PIXEL_FIELDS + tuple("delta_" + key for key in QUALITY_FIELDS)
COHORTS = ("all_fixed_pairs", "baseline_success", "baseline_failure")
BOUNDARY_REPOSITORY = "https://github.com/bowenc0221/boundary-iou-api"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_pixel_metrics():
    for parent in Path(__file__).resolve().parents:
        path = parent / "shared/evaluation/gt_prediction_mapping.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("multiview_pixel_metrics", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.pixel_metrics, path
    raise FileNotFoundError("Cannot locate existing shared/evaluation/gt_prediction_mapping.py")


class Means:
    """Field-wise arithmetic means with explicit valid/missing denominators."""
    def __init__(self, fields=FIELDS):
        self.rows = 0
        self.values = {key: {"sum": 0., "valid_count": 0, "missing_count": 0}
                       for key in fields}

    def add(self, row):
        self.rows += 1
        for key, acc in self.values.items():
            value = row.get(key)
            if value is None:
                acc["missing_count"] += 1
            else:
                check(type(value) in (int, float) and math.isfinite(value),
                      "Nonfinite or nonnumeric multiview value: " + key)
                acc["sum"] += value
                acc["valid_count"] += 1

    def finish(self):
        return {"row_count": self.rows, "fields": {
            key: acc | {"mean": acc["sum"] / acc["valid_count"]
                        if acc["valid_count"] else None}
            for key, acc in self.values.items()}}


def new_cohorts():
    return {name: Means() for name in COHORTS}


def record_metrics(method, baseline):
    return dict(method) | {
        "delta_" + key: method[key] - baseline[key]
        if method[key] is not None and baseline[key] is not None else None
        for key in QUALITY_FIELDS}


def outcome(baseline_iou, method_iou):
    if baseline_iou is None or method_iou is None:
        return {"baseline_success": None, "method_success": None,
                "repair": None, "damage": None, "transition": "unknown"}
    b, m = baseline_iou >= THRESHOLD, method_iou >= THRESHOLD
    return {"baseline_success": b, "method_success": m,
            "repair": not b and m, "damage": b and not m,
            "transition": "repair" if not b and m else "damage" if b and not m
            else "stable_success" if b else "stable_failure"}


def fixed_pairs_for_image(iid, sources, baseline_rows, coco):
    """Validate and index the existing detection association; never rematch."""
    indexed = {}
    for source in sources:
        index = source["detection_index"]
        check(type(index) is int and 0 <= index < len(baseline_rows),
              "Fixed detection index missing from native output")
        check(index not in indexed and source["image_id"] == iid,
              "Fixed detection association duplicated or in wrong image")
        baseline = baseline_rows[index]
        check(baseline["image_id"] == iid and baseline["category_id"] == source["category_id"],
              "Fixed class/image identity differs")
        if "raw_confidence" in baseline:
            check(baseline["raw_confidence"] == source["baseline_score"],
                  "Fixed source raw confidence differs")
        ann = coco.anns.get(source["annotation_id"])
        check(ann is not None and ann["image_id"] == iid
              and ann["category_id"] == source["category_id"]
              and not ann.get("iscrowd", 0) and not ann.get("ignore", 0),
              "Fixed source annotation identity is absent/ignored/inconsistent")
        check(source["baseline_success"] is (source["baseline_mask_iou"] >= THRESHOLD),
              "Fixed source success threshold changed")
        indexed[index] = source
    return indexed


def segmentation_records(records):
    """Fresh segmentation-only records; COCO loadRes mutates its input list.

    Drop bbox AND supplied area even if a previous loadRes contaminated them.
    This prevents the bbox branch from overriding segmentation area semantics.
    """
    return [{key: copy.deepcopy(row[key])
             for key in ("image_id", "category_id", "score", "segmentation")}
            for row in records]


def memory_observation():
    """OS process memory only, not GPU memory or a model deployment benchmark."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            result = Counters()
            result.cb = ctypes.sizeof(result)
            if not get_memory(get_process(), ctypes.byref(result), result.cb):
                raise OSError("GetProcessMemoryInfo failed")
            return {"current_working_set_bytes": result.WorkingSetSize,
                    "process_lifetime_peak_working_set_bytes": result.PeakWorkingSetSize,
                    "source": "Windows GetProcessMemoryInfo", "error": None}
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"process_lifetime_peak_working_set_bytes": int(value * (1 if sys.platform == "darwin" else 1024)),
                "current_working_set_bytes": None, "source": "getrusage ru_maxrss", "error": None}
    except Exception as error:
        return {"current_working_set_bytes": None, "process_lifetime_peak_working_set_bytes": None,
                "source": None, "error": repr(error)}


def check_boundary_provenance(vendor, provenance, expected_sha, lock):
    check(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") is not None,
          "A locked provenance SHA256 is required")
    check(lock.add(provenance) == expected_sha, "Boundary upstream provenance bytes differ")
    value = read_json(provenance)
    check(value.get("upstream_repository", "").rstrip("/") == BOUNDARY_REPOSITORY
          and re.fullmatch(r"[0-9a-f]{40}", value.get("upstream_revision", "")) is not None,
          "Boundary source needs its actual upstream repository and commit")
    files = value.get("files", {})
    actual = {p.relative_to(vendor).as_posix() for p in (vendor / "boundary_iou").rglob("*.py")}
    check(actual and actual.issubset(files), "Boundary manifest omits imported vendor Python sources")
    for relative, expected in files.items():
        path = (vendor / relative).resolve()
        check(path.is_relative_to(vendor) and path.is_file(), "Invalid Boundary manifest path")
        check(re.fullmatch(r"[0-9a-f]{64}", expected) is not None and lock.add(path) == expected,
              "Boundary vendor source hash differs: " + relative)
    return value


def run_boundary(args, ids, scorer, out, lock):
    if args.boundary_vendor is None:
        return {"status": "not_requested", "metrics": None, "dilation_ratio": DILATION_RATIO,
                "baseline_ordinary_AP_check": None, "seconds": None,
                "reason": "Boundary requires explicit vendor and verified upstream provenance"}
    began = time.perf_counter()
    vendor = args.boundary_vendor.resolve()
    provenance = check_boundary_provenance(vendor, args.boundary_provenance.resolve(),
                                            args.boundary_provenance_sha256, lock)
    sys.path.insert(0, str(vendor))
    from boundary_iou.coco_instance_api.coco import COCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval
    for module in tuple(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("boundary_iou") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            check(path.is_relative_to(vendor), "Another Boundary package shadows the locked vendor")
            lock.add(path)
    import cv2
    lock.add(cv2.__file__)
    directory = out / "boundary"
    directory.mkdir()
    shutil.copy2(args.boundary_provenance, directory / "UPSTREAM_PROVENANCE.json")

    def evaluate(coco, predictions, kind, name):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            records = segmentation_records(predictions)
            if records:
                detections = coco.loadRes(records)
            else:
                detections = COCO()
                detections.dataset = {"images": list(coco.imgs.values()),
                                      "categories": list(coco.cats.values()), "annotations": []}
                detections.createIndex()
            evaluator = COCOeval(coco, detections, iouType=kind, dilation_ratio=DILATION_RATIO)
            evaluator.params.imgIds = sorted(ids)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        (directory / (name + ".txt")).write_text(stream.getvalue(), encoding="utf-8")
        np.savez_compressed(directory / (name + "_ACCUMULATED.npz"),
                            precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                            iou_thresholds=evaluator.params.iouThrs,
                            max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
        return {key: float(value) if value >= 0 else None
                for key, value in zip(METRIC_NAMES, evaluator.stats)}

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotations.resolve()))
    baseline = list(json_array(args.run / "baseline/predictions.json"))
    ordinary = evaluate(coco, baseline, "segm", "BASELINE_ORDINARY_SEGM")
    expected = scorer["metrics"]["baseline"]["segm"]
    for key in METRIC_NAMES:
        if expected[key] is None:
            check(ordinary[key] is None, "Boundary vendor ordinary metric missingness differs")
        else:
            check(ordinary[key] is not None and math.isclose(ordinary[key], expected[key], rel_tol=0, abs_tol=1e-10),
                  "Boundary vendor baseline ordinary COCO metric differs from actual scorer: " + key)
    # All ordinary metrics must agree before any Boundary computation begins.
    coco.get_boundary, coco.dilation_ratio = True, DILATION_RATIO
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    metrics = {}
    for arm in ARMS:
        predictions = baseline if arm == "baseline" else list(json_array(args.run / arm / "predictions.json"))
        metrics[arm] = evaluate(coco, predictions, "boundary", arm)
        write_json(directory / "PROGRESS.json", {"completed_arms": list(metrics), "metrics": metrics})
        if arm == "baseline":
            del baseline
        del predictions
    return {"status": "completed", "metrics": metrics, "dilation_ratio": DILATION_RATIO,
            "baseline_ordinary_AP_check": {"passed": True, "actual": ordinary, "source": expected,
                                            "absolute_tolerance": 1e-10},
            "upstream": provenance, "seconds": time.perf_counter() - began,
            "metric_units": "COCO fractions, not AP points"}


def validate_inputs(args, lock):
    run, scoring = args.run.resolve(), args.scoring_run.resolve()
    for path in (args.annotations, args.protocol):
        lock.add(path)
    check(sha(args.annotations) == ANNOTATIONS_SHA, "Original val2017 annotation SHA required")
    def locked_json(path):
        lock.add(path)
        return read_json(path)
    inference = locked_json(run / "SUMMARY.json")
    terminal = locked_json(run / "EVALUATION_COMPLETE.json")
    inputs = locked_json(run / "EVALUATION_INPUTS.json")
    check(inference.get("status") == terminal.get("status") == "prediction_complete"
          and inference.get("passed") is True and terminal.get("summary_sha256") == sha(run / "SUMMARY.json"),
          "Inference not complete or sealed summary differs")
    config = inference.get("configuration") or locked_json(run / "baseline/COMPLETE.json").get("fingerprint")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    check(config == inputs["configuration"] and config_sha == inputs["configuration_sha256"]
          == inference["configuration_sha256"], "Inference configuration lock differs")
    ids = config["image_ids"]
    engineering = inference.get("engineering")
    check(type(engineering) is bool and ids and len(ids) == len(set(ids))
          and (len(ids) <= 64 if engineering else len(ids) == 5000), "Inference image/scope differs")
    check(set(inference["arm_names"]) == set(ARMS), "Fixed seven-arm scope changed")
    protocol_snapshot = run / "snapshots" / config["protocol_sha256"]
    check(lock.add(protocol_snapshot) == config["protocol_sha256"], "Original inference protocol snapshot differs")
    scorer = locked_json(scoring / "SUMMARY.json")
    complete = locked_json(scoring / "SCORE_COMPLETE.json")
    score_lock = locked_json(scoring / "SOURCE_LOCK.json")
    score_inputs = locked_json(scoring / "SCORE_INPUTS.json")
    check(scorer.get("status") == "complete" and complete.get("status") == "completed"
          and complete.get("formal_metrics_measured") is True and complete.get("identity_passed") is True
          and complete["summary_sha256"] == sha(scoring / "SUMMARY.json")
          and scorer["source_lock_sha256"] == sha(scoring / "SOURCE_LOCK.json"),
          "Completed ordinary scoring and identity receipts are required")
    check(score_lock.get("all_before_after_sha256_equal") is True,
          "Scorer input lock did not pass")
    check(scorer["source_inference_configuration_sha256"] == config_sha
          and score_inputs["image_ids"] == ids and scorer["engineering"] is engineering
          and scorer["arms"] == list(ARMS), "Scorer belongs to another inference scope")
    check(scorer.get("association_rematched") is False and scorer.get("GPU_or_model_execution") is False,
          "Scorer scope is incompatible")
    for name in ("INFERENCE_SUMMARY.json", "INFERENCE_COMPLETE.json", "INFERENCE_INPUTS.json"):
        file = scoring / "source_provenance" / name
        expected = {"INFERENCE_SUMMARY.json": "SUMMARY.json", "INFERENCE_COMPLETE.json": "EVALUATION_COMPLETE.json",
                    "INFERENCE_INPUTS.json": "EVALUATION_INPUTS.json"}[name]
        check(lock.add(file) == sha(run / expected), "Scorer sealed a different inference source")
    # Verify only small control/provenance artifacts and ordinary metric JSON;
    # this is not a second independent reconstruction of scorer tensor arrays.
    for relative in ("SCORE_INPUTS.json", "SOURCE_LOCK.json", "IDENTITY_IMAGES.jsonl"):
        check(lock.add(scoring / relative) == complete["artifact_sha256"][relative],
              "Scorer source/control artifact differs: " + relative)
    for arm in ARMS:
        relative = arm + "/COCO_METRICS.json"
        check(lock.add(scoring / relative) == complete["artifact_sha256"][relative]
              and read_json(scoring / relative) == scorer["metrics"][arm], "Scorer metric receipt differs")
        archived = scoring / "source_provenance" / (arm + "_COMPLETE.json")
        check(lock.add(archived) == lock.add(run / arm / "COMPLETE.json"), "Scorer arm receipt differs")
    parity = locked_json(run / "BASELINE_PARITY.json")
    check(parity.get("passed") is True or parity.get("status") == "passed", "Native baseline parity absent")
    lock.add(run / "NATIVE_DECISIONS.jsonl")
    # The completed scorer must have read this exact annotation and decisions file.
    for path in (args.annotations.resolve(), run / "NATIVE_DECISIONS.jsonl"):
        expected = sha(path)
        check(any(item.get("sha256_before") == item.get("sha256_after") == expected
                  for item in score_lock["files"].values()), "Readout input not in scorer's sealed input lock")
    readers = {arm: ArmReader(run / arm, lock, ids) for arm in ARMS}
    for arm, reader in readers.items():
        declared = inference["prediction_receipts"][arm]
        check(reader.receipt["fingerprint"] == config
              and reader.receipt["baseline_parity_receipt_sha256"] == sha(run / "BASELINE_PARITY.json")
              and declared["predictions_sha256"] == reader.predictions_sha256
              and declared["complete_sha256"] == sha(run / arm / "COMPLETE.json"), "Arm input fingerprint differs")
    fixed, fixed_source = load_fixed_source(args.source_paired, lock, ids, engineering)
    check(fixed_source["instances_sha256"] == scorer["fixed_source"]["instances_sha256"]
          and fixed_source["paired_summary_sha256"] == scorer["fixed_source"]["paired_summary_sha256"]
          and fixed_source["final_summary_sha256"] == scorer["fixed_source"]["final_summary_sha256"],
          "Original fixed association differs from scorer source")
    return ids, engineering, scorer, readers, fixed, fixed_source


def run_readout(args):
    began = time.perf_counter()
    args.run, args.scoring_run = args.run.resolve(), args.scoring_run.resolve()
    out = args.out_dir.resolve()
    for origin in (args.run, args.scoring_run, args.source_paired.resolve()):
        check(out != origin and not out.is_relative_to(origin) and not origin.is_relative_to(out),
              "Readout output must be a separate Run, not a source ancestor/descendant")
    out.mkdir(parents=True, exist_ok=True)
    check(not any((out / name).exists() for name in ("READOUT_INPUTS.json", "SUMMARY.json", "READOUT_COMPLETE.json", "READOUT_FAILURE.json")),
          "Readout history is immutable; use a new Run")
    lock = InputLock()
    pixel_metrics, pixel_source = load_pixel_metrics()
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for path in (Path(__file__), Path(__file__).with_name("comparison_io.py"),
                 Path(__file__).with_name("score_and_verify_comparison.py"), pixel_source):
        digest = lock.add(path)
        source_hashes[str(path.resolve())] = digest
        target = source_dir / path.name
        check(not target.exists() or sha(target) == digest, "Archived readout source differs")
        if not target.exists():
            shutil.copy2(path, target)
    ids, engineering, scorer, readers, fixed, fixed_source = validate_inputs(args, lock)
    import pycocotools.coco as coco_module
    import pycocotools.mask as mask_utils
    import pycocotools._mask as mask_binary
    for module in (coco_module, mask_utils, mask_binary, np):
        lock.add(module.__file__)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = coco_module.COCO(str(args.annotations.resolve()))
    check(set(ids).issubset(coco.imgs) and (engineering or set(ids) == set(coco.imgs)), "GT image scope differs")
    write_json(out / "READOUT_INPUTS.json", {
        "version": VERSION, "started_at": now(), "inference_run": str(args.run),
        "scoring_run": str(args.scoring_run), "annotations": str(args.annotations.resolve()),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": sha(args.protocol),
        "image_ids": ids, "arms": list(ARMS), "fixed_source": fixed_source,
        "engineering": engineering, "source_sha256": source_hashes,
        "boundary_requested": args.boundary_vendor is not None,
        "python": sys.version, "executable": sys.executable, "numpy": np.__version__})
    shutil.copy2(args.protocol, out / "READOUT_PROTOCOL.md")
    totals = {arm: Counter() for arm in ARMS}
    candidate = {arm: new_cohorts() for arm in ARMS}
    image_macro = {arm: new_cohorts() for arm in ARMS}
    unique_gt = set()
    decisions = iter(jsonl(args.run / "NATIVE_DECISIONS.jsonl"))
    pixel_start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        outputs = {}
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=True)
            outputs[arm] = {
                "candidates": stack.enter_context(gzip.open(folder / "CANDIDATES.jsonl.gz", "xt", encoding="utf-8")),
                "images": stack.enter_context((folder / "IMAGES.jsonl").open("x", encoding="utf-8"))}
        for ordinal, iid in enumerate(ids):
            decision = next(decisions, None)
            check(decision is not None and decision["image_id"] == iid
                  and decision["tau0_native_exact"] is True
                  and decision["all_arm_native_identity_exact"] is True and decision["gt_used"] is False,
                  "Native decision receipt is inconsistent")
            count = decision["native_rows"]
            check(type(count) is int and count >= 0 and len(decision["decisions"]) == count,
                  "Native decision count missing")
            rows = {arm: readers[arm].read(iid, count) for arm in ARMS}
            pairs = fixed_pairs_for_image(iid, fixed.get(iid, []), rows["baseline"], coco)
            gt_masks = {}
            per_image = {arm: new_cohorts() for arm in ARMS}
            image_counts = {arm: Counter() for arm in ARMS}
            expected_shape = (coco.imgs[iid]["height"], coco.imgs[iid]["width"])
            for index, baseline_row in enumerate(rows["baseline"]):
                detail = decision["decisions"][index]
                check(detail["detection_index"] == index and type(detail["raw_index"]) is int
                      and type(detail["native_output_row"]) is int, "Native ordinal/raw bridge differs")
                source = pairs.get(index)
                gt, base = None, None
                if source is not None:
                    annid = source["annotation_id"]
                    unique_gt.add((iid, annid))
                    if annid not in gt_masks:
                        gt_masks[annid] = coco.annToMask(coco.anns[annid]).astype(bool)
                    gt = gt_masks[annid]
                    check(gt.shape == expected_shape and np.any(gt), "Fixed GT mask is empty or has wrong grid")
                for arm in ARMS:
                    prediction = rows[arm][index]
                    check(all(prediction[key] == baseline_row[key]
                              for key in ("image_id", "category_id", "bbox", "score")),
                          "Arm candidate class/box/score/order differs")
                    check(prediction["segmentation"]["size"] == list(expected_shape), "Prediction RLE is not original-image grid")
                    mask = mask_utils.decode(encoded_rle(prediction["segmentation"])).astype(bool)
                    check(mask.shape == expected_shape, "Decoded mask shape differs")
                    empty = not np.any(mask)
                    counters = image_counts[arm]
                    counters["all_candidate_count"] += 1
                    counters["empty_prediction_count"] += int(empty)
                    record = {"image_id": iid, "detection_index": index,
                              "native_output_row": detail["native_output_row"], "raw_index": detail["raw_index"],
                              "annotation_id": source["annotation_id"] if source else None,
                              "category_id": prediction["category_id"], "empty_prediction": empty,
                              "association_status": "fixed_final8_pair" if source else "unassociated_in_fixed_source"}
                    if source is None:
                        counters["unassociated_candidate_count"] += 1
                        record.update({key: None for key in FIELDS})
                        record.update(outcome(None, None))
                        record["baseline_mask_iou"] = None
                    else:
                        measured = pixel_metrics(mask, gt)
                        if arm == "baseline":
                            base = measured
                            check(math.isclose(base["mask_iou"], source["baseline_mask_iou"], rel_tol=1e-14, abs_tol=1e-14),
                                  "Baseline original-image IoU differs from locked fixed pair")
                        record.update(record_metrics(measured, base))
                        record.update(outcome(base["mask_iou"], measured["mask_iou"]))
                        record["baseline_mask_iou"] = base["mask_iou"]
                        counters["fixed_pair_count"] += 1
                        counters["baseline_success_count"] += int(record["baseline_success"])
                        counters["baseline_failure_count"] += int(not record["baseline_success"])
                        counters["repair_count"] += int(record["repair"])
                        counters["damage_count"] += int(record["damage"])
                        counters["empty_paired_prediction_count"] += int(empty)
                        group = "baseline_success" if record["baseline_success"] else "baseline_failure"
                        for cohort in ("all_fixed_pairs", group):
                            candidate[arm][cohort].add(record)
                            per_image[arm][cohort].add(record)
                    outputs[arm]["candidates"].write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            for arm in ARMS:
                totals[arm].update(image_counts[arm])
                cohorts = {}
                for cohort, acc in per_image[arm].items():
                    table = acc.finish()
                    cohorts[cohort] = table
                    image_macro[arm][cohort].add({key: value["mean"] for key, value in table["fields"].items()})
                outputs[arm]["images"].write(json.dumps({"image_id": iid, "counts": dict(image_counts[arm]),
                                                          "candidate_means_within_image": cohorts}, allow_nan=False) + "\n")
            if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(ids):
                write_json(out / "PROGRESS.json", {"stage": "pixels", "images": ordinal + 1,
                                                    "total_images": len(ids), "seconds": time.perf_counter() - pixel_start})
                print("MULTIVIEW_PIXELS", ordinal + 1, len(ids), flush=True)
    check(next(decisions, None) is None, "Extra native decision rows")
    for arm, reader in readers.items():
        reader.finish(len(ids))
        expected_count = scorer["identity"]["arm_counts"][arm]
        check(totals[arm]["all_candidate_count"] == expected_count["candidate_count"]
              and totals[arm]["empty_prediction_count"] == expected_count["empty_mask_count"], "Native output/empty count differs from scorer")
        original = scorer["fixed_paired"][arm]["statistics"]
        for own, source_name in (("fixed_pair_count", "matched_detection_count"),
                                 ("baseline_success_count", "baseline_success_count"),
                                 ("baseline_failure_count", "baseline_failure_count"),
                                 ("repair_count", "repair_count"), ("damage_count", "damage_count")):
            check(totals[arm][own] == original[source_name], "Fixed outcome cardinality differs from scorer: " + arm + "/" + own)
    pixel_seconds = time.perf_counter() - pixel_start
    boundary = run_boundary(args, ids, scorer, out, lock)
    input_hashes = lock.finish()
    write_json(out / "SOURCE_LOCK.json", {"files": input_hashes,
                                          "all_before_after_sha256_equal": True,
                                          "executed_source_sha256": source_hashes})
    summaries = {}
    for arm in ARMS:
        counts = {key: totals[arm][key] for key in (
            "all_candidate_count", "fixed_pair_count", "unassociated_candidate_count",
            "empty_prediction_count", "empty_paired_prediction_count", "baseline_success_count",
            "baseline_failure_count", "repair_count", "damage_count")}
        summaries[arm] = {"counts": counts,
            "damage_rate_of_baseline_success": counts["damage_count"] / counts["baseline_success_count"] if counts["baseline_success_count"] else None,
            "repair_rate_of_baseline_failure": counts["repair_count"] / counts["baseline_failure_count"] if counts["baseline_failure_count"] else None,
            "candidate_macro": {key: value.finish() for key, value in candidate[arm].items()},
            "image_macro": {key: value.finish() for key, value in image_macro[arm].items()}}
    result = {"version": VERSION, "status": "complete", "image_count": len(ids), "engineering": engineering,
        "formal_5000": not engineering, "arms": summaries, "unique_associated_GT_count": len(unique_gt),
        "association": "original final8 detection-to-GT; duplicate GT associations retained; no rematching",
        "macro_definitions": {"candidate_macro": "equal weight per fixed paired detection; not unique-GT average",
            "image_macro": "equal weight per image's candidate mean, field-wise; images with no valid values counted missing",
            "delta": "paired difference only when both baseline and method values exist; denominator may differ by field"},
        "unassociated_policy": "all native rows retained with null GT-dependent metrics, excluded from fixed-pair means",
        "success_threshold": THRESHOLD, "boundary": boundary,
        "unmeasured": {"raw_geometry": None, "TAL": None, "AUC": None, "pixel_domain_FPR": None,
                       "crop_support": None, "AP_confidence_intervals": None, "seed_uncertainty": None,
                       "neighbor_error_partition": None},
        "ordinary_AP": {"recomputed": False, "source_scoring_run": str(args.scoring_run),
                         "source_summary_sha256": sha(args.scoring_run / "SUMMARY.json"),
                         "boundary_vendor_baseline_parity_only": args.boundary_vendor is not None},
        "GPU_or_model_execution": False, "GT_rematched": False,
        "fixed_source": fixed_source, "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
        "timing_seconds": {"pixel_readout": pixel_seconds, "boundary": boundary["seconds"],
                           "total_through_input_rehash": time.perf_counter() - began},
        "memory": memory_observation(), "completed_at": now(),
        "limitations": ["No new native forward; native parity is inherited from the sealed completed scorer and inference receipts",
            "Output-only fixed-pair cohort does not measure full raw-pool geometry or training assignment",
            "All input/source hashes rechecked; this run does not independently reconstruct ordinary scorer accumulated arrays",
            "Pixel fractions share TP/FP/FN bookkeeping and are not independent causal evidence",
            "CPU readout timings and process memory are not deployment inference costs"]}
    write_json(out / "SUMMARY.json", result)
    write_json(out / "READOUT_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"),
        "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*")
                            if p.is_file() and p.name not in ("run.json", "stdout.log", "SOURCE.json", "PROCESS.json", "READOUT_COMPLETE.json")
                            and "source" not in p.relative_to(out).parts}})
    print(json.dumps({"status": "complete", "images": len(ids), "boundary": boundary["status"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "scoring-run", "source-paired", "annotations", "protocol", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--boundary-vendor", type=Path)
    parser.add_argument("--boundary-provenance", type=Path)
    parser.add_argument("--boundary-provenance-sha256")
    args = parser.parse_args()
    supplied = [args.boundary_vendor is not None, args.boundary_provenance is not None,
                args.boundary_provenance_sha256 is not None]
    if any(supplied) and not all(supplied):
        parser.error("Boundary requires --boundary-vendor, --boundary-provenance, and --boundary-provenance-sha256 together")
    try:
        run_readout(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        failure = args.out_dir / "READOUT_FAILURE.json"
        if not failure.exists() and not (args.out_dir / "READOUT_COMPLETE.json").exists():
            write_json(failure, {"status": "failed", "version": VERSION, "error": repr(error),
                                 "traceback": traceback.format_exc(), "time": now(),
                                 "GPU_or_model_execution": False})
        raise


if __name__ == "__main__":
    main()
