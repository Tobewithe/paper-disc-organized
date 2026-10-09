"""Independent CPU evaluation of complete native baseline / I / H exports.

GT is used only here. Original final8 candidate-to-GT pairs are read verbatim;
normal COCO metrics use all COCO GT with crowd/ignore and full normal output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from target_scoring_common import (ARMS, ANN_SHA, BASE_PRED_SHA, FINAL8_HEAD_SHA,
    PROTOCOL_SHA, COCO_KEYS, IDENTITY_KEYS, METRICS, PAIR_FIELDS, InputLock,
    cache_from_evaluator, check, encoded_rle, json_array, jsonl, official_eval,
    paired_summary, read_json, segm_records, sha, write_json, seal_outputs)


def fixed_source(path, ids, engineering, lock):
    path = Path(path).resolve()
    instances = path / "INSTANCES.jsonl" if path.is_dir() else path
    paired_path, final_path = instances.parent / "SUMMARY.json", instances.parent.parent / "SUMMARY.json"
    for p in (instances, paired_path, final_path):
        lock.add(p)
    paired, final = read_json(paired_path), read_json(final_path)
    check(paired["paired_readout_valid"] is True and final["evaluation_epoch"] == 8 and final["image_count"] == 5000,
          "Require accepted original final8 fixed cohort")
    check(final["head_provenance"]["loaded_state_sha256"] == FINAL8_HEAD_SHA, "Final8 source head differs")
    check(final["paired"]["triflow"] == paired, "Original final8 paired summary binding differs")
    grouped, keys, full_success = defaultdict(list), set(), 0
    supplied = set(ids)
    for row in jsonl(instances):
        key = row["image_id"], row["detection_index"]
        check(key not in keys and row["baseline_success"] == (row["baseline_mask_iou"] >= .75), "Invalid fixed source identity/threshold")
        keys.add(key)
        full_success += row["baseline_success"]
        if row["image_id"] in supplied:
            grouped[row["image_id"]].append(row)
    check(len(keys) == 86600 and full_success == 36266, "Accepted fixed cohort cardinality changed")
    check(paired["statistics"]["matched_detection_count"] == len(keys) and paired["statistics"]["baseline_success_count"] == full_success,
          "Original fixed source total differs")
    if not engineering:
        check(sum(map(len, grouped.values())) == 86600, "Formal fixed cohort incomplete")
    return grouped, {"instances": str(instances), "instances_sha256": sha(instances),
        "paired_summary": str(paired_path), "paired_summary_sha256": sha(paired_path),
        "final_summary": str(final_path), "final_summary_sha256": sha(final_path),
        "definition": paired["definition"], "selected_rows": sum(map(len, grouped.values())),
        "full_rows": 86600, "full_baseline_success": 36266, "full_baseline_failure": 50334,
        "duplicate_matches_allowed": True, "association_rematched": False}


def arm_image(run, arm, iid, count, streams, lock):
    path = run / arm / "images" / f"{iid:012d}.json"
    lock.add(path)
    value = read_json(path)
    rows = value["detections"]
    check(value["image_id"] == iid and len(rows) == count, "Arm image count differs")
    exported = [next(streams[arm], None) for _ in range(count)]
    for i, row in enumerate(rows):
        check(row["detection_index"] == i and row["image_id"] == iid and all(k in row for k in IDENTITY_KEYS), "Native identity fields missing")
        check(exported[i] is not None and all(exported[i][k] == row[k] for k in COCO_KEYS), "Assembled COCO order/fields differ")
        encoded_rle(row["segmentation"])
    return value


def save_eval(path, evaluator, values, stdout, cache=False):
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "COCO_METRICS.json", values)
    (path / "COCO.txt").write_text(stdout, encoding="utf-8")
    np.savez_compressed(path / "COCO_ACCUMULATED.npz", precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                        scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs,
                        recall_thresholds=evaluator.params.recThrs, category_ids=evaluator.params.catIds,
                        max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
    if cache:
        np.savez_compressed(path / "COCO_BOOTSTRAP_CACHE.npz", **cache_from_evaluator(evaluator))


def boundary_score(args, ids, ordinary, lock):
    if args.boundary_vendor is None:
        return {"status": "unmeasured", "metrics": None, "dilation_ratio": .02}
    vendor, provenance_path = args.boundary_vendor.resolve(), args.boundary_provenance.resolve()
    check(lock.add(provenance_path) == args.boundary_provenance_sha256, "Boundary provenance SHA differs")
    provenance = read_json(provenance_path)
    files = provenance["files"]
    actual = {p.relative_to(vendor).as_posix() for p in (vendor / "boundary_iou").rglob("*.py")}
    check(len(actual) == 6 and set(files) == actual, "Boundary requires exact six-source content lock")
    for relative, expected in files.items():
        path = (vendor / relative).resolve()
        path.relative_to(vendor)
        check(lock.add(path) == expected, "Boundary vendor source changed")
    sys.path.insert(0, str(vendor))
    from boundary_iou.coco_instance_api.coco import COCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval
    import cv2
    cv2.setNumThreads(1)
    for mod in tuple(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("boundary_iou") and getattr(mod, "__file__", None):
            check(Path(mod.__file__).resolve().is_relative_to(vendor), "Another Boundary module shadows locked source")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotations.resolve()))
    base = segm_records(args.run / "baseline" / "predictions.json")
    ev, values, log = official_eval(coco, base, ids, COCO, COCOeval, dilation_ratio=.02)
    for name in METRICS:
        check(values[name] == ordinary["baseline"][name] or values[name] is not None and ordinary["baseline"][name] is not None
              and abs(values[name] - ordinary["baseline"][name]) <= 1e-10, "Boundary backend ordinary base parity differs: " + name)
    save_eval(args.out_dir / "boundary" / "BASELINE_ORDINARY_SEGM", ev, values, log)
    del ev, base
    coco.get_boundary, coco.dilation_ratio = True, .02
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    results = {}
    for arm in ARMS:
        records = segm_records(args.run / arm / "predictions.json")
        ev, stats, stdout = official_eval(coco, records, ids, COCO, COCOeval, "boundary", dilation_ratio=.02)
        save_eval(args.out_dir / "boundary" / arm, ev, stats, stdout)
        results[arm] = stats
        del ev, records
        print("boundary complete", arm, flush=True)
    return {"status": "completed", "dilation_ratio": .02, "metrics": results,
            "ordinary_baseline_parity": {"passed": True, "actual": values, "absolute_tolerance": 1e-10},
            "vendor_content_lock": provenance, "provenance_sha256": sha(provenance_path)}


def score(args):
    began = time.perf_counter()
    args.run, args.out_dir = args.run.resolve(), args.out_dir.resolve()
    out, run = args.out_dir, args.run
    check(out != run and not out.is_relative_to(run) and not run.is_relative_to(out), "Independent scoring Run required")
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / "SUMMARY.json").exists() and not (out / "SCORE_INPUTS.json").exists(), "New scoring Run required")
    lock = InputLock()
    check(lock.add(args.protocol) == PROTOCOL_SHA, "Frozen target protocol changed")
    check(lock.add(args.annotations) == ANN_SHA, "Original COCO val annotation bytes changed")
    for name in ("run.json", "SUMMARY.json", "COMPLETE.json", "INPUTS.json", "SOURCE_LOCK.json", "NATIVE_DECISIONS.jsonl", "BASELINE_PARITY_IMAGES.jsonl"):
        lock.add(run / name)
    observed, summary, terminal, inputs = [read_json(run / n) for n in ("run.json", "SUMMARY.json", "COMPLETE.json", "INPUTS.json")]
    check(observed["status"] == "completed" and observed["return_code"] == 0 and observed["artifact_completeness"] == "complete", "Producer process must actually complete")
    check(summary["status"] == "prediction_complete" and summary["passed"] is True and terminal["summary_sha256"] == sha(run / "SUMMARY.json"), "Producer terminal summary binding differs")
    check(summary["source_unchanged"] is True and summary["gt_parsed"] is False and summary["gt_used_in_inference"] is False
          and inputs["gt_parsed"] is False and inputs["gt_used_in_inference"] is False, "GT-free/source-immutability producer declaration differs")
    check(inputs["protocol_sha256"] == PROTOCOL_SHA and summary["inputs_sha256"] == sha(run / "INPUTS.json"), "Actual inference input/protocol binding differs")
    ids = summary["image_ids"]
    engineering = summary["engineering"]
    check(type(engineering) is bool and len(ids) == len(set(ids)) and (0 < len(ids) <= 4 if engineering else len(ids) == 5000), "Protocol image scope differs")
    check(summary["arm_names"] == list(ARMS), "Frozen arm naming differs")
    check(inputs["image_ids"] == ids and inputs["engineering"] is engineering, "Actual inference image scope differs")
    source_lock = read_json(run / "SOURCE_LOCK.json")
    check(summary["source_lock_sha256"] == sha(run / "SOURCE_LOCK.json"), "Producer source lock not bound")
    for value in source_lock["files"].values():
        check(lock.add(value["path"]) == value["sha256"], "Producer source/input changed")
        if value.get("snapshot"):
            check(lock.add(run / value["snapshot"]) == value["sha256"], "Producer source snapshot differs")
    portable_path = Path(source_lock["files"]["frozen/portable_risk.py"]["path"])
    spec = importlib.util.spec_from_file_location("target_score_frozen_portable", portable_path)
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    estimators = {}
    fit_summaries = []
    for arm in ARMS[1:]:
        model_root = Path(inputs["model_runs"][arm])
        check(lock.add(model_root / "model.json") == inputs["model_sha256"][arm], "Actual frozen numeric model SHA differs")
        check(lock.add(model_root / "SUMMARY.json") == inputs["model_summary_sha256"][arm], "Actual frozen fit summary SHA differs")
        fit = read_json(model_root / "SUMMARY.json")
        check(fit["passed"] and fit["status"] == "fit_complete" and fit["iterations"] == 100 and fit["features"] == 5
              and fit["engineering"] is engineering and fit["model_sha256"] == inputs["model_sha256"][arm], "Frozen fit budget/model binding differs")
        fit_summaries.append(fit)
        estimators[arm] = portable.PortableRisk(model_root / "model.json")
    for key in ("extraction_run", "extraction_summary_sha256", "extraction_complete_sha256", "full_hgb_parameters",
                "sample_weight", "weight_bytes_sha256", "training_feature_bytes_sha256", "supervised_binding_sha256",
                "training_rows", "training_images", "runtime", "engineering"):
        check(fit_summaries[0][key] == fit_summaries[1][key], "Targets changed shared fit input/capacity: " + key)
    reference = args.reference_baseline.resolve()
    check(lock.add(reference / "predictions.json") == BASE_PRED_SHA, "Accepted native full baseline bytes changed")
    lock.add(reference / "COMPLETE.json")
    ref_complete = read_json(reference / "COMPLETE.json")
    check(ref_complete["image_count"] == 5000, "Accepted baseline incomplete")
    old_parity_path = reference.parent / "BASELINE_PARITY_IMAGES.jsonl"
    lock.add(old_parity_path)
    old_parity = {r["image_id"]: r for r in jsonl(old_parity_path)}
    if not engineering:
        check(ids == ref_complete["fingerprint"]["image_ids"], "Formal original val order differs")
    grouped, source = fixed_source(args.source_paired, ids, engineering, lock)
    provenance = out / "source_provenance"
    provenance.mkdir(exist_ok=True)
    for origin, name in ((source["instances"], "FIXED_INSTANCES_SOURCE.jsonl"), (source["paired_summary"], "FIXED_PAIRED_SOURCE_SUMMARY.json"),
                         (source["final_summary"], "FINAL8_SOURCE_SUMMARY.json"), (args.protocol, "PROTOCOL.md"),
                         (run / "SUMMARY.json", "INFERENCE_SUMMARY.json"), (run / "SOURCE_LOCK.json", "INFERENCE_SOURCE_LOCK.json")):
        shutil.copy2(origin, provenance / name)
    for name in ("evaluate_target_comparison.py", "bootstrap_target_ap.py", "target_scoring_common.py", "verify_target_comparison.py"):
        path = Path(__file__).with_name(name)
        lock.add(path)
        shutil.copy2(path, provenance / name)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    import pycocotools.coco as coco_module
    import pycocotools.cocoeval as eval_module
    import pycocotools._mask as mask_binary
    for module in (coco_module, eval_module, mask_utils, mask_binary):
        lock.add(module.__file__)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotations.resolve()))
    check(set(ids).issubset(coco.imgs) and (engineering or set(ids) == set(coco.imgs)), "COCO image scope differs")
    streams = {arm: iter(json_array(run / arm / "predictions.json")) for arm in ARMS}
    arm_receipts = {}
    for arm in ARMS:
        arm_receipts[arm] = read_json(run / arm / "COMPLETE.json")
        lock.add(run / arm / "COMPLETE.json")
        digest = lock.add(run / arm / "predictions.json")
        check(arm_receipts[arm]["status"] == "prediction_complete" and arm_receipts[arm]["image_count"] == len(ids)
              and arm_receipts[arm]["predictions_sha256"] == digest, "Arm completion/SHA differs")
        check(arm_receipts[arm]["image_ids"] == ids and arm_receipts[arm]["input_sha256"] == sha(run / "INPUTS.json")
              and arm_receipts[arm]["empty_masks_preserved"] is True and arm_receipts[arm]["gt_used"] is False,
              "Arm identity/input/empty/GT-free binding differs")
        check(summary["prediction_receipts"][arm] == {"predictions_sha256": digest, "complete_sha256": sha(run / arm / "COMPLETE.json")},
              "Summary does not bind actual arm bytes")
        if arm != "baseline":
            check(arm_receipts[arm]["model_sha256"] == inputs["model_sha256"][arm], "Arm model SHA differs")
    check(len({v["frozen_state_sha256"] for v in arm_receipts.values()}) == 1, "Arms changed frozen backbone state")
    write_json(out / "SCORE_INPUTS.json", {"image_ids": ids, "arms": list(ARMS), "engineering": engineering,
               "fixed_source": source, "producer_run": str(run), "producer_summary_sha256": sha(run / "SUMMARY.json"),
               "protocol_sha256": PROTOCOL_SHA, "annotations_sha256": ANN_SHA, "reference_baseline": str(reference),
               "reference_predictions_sha256": BASE_PRED_SHA, "interpreter": sys.executable,
               "numpy": np.__version__, "pycocotools": importlib.metadata.version("pycocotools"),
               "GPU_or_model_execution": False, "GT_rematched": False})
    pairs = np.zeros((3, len(ids), len(PAIR_FIELDS)), dtype=np.float64)
    integer_rows, pair_keys, baseline_source_ious = [], [], []
    decisions = iter(jsonl(run / "NATIVE_DECISIONS.jsonl"))
    parities = iter(jsonl(run / "BASELINE_PARITY_IMAGES.jsonl"))
    same_strategy = True
    actual_counts = {key: 0 for key in ("native_rows", "selected_first64", "supported_first64", "unsupported_first64",
                    "empty_baseline_masks", "empty_trial_fallback", "target_I_applied", "target_H_applied", "different_decisions", "different_binary_outputs")}
    with (out / "IDENTITY_IMAGES.jsonl").open("x", encoding="utf-8") as identity_out:
        for pos, iid in enumerate(ids):
            decision, parity = next(decisions, None), next(parities, None)
            check(decision is not None and parity is not None and decision["image_id"] == parity["image_id"] == iid, "Decision/parity image sequence differs")
            ref_path = reference / "images" / f"{iid:012d}.json"
            lock.add(ref_path)
            ref = read_json(ref_path)
            count = len(ref["detections"])
            images = {arm: arm_image(run, arm, iid, count, streams, lock) for arm in ARMS}
            base = images["baseline"]["detections"]
            check(images["baseline"]["original_shape"] == ref["original_shape"] and base == ref["detections"], "New baseline differs from accepted all-normal native identity/RLE")
            check(parity["image_sha256"] == old_parity[iid]["image_sha256"] and parity["input_sha256"] == old_parity[iid]["input_sha256"], "Original JPEG/FP32 input chain differs")
            check(parity["coefficients_sha256"] == old_parity[iid]["coefficients_sha256"]
                  and parity["native_identity_sha256"] == old_parity[iid]["native_identity_sha256"] == ref["source_baseline_identity_sha256"]
                  and parity["source_image_cache_sha256"] == sha(ref_path), "Original native coefficient/identity/source-cache binding differs")
            policies = decision["decisions"]
            check(len(policies) == count, "Missing normal candidate policy rows")
            supported = [p for p in policies if p["action_supported"]]
            check(parity["native_rows"] == count and parity["selected_first64"] == min(64, count)
                  and parity["supported_first64"] == len(supported) and parity["unsupported_first64"] == min(64, count) - len(supported),
                  "Explicit full/first64/support zero and nonzero counts differ")
            image_counts = {"native_rows": count, "selected_first64": min(64, count), "supported_first64": len(supported),
                            "unsupported_first64": min(64, count) - len(supported),
                            "empty_baseline_masks": sum(int(mask_utils.area(encoded_rle(r["segmentation"]))) == 0 for r in base),
                            "empty_trial_fallback": sum(bool(p["fallback"]) for p in supported),
                            "target_I_applied": sum(p["target_I_applied"] for p in policies),
                            "target_H_applied": sum(p["target_H_applied"] for p in policies),
                            "different_decisions": sum(p["target_I_applied"] != p["target_H_applied"] for p in policies),
                            "different_binary_outputs": sum(a["segmentation"] != b["segmentation"] for a, b in zip(images["target_I"]["detections"], images["target_H"]["detections"]))}
            check(image_counts == parity["counts"], "Producer candidate/image statistics differ from actual exported rows")
            for key, value in image_counts.items():
                actual_counts[key] += value
            feature_array = np.asarray([p["features"] for p in supported], dtype=np.float64).reshape(-1, 5)
            predicted = {arm: estimators[arm].predict(feature_array) for arm in ARMS[1:]}
            for j, policy in enumerate(supported):
                check(policy["in_first64"] is True and policy["proto_roi_supported"] is True
                      and np.isfinite(feature_array[j]).all() and hashlib.sha256(feature_array[j].tobytes()).hexdigest() == policy["feature_sha256"],
                      "Supported five-feature bytes/qualification differ")
                for arm in ARMS[1:]:
                    check(float(predicted[arm][j]) == policy[arm + "_gain"], "Saved scalar differs from frozen numeric model")
            changed = [0, 0, 0]
            for index, policy in enumerate(policies):
                check(policy["detection_index"] == index, "Policy ordinal changed")
                check(policy["in_first64"] is (index < 64) and policy["action_supported"] is (index < 64 and policy["proto_roi_supported"]), "First64/protoROI qualification changed")
                check(policy["native_output_row"] == parity["native_output_rows"][index]
                      and policy["raw_index"] == parity["raw_indices"][index], "Native raw/output ordinal binding differs")
                same_strategy = same_strategy and policy["target_I_applied"] == policy["target_H_applied"]
                for a, arm in enumerate(ARMS):
                    row = images[arm]["detections"][index]
                    check(all(row[k] == base[index][k] for k in IDENTITY_KEYS), "Action changed class/box/score/raw identity/order")
                    if arm == "baseline":
                        continue
                    apply = policy[arm + "_applied"]
                    prediction = policy[arm + "_gain"]
                    support = policy["action_supported"]
                    if support:
                        check(index < 64 and prediction is not None and math.isfinite(prediction), "Supported scalar missing/nonfinite")
                        check(apply is (prediction > 0 and policy["trial_nonempty"] and not policy["fallback"]), "Frozen scalar>0/nonempty policy differs")
                    else:
                        check(prediction is None and not apply and policy["features"] is None and policy["feature_sha256"] is None
                              and policy["fallback"] is None and policy["trial_nonempty"] is None, "Unsupported/outside-first64 unknown fields must stay unknown")
                    if not apply:
                        check(row["segmentation"] == base[index]["segmentation"], "No-op/fallback changed baseline mask")
                    if row["segmentation"] != base[index]["segmentation"]:
                        changed[a] += 1
                        rle, base_rle = encoded_rle(row["segmentation"]), encoded_rle(base[index]["segmentation"])
                        area = int(mask_utils.area(rle))
                        check(area > 0 and int(mask_utils.area(mask_utils.merge([rle, base_rle], intersect=True))) == area, "Smooth action expanded/emptied baseline support")
            for source_row in grouped.get(iid, []):
                index, annid = source_row["detection_index"], source_row["annotation_id"]
                ann = coco.anns[annid]
                check(ann["image_id"] == iid and ann["category_id"] == base[index]["category_id"] == source_row["category_id"]
                      and not ann.get("iscrowd", 0) and not ann.get("ignore", 0), "Fixed source GT identity differs")
                check(base[index]["raw_confidence"] == source_row["baseline_score"], "Fixed original confidence differs")
                gt = coco.annToRLE(ann)
                gt_area = int(mask_utils.area(gt))
                check(gt_area > 0, "Invalid fixed GT mask has zero pixels")
                ledger = []
                for a, arm in enumerate(ARMS):
                    rle = encoded_rle(images[arm]["detections"][index]["segmentation"])
                    area = int(mask_utils.area(rle))
                    tp = int(mask_utils.area(mask_utils.merge([rle, gt], intersect=True)))
                    fp, fn = area - tp, gt_area - tp
                    union = tp + fp + fn
                    iou = tp / union
                    ledger.append((tp, fp, fn, area))
                    if a == 0:
                        base_iou, success = iou, 4 * tp >= 3 * union
                        check(abs(base_iou - source_row["baseline_mask_iou"]) <= 1e-14 and success is source_row["baseline_success"], "Original fixed baseline IoU/success differs")
                    method_success = 4 * tp >= 3 * union
                    delta = iou - base_iou
                    pairs[a, pos] += np.asarray([1, success, not success, success and not method_success, not success and method_success,
                        base_iou, iou, delta, delta if success else 0., tp / gt_area, 1, tp / area if area else 0., int(area > 0),
                        fp / gt_area, 1, fn / gt_area, 1], dtype=np.float64)
                pair_keys.append((pos, index, annid, source_row["category_id"]))
                baseline_source_ious.append(source_row["baseline_mask_iou"])
                integer_rows.append(ledger)
            identity_out.write(json.dumps({"image_id": iid, "counts": image_counts, "native_rows": count, "changed": dict(zip(ARMS, changed)),
                                          "scalar_counts": {arm: len(predicted[arm]) for arm in ARMS[1:]},
                                          "JPEG_input_native_baseline_allarm_identity_passed": True}) + "\n")
            if (pos + 1) % 100 == 0:
                print("fixed CPU pixel score", pos + 1, len(ids), flush=True)
    check(next(decisions, None) is None and next(parities, None) is None, "Trailing decision/parity image")
    for stream in streams.values():
        check(next(stream, None) is None, "Trailing standard normal detection")
    check(actual_counts == summary["totals"], "Full producer normal-candidate/image aggregate statistics differ")
    np.savez_compressed(out / "FIXED_PAIRS.npz", keys=np.asarray(pair_keys, dtype=np.int64),
                        ledger=np.asarray(integer_rows, dtype=np.int64), source_baseline_ious=np.asarray(baseline_source_ious), image_ids=ids,
                        ledger_fields=np.asarray(["TP", "FP", "FN", "prediction_area"]))
    np.savez_compressed(out / "PAIRED_IMAGE_SUMS.npz", values=pairs, image_ids=ids, arms=np.asarray(ARMS), fields=np.asarray(PAIR_FIELDS))
    ordinary = {}
    for arm in ARMS:
        records = segm_records(run / arm / "predictions.json")
        ev, stats, log = official_eval(coco, records, ids, COCO, COCOeval)
        save_eval(out / arm, ev, stats, log, cache=True)
        ordinary[arm] = stats
        del ev, records
        print("ordinary COCO complete", arm, flush=True)
    if not engineering:
        expected_path = args.reference_scoring / "baseline" / "COCO_METRICS.json"
        lock.add(expected_path)
        expected = read_json(expected_path)["segm"]
        check(all(abs(ordinary["baseline"][k] - expected[k]) <= 1e-10 for k in METRICS), "Accepted ordinary baseline12 parity differs")
    boundary = boundary_score(args, ids, ordinary, lock)
    locked = lock.finish()
    write_json(out / "SOURCE_LOCK.json", {"files": locked, "GPU_or_model_execution": False, "GT_rematched": False})
    artifact_names = ["SCORE_INPUTS.json", "FIXED_PAIRS.npz", "PAIRED_IMAGE_SUMS.npz", "IDENTITY_IMAGES.jsonl"]
    artifact_names += [f"{arm}/{name}" for arm in ARMS for name in ("COCO_METRICS.json", "COCO_ACCUMULATED.npz", "COCO_BOOTSTRAP_CACHE.npz")]
    if boundary["status"] == "completed":
        artifact_names += [f"boundary/{arm}/{name}" for arm in ("BASELINE_ORDINARY_SEGM",) + ARMS
                           for name in ("COCO_METRICS.json", "COCO_ACCUMULATED.npz")]
    artifacts = seal_outputs(out, artifact_names)
    summary_out = {"status": "scoring_complete", "passed": True, "engineering": engineering, "image_count": len(ids), "image_ids": ids,
        "arms": list(ARMS), "metrics": ordinary, "paired": {arm: paired_summary(pairs[a]) for a, arm in enumerate(ARMS)},
        "fixed_source": source, "boundary": boundary, "metric_units": "fraction 0..1; AP points multiply100",
        "same_strategy": same_strategy,
        "candidate_image_counts": actual_counts, "artifacts": artifacts,
        "confidence_intervals": None, "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
        "fixed_pairs_sha256": sha(out / "FIXED_PAIRS.npz"), "paired_image_sums_sha256": sha(out / "PAIRED_IMAGE_SUMS.npz"),
        "GPU_or_model_execution": False, "GT_rematched": False, "seconds": time.perf_counter() - began,
        "limitations": ["Fixed final8 detection-associated cohort allows duplicate GT; separate from normal all-GT COCO AP", "No training seed interval; conditional single fit"]}
    write_json(out / "SUMMARY.json", summary_out)
    write_json(out / "COMPLETE.json", {"status": "scoring_complete", "summary_sha256": sha(out / "SUMMARY.json"), "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "artifacts": artifacts})
    print(json.dumps({"status": "scoring_complete", "image_count": len(ids), "fixed_rows": len(pair_keys), "metrics": ordinary}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "annotations", "reference-baseline", "source-paired", "protocol", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reference-scoring", type=Path)
    p.add_argument("--boundary-vendor", type=Path)
    p.add_argument("--boundary-provenance", type=Path)
    p.add_argument("--boundary-provenance-sha256")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        score(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "SCORING_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "GPU_or_model_execution": False})
        raise
