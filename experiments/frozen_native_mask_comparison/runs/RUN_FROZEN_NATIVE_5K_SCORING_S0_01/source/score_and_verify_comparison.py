"""Independent CPU scoring of seven frozen native readouts and fixed final8 pairs.

No model, torch, CUDA, GT rematching, gate fitting or action selection is run.
Formal scoring requires numpy and the actual pycocotools implementation. The
engineering identity-only path leaves all AP and mask-IoU metrics unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

from comparison_io import ARMS, COCO_KEYS, IDENTITY_KEYS, ArmReader, InputLock, check, encoded_rle, json_array, jsonl, read_json, rle_area, sha, write_json

VERSION = "frozen_native_comparison_cpu_score_v1"
ANNOTATIONS_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
FINAL8_HEAD_STATE_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
METRIC_NAMES = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")


def now():
    return datetime.now(timezone.utc).isoformat()


def divide(a, b):
    return a/b if b else None


def new_pair_acc():
    return {"matched_detection_count": 0, "baseline_success_count": 0, "baseline_failure_count": 0, "damage_count": 0, "repair_count": 0,
        "baseline_iou_sum": 0., "method_iou_sum": 0., "mask_iou_delta_sum": 0., "baseline_success_delta_sum": 0.}


def pair_add(a, row):
    a["matched_detection_count"] += 1
    a["baseline_success_count"] += row["baseline_success"]
    a["baseline_failure_count"] += not row["baseline_success"]
    a["damage_count"] += row["damage"]
    a["repair_count"] += row["repair"]
    a["baseline_iou_sum"] += row["baseline_mask_iou"]
    a["method_iou_sum"] += row["method_mask_iou"]
    a["mask_iou_delta_sum"] += row["delta_mask_iou"]
    a["baseline_success_delta_sum"] += row["delta_mask_iou"] if row["baseline_success"] else 0.


def pair_finish(a):
    return a | {"baseline_mean_mask_iou": divide(a["baseline_iou_sum"], a["matched_detection_count"]),
        "method_mean_mask_iou": divide(a["method_iou_sum"], a["matched_detection_count"]),
        "mean_mask_iou_delta": divide(a["mask_iou_delta_sum"], a["matched_detection_count"]),
        "baseline_success_mean_iou_delta": divide(a["baseline_success_delta_sum"], a["baseline_success_count"]),
        "damage_rate_of_baseline_success": divide(a["damage_count"], a["baseline_success_count"]),
        "repair_rate_of_baseline_failure": divide(a["repair_count"], a["baseline_failure_count"])}


def load_fixed_source(path, lock, image_ids, engineering):
    path = path.resolve()
    directory = path if path.is_dir() else path.parent
    instances = directory/"INSTANCES.jsonl" if path.is_dir() else path
    paired_summary = directory/"SUMMARY.json"
    final_summary = directory.parent/"SUMMARY.json"
    for file in (instances, paired_summary, final_summary):
        lock.add(file)
    paired, final = read_json(paired_summary), read_json(final_summary)
    check(paired.get("paired_readout_valid") is True and final.get("evaluation_epoch") == 8 and final.get("image_count") == 5000,
          "Only verified fixed final8 associations may be used")
    check(final.get("paired", {}).get("triflow") == paired and final.get("head_provenance", {}).get("loaded_state_sha256") == FINAL8_HEAD_STATE_SHA,
          "Source paired table does not bind the fixed final8 module")
    grouped, keys, all_counts = defaultdict(list), set(), Counter()
    supplied = set(image_ids)
    for row in jsonl(instances):
        key = row["image_id"], row["detection_index"]
        check(key not in keys, "Fixed source association detection identity duplicated")
        keys.add(key)
        success = row["baseline_mask_iou"] >= .75
        check(row["baseline_success"] is success, "Fixed source baseline success definition differs")
        all_counts["matched_detection_count"] += 1
        all_counts["baseline_success_count"] += success
        all_counts["baseline_failure_count"] += not success
        if row["image_id"] in supplied:
            grouped[row["image_id"]].append(row)
    for key, value in all_counts.items():
        check(value == paired["statistics"][key], "Original fixed source total differs: "+key)
    if not engineering:
        check(set(image_ids) == set(final["configuration"]["image_ids"]) and len(keys) == paired["statistics"]["matched_detection_count"], "Formal fixed-source complete image/cardinality mismatch")
    return grouped, {"directory": str(directory), "instances_path": str(instances), "instances_sha256": sha(instances), "paired_summary_sha256": sha(paired_summary),
        "final_summary_sha256": sha(final_summary), "definition": paired["definition"], "full_source_statistics": paired["statistics"],
        "final8_head_state_sha256": FINAL8_HEAD_STATE_SHA, "fixed_final8_metrics": final["metrics"],
        "selected_fixed_rows": sum(len(rows) for rows in grouped.values()), "engineering_subset": engineering}


def coco_metrics(coco, predictions_path, ids, out, kind, COCO, COCOeval, np):
    began = time.perf_counter()
    predictions = [{key: row[key] for key in COCO_KEYS} for row in json_array(predictions_path)]
    if kind == "segm":
        for row in predictions:
            row.pop("bbox")
    else:
        for row in predictions:
            row.pop("segmentation")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        if predictions:
            detected = coco.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {"images": list(coco.imgs.values()), "categories": list(coco.cats.values()), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(coco, detected, kind)
        evaluator.params.imgIds = sorted(ids)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    out.mkdir(parents=True, exist_ok=True)
    (out/f"COCO_{kind}.txt").write_text(stream.getvalue(), encoding="utf-8")
    np.savez_compressed(out/f"COCO_{kind}_ACCUMULATED.npz", precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
        scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs, recall_thresholds=evaluator.params.recThrs,
        category_ids=evaluator.params.catIds, max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
    result = {key: float(value) if value >= 0 else None for key, value in zip(METRIC_NAMES, evaluator.stats)}
    print("COCO_CPU", kind, out.name, json.dumps(result), flush=True)
    return result, time.perf_counter()-began


def compute_fixed_pairs(iid, rows_by_arm, source_rows, coco, mask_utils, outputs, totals, gt_labels):
    grouped = defaultdict(list)
    for row in source_rows:
        index, annid = row["detection_index"], row["annotation_id"]
        check(0 <= index < len(rows_by_arm["baseline"]), "Fixed association candidate index absent")
        baseline = rows_by_arm["baseline"][index]
        check(baseline["category_id"] == row["category_id"] and ("raw_confidence" not in baseline or baseline["raw_confidence"] == row["baseline_score"]), "Fixed association class/rawscore identity differs")
        ann = coco.anns.get(annid)
        check(ann is not None and ann["image_id"] == iid and ann["category_id"] == row["category_id"] and not ann.get("iscrowd", 0), "Original fixed noncrowd annotation identity differs")
        grouped[annid].append(row)
    image = {arm: new_pair_acc() for arm in ARMS}
    for annid, fixed in grouped.items():
        gt = coco.annToRLE(coco.anns[annid])
        indices = [row["detection_index"] for row in fixed]
        baseline_values = mask_utils.iou([encoded_rle(rows_by_arm["baseline"][i]["segmentation"]) for i in indices], [gt], [0])[:, 0]
        for source, actual in zip(fixed, baseline_values):
            check(math.isclose(float(actual), source["baseline_mask_iou"], abs_tol=1e-14, rel_tol=1e-14), "Recomputed baseline maskIoU differs from original fixed source")
        for arm in ARMS:
            values = baseline_values if arm == "baseline" else mask_utils.iou([encoded_rle(rows_by_arm[arm][i]["segmentation"]) for i in indices], [gt], [0])[:, 0]
            association = {"image_id": iid, "annotation_id": annid, "category_id": fixed[0]["category_id"],
                "associated_candidates": len(fixed), "baseline_success_candidates": 0, "baseline_failure_candidates": 0, "damaged_candidates": 0, "repaired_candidates": 0,
                "detection_indices": indices}
            for source, b, m in zip(fixed, baseline_values, values):
                b, m = float(b), float(m)
                success, damage, repair = b >= .75, b >= .75 and m < .75, b < .75 and m >= .75
                row = {"image_id": iid, "detection_index": source["detection_index"], "annotation_id": annid, "category_id": source["category_id"],
                    "baseline_mask_iou": b, "method_mask_iou": m, "delta_mask_iou": m-b, "baseline_success": success, "damage": damage, "repair": repair,
                    "association": "unchanged final8 source (image_id,detection_index,annotation_id); no GT rematching"}
                outputs[arm]["instances"].write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                pair_add(totals[arm], row); pair_add(image[arm], row)
                association["baseline_success_candidates"] += success
                association["baseline_failure_candidates"] += not success
                association["damaged_candidates"] += damage
                association["repaired_candidates"] += repair
            association.update(any_damage_label=association["damaged_candidates"] > 0, all_candidates_damage_label=association["damaged_candidates"] == len(fixed),
                any_repair_label=association["repaired_candidates"] > 0, all_candidates_repair_label=association["repaired_candidates"] == len(fixed),
                duplicate_candidate_association=len(fixed) > 1)
            outputs[arm]["gt"].write(json.dumps(association, separators=(",", ":"), allow_nan=False)+"\n")
            for key in ("any_damage_label", "all_candidates_damage_label", "any_repair_label", "all_candidates_repair_label", "duplicate_candidate_association"):
                gt_labels[arm][key] += association[key]
            gt_labels[arm]["unique_associated_gt_count"] += 1
    for arm in ARMS:
        outputs[arm]["images"].write(json.dumps({"image_id": iid, **pair_finish(image[arm])}, separators=(",", ":"), allow_nan=False)+"\n")


def score(args):
    run, reference, out = args.run.resolve(), args.reference_baseline.resolve(), args.out_dir.resolve()
    check(run != out and out not in run.parents and run not in out.parents, "Scoring must use its own output Run")
    out.mkdir(parents=True, exist_ok=True)
    check(not any((out/name).exists() for name in ("SUMMARY.json", "SOURCE_LOCK.json", "IDENTITY_IMAGES.jsonl")), "Scoring history is immutable; use a new Run")
    lock = InputLock()
    for path in (run/"run.json", run/"SUMMARY.json", run/"EVALUATION_COMPLETE.json", run/"EVALUATION_INPUTS.json", reference/"COMPLETE.json", run/"NATIVE_DECISIONS.jsonl", args.annotations):
        lock.add(path)
    source = read_json(run/"SUMMARY.json")
    actual_execution = read_json(run/"run.json")
    check(actual_execution.get("status") == "completed" and actual_execution.get("return_code") == 0 and actual_execution.get("artifact_completeness") == "complete", "Inference process must actually exit0 with complete recorded artifacts before CPU scoring")
    check(source.get("status") == "prediction_complete", "Inference Run is not prediction-complete")
    terminal = read_json(run/"EVALUATION_COMPLETE.json")
    check(terminal.get("status") == "prediction_complete" and terminal.get("summary_sha256") == sha(run/"SUMMARY.json") and source.get("passed") is True,
        "Actual prediction terminal receipt is missing or stale")
    engineering = source.get("engineering")
    check(type(engineering) is bool, "Inference engineering scope must be explicit")
    check(not args.engineering_identity_only or engineering, "Identity-only scoring is limited to a declared engineering Run")
    config = source.get("configuration") or read_json(run/"baseline/COMPLETE.json").get("fingerprint")
    input_config = read_json(run/"EVALUATION_INPUTS.json")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    check(config == input_config.get("configuration") and config_sha == source.get("configuration_sha256") == input_config.get("configuration_sha256"), "Sealed actual inference configuration differs")
    protocol_snapshot = run/"snapshots"/config["protocol_sha256"]
    lock.add(protocol_snapshot)
    check(sha(protocol_snapshot) == config["protocol_sha256"], "Actual sealed inference protocol snapshot differs")
    ids = config.get("image_ids")
    check(isinstance(ids, list) and ids and len(ids) == len(set(ids)), "Actual inference image identities missing/duplicated")
    check(len(ids) == 5000 if not engineering else len(ids) <= 64, "Formal/engineering image scope differs from fixed protocol")
    check(sha(args.annotations) == ANNOTATIONS_SHA, "Original COCO val2017 annotation bytes required")
    reference_receipt = read_json(reference/"COMPLETE.json")
    check(reference_receipt.get("status") == "prediction_complete" and reference_receipt.get("image_count") == 5000, "Reference official cache is not complete5000")
    check(source.get("arm_names") == list(ARMS), "Seven fixed arms must exist in their declared order")
    sources = out/"source"
    sources.mkdir(exist_ok=True)
    executed_sources = {}
    for name in ("score_and_verify_comparison.py", "comparison_io.py"):
        path = Path(__file__).with_name(name)
        executed_sources[name] = sha(path)
        target = sources/name
        check(not target.exists() or sha(target) == executed_sources[name], "Executed score source differs from archived source")
        if not target.exists():
            shutil.copy2(path, target)
        lock.add(path)
    for path in (run/"source").glob("*.py"):
        lock.add(path)
    check(source.get("source_unchanged") is True, "Actual native execution source immutability was not established")
    for name, expected_hash in source.get("source_sha256", {}).items():
        check(lock.add(run/"source"/name) == expected_hash, "Sealed native inference source snapshot differs: "+name)
    for name in ("BASELINE_PARITY.json", "BASELINE_PARITY_IMAGES.jsonl"):
        lock.add(run/name)
    recorded_parity = read_json(run/"BASELINE_PARITY.json")
    check(recorded_parity.get("passed") is True or recorded_parity.get("status") == "passed", "Recorded native baseline parity did not pass")
    readers = {arm: ArmReader(run/arm, lock, ids) for arm in ARMS}
    for arm, reader in readers.items():
        check(reader.receipt.get("fingerprint") == config and reader.receipt.get("baseline_parity_receipt_sha256") == sha(run/"BASELINE_PARITY.json"), "Arm actual configuration/parity receipt differs")
        declared = source.get("prediction_receipts", {}).get(arm, {})
        check(declared.get("predictions_sha256") == reader.predictions_sha256 and declared.get("complete_sha256") == sha(run/arm/"COMPLETE.json"), "Arm source SUMMARY does not bind actual prediction/receipt bytes")
    fixed, fixed_source = load_fixed_source(args.source_paired, lock, ids, engineering)
    provenance = out/"source_provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(protocol_snapshot, provenance/"PROTOCOL.md")
    for origin, name in ((run/"run.json", "INFERENCE_RUN.json"), (run/"SUMMARY.json", "INFERENCE_SUMMARY.json"), (run/"EVALUATION_COMPLETE.json", "INFERENCE_COMPLETE.json"),
        (run/"EVALUATION_INPUTS.json", "INFERENCE_INPUTS.json"), (run/"BASELINE_PARITY.json", "NATIVE_BASELINE_PARITY.json"),
        (reference/"COMPLETE.json", "REFERENCE_COMPLETE.json"), (Path(fixed_source["directory"])/"SUMMARY.json", "FIXED_PAIRED_SOURCE_SUMMARY.json"),
        (Path(fixed_source["directory"]).parent/"SUMMARY.json", "FINAL8_SOURCE_SUMMARY.json")):
        shutil.copy2(origin, provenance/name)
    for arm in ARMS:
        shutil.copy2(run/arm/"COMPLETE.json", provenance/(arm+"_COMPLETE.json"))
    shutil.copy2(fixed_source["instances_path"], provenance/"FIXED_INSTANCES_SOURCE.jsonl")
    np, mask_utils, coco, COCO, COCOeval = None, None, None, None, None
    packages = {"python": sys.version, "executable": sys.executable, "numpy": None, "pycocotools": None, "torch_imported": "torch" in sys.modules}
    if not args.engineering_identity_only:
        try:
            import numpy as np
            from pycocotools import mask as mask_utils
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as error:
            raise RuntimeError("Formal CPU scoring requires existing numpy and pycocotools; no substitute or network install is used") from error
        try:
            coco_package_version = importlib.metadata.version("pycocotools")
        except importlib.metadata.PackageNotFoundError:
            coco_package_version = "unknown_metadata; actual module bytes fingerprinted"
        packages.update(numpy=np.__version__, pycocotools=coco_package_version)
        import pycocotools.coco as coco_source
        import pycocotools.cocoeval as eval_source
        import pycocotools.mask as mask_source
        import pycocotools._mask as mask_binary
        for module in (coco_source, eval_source, mask_source, mask_binary):
            lock.add(module.__file__)
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(args.annotations.resolve()))
        check(set(ids) == set(coco.imgs) if not engineering else set(ids).issubset(coco.imgs), "COCO GT image scope differs")
    write_json(out/"SCORE_INPUTS.json", {"version": VERSION, "inference_run": str(run), "reference_baseline": str(reference),
        "annotations": str(args.annotations.resolve()), "fixed_source": fixed_source, "engineering": engineering,
        "identity_only": args.engineering_identity_only, "image_ids": ids, "source_sha256": executed_sources, "packages": packages,
        "GPU_or_model_execution": False, "association_rematched": False, "started_at": now()})
    totals, gt_labels = {arm: new_pair_acc() for arm in ARMS}, {arm: Counter() for arm in ARMS}
    identity_counts = {arm: Counter() for arm in ARMS}
    empty_indices_sha = {arm: hashlib.sha256() for arm in ARMS}
    baseline_chain, reference_chain = [], []
    decisions = iter(jsonl(run/"NATIVE_DECISIONS.jsonl"))
    began = time.perf_counter()
    with contextlib.ExitStack() as stack:
        identity_output = stack.enter_context((out/"IDENTITY_IMAGES.jsonl").open("x", encoding="utf-8"))
        pair_outputs = {}
        if not args.engineering_identity_only:
            for arm in ARMS:
                directory = out/"paired"/arm
                directory.mkdir(parents=True, exist_ok=True)
                pair_outputs[arm] = {key: stack.enter_context((directory/name).open("x", encoding="utf-8"))
                    for key, name in (("instances", "INSTANCES.jsonl"), ("images", "IMAGES.jsonl"), ("gt", "GT_LABELS.jsonl"))}
        for position, iid in enumerate(ids):
            decision = next(decisions, None)
            check(decision is not None and decision["image_id"] == iid and decision.get("tau0_native_exact") is True
                and decision.get("all_arm_native_identity_exact") is True and decision.get("gt_used") is False, "Recorded native decision/identity scope differs")
            reference_path = reference/"images"/f"{iid:012d}.json"
            data = reference_path.read_bytes()
            reference_chain.append(lock.add(reference_path, data))
            reference_rows = json.loads(data)["detections"]
            rows = {arm: readers[arm].read(iid, len(reference_rows)) for arm in ARMS}
            check(decision["native_rows"] == len(reference_rows) == len(decision["decisions"]), "Native row or empty-preserving candidate count differs")
            image_counts = {arm: Counter() for arm in ARMS}
            baseline_chain.append(hashlib.sha256(json.dumps([{key: row[key] for key in COCO_KEYS} for row in rows["baseline"]], sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            for index, original in enumerate(reference_rows):
                check(original["detection_index"] == index and original["image_id"] == iid, "Official reference explicit ordinal differs")
                baseline = rows["baseline"][index]
                check(all(original[k] == baseline[k] for k in COCO_KEYS), "Baseline official cache COCO identity/RLE differs")
                for key in IDENTITY_KEYS:
                    if key in baseline:
                        check(baseline[key] == original[key], "Baseline official cache native field differs: "+key)
                detail = decision["decisions"][index]
                check(detail["detection_index"] == index and type(detail["native_output_row"]) is int and type(detail["raw_index"]) is int, "Recorded native raw/index bridge differs")
                check(detail["in_first64"] is (index < 64) and detail["first64_proto_supported"] is (index < 64 and detail["proto_roi_supported"]), "first64 unsupported domain changed")
                areas = {}
                for arm in ARMS:
                    row = rows[arm][index]
                    check(all(row[k] == baseline[k] for k in ("image_id", "category_id", "bbox", "score")), "Arm output class/score/bbox/order changed: "+arm)
                    for key in IDENTITY_KEYS:
                        if key in row and key in baseline:
                            check(row[key] == baseline[key], "Arm native field changed: "+arm+"/"+key)
                    check(row["segmentation"]["size"] == original["segmentation"]["size"], "Arm mask grid changed")
                    area = rle_area(row["segmentation"], mask_utils)
                    areas[arm] = area
                    identity_counts[arm]["candidate_count"] += 1
                    image_counts[arm]["candidate_count"] += 1
                    identity_counts[arm]["empty_mask_count"] += area == 0
                    image_counts[arm]["empty_mask_count"] += area == 0
                    if area == 0:
                        empty_indices_sha[arm].update(f"{iid}:{index}\n".encode())
                check(detail["baseline_empty"] is (areas["baseline"] == 0) and detail["global025_empty"] is (areas["global_minus025_full"] == 0), "Recorded empty identity differs")
                check(detail["baseline_area"] == areas["baseline"], "RCMC original-grid baseline area differs")
                check(math.isclose(detail["smooth_tau"], .75/(1+(areas["baseline"]/2304)**2), abs_tol=1e-7, rel_tol=1e-6), "Frozen smooth tau formula changed")
                availability, gains, choice = detail["trial_nonempty"], detail["multi_gains"], detail["multi_choice"]
                check(len(availability) == len(gains) == 5 and all(type(v) is bool for v in availability) and all(math.isfinite(v) for v in gains), "Fixed five-action eligibility/gains missing")
                expected_choice = max(range(5), key=lambda j: gains[j] if availability[j] else -math.inf)
                if not availability[expected_choice] or gains[expected_choice] <= 0:
                    expected_choice = -1
                check(choice == expected_choice and detail["response_chosen"] is (detail["response_gain"] > 0 and availability[0]), "Frozen action argmax/gain rule differs")
                for arm in ("RCMC_full", "RCMC_first64", "multi_local_full", "multi_local_first64"):
                    check(areas["baseline"] == 0 or areas[arm] > 0, "Gate produced a forbidden new empty output: "+arm)
                for arm in ARMS:
                    check(areas[arm] <= areas["baseline"], "Tightening action expanded output area")
                if not detail["response_chosen"]:
                    check(rows["RCMC_full"][index]["segmentation"] == baseline["segmentation"], "Unchosen RCMC output changed")
                if choice < 0:
                    check(rows["multi_local_full"][index]["segmentation"] == baseline["segmentation"], "Unchosen multi_local output changed")
                for limited, full in (("RCMC_first64", "RCMC_full"), ("global_minus025_first64", "global_minus025_full"), ("multi_local_first64", "multi_local_full")):
                    target = rows[full][index] if detail["first64_proto_supported"] else baseline
                    check(rows[limited][index]["segmentation"] == target["segmentation"], "first64 arm replaced unsupported/outside rows or differs from full action")
            check(decision["selected_first64"] == min(64, len(reference_rows)) and decision["supported_first64"] == sum(d["first64_proto_supported"] for d in decision["decisions"])
                and decision["unsupported_first64"] == min(64, len(reference_rows))-decision["supported_first64"], "Declared first64 supported/unsupported counts differ")
            identity_output.write(json.dumps({"image_id": iid, "candidate_count": len(reference_rows), "all_arms_identity_exact": True,
                "official_baseline_rle_exact": True, "all_empty_ordinals_retained": True, "first64_domain_exact": True,
                "arm_counts": {arm: dict(image_counts[arm]) for arm in ARMS}, "reference_image_sha256": sha(reference_path)}, separators=(",", ":"))+"\n")
            if not args.engineering_identity_only:
                compute_fixed_pairs(iid, rows, fixed.get(iid, []), coco, mask_utils, pair_outputs, totals, gt_labels)
            if position == 0 or (position+1) % 100 == 0:
                print(f"CPU_IDENTITIES images={position+1}/{len(ids)} candidates={identity_counts['baseline']['candidate_count']}", flush=True)
    check(next(decisions, None) is None, "Unexpected native decision images beyond fixed scope")
    for reader in readers.values():
        reader.finish(len(ids))
    identity = {"status": "passed", "passed": True, "images": len(ids), "arm_counts": {arm: dict(identity_counts[arm]) for arm in ARMS},
        "empty_ordinal_sha256": {arm: h.hexdigest() for arm, h in empty_indices_sha.items()}, "all_arms_box_class_score_count_order_exact": True,
        "all_official_baseline_masks_exact": True, "all_empty_candidate_ordinals_retained": True, "all_first64_unsupported_invariants_passed": True,
        "arm_source_views": {arm: reader.mode for arm, reader in readers.items()}, "baseline_reference_image_sha256_chain": hashlib.sha256("".join(reference_chain).encode()).hexdigest(),
        "native_fields_scope": {arm: "explicit per-image native fields crosschecked" if reader.mode == "per_image_native_fields" else "five-field COCO ordinal checked; independent raw confidence/model class/exact coordinates unavailable; recorded native parity only" for arm, reader in readers.items()},
        "baseline_standard_COCO_rows_sha256_chain": hashlib.sha256("".join(baseline_chain).encode()).hexdigest(),
        "scope": "Independent CPU exported identity/mask/action-domain checks; native forward parity retained from original Run receipt",
        "source_native_parity_receipt_sha256": sha(run/"BASELINE_PARITY.json"), "seconds": time.perf_counter()-began}
    check(identity_counts["baseline"]["candidate_count"] == source["native_rows"] == recorded_parity["native_rows"], "Actual native total differs from source SUMMARY/parity")
    write_json(out/"IDENTITY_VERIFICATION.json", identity)
    metrics, pair_metrics, metric_seconds = {}, {}, {}
    if args.engineering_identity_only:
        metrics = {arm: {"segm": None, "bbox": None, "status": "unknown_engineering_identity_only"} for arm in ARMS}
        pair_metrics = {arm: None for arm in ARMS}
    else:
        for arm in ARMS:
            metrics[arm] = {"predictions_sha256": readers[arm].predictions_sha256, "image_ids": sorted(ids), "units": "fraction, AP report layer multiplies100"}
            metrics[arm]["segm"], metric_seconds[arm+"_segm"] = coco_metrics(coco, run/arm/"predictions.json", ids, out/arm, "segm", COCO, COCOeval, np)
        metrics["baseline"]["bbox"], metric_seconds["baseline_bbox"] = coco_metrics(coco, run/"baseline/predictions.json", ids, out/"baseline", "bbox", COCO, COCOeval, np)
        for arm in ARMS:
            if arm != "baseline":
                metrics[arm]["bbox"] = dict(metrics["baseline"]["bbox"])
            metrics[arm]["bbox_evaluated_this_arm"] = arm == "baseline"
            metrics[arm]["bbox_identity_equal_baseline"] = True
            write_json(out/arm/"COCO_METRICS.json", metrics[arm])
            expected_count = fixed_source["selected_fixed_rows"]
            check(totals[arm]["matched_detection_count"] == expected_count, "Fixed association count changed: "+arm)
            pair_metrics[arm] = {"statistics": pair_finish(totals[arm]), "unique_GT_association_labels": dict(gt_labels[arm]),
                "association_source_sha256": fixed_source["instances_sha256"], "definition": fixed_source["definition"], "confidence_intervals": None,
                "GT_label_scope": "candidate-associated GT any/all labels, not candidate rates renamed as GT rates"}
            write_json(out/"paired"/arm/"SUMMARY.json", pair_metrics[arm])
        if not engineering:
            for key in ("matched_detection_count", "baseline_success_count", "baseline_failure_count"):
                check(totals["baseline"][key] == fixed_source["full_source_statistics"][key], "Full final8 fixed baseline denominator differs")
        check(totals["baseline"]["damage_count"] == totals["baseline"]["repair_count"] == 0 and totals["baseline"]["mask_iou_delta_sum"] == 0, "Baseline zero-action paired invariant failed")
        reference_metrics = reference/"COCO_METRICS.json"
        if not engineering and reference_metrics.is_file():
            lock.add(reference_metrics)
            old_metrics = read_json(reference_metrics)
            for kind in ("segm", "bbox"):
                for key in METRIC_NAMES:
                    actual, expected = metrics["baseline"][kind][key], old_metrics[kind][key]
                    check(actual is None and expected is None or actual is not None and expected is not None and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), "Actual recomputed baseline COCO metric differs from verified official cache")
    source_lock = {"files": lock.finish(), "all_before_after_sha256_equal": True, "executed_source_sha256": executed_sources}
    write_json(out/"SOURCE_LOCK.json", source_lock)
    result = {"version": VERSION, "status": "engineering_identity_verified_metrics_unknown" if args.engineering_identity_only else "complete",
        "engineering": engineering, "formal_5000": not engineering, "image_count": len(ids), "arms": list(ARMS), "metrics": metrics, "fixed_paired": pair_metrics,
        "identity": identity, "fixed_source": fixed_source, "source_lock_sha256": sha(out/"SOURCE_LOCK.json"), "inference_run": str(run),
        "metric_units": "AP and AR fraction0..1; AP points only in report layer", "confidence_intervals": {"AP": None, "fixed_paired": None},
        "source_inference_configuration_sha256": config_sha, "source_protocol_sha256": config["protocol_sha256"],
        "native_forward_replayed": False, "association_rematched": False, "GPU_or_model_execution": False, "packages": packages,
        "timing_seconds": {"identity_and_fixed_pairs": identity["seconds"], "COCOeval": metric_seconds}, "completed_at": now()}
    write_json(out/"SUMMARY.json", result)
    artifacts = {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file() and path.name not in ("run.json", "stdout.log", "SOURCE.json", "PROCESS.json", "SCORE_COMPLETE.json") and "source" not in path.relative_to(out).parts}
    write_json(out/"SCORE_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out/"SUMMARY.json"), "artifact_sha256": artifacts,
        "formal_metrics_measured": not args.engineering_identity_only, "identity_passed": True})
    print(json.dumps({"status": result["status"], "image_count": len(ids), "identity_passed": True, "metrics_unknown": args.engineering_identity_only}), flush=True)
    return result


def npz_metrics(path, np):
    with np.load(path, allow_pickle=False) as data:
        p, r = data["precision"], data["recall"]
        check(p.ndim == 5 and r.ndim == 4 and p.shape[0] == r.shape[0] == 10 and p.shape[1] == 101 and p.shape[3:] == (4, 3), "Standard COCO accumulated tensor dimensions differ")
        check(np.array_equal(data["max_detections"], [1, 10, 100]) and np.allclose(data["iou_thresholds"], np.linspace(.5, .95, 10), rtol=0, atol=1e-12)
            and np.allclose(data["recall_thresholds"], np.linspace(0, 1, 101), rtol=0, atol=1e-12), "Standard COCO threshold/maxDets metadata differs")
        check(np.isfinite(p).all() and np.isfinite(r).all(), "COCO accumulated arrays nonfinite")
        check(np.array_equal(data["area_ranges"], [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]), "Standard COCO area ranges differ")
        def mean(values):
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else None
        ap = [mean(p[:, :, :, 0, 2]), mean(p[0, :, :, 0, 2]), mean(p[5, :, :, 0, 2])]
        ap += [mean(p[:, :, :, area, 2]) for area in (1, 2, 3)]
        ar = [mean(r[:, :, 0, maximum]) for maximum in (0, 1, 2)]
        ar += [mean(r[:, :, area, 2]) for area in (1, 2, 3)]
        return dict(zip(METRIC_NAMES, ap+ar))


def equal_value(actual, expected, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), "Aggregate dictionary differs: "+label)
        for key in expected:
            equal_value(actual[key], expected[key], label+"."+key)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        check(type(actual) in (float, int) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-11, rel_tol=1e-11), "Aggregate numeric value differs: "+label)
    else:
        check(type(actual) is type(expected) and actual == expected, "Aggregate value differs: "+label)


def verify_scoring(run, out):
    """No GT/COCO matching replay: validate saved arrays, fixed rows and provenance."""
    run, out = run.resolve(), out.resolve()
    check(run != out and run not in out.parents, "Independent verification requires a separate Run")
    out.mkdir(parents=True, exist_ok=True)
    check(not (out/"VERIFICATION.json").exists(), "Independent verification history is immutable")
    summary, complete, inputs, lock = (read_json(run/name) for name in ("SUMMARY.json", "SCORE_COMPLETE.json", "SCORE_INPUTS.json", "SOURCE_LOCK.json"))
    check(complete.get("status") == "completed" and sha(run/"SUMMARY.json") == complete["summary_sha256"] and sha(run/"SOURCE_LOCK.json") == summary["source_lock_sha256"], "Saved scoring summary/source lock not bound")
    for relative, expected_hash in complete["artifact_sha256"].items():
        check(sha(run/relative) == expected_hash, "Scoring artifact bytes changed: "+relative)
    check(summary.get("GPU_or_model_execution") is False and summary.get("association_rematched") is False and summary.get("native_forward_replayed") is False, "Scoring scope inflated")
    check(lock.get("all_before_after_sha256_equal") is True, "Actual scorer did not establish before/after input identity")
    available, unavailable = [], []
    for path, evidence in lock["files"].items():
        check(evidence["sha256_before"] == evidence["sha256_after"], "Original scored input changed during scoring")
        if Path(path).is_file():
            check(sha(path) == evidence["sha256_before"], "Available original scored source bytes changed: "+path)
            available.append(path)
        else:
            unavailable.append(path)
    for name, original in lock["executed_source_sha256"].items():
        check(sha(run/"source"/name) == original, "Actual CPU scorer source archive differs")
    check(sha(Path(__file__)) == lock["executed_source_sha256"]["score_and_verify_comparison.py"]
        and sha(Path(__file__).with_name("comparison_io.py")) == lock["executed_source_sha256"]["comparison_io.py"], "Independent reconstruction uses different declared scoring/helper versions")
    inference = read_json(run/"source_provenance/INFERENCE_SUMMARY.json")
    inference_run = read_json(run/"source_provenance/INFERENCE_RUN.json")
    check(inference_run.get("status") == "completed" and inference_run.get("return_code") == 0 and inference_run.get("artifact_completeness") == "complete", "Scoring source process completion is not established")
    inferred_complete = read_json(run/"source_provenance/INFERENCE_COMPLETE.json")
    inference_inputs = read_json(run/"source_provenance/INFERENCE_INPUTS.json")
    check(sha(run/"source_provenance/PROTOCOL.md") == inference_inputs["configuration"]["protocol_sha256"] == summary["source_protocol_sha256"], "Actual fixed protocol source snapshot differs")
    native_parity = read_json(run/"source_provenance/NATIVE_BASELINE_PARITY.json")
    original_fixed = read_json(run/"source_provenance/FIXED_PAIRED_SOURCE_SUMMARY.json")
    final8 = read_json(run/"source_provenance/FINAL8_SOURCE_SUMMARY.json")
    check(sha(run/"source_provenance/FIXED_INSTANCES_SOURCE.jsonl") == summary["fixed_source"]["instances_sha256"], "Actual original fixed association source bytes not bound")
    fixed_original = {}
    supplied_ids = set(inputs["image_ids"])
    for row in jsonl(run/"source_provenance/FIXED_INSTANCES_SOURCE.jsonl"):
        if row["image_id"] in supplied_ids:
            key = row["image_id"], row["detection_index"]
            check(key not in fixed_original, "Original fixed candidate identity duplicated")
            fixed_original[key] = row
    check(inference["status"] == inferred_complete["status"] == "prediction_complete" and inferred_complete["summary_sha256"] == sha(run/"source_provenance/INFERENCE_SUMMARY.json"), "Sealed original inference terminal receipt differs")
    ids = inference_inputs["configuration"]["image_ids"]
    check(ids == inputs["image_ids"] and len(ids) == summary["image_count"] and len(ids) == len(set(ids)), "Scoring image identity scope differs")
    check(set(summary["arms"]) == set(ARMS) and inference["arm_names"] == list(ARMS), "Seven fixed arm identities differ")
    check(native_parity.get("passed") is True and sha(run/"source_provenance/NATIVE_BASELINE_PARITY.json") == summary["identity"]["source_native_parity_receipt_sha256"], "Recorded native source parity not bound")
    check(final8["paired"]["triflow"] == original_fixed and final8["head_provenance"]["loaded_state_sha256"] == FINAL8_HEAD_STATE_SHA, "Fixed final8 source identity differs")
    image_counts = {arm: Counter() for arm in ARMS}
    image_ids, image_reference_hashes = set(), []
    for row in jsonl(run/"IDENTITY_IMAGES.jsonl"):
        iid = row["image_id"]
        check(iid not in image_ids and iid in ids and all(row[key] is True for key in ("all_arms_identity_exact", "official_baseline_rle_exact", "all_empty_ordinals_retained", "first64_domain_exact")), "Actual exported identity audit row differs")
        image_ids.add(iid)
        image_reference_hashes.append(row["reference_image_sha256"])
        for arm in ARMS:
            check(row["arm_counts"][arm]["candidate_count"] == row["candidate_count"] and 0 <= row["arm_counts"][arm]["empty_mask_count"] <= row["candidate_count"], "Empty-preserving arm cardinality differs")
            image_counts[arm].update(row["arm_counts"][arm])
    check(image_ids == set(ids), "Identity audit omitted images")
    equal_value(summary["identity"]["arm_counts"], {arm: dict(image_counts[arm]) for arm in ARMS}, "identity_counts")
    check(hashlib.sha256("".join(image_reference_hashes).encode()).hexdigest() == summary["identity"]["baseline_reference_image_sha256_chain"], "Saved baseline reference identity chain differs")
    check(image_counts["baseline"]["candidate_count"] == inference["native_rows"] == native_parity["native_rows"], "Full output count differs from sealed native source")
    metrics_unknown = summary["status"] == "engineering_identity_verified_metrics_unknown"
    ap_recomputed, pair_recomputed = {}, {}
    if metrics_unknown:
        check(summary["engineering"] is True and complete["formal_metrics_measured"] is False and all(summary["metrics"][arm]["segm"] is None and summary["fixed_paired"][arm] is None for arm in ARMS), "Engineering unknown metrics were fabricated")
    else:
        import numpy as np
        for arm in ARMS:
            ap_recomputed[arm] = {"segm": npz_metrics(run/arm/"COCO_segm_ACCUMULATED.npz", np)}
            equal_value(summary["metrics"][arm]["segm"], ap_recomputed[arm]["segm"], arm+".segm")
            equal_value(read_json(run/arm/"COCO_METRICS.json"), summary["metrics"][arm], arm+".COCO_METRICS")
        bbox = npz_metrics(run/"baseline/COCO_bbox_ACCUMULATED.npz", np)
        for arm in ARMS:
            equal_value(summary["metrics"][arm]["bbox"], bbox, arm+".bbox_identity_equal_baseline")
            check(summary["metrics"][arm]["bbox_evaluated_this_arm"] is (arm == "baseline"), "BBox evaluation cadence was misstated")
        common_keys = None
        for arm in ARMS:
            rebuilt, per_image, per_gt, keys = new_pair_acc(), defaultdict(new_pair_acc), {}, set()
            for row in jsonl(run/"paired"/arm/"INSTANCES.jsonl"):
                key = row["image_id"], row["detection_index"], row["annotation_id"]
                detection_key = row["image_id"], row["detection_index"]
                check(detection_key not in keys and row["image_id"] in ids, "Fixed candidate association duplicated/missing")
                original = fixed_original.get(detection_key)
                check(original is not None and original["annotation_id"] == row["annotation_id"] and original["category_id"] == row["category_id"]
                    and math.isclose(original["baseline_mask_iou"], row["baseline_mask_iou"], abs_tol=1e-14, rel_tol=1e-14), "Fixed original candidate-to-GT association or baseline IoU changed")
                keys.add(detection_key)
                b, m, delta = row["baseline_mask_iou"], row["method_mask_iou"], row["delta_mask_iou"]
                check(0 <= b <= 1 and 0 <= m <= 1 and math.isclose(m-b, delta, abs_tol=1e-14), "Saved fixed pair IoU values differ")
                check(row["baseline_success"] is (b >= .75) and row["damage"] is (b >= .75 and m < .75) and row["repair"] is (b < .75 and m >= .75), "Fixed candidate success/damage/repair definitions differ")
                pair_add(rebuilt, row); pair_add(per_image[row["image_id"]], row)
                gkey = row["image_id"], row["annotation_id"]
                g = per_gt.setdefault(gkey, {"image_id": row["image_id"], "annotation_id": row["annotation_id"], "category_id": row["category_id"],
                    "associated_candidates": 0, "baseline_success_candidates": 0, "baseline_failure_candidates": 0,
                    "damaged_candidates": 0, "repaired_candidates": 0, "detection_indices": []})
                g["associated_candidates"] += 1
                g["baseline_success_candidates"] += row["baseline_success"]
                g["baseline_failure_candidates"] += not row["baseline_success"]
                g["damaged_candidates"] += row["damage"]
                g["repaired_candidates"] += row["repair"]
                g["detection_indices"].append(row["detection_index"])
            associations = {(iid, index, g["annotation_id"]) for (iid, _), g in per_gt.items() for index in g["detection_indices"]}
            if common_keys is None:
                common_keys = associations
            else:
                check(associations == common_keys, "Arms used different fixed GT associations")
            image_rows = list(jsonl(run/"paired"/arm/"IMAGES.jsonl"))
            check(len(image_rows) == len(ids) and {row["image_id"] for row in image_rows} == set(ids), "Fixed paired image aggregation scope differs")
            for row in image_rows:
                equal_value(row, {"image_id": row["image_id"], **pair_finish(per_image[row["image_id"]])}, arm+".image_aggregate")
            labels, seen_gt = Counter(), set()
            for row in jsonl(run/"paired"/arm/"GT_LABELS.jsonl"):
                key = row["image_id"], row["annotation_id"]
                check(key in per_gt and key not in seen_gt, "Candidate-associated GT identity duplicated")
                seen_gt.add(key)
                g = per_gt[key]
                g.update(any_damage_label=g["damaged_candidates"] > 0, all_candidates_damage_label=g["damaged_candidates"] == g["associated_candidates"],
                    any_repair_label=g["repaired_candidates"] > 0, all_candidates_repair_label=g["repaired_candidates"] == g["associated_candidates"],
                    duplicate_candidate_association=g["associated_candidates"] > 1)
                equal_value(row, g, arm+".GT_association_label")
                for k in ("any_damage_label", "all_candidates_damage_label", "any_repair_label", "all_candidates_repair_label", "duplicate_candidate_association"):
                    labels[k] += g[k]
                labels["unique_associated_gt_count"] += 1
            check(seen_gt == set(per_gt), "GT association label rows incomplete")
            paired = summary["fixed_paired"][arm]
            check(keys == set(fixed_original), "Fixed source association completeness differs")
            equal_value(paired["statistics"], pair_finish(rebuilt), arm+".fixed_paired")
            equal_value(paired["unique_GT_association_labels"], dict(labels), arm+".GT_labels")
            equal_value(read_json(run/"paired"/arm/"SUMMARY.json"), paired, arm+".paired_summary")
            check(rebuilt["matched_detection_count"] == summary["fixed_source"]["selected_fixed_rows"], "Actual fixed-source row cardinality differs")
            if not summary["engineering"]:
                for key in ("matched_detection_count", "baseline_success_count", "baseline_failure_count"):
                    check(rebuilt[key] == original_fixed["statistics"][key], "Actual original fixed success/failure denominator differs")
                check(len(per_gt) == original_fixed["unique_matched_gt_instances"], "Actual unique associatedGT count differs")
            if arm == "baseline":
                check(rebuilt["damage_count"] == rebuilt["repair_count"] == 0 and rebuilt["mask_iou_delta_sum"] == 0, "Saved zero-action baseline invariant failed")
            pair_recomputed[arm] = pair_finish(rebuilt)
    report = {"version": VERSION, "status": "passed", "passed": True, "scoring_run": str(run), "image_count": len(ids),
        "engineering": summary["engineering"], "AP_recomputed_from_saved_arrays": ap_recomputed if not metrics_unknown else None,
        "fixed_pairs_reaggregated": pair_recomputed if not metrics_unknown else None, "identity_counts_reaggregated": {arm: dict(image_counts[arm]) for arm in ARMS},
        "available_original_files_freshly_hashed": len(available), "unavailable_original_files_not_rehashed": len(unavailable),
        "source_before_after_hash_receipts_verified": True, "source_unavailable_paths": unavailable,
        "summary_sha256": sha(run/"SUMMARY.json"), "COCO_matching_or_GT_mask_replayed": False, "GPU_or_model_execution": False,
        "scope": "Saved accumulated arrays and fixed-line aggregation/provenance reconstruction; no new COCO matching or GT mask recalculation",
        "limitations": ["Exported mask/action/native identity checks are scoring-Run evidence, not repeated native forward",
            "Unavailable original remote inputs are checked through locked receipts and transferred artifact hashes, not freshly re-read"], "verified_at": now()}
    write_json(out/"VERIFICATION.json", report)
    print(json.dumps({"status": "passed", "images": len(ids), "metrics_unknown": metrics_unknown}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--reference-baseline", type=Path)
    parser.add_argument("--source-paired", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engineering-identity-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Independent stdlib/NumPy reconstruction of an existing scoring Run; no pycoco or GT-mask recalculation")
    args = parser.parse_args()
    if not args.verify_only:
        for name in ("annotations", "reference_baseline", "source_paired"):
            if getattr(args, name) is None:
                parser.error("scoring requires --"+name.replace("_", "-"))
    try:
        if args.verify_only:
            verify_scoring(args.run, args.out_dir)
        else:
            score(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir/"SCORING_FAILURE.json", {"status": "failed", "version": VERSION, "error": repr(error),
            "traceback": traceback.format_exc(), "GPU_or_model_execution": False, "time": now()})
        raise


if __name__ == "__main__":
    main()
