"""Independently verify a completed formal seven-arm Multiview Run on CPU.

Uses only the standard library and NumPy. Reconstructs persisted pixel counts,
field-wise candidate/image aggregates and COCO summaries from accumulated
arrays. It does not decode GT/prediction masks, repeat COCO matching, import
the producer, run a model, or treat an output-only cohort as full raw geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np

VERSION = "frozen_native_multiview_independent_verify_v1"
PRODUCER_VERSION = "frozen_native_multiview_v2"
ARMS = ("baseline", "RCMC_full", "global_minus025_full", "RCMC_first64",
        "global_minus025_first64", "multi_local_full", "multi_local_first64")
PIXELS = ("tp", "fp", "fn", "gt_pixel_area", "prediction_pixel_area",
          "mask_iou", "target_coverage", "prediction_purity", "fp_per_gt_area", "fn_per_gt_area")
QUALITY = PIXELS[5:]
FIELDS = PIXELS + tuple("delta_" + name for name in QUALITY)
COHORTS = ("all_fixed_pairs", "baseline_success", "baseline_failure")
COUNTS = ("all_candidate_count", "fixed_pair_count", "unassociated_candidate_count",
          "empty_prediction_count", "empty_paired_prediction_count", "baseline_success_count",
          "baseline_failure_count", "repair_count", "damage_count", "zero_GT_pixel_area_count",
          "undefined_pixel_mask_iou_count")
METRICS = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
           "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
SOURCE_HASHES = {
    "multiview_readout.py": "1076fc8da0de20b20d84b6aee2d871b17e69a39a7cebe41264e00f47f8242172",
    "comparison_io.py": "000ea7e883307ed1383bf005f543f099897ed2b523f8ad3960645049e594cd34",
    "score_and_verify_comparison.py": "d0c0a5e4b4f3ae5236a295cf5255dd93e221278dc391c04c11dadce6b2e5eb8c",
    "gt_prediction_mapping.py": "d8b7916dde76dbf0fc4502d948fa8ff6b946dabedbf61f7a272748a689afaf54"}
VENDOR_HASHES = {
    "boundary_iou/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "boundary_iou/coco_instance_api/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "boundary_iou/coco_instance_api/coco.py": "2fcbf65890836179db481312bb14a31cef808462e2d742f139a8970ee48255a2",
    "boundary_iou/coco_instance_api/cocoeval.py": "18ac1a7dc3c8c038e3ddfd3675ee02ff88a69ed2045c0e2d23bd9738f1818724",
    "boundary_iou/utils/__init__.py": "6cbde9611ad3b83a8f8dcdb00d7a3e80c43a3190ee3e6535f98c2c19dabf4507",
    "boundary_iou/utils/boundary_utils.py": "f67cf8e160523c179f056cd0a6fdf8e2d13c04e12e6d31245d750725cc466d33"}
ANNOTATIONS_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
FINAL8_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
EXPECTED_NATIVE = 555445
EXPECTED_FIXED = 86600
EXPECTED_SUCCESS = 36266
EXPECTED_FAILURE = 50334
EXPECTED_UNIQUE_GT = 33644


def check(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON literal: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


class Fingerprint:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        check(path.is_file(), "Required actual input is unavailable: " + str(path))
        key = str(path)
        if key not in self.files:
            self.files[key] = {"sha256_before": sha(path), "bytes": path.stat().st_size}
        value = self.files[key]["sha256_before"]
        check(expected is None or value == expected, "Actual input hash differs: " + key)
        return value

    def json(self, path, expected=None):
        self.add(path, expected)
        return read_json(path)

    def finish(self):
        for path, value in self.files.items():
            value["sha256_after"] = sha(path)
            check(value["sha256_before"] == value["sha256_after"]
                  and Path(path).stat().st_size == value["bytes"], "Input changed during verification: " + path)
        return {"files": self.files, "all_before_after_sha256_equal": True}


class Rows:
    """Strict JSON-lines iterator with lookahead, including legitimate empty files."""
    def __init__(self, path, stack):
        self.path = Path(path)
        opener = gzip.open if self.path.suffix == ".gz" else Path.open
        self.stream = stack.enter_context(opener(self.path, "rt", encoding="utf-8"))
        self.line = 0
        self.cached = None
        self.ended = False

    def peek(self):
        if self.cached is None and not self.ended:
            line = self.stream.readline()
            if not line:
                self.ended = True
            else:
                self.line += 1
                check(bool(line.strip()), "Blank JSON line: " + str(self.path))
                self.cached = json.loads(line, parse_constant=reject_constant)
                check(type(self.cached) is dict, "JSON line is not an object: " + str(self.path))
        return self.cached

    def take(self):
        value = self.peek()
        check(value is not None, "Premature end of rows: " + str(self.path))
        self.cached = None
        return value

    def finish(self):
        check(self.peek() is None, "Unexpected extra rows: " + str(self.path))


def integer(value, label, minimum=0):
    check(type(value) is int and value >= minimum, "Invalid integer " + label)
    return value


def finite(value, label):
    check(type(value) in (int, float) and math.isfinite(value), "Invalid finite numeric " + label)
    return value


def same(actual, expected, label):
    if isinstance(expected, dict):
        check(type(actual) is dict and set(actual) == set(expected), "Dictionary keys differ: " + label)
        for key, value in expected.items():
            same(actual[key], value, label + "/" + str(key))
    elif isinstance(expected, list):
        check(type(actual) is list and len(actual) == len(expected), "List differs: " + label)
        for index, value in enumerate(expected):
            same(actual[index], value, label + "/" + str(index))
    elif type(expected) is float:
        check(type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Numeric value differs: " + label)
    else:
        check(type(actual) is type(expected) and actual == expected, "Value or type differs: " + label)


class Aggregate:
    """Independent field-wise arithmetic accumulation; no producer imports."""
    def __init__(self):
        self.rows = 0
        self.sums = dict.fromkeys(FIELDS, 0.)
        self.valid = dict.fromkeys(FIELDS, 0)
        self.missing = dict.fromkeys(FIELDS, 0)

    def add(self, record):
        check(all(key in record for key in FIELDS), "Metric field missing from saved record")
        self.rows += 1
        for key in FIELDS:
            value = record[key]
            if value is None:
                self.missing[key] += 1
            else:
                self.sums[key] += finite(value, key)
                self.valid[key] += 1

    def finish(self):
        return {"row_count": self.rows, "fields": {key: {
            "sum": self.sums[key], "valid_count": self.valid[key], "missing_count": self.missing[key],
            "mean": self.sums[key] / self.valid[key] if self.valid[key] else None} for key in FIELDS}}


def cohorts():
    return {name: Aggregate() for name in COHORTS}


def compare_table(actual, expected, label):
    same(actual, expected, label)
    # Integer pixel-count sums are exactly representable at this scope.
    for key in PIXELS[:5]:
        check(actual["fields"][key]["sum"] == expected["fields"][key]["sum"], "Pixel sum differs: " + label + "/" + key)


def verify_recorded_lock(path, lock):
    record = lock.json(path)
    check(record.get("all_before_after_sha256_equal") is True and type(record.get("files")) is dict
          and record["files"], "Recorded source lock absent or unsuccessful: " + str(path))
    for source, evidence in record["files"].items():
        check(type(evidence) is dict and evidence["sha256_before"] == evidence["sha256_after"]
              and re.fullmatch(r"[0-9a-f]{64}", evidence["sha256_before"]) is not None,
              "Recorded before/after source differs: " + source)
        lock.add(source, evidence["sha256_before"])
        check(Path(source).stat().st_size == evidence["bytes"], "Recorded byte count differs: " + source)
    return record


def execution(path, lock):
    value = lock.json(path)
    check(value.get("source_kind") == "runner_observed" and value.get("status") == "completed"
          and value.get("return_code") == 0 and value.get("artifact_completeness") == "complete",
          "Actual runner did not complete successfully: " + str(path))
    for item in value.get("snapshots", []):
        relative = Path(item["snapshot"])
        check(not relative.is_absolute() and ".." not in relative.parts, "Invalid source snapshot path")
        lock.add(Path(path).parent / relative, item["revision"])
    check(value.get("snapshots"), "Actual source snapshots are absent")
    return value


def receipt_artifacts(run, receipt, lock):
    entries = receipt["artifact_sha256"]
    check(type(entries) is dict and entries, "Artifact receipt is empty")
    for relative, digest in entries.items():
        target = (run / relative).resolve()
        check(target.is_relative_to(run) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
              "Invalid sealed artifact path/hash")
        lock.add(target, digest)


def reconstruct_metrics(path, lock):
    lock.add(path)
    with np.load(path, allow_pickle=False) as saved:
        required = {"precision", "recall", "scores", "iou_thresholds", "recall_thresholds",
                    "category_ids", "max_detections", "area_ranges"}
        check(set(saved.files) == required, "Accumulated array fields differ: " + str(path))
        p, r, scores = saved["precision"], saved["recall"], saved["scores"]
        categories = saved["category_ids"]
        check(categories.ndim == 1 and len(categories) == 80 and categories.dtype.kind in "iu"
              and len(np.unique(categories)) == 80 and np.all(categories > 0), "COCO category metadata differs")
        check(p.shape == (10, 101, 80, 4, 3) and r.shape == (10, 80, 4, 3)
              and scores.shape == p.shape, "Standard COCO array dimensions differ")
        check(np.array_equal(saved["max_detections"], [1, 10, 100])
              and np.array_equal(saved["area_ranges"], [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]])
              and np.allclose(saved["iou_thresholds"], np.linspace(.5, .95, 10), rtol=0, atol=1e-12)
              and np.allclose(saved["recall_thresholds"], np.linspace(0, 1, 101), rtol=0, atol=1e-12),
              "Standard COCO threshold/area/maxDets metadata differs")
        for name, values in (("precision", p), ("recall", r), ("scores", scores)):
            check(np.isfinite(values).all() and np.all((values == -1) | ((values >= 0) & (values <= 1))),
                  "Invalid accumulated " + name)
        def mean(values):
            selected = values[values >= 0]
            return float(np.mean(selected)) if selected.size else None
        values = [mean(p[:, :, :, 0, 2]), mean(p[0, :, :, 0, 2]), mean(p[5, :, :, 0, 2])]
        values += [mean(p[:, :, :, area, 2]) for area in (1, 2, 3)]
        values += [mean(r[:, :, 0, maximum]) for maximum in (0, 1, 2)]
        values += [mean(r[:, :, area, 2]) for area in (1, 2, 3)]
        return dict(zip(METRICS, values)), categories.tolist()


def fixed_source(scoring, summary, ids, lock):
    directory = scoring / "source_provenance"
    path = directory / "FIXED_INSTANCES_SOURCE.jsonl"
    lock.add(path, summary["fixed_source"]["instances_sha256"])
    paired = lock.json(directory / "FIXED_PAIRED_SOURCE_SUMMARY.json",
                       summary["fixed_source"]["paired_summary_sha256"])
    final = lock.json(directory / "FINAL8_SOURCE_SUMMARY.json", summary["fixed_source"]["final_summary_sha256"])
    check(final["paired"]["triflow"] == paired and paired["paired_readout_valid"] is True
          and final["head_provenance"]["loaded_state_sha256"] == FINAL8_SHA
          and final["evaluation_epoch"] == 8 and final["image_count"] == 5000
          and final["configuration"]["image_ids"] == ids, "Original final8 fixed source identity differs")
    grouped, keys, unique_gt = defaultdict(dict), set(), set()
    success = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line, parse_constant=reject_constant)
            iid = integer(row["image_id"], "source image_id")
            index = integer(row["detection_index"], "source detection_index")
            annid = integer(row["annotation_id"], "source annotation_id")
            integer(row["category_id"], "source category_id", 1)
            key = (iid, index)
            check(iid in ids and key not in keys, "Original fixed source duplicate/missing image")
            finite(row["baseline_mask_iou"], "original baseline IoU")
            check(0 <= row["baseline_mask_iou"] <= 1
                  and row["baseline_success"] is (row["baseline_mask_iou"] >= .75), "Original baseline label differs")
            keys.add(key)
            unique_gt.add((iid, annid))
            success += row["baseline_success"]
            grouped[iid][index] = row
    check(len(keys) == EXPECTED_FIXED and success == EXPECTED_SUCCESS
          and len(keys) - success == EXPECTED_FAILURE and len(unique_gt) == EXPECTED_UNIQUE_GT,
          "Original fixed candidate/unique GT denominators differ")
    for name, expected in (("matched_detection_count", len(keys)), ("baseline_success_count", success),
                           ("baseline_failure_count", len(keys) - success)):
        check(paired["statistics"][name] == expected, "Original fixed summary count differs: " + name)
    return grouped, len(unique_gt)


def measured_pixel_values(row):
    values = {key: row[key] for key in PIXELS}
    tp, fp, fn, ga, pa = [integer(values[key], key) for key in PIXELS[:5]]
    check(ga == tp + fn and pa == tp + fp, "Saved pixel counts do not close")
    denominator = tp + fp + fn
    expected = {"mask_iou": tp / denominator if denominator else None,
                "target_coverage": tp / ga if ga else None,
                "prediction_purity": tp / pa if pa else None,
                "fp_per_gt_area": fp / ga if ga else None,
                "fn_per_gt_area": fn / ga if ga else None}
    same({key: values[key] for key in QUALITY}, expected, "candidate pixel ratios")
    check(row["empty_prediction"] is (pa == 0) and row["zero_GT_pixel_area"] is (ga == 0)
          and row["pixel_mask_iou_undefined"] is (denominator == 0), "Pixel empty/undefined flags differ")
    return values


def score_pair_rows(stream, iid, sources):
    rows = {}
    for _ in range(len(sources)):
        row = stream.take()
        check(row["image_id"] == iid and row["detection_index"] in sources
              and row["detection_index"] not in rows, "Scorer fixed association order/identity differs")
        original = sources[row["detection_index"]]
        check(row["annotation_id"] == original["annotation_id"] and row["category_id"] == original["category_id"],
              "Scorer changed original fixed annotation/category")
        same(row["baseline_mask_iou"], original["baseline_mask_iou"], "score baseline fixed IoU")
        m, b = finite(row["method_mask_iou"], "score method IoU"), row["baseline_mask_iou"]
        check(0 <= m <= 1 and row["baseline_success"] is (b >= .75)
              and row["damage"] is (b >= .75 and m < .75)
              and row["repair"] is (b < .75 and m >= .75), "Scorer fixed outcome label differs")
        same(row["delta_mask_iou"], m - b, "score fixed IoU delta")
        rows[row["detection_index"]] = row
    check(stream.peek() is None or stream.peek()["image_id"] != iid, "Extra scorer fixed rows in image")
    return rows


def verify_pixels(run, scoring, inputs, summary, pixel_summary, scorer, lock):
    import contextlib
    ids = inputs["image_ids"]
    sources, unique_gt = fixed_source(scoring, summary, ids, lock)
    totals = {arm: Counter() for arm in ARMS}
    candidate = {arm: cohorts() for arm in ARMS}
    image_macro = {arm: cohorts() for arm in ARMS}
    empty_ordinals = {arm: hashlib.sha256() for arm in ARMS}
    zero_images, seen_pairs = [], 0
    inference = Path(inputs["inference_run"]).resolve()
    for path in (inference / "NATIVE_DECISIONS.jsonl", scoring / "IDENTITY_IMAGES.jsonl"):
        lock.add(path)
    with contextlib.ExitStack() as stack:
        streams, image_streams, score_streams = {}, {}, {}
        for arm in ARMS:
            streams[arm] = Rows(run / arm / "CANDIDATES.jsonl.gz", stack)
            image_streams[arm] = Rows(run / arm / "IMAGES.jsonl", stack)
            pair_path = scoring / "paired" / arm / "INSTANCES.jsonl"
            lock.add(pair_path)
            score_streams[arm] = Rows(pair_path, stack)
        identities = Rows(scoring / "IDENTITY_IMAGES.jsonl", stack)
        decisions = Rows(inference / "NATIVE_DECISIONS.jsonl", stack)
        for ordinal, iid in enumerate(ids):
            audit, native = identities.take(), decisions.take()
            check(audit["image_id"] == native["image_id"] == iid
                  and all(audit[key] is True for key in ("all_arms_identity_exact", "official_baseline_rle_exact",
                                                        "all_empty_ordinals_retained", "first64_domain_exact"))
                  and native["tau0_native_exact"] is True and native["all_arm_native_identity_exact"] is True
                  and native["gt_used"] is False, "Recorded image identity/native parity differs")
            count = integer(audit["candidate_count"], "Score image native count")
            check(count == native["native_rows"] == len(native["decisions"]), "Native count bridge differs")
            fixed = sources.get(iid, {})
            check(all(index < count for index in fixed), "Fixed detection absent from native output")
            paired = {arm: score_pair_rows(score_streams[arm], iid, fixed) for arm in ARMS}
            per_image, image_counts = {arm: cohorts() for arm in ARMS}, {arm: Counter() for arm in ARMS}
            gt_areas = {}
            if count == 0:
                zero_images.append(iid)
                check(not fixed, "Zero candidate image has fixed associations")
                for arm in ARMS:
                    check(audit["arm_counts"][arm] == {}, "Zero candidate Score counter is not empty")
                    check(streams[arm].peek() is None or streams[arm].peek()["image_id"] != iid,
                          "Zero candidate image contains persisted candidate rows")
            for index in range(count):
                saved = {arm: streams[arm].take() for arm in ARMS}
                baseline = saved["baseline"]
                bridge = native["decisions"][index]
                check(bridge["detection_index"] == index, "Native bridge ordinal differs")
                integer(bridge["raw_index"], "raw_index")
                integer(bridge["native_output_row"], "native_output_row")
                original = fixed.get(index)
                base_values = None
                for arm in ARMS:
                    row = saved[arm]
                    check(row["image_id"] == iid and row["detection_index"] == index
                          and type(row["image_id"]) is int and type(row["detection_index"]) is int
                          and row["native_output_row"] == bridge["native_output_row"]
                          and row["raw_index"] == bridge["raw_index"], "Candidate persisted native/raw/order differs")
                    for key in ("image_id", "detection_index", "native_output_row", "raw_index", "annotation_id", "category_id"):
                        check(type(row[key]) is type(baseline[key]) and row[key] == baseline[key], "Arm identity differs: " + key)
                    integer(row["category_id"], "candidate category_id", 1)
                    integer(row["native_output_row"], "persisted native_output_row")
                    integer(row["raw_index"], "persisted raw_index")
                    check(type(row["empty_prediction"]) is bool, "Empty prediction flag missing")
                    if row["empty_prediction"]:
                        empty_ordinals[arm].update(f"{iid}:{index}\n".encode())
                    if arm == "baseline":
                        check(row["empty_prediction"] is bridge["baseline_empty"], "Saved baseline empty ordinal differs")
                    if arm == "global_minus025_full":
                        check(row["empty_prediction"] is bridge["global025_empty"], "Saved global empty ordinal differs")
                    counters = image_counts[arm]
                    counters["all_candidate_count"] += 1
                    counters["empty_prediction_count"] += int(row["empty_prediction"])
                    if original is None:
                        check(row["association_status"] == "unassociated_in_fixed_source" and row["annotation_id"] is None
                              and all(row[key] is None for key in FIELDS)
                              and all(row[key] is None for key in ("baseline_mask_iou", "baseline_success", "method_success", "repair", "damage"))
                              and row["transition"] == "unknown", "Unassociated row has invented GT quality/outcome")
                        counters["unassociated_candidate_count"] += 1
                        continue
                    check(row["association_status"] == "fixed_final8_pair" and row["annotation_id"] == original["annotation_id"]
                          and row["category_id"] == original["category_id"], "Original candidate association changed")
                    values = measured_pixel_values(row)
                    if arm == "baseline":
                        base_values = values
                        annid = original["annotation_id"]
                        check(annid not in gt_areas or gt_areas[annid] == values["gt_pixel_area"], "Repeated GT has inconsistent pixel area")
                        gt_areas[annid] = values["gt_pixel_area"]
                    check(values["gt_pixel_area"] == base_values["gt_pixel_area"], "Arm GT area differs")
                    for key in QUALITY:
                        expected = values[key] - base_values[key] if values[key] is not None and base_values[key] is not None else None
                        same(row["delta_" + key], expected, "paired quality delta " + key)
                    same(row["baseline_mask_iou"], original["baseline_mask_iou"], "saved baseline fixed IoU")
                    score_row = paired[arm][index]
                    if values["mask_iou"] is not None:
                        same(values["mask_iou"], score_row["method_mask_iou"], "pixel IoU vs scorer fixed IoU")
                    b, m = original["baseline_success"], score_row["method_mask_iou"] >= .75
                    check(row["baseline_success"] is b and row["method_success"] is m
                          and row["repair"] is (not b and m) and row["damage"] is (b and not m), "Saved fixed outcome differs")
                    transition = "repair" if not b and m else "damage" if b and not m else "stable_success" if b else "stable_failure"
                    check(row["transition"] == transition, "Saved transition differs")
                    for name, amount in (("zero_GT_pixel_area_count", row["zero_GT_pixel_area"]),
                                         ("undefined_pixel_mask_iou_count", row["pixel_mask_iou_undefined"]),
                                         ("fixed_pair_count", 1), ("baseline_success_count", b),
                                         ("baseline_failure_count", not b), ("repair_count", row["repair"]),
                                         ("damage_count", row["damage"]), ("empty_paired_prediction_count", row["empty_prediction"])):
                        counters[name] += int(amount)
                    for name in ("all_fixed_pairs", "baseline_success" if b else "baseline_failure"):
                        candidate[arm][name].add(row)
                        per_image[arm][name].add(row)
                if original is not None:
                    seen_pairs += 1
            for arm in ARMS:
                check(streams[arm].peek() is None or streams[arm].peek()["image_id"] != iid, "Extra persisted native rows in image")
                image = image_streams[arm].take()
                check(image["image_id"] == iid and type(image["image_id"]) is int, "Image summary order differs")
                same(image["counts"], dict(image_counts[arm]), "saved image count " + arm)
                # Nonempty Score images always wrote both counters, including zero empties.
                if count:
                    same(audit["arm_counts"][arm], {"candidate_count": count,
                        "empty_mask_count": image_counts[arm]["empty_prediction_count"]}, "Score image counts")
                totals[arm].update(image_counts[arm])
                check(set(image["candidate_means_within_image"]) == set(COHORTS), "Image cohort set differs")
                for name in COHORTS:
                    table = per_image[arm][name].finish()
                    compare_table(image["candidate_means_within_image"][name], table, "within image " + arm + "/" + name)
                    image_macro[arm][name].add({key: value["mean"] for key, value in table["fields"].items()})
            if (ordinal + 1) % 250 == 0:
                print("MULTIVIEW_VERIFY_PIXELS", ordinal + 1, len(ids), flush=True)
        for stream in (*streams.values(), *image_streams.values(), *score_streams.values(), identities, decisions):
            stream.finish()
    check(seen_pairs == EXPECTED_FIXED, "Persisted fixed pair count differs")
    rebuilt = {}
    for arm in ARMS:
        counts = {key: totals[arm][key] for key in COUNTS}
        check(counts["all_candidate_count"] == EXPECTED_NATIVE == scorer["identity"]["arm_counts"][arm]["candidate_count"]
              and counts["empty_prediction_count"] == scorer["identity"]["arm_counts"][arm]["empty_mask_count"], "Native total/empty count differs")
        check(empty_ordinals[arm].hexdigest() == scorer["identity"]["empty_ordinal_sha256"][arm], "Persisted empty-mask ordinal chain differs")
        check(counts["fixed_pair_count"] == EXPECTED_FIXED and counts["baseline_success_count"] == EXPECTED_SUCCESS
              and counts["baseline_failure_count"] == EXPECTED_FAILURE
              and counts["all_candidate_count"] == counts["fixed_pair_count"] + counts["unassociated_candidate_count"], "Fixed candidate denominators differ")
        source_stats = scorer["fixed_paired"][arm]["statistics"]
        for name, source_name in (("fixed_pair_count", "matched_detection_count"), ("baseline_success_count", "baseline_success_count"),
                                  ("baseline_failure_count", "baseline_failure_count"), ("repair_count", "repair_count"), ("damage_count", "damage_count")):
            same(counts[name], source_stats[source_name], "Scorer outcome totals " + arm + "/" + name)
        rebuilt[arm] = {"counts": counts,
            "damage_rate_of_baseline_success": counts["damage_count"] / counts["baseline_success_count"],
            "repair_rate_of_baseline_failure": counts["repair_count"] / counts["baseline_failure_count"],
            "candidate_macro": {name: table.finish() for name, table in candidate[arm].items()},
            "image_macro": {name: table.finish() for name, table in image_macro[arm].items()}}
        same(summary["arms"][arm]["counts"], counts, "Final summary counts " + arm)
        same(pixel_summary["arms"][arm]["counts"], dict(totals[arm]), "Pixel summary Counter counts " + arm)
        for kind in ("candidate_macro", "image_macro"):
            check(set(summary["arms"][arm][kind]) == set(pixel_summary["arms"][arm][kind]) == set(COHORTS), "Summary cohort set differs")
            for name in COHORTS:
                compare_table(summary["arms"][arm][kind][name], rebuilt[arm][kind][name], "Final " + arm + "/" + kind + "/" + name)
                compare_table(pixel_summary["arms"][arm][kind][name], rebuilt[arm][kind][name], "Pixel " + arm + "/" + kind + "/" + name)
        for key in ("damage_rate_of_baseline_success", "repair_rate_of_baseline_failure"):
            same(summary["arms"][arm][key], rebuilt[arm][key], "Conditional candidate rate " + arm + "/" + key)
    check(summary["unique_associated_GT_count"] == pixel_summary["unique_associated_GT_count"] == unique_gt,
          "Unique associated GT count differs; candidate count is not GT count")
    return {"arms": rebuilt, "unique_associated_GT_count": unique_gt, "zero_native_candidate_images": zero_images,
            "empty_ordinal_sha256": {arm: value.hexdigest() for arm, value in empty_ordinals.items()}}


def verify_boundary(run, summary, scorer, source_lock, lock):
    boundary = summary["boundary"]
    check(boundary["status"] == "completed" and boundary["dilation_ratio"] == .02
          and boundary["metric_units"] == "COCO fractions, not AP points", "Formal Boundary computation is incomplete or changed")
    vendor = lock.json(run / "boundary/VENDOR_CONTENT_LOCK.json")
    same(vendor, boundary["vendor_content_lock"], "Boundary content lock receipt")
    check(vendor["upstream_repository"].rstrip("/") == "https://github.com/bowenc0221/boundary-iou-api"
          and vendor["upstream_revision"] is None and boundary["actual_upstream_revision"] is None
          and vendor["files"] == VENDOR_HASHES, "Locked vendor identity/revision facts differ")
    paths = boundary["imported_module_paths"]
    check(type(paths) is dict and paths and all(name.startswith("boundary_iou") for name in paths), "Boundary imported module paths absent")
    coco_path = Path(paths["boundary_iou.coco_instance_api.coco"]).resolve()
    vendor_root = coco_path.parents[2]
    actual_files = {p.relative_to(vendor_root).as_posix() for p in (vendor_root / "boundary_iou").rglob("*.py")}
    check(actual_files == set(VENDOR_HASHES), "Boundary vendor Python source set differs")
    for relative, digest in VENDOR_HASHES.items():
        path = (vendor_root / relative).resolve()
        lock.add(path, digest)
        check(str(path) in source_lock["files"] and source_lock["files"][str(path)]["sha256_before"] == digest,
              "Vendor file absent from actual readout source lock")
    for name, path in paths.items():
        actual = Path(path).resolve()
        check(actual.is_relative_to(vendor_root) and actual.relative_to(vendor_root).as_posix() in VENDOR_HASHES,
              "A different Boundary package was imported: " + name)
    ordinary, categories = reconstruct_metrics(run / "boundary/BASELINE_ORDINARY_SEGM_ACCUMULATED.npz", lock)
    baseline_check = boundary["baseline_ordinary_AP_check"]
    check(baseline_check["passed"] is True and baseline_check["absolute_tolerance"] == 1e-10, "Ordinary baseline parity absent")
    same(ordinary, baseline_check["actual"], "Independent ordinary baseline summary")
    same(baseline_check["source"], scorer["metrics"]["baseline"]["segm"], "Ordinary original Score source")
    for name in METRICS:
        a, b = ordinary[name], scorer["metrics"]["baseline"]["segm"][name]
        check(a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-10,
              "Actual ordinary baseline vs Score differs: " + name)
    check(set(boundary["metrics"]) == set(ARMS), "Boundary arm set differs")
    rebuilt = {}
    for arm in ARMS:
        rebuilt[arm], cat_ids = reconstruct_metrics(run / "boundary" / (arm + "_ACCUMULATED.npz"), lock)
        check(cat_ids == categories, "Boundary category order differs across arms")
        same(boundary["metrics"][arm], rebuilt[arm], "Independent Boundary summary " + arm)
    return {"ordinary_baseline": ordinary, "boundary": rebuilt, "category_ids": categories,
            "vendor_source_hashes": VENDOR_HASHES, "actual_upstream_revision": None, "dilation_ratio": .02}


def verify(run, out, lock):
    began = time.perf_counter()
    observed = execution(run / "run.json", lock)
    summary = lock.json(run / "SUMMARY.json")
    inputs = lock.json(run / "READOUT_INPUTS.json")
    terminal = lock.json(run / "READOUT_COMPLETE.json")
    check(summary["version"] == inputs["version"] == PRODUCER_VERSION and summary["status"] == "complete"
          and summary["engineering"] is inputs["engineering"] is False and summary["formal_5000"] is True
          and summary["image_count"] == 5000 and set(summary["arms"]) == set(ARMS)
          and inputs["arms"] == list(ARMS) and inputs["boundary_requested"] is True
          and summary["success_threshold"] == .75, "Only complete formal fixed seven-arm output can be verified")
    ids = inputs["image_ids"]
    check(type(ids) is list and len(ids) == len(set(ids)) == 5000 and all(type(iid) is int for iid in ids), "Formal image scope differs")
    check(terminal["status"] == "completed" and terminal["full_multiview_completed"] is True
          and terminal["scope"] == "pixels_and_boundary"
          and terminal["summary_sha256"] == lock.add(run / "SUMMARY.json"), "Formal Multiview terminal receipt differs")
    receipt_artifacts(run, terminal, lock)
    required = {"SUMMARY.json", "READOUT_INPUTS.json", "READOUT_PROTOCOL.md", "SOURCE_LOCK.json",
                "PIXEL_SUMMARY.json", "PIXEL_COMPLETE.json", "PIXEL_SOURCE_LOCK.json", "boundary/VENDOR_CONTENT_LOCK.json",
                "boundary/BASELINE_ORDINARY_SEGM_ACCUMULATED.npz"}
    required |= {arm + "/" + name for arm in ARMS for name in ("CANDIDATES.jsonl.gz", "IMAGES.jsonl")}
    required |= {"boundary/" + arm + "_ACCUMULATED.npz" for arm in ARMS}
    check(required.issubset(terminal["artifact_sha256"]), "Completion receipt omits required scientific artifacts")
    lock.add(run / "READOUT_PROTOCOL.md", inputs["protocol_sha256"])
    lock.add(inputs["annotations"], ANNOTATIONS_SHA)
    source_lock = verify_recorded_lock(run / "SOURCE_LOCK.json", lock)
    check(summary["source_lock_sha256"] == lock.add(run / "SOURCE_LOCK.json"), "Summary source lock binding differs")
    pixel_lock = verify_recorded_lock(run / "PIXEL_SOURCE_LOCK.json", lock)
    pixel_terminal = lock.json(run / "PIXEL_COMPLETE.json")
    pixel = lock.json(run / "PIXEL_SUMMARY.json")
    check(pixel_terminal["status"] == "completed" and pixel_terminal["scope"] == "pixel_stage_only"
          and pixel_terminal["summary_sha256"] == lock.add(run / "PIXEL_SUMMARY.json")
          and pixel_terminal["source_lock_sha256"] == lock.add(run / "PIXEL_SOURCE_LOCK.json")
          and pixel["version"] == PRODUCER_VERSION and pixel["status"] == "pixel_stage_completed"
          and pixel["image_count"] == 5000 and pixel["engineering"] is False and pixel["boundary_metrics"] is None
          and set(pixel["arms"]) == set(ARMS), "Pixel stage terminal/source receipt differs")
    for path, evidence in pixel_lock["files"].items():
        check(path in source_lock["files"] and evidence == source_lock["files"][path], "Pixel source lock is not retained in final source lock")
    same(summary["fixed_source"], inputs["fixed_source"], "Final original fixed source")
    same(pixel["fixed_source"], inputs["fixed_source"], "Pixel original fixed source")
    for record in (summary, pixel):
        check(record["GPU_or_model_execution"] is False and record["GT_rematched"] is False, "Readout execution/rematching scope changed")
    check(all(value is None for value in summary["unmeasured"].values()), "Unmeasured scientific fields became fabricated values")
    declared_sources = inputs["source_sha256"]
    same(declared_sources, source_lock["executed_source_sha256"], "Actual executed source declaration")
    check({Path(path).name: digest for path, digest in declared_sources.items()} == SOURCE_HASHES,
          "Readout executed different locked source versions")
    for path, digest in declared_sources.items():
        lock.add(path, digest)
        lock.add(run / "source" / Path(path).name, digest)
        check(str(Path(path).resolve()) in source_lock["files"], "Executed code absent from original source lock")
    scoring = Path(inputs["scoring_run"]).resolve()
    for origin in (scoring, Path(inputs["inference_run"]).resolve()):
        check(out != origin and not out.is_relative_to(origin) and not origin.is_relative_to(out),
              "Verification output overlaps an inference or scoring source")
    execution(scoring / "run.json", lock)
    scorer = lock.json(scoring / "SUMMARY.json")
    score_complete = lock.json(scoring / "SCORE_COMPLETE.json")
    score_inputs = lock.json(scoring / "SCORE_INPUTS.json", score_complete["artifact_sha256"]["SCORE_INPUTS.json"])
    check(scorer["status"] == "complete" and scorer["engineering"] is False and scorer["formal_5000"] is True
          and scorer["image_count"] == 5000 and scorer["arms"] == list(ARMS) and score_inputs["image_ids"] == ids
          and score_complete["status"] == "completed" and score_complete["formal_metrics_measured"] is True
          and score_complete["identity_passed"] is True and score_complete["summary_sha256"] == lock.add(scoring / "SUMMARY.json")
          and scorer["association_rematched"] is False and scorer["GPU_or_model_execution"] is False,
          "Actual ordinary scoring scope/completion differs")
    same(scorer["fixed_source"], inputs["fixed_source"], "Score/readout original fixed association")
    for relative in ["IDENTITY_IMAGES.jsonl", "source_provenance/FIXED_INSTANCES_SOURCE.jsonl",
                     "source_provenance/FINAL8_SOURCE_SUMMARY.json", "source_provenance/FIXED_PAIRED_SOURCE_SUMMARY.json"]:
        lock.add(scoring / relative, score_complete["artifact_sha256"][relative])
    for arm in ARMS:
        relative = "paired/" + arm + "/INSTANCES.jsonl"
        lock.add(scoring / relative, score_complete["artifact_sha256"][relative])
        relative = arm + "/COCO_METRICS.json"
        same(lock.json(scoring / relative, score_complete["artifact_sha256"][relative]), scorer["metrics"][arm], "Scorer actual metric receipt")
    check(summary["ordinary_AP"]["recomputed"] is False
          and summary["ordinary_AP"]["boundary_vendor_baseline_parity_only"] is True
          and Path(summary["ordinary_AP"]["source_scoring_run"]).resolve() == scoring
          and summary["ordinary_AP"]["source_summary_sha256"] == lock.add(scoring / "SUMMARY.json"), "Ordinary AP source claim differs")
    pixels = verify_pixels(run, scoring, inputs, summary, pixel, scorer, lock)
    boundary = verify_boundary(run, summary, scorer, source_lock, lock)
    fingerprint = lock.finish()
    write_json(out / "SOURCE_LOCK.json", fingerprint)
    result = {"version": VERSION, "status": "passed", "passed": True, "formal_5000": True,
              "image_count": 5000, "arm_names": list(ARMS), "source_run": str(run),
              "source_run_id": observed["run_id"], "source_summary_sha256": lock.add(run / "SUMMARY.json"),
              "source_readout_complete_sha256": lock.add(run / "READOUT_COMPLETE.json"),
              "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
              "verification_source_sha256": sha(__file__), "python": sys.version, "executable": sys.executable,
              "numpy": np.__version__, "reconstructed_pixels": pixels, "reconstructed_AP_AR": boundary,
              "GPU_or_model_execution": False, "GT_rematched": False, "torch_imported": "torch" in sys.modules,
              "pycocotools_imported": any(name.startswith("pycocotools") for name in sys.modules),
              "scope": "Independent saved-record formulas, native/raw/category ordinal bridges, fixed final8 associations, field-wise candidate/image means, counts, labels, and 12 standard accumulated-array summaries per evaluation",
              "limitations": ["GT/prediction masks are not decoded or independently recounted; saved TP/FP/FN closure and ratios are checked",
                  "COCO matching and boundary-mask construction are not replayed; AP/AR are reconstructed from sealed accumulated arrays",
                  "Candidate streams omit bbox/score; their exact identity is inherited from the completed original Score audit",
                  "Undefined pixel-IoU outcome convention is checked against sealed Score paired rows, not reconstructed from masks",
                  "No new model forward, training, raw-pool geometry, TAL, AUC, exact-n, confidence interval, or causal proof"],
              "seconds": time.perf_counter() - began, "completed_at": utc()}
    check(result["torch_imported"] is False and result["pycocotools_imported"] is False, "Unexpected model/COCO module import")
    write_json(out / "VERIFICATION.json", result)
    write_json(out / "VERIFICATION_COMPLETE.json", {"status": "completed", "passed": True,
        "verification_sha256": sha(out / "VERIFICATION.json"), "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
        "verification_source_sha256": sha(__file__)})
    print(json.dumps({"status": "passed", "images": 5000, "arms": len(ARMS),
                      "native_rows_per_arm": EXPECTED_NATIVE, "fixed_pairs_per_arm": EXPECTED_FIXED}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out_dir.resolve()
    check(run != out and not run.is_relative_to(out) and not out.is_relative_to(run),
          "Verification output must be a separate Run, not a source ancestor/descendant")
    out.mkdir(parents=True, exist_ok=True)
    check(not any((out / name).exists() for name in ("VERIFICATION.json", "VERIFICATION_COMPLETE.json", "VERIFICATION_FAILURE.json")),
          "Verification history is immutable; use a new Run")
    lock = Fingerprint()
    lock.add(__file__)
    lock.add(np.__file__)
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    target = source_dir / Path(__file__).name
    check(not target.exists() or sha(target) == sha(__file__), "Verifier source archive differs")
    if not target.exists():
        shutil.copy2(__file__, target)
    lock.add(target, sha(__file__))
    try:
        verify(run, out, lock)
    except Exception as error:
        failure = {"version": VERSION, "status": "failed", "passed": False, "source_run": str(run),
                   "error": repr(error), "traceback": traceback.format_exc(), "time": utc(),
                   "verification_source_sha256": sha(__file__), "python": sys.version,
                   "executable": sys.executable, "numpy": np.__version__, "GPU_or_model_execution": False,
                   "GT_rematched": False, "independent_reconstruction_completed": False}
        try:
            write_json(out / "SOURCE_LOCK.json", lock.finish())
            failure["source_lock_sha256"] = sha(out / "SOURCE_LOCK.json")
        except Exception as changed:
            write_json(out / "PARTIAL_INPUT_FINGERPRINT.json", {"files": lock.files, "all_before_after_sha256_equal": False})
            failure["input_rehash_error"] = repr(changed)
        write_json(out / "VERIFICATION_FAILURE.json", failure)
        write_json(out / "VERIFICATION.json", failure)
        raise


if __name__ == "__main__":
    main()
