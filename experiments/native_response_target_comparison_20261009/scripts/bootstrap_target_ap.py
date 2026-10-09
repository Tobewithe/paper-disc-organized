"""Paired image-copy bootstrap of actual joint COCO AP, conditional on one fit."""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import numpy as np
from target_scoring_common import (ARMS, ANN_SHA, BASE_PRED_SHA, PAIR_FIELDS, InputLock, cache_from_evaluator, check,
    json_array, load_npz, official_eval, paired_summary, read_json, sha, write_json)
from target_scoring_common import (completed_receipt, seal_outputs, verify_output_seal, validate_completed_record)

SEED = 20261009
REPLICATES = 1000


def expanded_indices(cache, category, draw):
    """Ordered concatenation of every sampled image copy, before score sort."""
    offsets = cache["offsets"][category]
    starts = offsets[draw]
    lengths = offsets[draw + 1] - starts
    total = int(lengths.sum())
    if not total:
        return np.empty(0, np.int64)
    copy_starts = np.cumsum(lengths) - lengths
    return np.repeat(starts, lengths) + np.arange(total, dtype=np.int64) - np.repeat(copy_starts, lengths)


def category_precision(cache, category, draw, ordered_indices=None):
    t, r = len(cache["iou_thresholds"]), len(cache["recall_thresholds"])
    gt_n = int(cache["gt_nonignore"][category, draw].sum())
    if gt_n == 0:
        return np.full((t, r), -1.), np.full(t, -1.)
    if ordered_indices is None:
        idx = expanded_indices(cache, category, draw)
        ordered_indices = idx[np.argsort(-cache["dt_scores"][idx], kind="mergesort")]
    matched = cache["dt_matches"][:, ordered_indices]
    ignored = cache["dt_ignore"][:, ordered_indices]
    tp = np.cumsum(matched & ~ignored, axis=1, dtype=np.float64)
    fp = np.cumsum(~matched & ~ignored, axis=1, dtype=np.float64)
    precision = np.zeros((t, r), dtype=np.float64)
    recall = np.zeros(t, dtype=np.float64)
    if tp.shape[1]:
        rc = tp / gt_n
        pr = tp / (tp + fp + np.spacing(1))
        # Same backward precision envelope as actual COCOeval.accumulate.
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        recall = rc[:, -1]
        for threshold in range(t):
            locations = np.searchsorted(rc[threshold], cache["recall_thresholds"], side="left")
            valid = locations < len(ordered_indices)
            precision[threshold, valid] = pr[threshold, locations[valid]]
    return precision, recall


def reaccumulate(cache, draw):
    draw = np.asarray(draw, dtype=np.int64)
    check(draw.ndim == 1 and np.all(draw >= 0) and np.all(draw < len(cache["image_ids"])), "Invalid ordered bootstrap draw")
    precision = np.empty((10, 101, len(cache["category_ids"])), dtype=np.float64)
    recall = np.empty((10, len(cache["category_ids"])), dtype=np.float64)
    for cat in range(len(cache["category_ids"])):
        precision[:, :, cat], recall[:, cat] = category_precision(cache, cat, draw)
    valid = precision >= 0
    return float(precision[valid].mean()) if valid.any() else None, precision, recall


def paired_aps(caches, draw):
    # The producer retains class/score/order in all arms. Check cache ordering
    # identity before reusing a single stable score sort for all three masks.
    base = caches[0]
    sums, counts = np.zeros(3), np.zeros(3, dtype=np.int64)
    for cat in range(len(base["category_ids"])):
        idx = expanded_indices(base, cat, draw)
        ordered = idx[np.argsort(-base["dt_scores"][idx], kind="mergesort")]
        for a, cache in enumerate(caches):
            precision, _ = category_precision(cache, cat, draw, ordered)
            valid = precision >= 0
            sums[a] += precision[valid].sum()
            counts[a] += valid.sum()
    check(np.all(counts > 0), "AP undefined on bootstrap draw")
    return sums / counts


def cache_contract(caches):
    base = caches[0]
    check(base["offsets"].shape == (len(base["category_ids"]), len(base["image_ids"]) + 1), "Cache offset dimensions differ")
    for other in caches[1:]:
        for key in ("image_ids", "category_ids", "offsets", "dt_scores", "gt_nonignore", "iou_thresholds", "recall_thresholds"):
            check(np.array_equal(base[key], other[key]), "Arms changed cached ordering/score/GT identity: " + key)
    for cache in caches:
        check(cache["dt_matches"].shape == cache["dt_ignore"].shape == (10, len(cache["dt_scores"])), "Cache match arrays differ")
        check(np.isfinite(cache["dt_scores"]).all(), "Nonfinite detection score")


def clones_from_draw(dataset, records, image_ids, draw):
    """Independent official reference: genuinely new GT/DT/image IDs per copy."""
    source_images = {r["id"]: r for r in dataset["images"]}
    source_gt = {iid: [] for iid in image_ids}
    source_dt = {iid: [] for iid in image_ids}
    for row in dataset["annotations"]:
        if row["image_id"] in source_gt:
            source_gt[row["image_id"]].append(row)
    for row in records:
        if row["image_id"] in source_dt:
            source_dt[row["image_id"]].append(row)
    copied = {"info": copy.deepcopy(dataset.get("info", {})), "images": [], "categories": copy.deepcopy(dataset["categories"]), "annotations": []}
    predictions = []
    annid, dtid = 1, 1
    for position, index in enumerate(draw):
        iid, copy_id = image_ids[int(index)], position + 1
        image = copy.deepcopy(source_images[iid])
        image["id"] = copy_id
        copied["images"].append(image)
        for row in source_gt[iid]:
            ann = copy.deepcopy(row)
            ann.update(id=annid, image_id=copy_id)
            annid += 1
            copied["annotations"].append(ann)
        for row in source_dt[iid]:
            dt = copy.deepcopy(row)
            dt.update(id=dtid, image_id=copy_id)
            dtid += 1
            predictions.append(dt)
    check(len({r["id"] for r in copied["annotations"]}) == len(copied["annotations"]), "Reference GT copy IDs duplicate")
    check(len({r["id"] for r in predictions}) == len(predictions), "Reference DT copy IDs duplicate")
    return copied, predictions


def official_dataset(dataset, records):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = copy.deepcopy(dataset)
        coco.createIndex()
    return official_eval(coco, records, [v["id"] for v in dataset["images"]], COCO, COCOeval)[0]


def synthetic_fixture():
    from pycocotools import mask as m
    def rectangle(x, y, w, h):
        value = np.zeros((20, 20), np.uint8)
        value[y:y + h, x:x + w] = 1
        rle = m.encode(np.asfortranarray(value))
        return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}
    gt = [dict(id=1, image_id=11, category_id=1, segmentation=rectangle(1, 1, 5, 5), area=25, bbox=[1, 1, 5, 5], iscrowd=0),
          dict(id=2, image_id=11, category_id=1, segmentation=rectangle(11, 11, 7, 7), area=49, bbox=[11, 11, 7, 7], iscrowd=1),
          dict(id=3, image_id=22, category_id=2, segmentation=rectangle(1, 1, 4, 4), area=16, bbox=[1, 1, 4, 4], iscrowd=0),
          dict(id=4, image_id=11, category_id=3, segmentation=rectangle(1, 1, 5, 5), area=25, bbox=[1, 1, 5, 5], iscrowd=1, ignore=1)]
    dataset = {"info": {}, "images": [{"id": i, "width": 20, "height": 20} for i in (11, 22, 33)],
               "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)], "annotations": gt}
    records = []
    # >100 same-score candidates exercise truncation and original stable tie order.
    for index in range(105):
        records.append(dict(image_id=11, category_id=1, score=.5,
                            segmentation=rectangle(1, 1, 5, 5) if index == 0 else rectangle(11, 11, 7, 7) if index == 1 else rectangle(5, 5, 2, 2)))
    records += [dict(image_id=11, category_id=3, score=.5, segmentation=rectangle(1, 1, 5, 5)),
                dict(image_id=33, category_id=1, score=.5, segmentation=rectangle(1, 1, 5, 5)),
                dict(image_id=33, category_id=1, score=.5, segmentation=rectangle(0, 0, 0, 0))]
    return dataset, records


def compare_reference(dataset, records, draws, out, name):
    evaluator = official_dataset(dataset, records)
    cache = cache_from_evaluator(evaluator)
    ids = cache["image_ids"].tolist()
    findings = []
    for draw in draws:
        draw = np.asarray(draw, np.int64)
        actual_ap, actual_precision, actual_recall = reaccumulate(cache, draw)
        cloned_gt, cloned_dt = clones_from_draw(dataset, records, ids, draw)
        direct = official_dataset(cloned_gt, cloned_dt)
        expected_precision = direct.eval["precision"][:, :, :, 0, 2]
        expected_recall = direct.eval["recall"][:, :, 0, 2]
        check(np.array_equal(actual_precision < 0, expected_precision < 0), "Independent official copy precision missingness differs")
        check(np.array_equal(actual_recall < 0, expected_recall < 0), "Independent official copy recall missingness differs")
        maximum = max(float(np.max(np.abs(actual_precision - expected_precision))), float(np.max(np.abs(actual_recall - expected_recall))))
        check(maximum <= 1e-14, "Independent official copy precision/recall differs")
        official_ap = float(direct.stats[0]) if direct.stats[0] >= 0 else None
        check(actual_ap == official_ap or actual_ap is not None and official_ap is not None and abs(actual_ap - official_ap) <= 1e-14,
              "Independent official copy AP differs")
        findings.append({"draw": draw.tolist(), "copy_image_ids": list(range(1, len(draw) + 1)),
                         "gt_copies": len(cloned_gt["annotations"]), "dt_copies": len(cloned_dt),
                         "AP": actual_ap, "official_AP": official_ap, "max_array_abs_error": maximum})
    write_json(out / (name + "_OFFICIAL_COPY_COMPARISON.json"), {"passed": True, "cases": findings, "source_image_ids": ids,
               "matching_source": "actual pycocotools.COCOeval.evaluate; reference reruns full matching on genuinely unique GT/DT/image copies",
               "accumulation_scope": "10 IoU thresholds x 101 recall points x categories; all area, max100"})
    np.savez_compressed(out / (name + "_MATCH_CACHE.npz"), **cache)
    return {"passed": True, "cases": len(findings), "maximum_array_error": max(v["max_array_abs_error"] for v in findings)}


def engineering(args):
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = InputLock()
    source_dir = out / "source"
    source_dir.mkdir()
    executed_sources = {}
    for file in (Path(__file__), Path(__file__).with_name("target_scoring_common.py")):
        lock.add(file)
        target = source_dir / file.name
        shutil.copy2(file, target)
        lock.files[str(file.resolve())]["snapshot"] = target.relative_to(out).as_posix()
        executed_sources[file.name] = {"path": str(file.resolve()), "sha256": sha(file), "snapshot": target.relative_to(out).as_posix()}
    import pycocotools.coco as coco_module
    import pycocotools.cocoeval as eval_module
    import pycocotools.mask as mask_module
    import pycocotools._mask as mask_binary
    coco_sources = {}
    for module in (coco_module, eval_module, mask_module, mask_binary):
        path = Path(module.__file__).resolve()
        lock.add(path)
        target = source_dir / (module.__name__.replace(".", "_") + path.suffix)
        shutil.copy2(path, target)
        lock.files[str(path)]["snapshot"] = target.relative_to(out).as_posix()
        coco_sources[module.__name__] = {"path": str(path), "sha256": sha(path), "snapshot": target.relative_to(out).as_posix()}
    dataset, records = synthetic_fixture()
    write_json(out / "SYNTHETIC_GT.json", dataset)
    write_json(out / "SYNTHETIC_DT.json", records)
    draws = [[0, 0, 1], [2, 0, 0], [2, 2, 1], [1, 1, 1], [0, 2, 0], [0, 1, 2], [2, 0, 1]]
    result = {"synthetic": compare_reference(dataset, records, draws, out, "SYNTHETIC")}
    if args.annotations is not None and args.real_predictions is not None:
        check(lock.add(args.annotations) == ANN_SHA, "Engineering real-first4 original val annotation SHA differs")
        check(lock.add(args.real_predictions) == BASE_PRED_SHA, "Engineering real-first4 accepted full native baseline SHA differs")
        dataset = read_json(args.annotations)
        ids = sorted(v["id"] for v in dataset["images"])[:4]
        scope = set(ids)
        short = {k: copy.deepcopy(v) for k, v in dataset.items() if k not in ("images", "annotations")}
        short["images"] = [v for v in dataset["images"] if v["id"] in scope]
        short["annotations"] = [v for v in dataset["annotations"] if v["image_id"] in scope]
        records = []
        # The accepted full native file's fixed SHA binds its original ascending
        # image assembly. Parse only the fixed first4; hash its full bytes before
        # and after, without decoding the remaining 4996 images' JSON/RLE.
        for row in json_array(args.real_predictions):
            if row["image_id"] > ids[-1]:
                break
            check(row["image_id"] in scope, "Real engineering native first4 source order differs")
            records.append({k: copy.deepcopy(row[k]) for k in ("image_id", "category_id", "score", "segmentation")})
        result["real_fixed_first4"] = compare_reference(short, records, [[0, 0, 1, 3], [3, 2, 3, 0], [1, 1, 1, 1]], out, "REAL_FIRST4")
    else:
        result["real_fixed_first4"] = {"passed": None, "status": "unmeasured"}
    # Exercise the consumer's actual provenance guards, not statistical formulas.
    guard_results = {}
    passed_summary = {"status": "engineering_complete", "passed": True, "source_lock_sha256": "source"}
    passed_complete = {"status": "engineering_complete", "summary_sha256": "summary", "source_lock_sha256": "source"}
    for bad_status in ("failed", "cancelled", "running"):
        record = {"status": bad_status, "return_code": 1 if bad_status == "failed" else 0, "artifact_completeness": "complete"}
        try:
            validate_completed_record(record, "engineering_complete", passed_summary, passed_complete, "summary", "source")
        except ValueError as error:
            guard_results[bad_status + "_despite_passed_summary"] = {"rejected": True, "reason": str(error)}
        else:
            raise AssertionError("A non-completed process passed the engineering consumer gate")
    negative_dir = out / "negative_fixtures"
    negative_dir.mkdir()
    np.savez(negative_dir / "CACHE.npz", values=np.zeros(4))
    origin_seal = seal_outputs(negative_dir, ["CACHE.npz"])
    verify_output_seal(negative_dir, origin_seal, ["CACHE.npz"])
    np.savez(negative_dir / "CACHE.npz", values=np.ones(4))
    try:
        verify_output_seal(negative_dir, origin_seal, ["CACHE.npz"])
    except ValueError as error:
        guard_results["modified_NPZ"] = {"rejected": True, "reason": str(error), "original_seal": origin_seal,
                                         "actual_sha256_after_intentional_change": sha(negative_dir / "CACHE.npz")}
    else:
        raise AssertionError("An intentionally modified match cache passed the producer-seal guard")
    write_json(out / "ENGINEERING_CONTRACT_GUARDS.json", {"passed": True, "cases": guard_results})
    write_json(out / "SOURCE_LOCK.json", {"files": lock.finish(), "executed_sources": executed_sources,
               "actual_coco_modules": coco_sources, "pycocotools_version": importlib.metadata.version("pycocotools")})
    artifact_paths = ["SYNTHETIC_GT.json", "SYNTHETIC_DT.json", "SYNTHETIC_OFFICIAL_COPY_COMPARISON.json", "SYNTHETIC_MATCH_CACHE.npz",
                      "ENGINEERING_CONTRACT_GUARDS.json", "negative_fixtures/CACHE.npz"]
    if result["real_fixed_first4"]["passed"] is True:
        artifact_paths += ["REAL_FIRST4_OFFICIAL_COPY_COMPARISON.json", "REAL_FIRST4_MATCH_CACHE.npz"]
    artifact_paths += [v["snapshot"] for v in executed_sources.values()] + [v["snapshot"] for v in coco_sources.values()]
    artifacts = seal_outputs(out, artifact_paths)
    write_json(out / "SUMMARY.json", {"status": "engineering_complete", "passed": True, "engineering": True,
               "checks": result, "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "formal_metrics": None,
               "artifacts": artifacts, "provenance_negative_guards_passed": True, "actual_coco_modules": coco_sources,
               "pycocotools_version": importlib.metadata.version("pycocotools"), "numpy": np.__version__,
               "GPU_or_model_execution": False, "interpreter": sys.executable, "host": platform.node(), "platform": platform.platform()})
    write_json(out / "COMPLETE.json", {"status": "engineering_complete", "summary_sha256": sha(out / "SUMMARY.json"),
               "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "artifacts": artifacts})


def ci(values, point):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return {"point": float(point) if point is not None else None,
            "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
            "valid_replicates": len(valid), "interval": "percentile 2.5/97.5; conditional on the single fitted models"}


def paired_replicate_values(pairs, draw):
    counts = np.bincount(draw, minlength=pairs.shape[1])
    total = np.einsum("aif,i->af", pairs, counts, optimize=True)
    values = {}
    metrics = (("damage", "damage", "baseline_success"), ("repair", "repair", "baseline_failure"),
               ("mask_iou", "method_iou_sum", "matched_count"), ("delta_iou", "delta_iou_sum", "matched_count"),
               ("coverage", "coverage_sum", "coverage_n"), ("purity", "purity_sum", "purity_n"),
               ("FP_G", "fp_g_sum", "fp_g_n"), ("FN_G", "fn_g_sum", "fn_g_n"))
    for name, numerator, denominator in metrics:
        ni, di = PAIR_FIELDS.index(numerator), PAIR_FIELDS.index(denominator)
        for a, arm in enumerate(ARMS):
            values[f"{arm}.{name}"] = total[a, ni] / total[a, di] if total[a, di] else np.nan
            valid = pairs[a, :, di] > 0
            den = counts[valid].sum()
            values[f"{arm}.image_macro_{name}"] = np.sum(counts[valid] * pairs[a, valid, ni] / pairs[a, valid, di]) / den if den else np.nan
        values[f"H_minus_I.{name}"] = values[f"target_H.{name}"] - values[f"target_I.{name}"]
        values[f"H_minus_I.image_macro_{name}"] = values[f"target_H.image_macro_{name}"] - values[f"target_I.image_macro_{name}"]
    return values


def bootstrap(args):
    began = time.perf_counter()
    out, score = args.out_dir.resolve(), args.scoring_run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = InputLock()
    summary, score_complete, score_sources = completed_receipt(score, "scoring_complete", lock)
    check(summary["status"] == "scoring_complete" and summary["passed"] is True and not summary["engineering"] and summary["image_count"] == 5000, "Formal scored5000 input required")
    required_score_outputs = ["FIXED_PAIRS.npz", "PAIRED_IMAGE_SUMS.npz", "IDENTITY_IMAGES.jsonl", "SCORE_INPUTS.json"]
    required_score_outputs += [f"{arm}/{name}" for arm in ARMS for name in ("COCO_BOOTSTRAP_CACHE.npz", "COCO_ACCUMULATED.npz", "COCO_METRICS.json")]
    verify_output_seal(score, summary["artifacts"], required_score_outputs, lock)
    check(sha(score / "PAIRED_IMAGE_SUMS.npz") == summary["paired_image_sums_sha256"]
          and sha(score / "FIXED_PAIRS.npz") == summary["fixed_pairs_sha256"], "Original fixed/image-stat NPZ seal differs")
    check(args.engineering_run is not None, "Official-copy engineering hard gate required")
    engineer, engineering_complete, engineering_sources = completed_receipt(args.engineering_run, "engineering_complete", lock)
    check(engineer["status"] == "engineering_complete" and engineer["passed"] and engineer["checks"]["synthetic"]["passed"]
          and engineer["checks"]["real_fixed_first4"]["passed"], "Synthetic AND real official-copy engineering must pass")
    check(engineer["provenance_negative_guards_passed"] is True, "Meaningful failed/cancelled/modified-cache guards were not measured")
    for name in ("bootstrap_target_ap.py", "target_scoring_common.py"):
        expected = engineering_sources["executed_sources"][name]
        check(sha(Path(__file__).with_name(name)) == expected["sha256"]
              and sha(args.engineering_run / expected["snapshot"]) == expected["sha256"], "Bootstrap executable source changed since independent engineering")
        lock.add(Path(__file__).with_name(name))
    import pycocotools.coco as coco_module
    import pycocotools.cocoeval as eval_module
    import pycocotools.mask as mask_module
    import pycocotools._mask as mask_binary
    actual_modules = {}
    for module in (coco_module, eval_module, mask_module, mask_binary):
        actual_modules[module.__name__] = {"path": str(Path(module.__file__).resolve()), "sha256": lock.add(module.__file__)}
        if module is not mask_binary:
            check(actual_modules[module.__name__]["sha256"] == engineering_sources["actual_coco_modules"][module.__name__]["sha256"],
                  "Actual COCO Python semantics differ from independent engineering")
    check(importlib.metadata.version("pycocotools") == engineering_sources["pycocotools_version"], "Actual COCO package version differs from independent engineering")
    caches = []
    for arm in ARMS:
        path = score / arm / "COCO_BOOTSTRAP_CACHE.npz"
        lock.add(path)
        caches.append(load_npz(path))
    cache_contract(caches)
    n = len(caches[0]["image_ids"])
    check(n == 5000 and caches[0]["image_ids"].tolist() == summary["image_ids"], "Formal cached image order differs")
    pairs_npz = load_npz(score / "PAIRED_IMAGE_SUMS.npz")
    pairs = pairs_npz["values"]
    check(pairs_npz["image_ids"].tolist() == summary["image_ids"] and pairs.shape == (3, n, len(PAIR_FIELDS)), "Paired image cluster shape/order differs")
    for a, cache in enumerate(caches):
        point, precision, recall = reaccumulate(cache, np.arange(n))
        check(np.allclose(precision, cache["official_precision"], rtol=0, atol=1e-14) and np.allclose(recall, cache["official_recall"], rtol=0, atol=1e-14), "Point reaccumulation differs from actual COCOeval")
        check(abs(point - summary["metrics"][ARMS[a]]["AP"]) <= 1e-14, "Point AP cache parity differs")
    # Draw as default_rng integers default int64, then store losslessly as int32.
    # Do not generate using dtype=int32: that changes RNG consumption/sequence.
    generator = np.random.default_rng(SEED)
    draws = np.empty((REPLICATES, n), dtype=np.int32)
    ap_values = np.empty((REPLICATES, 3), dtype=np.float64)
    extra = None
    for rep in range(REPLICATES):
        draw = generator.integers(0, n, size=n)
        draws[rep] = draw
        ap_values[rep] = paired_aps(caches, draw)
        row = paired_replicate_values(pairs, draw)
        if extra is None:
            extra = {key: np.empty(REPLICATES, dtype=np.float64) for key in row}
        for key, value in row.items():
            extra[key][rep] = value
        if (rep + 1) % 10 == 0:
            print("paired joint AP bootstrap", rep + 1, REPLICATES, "seconds", time.perf_counter() - began, flush=True)
    np.savez_compressed(out / "BOOTSTRAP_DRAWS.npz", draws=draws, image_ids=caches[0]["image_ids"], seed=np.int64(SEED))
    np.savez_compressed(out / "BOOTSTRAP_REPLICATES.npz", AP=ap_values, AP_H_minus_I=ap_values[:, 2] - ap_values[:, 1], **extra)
    point_aps = [summary["metrics"][a]["AP"] for a in ARMS]
    point_extra = paired_replicate_values(pairs, np.arange(n))
    intervals = {"AP." + arm: ci(ap_values[:, a], point_aps[a]) for a, arm in enumerate(ARMS)}
    intervals["AP.H_minus_I"] = ci(ap_values[:, 2] - ap_values[:, 1], point_aps[2] - point_aps[1])
    intervals.update({key: ci(value, point_extra[key]) for key, value in extra.items()})
    diff = intervals["AP.H_minus_I"]
    damage_h, damage_i = point_extra["target_H.damage"], point_extra["target_I.damage"]
    same_strategy = summary.get("same_strategy")
    if same_strategy is True:
        decision = "current_configuration_no_support_identical_policy"
    elif diff["ci95"][0] > 0 and damage_h <= damage_i:
        decision = "direction_worth_later_independent_confirmation"
    elif point_aps[2] > point_aps[1] and damage_h > damage_i:
        decision = "AP_damage_tradeoff"
    elif diff["ci95"][1] <= 0:
        decision = "current_configuration_no_support"
    else:
        decision = "not_demonstrated_interval_includes_zero"
    write_json(out / "SOURCE_LOCK.json", {"files": lock.finish()})
    artifacts = seal_outputs(out, ["BOOTSTRAP_DRAWS.npz", "BOOTSTRAP_REPLICATES.npz"])
    result = {"status": "bootstrap_complete", "passed": True, "seed": SEED, "replicates": REPLICATES, "draw_size": n,
              "ordered_independent_image_copies": True, "score_tie_sort": "stable mergesort; sampled copy position then original class candidate sequence",
              "metric": "actual COCOeval matching cache reaccumulated across copied images; 10 thresholds x101 recall grid; all-area/max100",
              "intervals": intervals, "decision": decision, "decision_is_safety_guarantee": False,
              "scoring_run": str(score), "scoring_summary_sha256": sha(score / "SUMMARY.json"),
              "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "draws_sha256": sha(out / "BOOTSTRAP_DRAWS.npz"),
              "artifacts": artifacts, "actual_coco_modules": actual_modules, "pycocotools_version": importlib.metadata.version("pycocotools"),
              "replicates_sha256": sha(out / "BOOTSTRAP_REPLICATES.npz"), "seconds": time.perf_counter() - began,
              "boundary_CI": None, "training_seed_uncertainty_included": False, "GPU_or_model_execution": False,
              "fixed_association_scope": "original final8 86600 detection-associated pairs; repeated GT is retained; separate from normal all-GT AP",
              "limits": ["Conditional single fit", "Previously studied val5k; not an untouched final test", "Finite outcome; no parameter or seed search"]}
    write_json(out / "SUMMARY.json", result)
    write_json(out / "COMPLETE.json", {"status": "bootstrap_complete", "summary_sha256": sha(out / "SUMMARY.json"),
               "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "artifacts": artifacts})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scoring-run", type=Path)
    p.add_argument("--engineering-run", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--engineering", action="store_true")
    p.add_argument("--annotations", type=Path)
    p.add_argument("--real-predictions", type=Path)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        engineering(args) if args.engineering else bootstrap(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "BOOTSTRAP_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "GPU_or_model_execution": False})
        raise
