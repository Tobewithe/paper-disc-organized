"""Independent saved-array/ledger/provenance checker; no GT rematching/model."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import traceback
import numpy as np
from target_scoring_common import (ARMS, METRICS, PAIR_FIELDS, InputLock, check,
    jsonl, load_npz, paired_summary, read_json, sha, write_json)
from target_scoring_common import completed_receipt, verify_output_seal
from bootstrap_target_ap import SEED, REPLICATES, cache_contract, ci, paired_aps, paired_replicate_values, reaccumulate


def close(actual, expected, message, tolerance=1e-12):
    if isinstance(actual, dict) and isinstance(expected, dict):
        check(set(actual) == set(expected), message + " keys")
        for key in actual:
            close(actual[key], expected[key], message + "." + key, tolerance)
    elif isinstance(actual, list) and isinstance(expected, list):
        check(len(actual) == len(expected), message + " length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, message + f"[{i}]", tolerance)
    elif type(actual) in (float, int) and type(expected) in (float, int):
        check(math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), message)
    else:
        check(actual == expected, message)


def metrics_from_arrays(path):
    z = load_npz(path)
    p, r = z["precision"], z["recall"]
    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if len(valid) else None
    indices50 = np.flatnonzero(np.isclose(z["iou_thresholds"], .5, rtol=0, atol=1e-14))
    indices75 = np.flatnonzero(np.isclose(z["iou_thresholds"], .75, rtol=0, atol=1e-14))
    check(len(indices50) == len(indices75) == 1 and p.shape[:2] == (10, 101), "Official grid differs")
    values = [mean(p[:, :, :, 0, 2]), mean(p[indices50, :, :, 0, 2]), mean(p[indices75, :, :, 0, 2])]
    values += [mean(p[:, :, :, area, 2]) for area in (1, 2, 3)]
    values += [mean(r[:, :, 0, maximum]) for maximum in (0, 1, 2)]
    values += [mean(r[:, :, area, 2]) for area in (1, 2, 3)]
    return dict(zip(METRICS, values))


def rebuild_pairs(score, ids):
    z = load_npz(score / "FIXED_PAIRS.npz")
    keys, ledger = z["keys"], z["ledger"]
    check(keys.shape == (len(ledger), 4) and ledger.shape == (len(keys), 3, 4), "Fixed pixel ledger dimensions differ")
    check(z["image_ids"].tolist() == ids, "Fixed ledger image order differs")
    check(np.all(ledger >= 0) and np.all(ledger[:, :, 0] + ledger[:, :, 1] == ledger[:, :, 3]), "Integer TP/FP/prediction-area accounting differs")
    gt_area = ledger[:, :, 0] + ledger[:, :, 2]
    check(np.all(gt_area > 0) and np.array_equal(gt_area[:, 0], gt_area[:, 1]) and np.array_equal(gt_area[:, 0], gt_area[:, 2]), "Arms changed fixed GT pixels")
    source = {(r["image_id"], r["detection_index"]): r for r in jsonl(score / "source_provenance" / "FIXED_INSTANCES_SOURCE.jsonl")}
    selected_keys = set()
    rebuilt = np.zeros((3, len(ids), len(PAIR_FIELDS)), np.float64)
    for pos, (key, pixel) in enumerate(zip(keys, ledger)):
        image_index, detection_index, annid, category = map(int, key)
        check(0 <= image_index < len(ids), "Fixed ledger image index invalid")
        identity = ids[image_index], detection_index
        check(identity not in selected_keys and identity in source, "Fixed candidate key missing/duplicated")
        selected_keys.add(identity)
        original = source[identity]
        check(original["annotation_id"] == annid and original["category_id"] == category, "Original final8 association changed")
        base_tp, base_fp, base_fn, _ = map(int, pixel[0])
        base_union = base_tp + base_fp + base_fn
        b = base_tp / base_union
        success = 4 * base_tp >= 3 * base_union
        close(b, original["baseline_mask_iou"], "Original baseline IoU differs", 1e-14)
        close(b, z["source_baseline_ious"][pos].item(), "Recorded source baseline IoU differs", 1e-14)
        check(success is original["baseline_success"], "Original baseline success differs")
        for a, row in enumerate(pixel):
            tp, fp, fn, area = map(int, row)
            union, gt = tp + fp + fn, tp + fn
            m = tp / union
            method_success = 4 * tp >= 3 * union
            delta = m - b
            rebuilt[a, image_index] += [1, success, not success, success and not method_success, not success and method_success,
                b, m, delta, delta if success else 0, tp / gt, 1, tp / area if area else 0, int(area > 0), fp / gt, 1, fn / gt, 1]
    expected_keys = {k for k in source if k[0] in set(ids)}
    check(selected_keys == expected_keys, "Fixed original selected cohort incomplete")
    return rebuilt


def terminal(path, status, lock):
    summary, complete, source = completed_receipt(path, status, lock)
    for original, entry in source["files"].items():
        check(entry["sha256_before"] == entry["sha256_after"], "Source changed during original Run")
        check(lock.add(original) == entry["sha256_after"], "Original source bytes changed since scoring/bootstrap")
    return summary


def verify(args):
    score, boot, out = args.scoring_run.resolve(), args.bootstrap_run.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = InputLock()
    summary = terminal(score, "scoring_complete", lock)
    bootstrap = terminal(boot, "bootstrap_complete", lock)
    required_score_outputs = ["FIXED_PAIRS.npz", "PAIRED_IMAGE_SUMS.npz", "IDENTITY_IMAGES.jsonl", "SCORE_INPUTS.json"]
    required_score_outputs += [f"{arm}/{name}" for arm in ARMS for name in ("COCO_BOOTSTRAP_CACHE.npz", "COCO_ACCUMULATED.npz", "COCO_METRICS.json")]
    verify_output_seal(score, summary["artifacts"], required_score_outputs, lock)
    check(sha(score / "PAIRED_IMAGE_SUMS.npz") == summary["paired_image_sums_sha256"]
          and sha(score / "FIXED_PAIRS.npz") == summary["fixed_pairs_sha256"], "Original scored paired/candidate ledger seal differs")
    check(not summary["engineering"] and summary["image_count"] == 5000 and bootstrap["seed"] == SEED
          and bootstrap["replicates"] == REPLICATES and bootstrap["draw_size"] == 5000, "Formal protocol/bootstrap scope differs")
    ids = summary["image_ids"]
    check(bootstrap["scoring_summary_sha256"] == sha(score / "SUMMARY.json"), "Bootstrap bound another score Run")
    ap_metrics = {}
    caches = []
    for arm in ARMS:
        arrays = score / arm / "COCO_ACCUMULATED.npz"
        cache_path = score / arm / "COCO_BOOTSTRAP_CACHE.npz"
        lock.add(arrays)
        lock.add(cache_path)
        metrics = metrics_from_arrays(arrays)
        close(metrics, summary["metrics"][arm], "Saved official AP/AR array reconstruction differs", 1e-14)
        ap_metrics[arm] = metrics
        cache = load_npz(cache_path)
        ap, p, r = reaccumulate(cache, np.arange(len(ids)))
        check(np.allclose(p, cache["official_precision"], rtol=0, atol=1e-14) and np.allclose(r, cache["official_recall"], rtol=0, atol=1e-14), "Cache/actual accumulated point parity differs")
        close(ap, metrics["AP"], "Joint point AP cache parity differs", 1e-14)
        caches.append(cache)
    cache_contract(caches)
    for name in ("FIXED_PAIRS.npz", "PAIRED_IMAGE_SUMS.npz", "source_provenance/FIXED_INSTANCES_SOURCE.jsonl"):
        lock.add(score / name)
    pairs = rebuild_pairs(score, ids)
    stored = load_npz(score / "PAIRED_IMAGE_SUMS.npz")
    check(np.allclose(pairs, stored["values"], rtol=0, atol=1e-12), "Independently rebuilt fixed pixel ledger image sums differ")
    for a, arm in enumerate(ARMS):
        close(paired_summary(pairs[a]), summary["paired"][arm], "Fixed pair reaggregation differs")
        for field, expected in (("matched_count", 86600), ("baseline_success", 36266), ("baseline_failure", 50334)):
            check(int(pairs[a, :, PAIR_FIELDS.index(field)].sum()) == expected, "Original fixed denominator differs")
    check(np.all(pairs[0, :, PAIR_FIELDS.index("damage")] == 0) and np.all(pairs[0, :, PAIR_FIELDS.index("repair")] == 0), "Baseline damage/repair nonzero")
    draws_path, reps_path = boot / "BOOTSTRAP_DRAWS.npz", boot / "BOOTSTRAP_REPLICATES.npz"
    check(lock.add(draws_path) == bootstrap["draws_sha256"] and lock.add(reps_path) == bootstrap["replicates_sha256"], "Raw bootstrap artifact hashes differ")
    draws, replicates = load_npz(draws_path), load_npz(reps_path)
    check(draws["draws"].shape == (1000, 5000) and draws["image_ids"].tolist() == ids, "Saved bootstrap ordered draw scope differs")
    generator = np.random.default_rng(SEED)
    # Reconstruct every paired cluster ratio from the exact saved image copies.
    metric_values = None
    for rep, saved in enumerate(draws["draws"]):
        expected = generator.integers(0, 5000, size=5000)
        check(np.array_equal(saved, expected), "Saved default_rng ordered draw differs")
        values = paired_replicate_values(pairs, saved)
        if metric_values is None:
            metric_values = {key: np.empty(1000) for key in values}
        for key, value in values.items():
            metric_values[key][rep] = value
            check(np.isclose(value, replicates[key][rep], rtol=0, atol=1e-14, equal_nan=True), "Saved paired bootstrap replicate differs")
    ap_replays = {}
    for rep in (0, 1, 499, 999):
        values = paired_aps(caches, draws["draws"][rep])
        check(np.allclose(values, replicates["AP"][rep], rtol=0, atol=1e-14), "Selected true joint AP bootstrap replay differs")
        ap_replays[str(rep)] = values.tolist()
    check(np.array_equal(replicates["AP_H_minus_I"], replicates["AP"][:, 2] - replicates["AP"][:, 1]), "AP difference raw replicates differ")
    intervals = {"AP." + arm: ci(replicates["AP"][:, a], summary["metrics"][arm]["AP"]) for a, arm in enumerate(ARMS)}
    intervals["AP.H_minus_I"] = ci(replicates["AP_H_minus_I"], summary["metrics"]["target_H"]["AP"] - summary["metrics"]["target_I"]["AP"])
    point = paired_replicate_values(pairs, np.arange(5000))
    intervals.update({key: ci(value, point[key]) for key, value in metric_values.items()})
    close(intervals, bootstrap["intervals"], "Bootstrap percentile interval reconstruction differs", 1e-14)
    boundary = summary["boundary"]
    if boundary["status"] == "completed":
        ordinary_base = metrics_from_arrays(score / "boundary" / "BASELINE_ORDINARY_SEGM" / "COCO_ACCUMULATED.npz")
        close(ordinary_base, ap_metrics["baseline"], "Boundary backend ordinary baseline parity differs", 1e-10)
        for arm in ARMS:
            metrics = metrics_from_arrays(score / "boundary" / arm / "COCO_ACCUMULATED.npz")
            close(metrics, boundary["metrics"][arm], "Boundary saved arrays differ", 1e-14)
    for name in ("verify_target_comparison.py", "bootstrap_target_ap.py", "target_scoring_common.py"):
        path = Path(__file__).with_name(name)
        lock.add(path)
    write_json(out / "SOURCE_LOCK.json", {"files": lock.finish()})
    write_json(out / "VERIFICATION.json", {"status": "passed", "passed": True,
        "image_count": 5000, "fixed_associations_rebuilt": 86600,
        "official_12_metrics_recomputed_from_accumulated_arrays": ap_metrics,
        "ordered_rng_draws_rebuilt": 1000, "all_fixed_cluster_bootstrap_replicates_rebuilt": 1000,
        "true_joint_AP_replicates_replayed": ap_replays,
        "percentile_intervals_recomputed": True, "source_before_after_and_current_sha_verified": True,
        "boundary_arrays_and_ordinary_base_parity_verified": boundary["status"] == "completed",
        "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "scoring_summary_sha256": sha(score / "SUMMARY.json"),
        "bootstrap_summary_sha256": sha(boot / "SUMMARY.json"), "GPU_or_model_execution": False,
        "GT_rematched": False, "limitation": "This checker reconstructs saved integer ledgers and actual COCO arrays; independent matching reference is the separate engineering Run"})
    write_json(out / "COMPLETE.json", {"status": "verification_complete", "verification_sha256": sha(out / "VERIFICATION.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scoring-run", type=Path, required=True)
    p.add_argument("--bootstrap-run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        verify(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "VERIFICATION_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "GPU_or_model_execution": False})
        raise
