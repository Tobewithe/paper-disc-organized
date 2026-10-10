"""Independent, read-only verification of native coefficient conflict outputs.

This program deliberately does not import the audit implementation.  It verifies
the saved candidate/pair tables, reconstructs pairing from native state, redoes
the statistics, and applies the pre-registered gate.  It never loads a model or
updates a coefficient, target, oracle, or audit output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


FIELDS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
SPLITS = ("fit", "dev", "val")
METRICS = ("same_neg_frac", "cross_neg_frac", "same_cos_mean", "cross_cos_mean", "same_neg_mass", "cross_neg_mass")
SEED, REPLICATES = 20261011, 2000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(row):
    return tuple(row[field] for field in FIELDS)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def close(a, b, tolerance=1e-9):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.failures = []

    def check(self, condition, label, detail=None):
        self.counts[label] += 1
        if not condition:
            self.failures.append({"check": label, "detail": detail})

    def numeric(self, actual, expected, label, detail=None):
        self.check(close(actual, expected), label, {"actual": actual, "expected": expected, "context": detail})


def correlation(x, y):
    """Spearman using independently implemented tie ranks and dot products."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    ranks = []
    for values in (a, b):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        upper = np.cumsum(counts)
        ranks.append((upper - (counts - 1) / 2.0)[inverse])
    a, b = (r - r.mean() for r in ranks)
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def percentile_summary(samples, estimate, **counts):
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [2.5, 97.5]) if values.size else (float("nan"), float("nan"))
    return {"estimate": float(estimate), "lo": float(low), "hi": float(high), **counts}


def candidate_difference(rows, metric):
    groups = [[float(r[metric]) for r in rows if r[g] and finite(r[metric])] for g in ("recoverable", "unrecoverable")]
    a, b = [np.array(values, dtype=np.float64) for values in groups]
    if not a.size or not b.size:
        return percentile_summary([], float("nan"), n_a=int(a.size), n_b=int(b.size))
    generator = np.random.default_rng(SEED)
    samples = []
    for _ in range(REPLICATES):
        ai = generator.integers(a.size, size=a.size)
        bi = generator.integers(b.size, size=b.size)
        samples.append(a[ai].mean() - b[bi].mean())
    return percentile_summary(samples, a.mean() - b.mean(), n_a=int(a.size), n_b=int(b.size))


def image_difference(rows, metric):
    """Resample whole images jointly, preserving groups that share an image.

    Each image contributes its group-specific candidate mean.  Missing groups
    remain missing.  The sampling unit is the union of metric-valid images.
    """
    buckets = defaultdict(lambda: {"recoverable": [], "unrecoverable": []})
    for row in rows:
        if not finite(row[metric]):
            continue
        for group in ("recoverable", "unrecoverable"):
            if row[group]:
                buckets[row["image_id"]][group].append(float(row[metric]))
    ordered = sorted(buckets)
    values = np.array([[np.mean(buckets[i][g]) if buckets[i][g] else float("nan") for g in ("recoverable", "unrecoverable")] for i in ordered], dtype=np.float64).reshape(-1, 2)
    counts = {"n_a": int(np.isfinite(values[:, 0]).sum()), "n_b": int(np.isfinite(values[:, 1]).sum()), "n_images": len(ordered)}
    if not counts["n_a"] or not counts["n_b"]:
        return percentile_summary([], float("nan"), **counts)
    estimate = np.nanmean(values[:, 0]) - np.nanmean(values[:, 1])
    generator = np.random.default_rng(SEED + 17)
    samples = []
    for _ in range(REPLICATES):
        selected = values[generator.integers(len(values), size=len(values))]
        if np.isfinite(selected[:, 0]).any() and np.isfinite(selected[:, 1]).any():
            samples.append(np.nanmean(selected[:, 0]) - np.nanmean(selected[:, 1]))
    return percentile_summary(samples, estimate, **counts)


def correlation_interval(rows, image_macro=False):
    valid = [r for r in rows if finite(r["cross_neg_frac"]) and finite(r["oracle_gap"])]
    if image_macro:
        buckets = defaultdict(list)
        for row in valid:
            buckets[row["image_id"]].append((float(row["cross_neg_frac"]), float(row["oracle_gap"])))
        values = np.array([np.mean(v, axis=0) for v in buckets.values()], dtype=np.float64).reshape(-1, 2)
    else:
        values = np.array([(r["cross_neg_frac"], r["oracle_gap"]) for r in valid], dtype=np.float64).reshape(-1, 2)
    if len(values) < 3:
        return percentile_summary([], float("nan"), n=len(values))
    generator = np.random.default_rng(SEED + (31 if image_macro else 0))
    samples = []
    for _ in range(REPLICATES):
        selected = values[generator.integers(len(values), size=len(values))]
        samples.append(correlation(selected[:, 0], selected[:, 1]))
    return percentile_summary(samples, correlation(values[:, 0], values[:, 1]), n=len(values))


def verify(args):
    audit, out = Path(args.audit), Path(args.out)
    if audit.resolve() == out.resolve():
        raise ValueError("verification must use a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    filenames = ("PER_INSTANCE.jsonl", "PER_PAIR.jsonl", "INPUT_MANIFEST.json", "PAIR_QUANTILES.json", "BOOTSTRAP.json", "SUMMARY.json", "DECISION.json", "run.json", "COMPLETE.json")
    before = {name: digest(audit / name) for name in filenames}
    rows, pairs = read_lines(audit / "PER_INSTANCE.jsonl"), read_lines(audit / "PER_PAIR.jsonl")
    manifest, quantiles, boot, summary, decision, source_run = [read_json(audit / name) for name in ("INPUT_MANIFEST.json", "PAIR_QUANTILES.json", "BOOTSTRAP.json", "SUMMARY.json", "DECISION.json", "run.json")]
    checks = Checks()
    oracle_rows = read_lines(args.oracle)
    oracle = {identity(r): r for r in oracle_rows}
    checks.check(len(oracle) == len(oracle_rows), "oracle_identity_unique")
    by_identity = {identity(r): r for r in rows}
    checks.check(len(by_identity) == len(rows), "candidate_identity_unique")
    checks.check(manifest["identity_count"] == len(rows), "manifest_identity_count")
    identities = [{field: r[field] for field in FIELDS} for r in rows]
    calculated_hash = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    checks.check(calculated_hash == manifest["identity_sha256"], "manifest_identity_hash")
    checks.check(digest(args.oracle) == manifest["oracle_sha256"], "oracle_hash")
    checks.check(digest(args.cache_index) == manifest["cache_index_sha256"], "cache_index_hash")
    index = read_json(args.cache_index)
    selection = manifest["selected_images"]
    checks.check(selection == summary["selected_images"], "selected_images_summary")
    for split in SPLITS:
        source_ids = [int(r["image_id"]) for r in index[split]]
        checks.check(len(source_ids) == len(set(source_ids)), "source_split_unique", split)
        expected = sorted(source_ids, key=lambda iid: (hashlib.sha256(f"native-conflict-v1/{split}/{iid}".encode()).hexdigest(), iid))[:int(source_run["max_images"])]
        checks.check(selection[split] == expected, "deterministic_image_selection", split)
        checks.check(len(selection[split]) == len(set(selection[split])), "selected_images_unique", split)
        checks.check({int(r["image_id"]) for r in rows if r["split"] == split}.issubset(set(selection[split])), "candidate_images_selected", split)
    for left, right in (("fit", "dev"), ("fit", "val"), ("dev", "val")):
        checks.check(not (set(selection[left]) & set(selection[right])), "split_images_disjoint", [left, right])

    for row in rows:
        key = identity(row)
        checks.check(row["split"] in SPLITS and row["branch"] == "one2one" and row["level"] == row["pyramid_level"], "candidate_identity_fields", key)
        checks.check(key in oracle, "oracle_identity_join", key)
        if key in oracle:
            for name in ("iou_A", "iou_C", "objective_A", "objective_C"):
                checks.numeric(row[name], oracle[key][name], "oracle_value", [key, name])
        checks.numeric(row["oracle_gap"], row["objective_A"] - row["objective_C"], "oracle_gap", key)
        checks.check(row["recoverable"] == (row["iou_A"] < .75 <= row["iou_C"]), "recoverable_definition", key)
        checks.check(row["unrecoverable"] == (row["iou_A"] < .75 and row["iou_C"] < .75), "unrecoverable_definition", key)
        checks.check(finite(row["gradient_norm"]) and row["gradient_norm"] > 0, "gradient_norm_finite_positive", key)
        checks.check(all(finite(row[name]) and row[name] >= 0 for name in ("h_norm", "c_norm")), "matching_state_finite", key)

    # Reconstruct the complete expected same-image pairs and deterministic
    # matched cross-image pairs from identities/native-state norms only.
    by_image, by_level = defaultdict(list), defaultdict(list)
    for row in rows:
        by_image[(row["split"], row["level"], row["image_id"])].append(identity(row))
        by_level[(row["split"], row["level"])].append(identity(row))
    expected_same = set()
    for keys in by_image.values():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                expected_same.add(frozenset((keys[i], keys[j])))
    expected_cross = {}
    supplied_quantiles = {(q["split"], q["level"]): q for q in quantiles}
    checks.check(len(supplied_quantiles) == len(quantiles) and set(supplied_quantiles) == set(by_level), "quantile_groups")
    for level_key, keys in by_level.items():
        hq = np.percentile([by_identity[k]["h_norm"] for k in keys], [25, 50, 75])
        cq = np.percentile([by_identity[k]["c_norm"] for k in keys], [25, 50, 75])
        recorded = supplied_quantiles.get(level_key, {})
        checks.check(recorded.get("n") == len(keys), "quantile_candidate_count", level_key)
        for name, expected in (("h_quantiles", hq), ("c_quantiles", cq)):
            actual = recorded.get(name, [])
            checks.check(len(actual) == 3 and all(close(a, b) for a, b in zip(actual, expected)), "quantile_values", [level_key, name])
        cells = defaultdict(list)
        for key in keys:
            row = by_identity[key]
            cell = (int(np.count_nonzero(hq <= row["h_norm"])), int(np.count_nonzero(cq <= row["c_norm"])))
            cells[cell].append(key)
        for cell, members in cells.items():
            pending = sorted(members, key=lambda k: (by_identity[k]["image_id"], by_identity[k]["annotation_id"], by_identity[k]["raw_id"]))
            while pending:
                first = pending.pop(0)
                partner_position = next((i for i, key in enumerate(pending) if by_identity[key]["image_id"] != by_identity[first]["image_id"]), None)
                if partner_position is None:
                    continue
                second = pending.pop(partner_position)
                expected_cross[frozenset((first, second))] = cell

    actual_same, actual_cross = set(), set()
    accum = defaultdict(list)
    cross_uses = Counter()
    for pair in pairs:
        left, right = identity(pair["i_identity"]), identity(pair["j_identity"])
        pair_key = frozenset((left, right))
        checks.check(left != right and left in by_identity and right in by_identity, "pair_endpoints_exist_distinct", [left, right])
        if left not in by_identity or right not in by_identity:
            continue
        a, b = by_identity[left], by_identity[right]
        checks.check(a["split"] == b["split"] == pair["split"], "pair_same_split", [left, right])
        checks.check(a["level"] == b["level"] == pair["level"], "pair_same_level", [left, right])
        checks.check(pair["image_i"] == a["image_id"] and pair["image_j"] == b["image_id"], "pair_image_fields", [left, right])
        c = pair["cosine"]
        checks.check(finite(c) and -1.000001 <= c <= 1.000001, "pair_cosine_finite_bounded", [left, right])
        checks.numeric(pair["negative"], int(c < 0), "pair_negative", [left, right])
        checks.numeric(pair["negative_mass"], max(0.0, -c), "pair_negative_mass", [left, right])
        pair_type = pair["pair_type"]
        checks.check(pair_type in ("same_image", "cross_image_matched"), "pair_type", pair_type)
        if pair_type == "same_image":
            checks.check(a["image_id"] == b["image_id"], "same_pair_image", [left, right])
            checks.check(pair_key not in actual_same, "same_pair_unique", [left, right])
            actual_same.add(pair_key)
            prefix = "same"
        elif pair_type == "cross_image_matched":
            checks.check(a["image_id"] != b["image_id"], "cross_pair_image", [left, right])
            checks.check(pair_key not in actual_cross, "cross_pair_unique", [left, right])
            actual_cross.add(pair_key)
            cross_uses.update((left, right))
            checks.check((pair.get("cell_h"), pair.get("cell_c")) == expected_cross.get(pair_key), "cross_pair_cell", [left, right])
            checks.numeric(pair["match_distance"], abs(a["h_norm"] - b["h_norm"]) + abs(a["c_norm"] - b["c_norm"]), "cross_pair_distance", [left, right])
            prefix = "cross"
        else:
            continue
        for key in (left, right):
            accum[(key, prefix)].append(c)
    checks.check(actual_same == expected_same, "complete_same_pair_manifest", {"actual": len(actual_same), "expected": len(expected_same)})
    checks.check(actual_cross == set(expected_cross), "exact_cross_pair_manifest", {"actual": len(actual_cross), "expected": len(expected_cross)})
    checks.check(all(count <= 1 for count in cross_uses.values()), "cross_candidate_used_once")
    for row in rows:
        key = identity(row)
        for prefix in ("same", "cross"):
            values = np.array(accum[(key, prefix)], dtype=np.float64)
            checks.check(row[prefix + "_pair_count"] == len(values), "candidate_pair_count", [key, prefix])
            metrics = {"cos_mean": float(values.mean()) if values.size else float("nan"), "neg_frac": float((values < 0).mean()) if values.size else float("nan"), "neg_mass": float(np.maximum(0, -values).mean()) if values.size else float("nan")}
            for suffix, expected in metrics.items():
                checks.numeric(row[prefix + "_" + suffix], expected, "candidate_pair_summary", [key, prefix, suffix])

    checks.check(boot["seed"] == SEED and boot["B"] == REPLICATES, "bootstrap_preregistered_settings")
    recomputed = {"seed": SEED, "B": REPLICATES, "metrics": {}, "correlation": {}}
    group_counts = {}
    for split in SPLITS:
        failures = [r for r in rows if r["split"] == split and r["iou_A"] < .75]
        counts = {g: sum(bool(r[g]) for r in failures) for g in ("recoverable", "unrecoverable")}
        counts.update({g + "_images": len({r["image_id"] for r in failures if r[g]}) for g in ("recoverable", "unrecoverable")})
        group_counts[split] = counts
        for name, value in counts.items():
            checks.check(summary["groups"][split][name] == value, "summary_group_count", [split, name])
        recomputed["metrics"][split] = {}
        for metric in METRICS:
            result = {"candidate": candidate_difference(failures, metric), "image_macro": image_difference(failures, metric)}
            recomputed["metrics"][split][metric] = result
            for unit, values in result.items():
                recorded = boot["metrics"][split][metric][unit]
                for name in ("estimate", "lo", "hi", "n_a", "n_b"):
                    checks.numeric(recorded.get(name), values[name], "bootstrap_difference", [split, metric, unit, name])
            for group in ("recoverable", "unrecoverable"):
                valid = [r[metric] for r in failures if r[group] and finite(r[metric])]
                recorded = summary["means"][split][metric][group]
                checks.check(recorded["n"] == len(valid), "summary_metric_count", [split, metric, group])
                checks.numeric(recorded["mean"], np.mean(valid) if valid else float("nan"), "summary_metric_mean", [split, metric, group])
        matched = [r for r in failures if r["cross_pair_count"] > 0]
        recomputed["correlation"][split] = {}
        for macro, name in ((False, "candidate_cross_conflict_vs_oracle_gap"), (True, "image_macro_cross_conflict_vs_oracle_gap")):
            result = correlation_interval(matched, macro)
            recomputed["correlation"][split][name] = result
            for field in ("estimate", "lo", "hi", "n"):
                checks.numeric(boot["correlation"][split][name].get(field), result[field], "bootstrap_spearman", [split, name, field])

    v = group_counts["val"]
    group_gate = v["recoverable"] >= 50 and v["unrecoverable"] >= 50 and v["recoverable_images"] >= 20 and v["unrecoverable_images"] >= 20
    metric_gate = all(recomputed["metrics"]["val"][metric]["candidate"]["n_" + suffix] >= 50 for metric in ("same_neg_frac", "cross_neg_frac") for suffix in ("a", "b"))
    count_gate = bool(group_gate and metric_gate)
    lower_bounds = [recomputed["metrics"]["val"][m][unit]["lo"] for m in ("same_neg_frac", "cross_neg_frac") for unit in ("candidate", "image_macro")]
    corr_low = recomputed["correlation"]["val"]["candidate_cross_conflict_vs_oracle_gap"]["lo"]
    dev_point = recomputed["correlation"]["dev"]["candidate_cross_conflict_vs_oracle_gap"]["estimate"]
    positive = count_gate and all(finite(x) and x > 0 for x in lower_bounds + [corr_low, dev_point])
    expected_status = "positive_authorize_conflict_routing" if positive else "negative_close_method_line" if count_gate else "unknown_insufficient_groups"
    checks.check(decision["gate_counts"] == count_gate, "decision_effective_count_gate")
    checks.check(decision["status"] == expected_status, "decision_status", {"actual": decision["status"], "expected": expected_status})
    checks.check(len(decision["required_lower_bounds"]) == len(lower_bounds) and all(close(a, b) for a, b in zip(decision["required_lower_bounds"], lower_bounds)), "decision_lower_bounds")
    checks.numeric(decision["val_correlation_lower"], corr_low, "decision_val_correlation")
    checks.numeric(decision["dev_correlation_point"], dev_point, "decision_dev_correlation")
    checks.check(summary["counts"]["rows"] == len(rows) and summary["counts"]["pairs"] == len(pairs), "summary_total_counts")
    for name in filenames:
        checks.check(before[name] == digest(audit / name), "source_output_unchanged", name)
    report = {"status": "passed" if not checks.failures else "failed", "audit": str(audit), "checks": dict(checks.counts), "check_count": sum(checks.counts.values()), "failures": checks.failures, "rows": len(rows), "pairs": len(pairs), "recomputed_decision": expected_status, "scope": "Identity, pairing, per-candidate summaries, recovery groups, all six bootstrap contrasts, Spearman, and decision. Gradient formula validation remains in the audit's independently checked smoke evidence.", "source_sha256": before, "verification_script_sha256": digest(__file__)}
    (out / "VALIDATION.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    (out / "RECOMPUTED_BOOTSTRAP.json").write_text(json.dumps(recomputed, indent=2, allow_nan=True), encoding="utf-8")
    run = {"run_id": out.name, "status": "complete" if not checks.failures else "failed", "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": sys.argv, "python": platform.python_version(), "numpy": np.__version__, "script_sha256": digest(__file__), "audit": str(audit), "oracle": str(args.oracle), "cache_index": str(args.cache_index)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "COMPLETE.json").write_text(json.dumps({"status": report["status"], "check_count": report["check_count"], "failure_count": len(checks.failures)}, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["check_count"], "failure_count": len(checks.failures), "decision": expected_status}, indent=2))
    return 0 if not checks.failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="completed audit Run directory")
    parser.add_argument("--oracle", required=True, help="unchanged existing PER_CANDIDATE.jsonl")
    parser.add_argument("--cache-index", required=True, help="official cache INDEX.json")
    parser.add_argument("--out", required=True, help="separate verification Run directory")
    raise SystemExit(verify(parser.parse_args()))
