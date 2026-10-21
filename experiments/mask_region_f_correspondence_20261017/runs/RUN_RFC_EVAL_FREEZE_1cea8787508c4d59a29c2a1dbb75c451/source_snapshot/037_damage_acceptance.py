"""Exact design-based damage bound for a frozen finite baseline-success population.

No iid-GT or independent-image model is assumed. Randomness is the preregistered
simple random sample without replacement, after checkpoint/configuration freeze.
The bound applies to this finite development population, not future images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
from fractions import Fraction

from posttrain_audit import (compare_outputs, key, population, read_json,
                             read_jsonl, require, sha256, write_new)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def hypergeom_cdf_counts(N, K, n, x):
    """Integer numerator/denominator; no tail underflow or floating point rounding."""
    require(all(type(v) is int for v in (N, K, n, x)), "integer counts required")
    require(0 <= K <= N and 0 <= n <= N, "invalid population/sample counts")
    denominator = math.comb(N, n)
    lo, hi = max(0, n - (N - K)), min(x, n, K)
    if hi < lo:
        return 0, denominator
    term = math.comb(K, lo) * math.comb(N - K, n - lo)
    total = term
    for j in range(lo, hi):
        numerator = term * (K - j) * (n - j)
        divisor = (j + 1) * (N - K - n + j + 1)
        term, remainder = divmod(numerator, divisor)
        require(remainder == 0, "internal hypergeometric recurrence error")
        total += term
    return total, denominator


def damage_upper_bound(N, n, x, alpha=Fraction(1, 20)):
    require(all(type(v) is int for v in (N, n, x)), "integer counts required")
    require(N > 0 and 0 < n <= N and 0 <= x <= n, "invalid damage counts")
    alpha = Fraction(alpha)
    require(0 < alpha < 1, "invalid alpha")
    lo, hi = x, N - (n - x)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        a, b = hypergeom_cdf_counts(N, mid, n, x)
        if a * alpha.denominator >= b * alpha.numerator:
            lo = mid
        else:
            hi = mid - 1
    return {
        "population_successes": N, "sample_size": n, "damaged": x,
        "point_rate": x / n, "upper_damaged_count": lo, "upper_rate": lo / N,
        "confidence": float(1 - alpha), "unit": "fraction",
        "method": "hypergeometric_CDF_inversion_inclusive",
        "scope": "frozen_finite_baseline_success_GT_population",
        "census": n == N,
    }


def success_population(rows):
    require(len({key(r) for r in rows}) == len(rows), "duplicate baseline GT")
    ids = sorted(key(r) for r in rows if population(r) == "S")
    require(ids, "empty baseline-success population")
    return ids


def population_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def select_damage_panel(rows, ledger_sha, checkpoint_sha, inference_sha, n=1024, seed=None):
    require(type(n) is int and n > 0, "positive integer sample size required")
    for value in (ledger_sha, checkpoint_sha, inference_sha):
        require(valid_sha(value), "invalid frozen artifact SHA")
    ids = success_population(rows)
    sources = {r["source_prediction_sha"] for r in rows}
    require(len(sources) == 1, "mixed prediction sources")
    seed = secrets.randbits(128) if seed is None else seed
    require(type(seed) is int and seed >= 0, "invalid seed")
    selected = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    return {
        "schema_version": 1, "design": "SRSWOR_success_GT_after_checkpoint_freeze",
        "baseline_ledger_sha256": ledger_sha, "source_prediction_sha": sources.pop(),
        "checkpoint_sha256": checkpoint_sha, "inference_contract_sha256": inference_sha,
        "population_sha256": population_digest(ids), "population_size": len(ids),
        "requested_sample_size": n, "sample_size": len(selected), "seed": seed,
        "alpha": "1/20", "maximum_damage_rate": "1/100",
        "sample": [dict(image_id=i, annotation_id=a) for i, a in selected],
        "image_ids": sorted({i for i, _ in selected}),
        "no_outcome_adaptive_selection": True,
    }


def score_damage(panel, baseline, method):
    ids = success_population(baseline)
    require(panel["design"] == "SRSWOR_success_GT_after_checkpoint_freeze", "wrong sampling design")
    require(panel["population_size"] == len(ids), "population size changed")
    require(panel["population_sha256"] == population_digest(ids), "population changed")
    require(panel["alpha"] == "1/20" and panel["maximum_damage_rate"] == "1/100", "gate changed")
    expected = sorted(random.Random(panel["seed"]).sample(ids, min(panel["requested_sample_size"], len(ids))))
    sampled = [key(r) for r in panel["sample"]]
    require(sampled == expected and panel["sample_size"] == len(expected), "sample altered")
    images = set(panel["image_ids"])
    require(images == {i for i, _ in expected}, "sample image list changed")
    require(len({key(r) for r in method}) == len(method), "duplicate method GT")
    wanted = {key(r): r for r in baseline if r["image_id"] in images}
    actual = {key(r): r for r in method}
    require(set(actual) == set(wanted), "evaluate ALL baseline GT in sampled images; no missing/extra GT")
    for k, row in actual.items():
        require(row["is_ignored"] == wanted[k]["is_ignored"], "ignore policy changed")
        require(row["gt_category_id"] == wanted[k]["gt_category_id"], "GT category changed")
        require(row["task_mask75"] in (("ignored",) if row["is_ignored"] else ("success", "failure")),
                "invalid task status")
    damaged = sum(actual[k]["task_mask75"] == "failure" for k in sampled)
    result = damage_upper_bound(len(ids), len(sampled), damaged)
    # Compare rational counts exactly at the 1% gate.
    if result["upper_damaged_count"] * 100 <= len(ids):
        gate = "damage_guardrail_passed"
    elif damaged * 100 > len(sampled):
        gate = "damage_guardrail_failed"
    else:
        gate = "blocked_insufficient_precision"
    return {**result, "gate": gate, "evaluated_images": len(images),
            "evaluated_GT": len(method),
            "counted_GT": len(sampled),
            "note": "Count only sampled S; other GT in the same image are not additional sampled units."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--baseline-ledger", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--inference-contract", required=True)
    p.add_argument("--sample-size", type=int, default=1024)
    p.add_argument("--seed", type=int, help="For a pre-recorded random draw/reproduction; never search seeds.")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    for name in ("panel", "baseline-ledger", "method-ledger", "baseline-predictions",
                 "method-predictions", "checkpoint", "inference-contract", "out"):
        p.add_argument("--" + name, required=True)
    a = parser.parse_args()
    baseline = read_jsonl(a.baseline_ledger)
    if a.command == "select":
        result = select_damage_panel(baseline, sha256(a.baseline_ledger), sha256(a.checkpoint),
                                     sha256(a.inference_contract), a.sample_size, a.seed)
    else:
        panel = read_json(a.panel)
        for field, path in (("baseline_ledger_sha256", a.baseline_ledger),
                            ("checkpoint_sha256", a.checkpoint),
                            ("inference_contract_sha256", a.inference_contract),
                            ("source_prediction_sha", a.baseline_predictions)):
            require(panel[field] == sha256(path), f"changed artifact: {field}")
        require({r["source_prediction_sha"] for r in baseline} == {panel["source_prediction_sha"]},
                "baseline ledger/prediction identity mismatch")
        images = set(panel["image_ids"])
        base_predictions = [r for r in read_json(a.baseline_predictions) if r["image_id"] in images]
        method_predictions = read_json(a.method_predictions)
        require(all(r["image_id"] in images for r in method_predictions), "unexpected prediction image")
        compare_outputs(base_predictions, method_predictions, zero_delta=False)
        method = read_jsonl(a.method_ledger)
        require({r["source_prediction_sha"] for r in method} == {sha256(a.method_predictions)},
                "method ledger/prediction identity mismatch")
        result = score_damage(panel, baseline, method)
        result.update(panel_sha256=sha256(a.panel), method_predictions_sha256=sha256(a.method_predictions),
                      checkpoint_sha256=sha256(a.checkpoint),
                      inference_contract_sha256=sha256(a.inference_contract))
    write_new(a.out, result)


if __name__ == "__main__":
    main()
