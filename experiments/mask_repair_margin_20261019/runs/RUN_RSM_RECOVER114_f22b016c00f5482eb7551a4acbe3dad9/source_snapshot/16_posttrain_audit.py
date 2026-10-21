"""Fail-closed, standard-library checks for fixed-output post-training evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    """Never silently overwrite an existing scientific artifact."""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, lo=None, hi=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"not a finite number: {value!r}")
    require(lo is None or value >= lo, f"value below {lo}")
    require(hi is None or value <= hi, f"value above {hi}")
    return value


def key(row):
    return (row["image_id"], row["annotation_id"])


def population(row):
    if row["is_ignored"]:
        return "ignored"
    require(row["task_mask75"] in ("success", "failure"), "invalid mask task status")
    if row["task_mask75"] == "success":
        return "S"
    if row["bbox75_match_output_id"] is None:
        return "F_without_box75"
    finite(row["box_iou_of_bbox_match"], .75, 1)
    miou = finite(row["mask_iou_of_bbox_match"], 0, 1)
    return "T" if miou < .75 else "U"


def make_panel(rows, ledger_sha, limit=None):
    """Select by baseline only. Default: ALL T; limited panels use fixed hash order."""
    require(limit is None or limit > 0, "limit must be positive")
    require(len({key(r) for r in rows}) == len(rows), "duplicate GT rows")
    counts = Counter()
    candidates = []
    source_shas = set()
    for r in rows:
        counts[population(r)] += 1
        source_shas.add(r["source_prediction_sha"])
        if population(r) == "T":
            candidates.append({
                "image_id": r["image_id"], "annotation_id": r["annotation_id"],
                "output_detection_id": r["bbox75_match_output_id"],
                "baseline_mask_iou": r["mask_iou_of_bbox_match"],
                "gt_area": r["gt_area"], "category_id": r["gt_category_id"],
            })
    require(len(source_shas) == 1, "mixed or absent source predictions")
    require(len({r["output_detection_id"] for r in candidates}) == len(candidates),
            "bbox matching must be one-to-one for ordinary GT")
    candidates.sort(key=lambda r: hashlib.sha256(
        f"{r['image_id']}:{r['annotation_id']}".encode()).hexdigest())
    targets = candidates if limit is None else candidates[:limit]
    return {
        "schema_version": 1, "ledger_sha256": ledger_sha,
        "source_prediction_sha": next(iter(source_shas)),
        "selection": "all_T" if limit is None else "sha256_image_id_colon_annotation_id",
        "limit": limit, "counts": dict(counts),
        "ordinary_gt": len(rows) - counts["ignored"],
        "all_task_failures": counts["T"] + counts["U"] + counts["F_without_box75"],
        "eligible_T": len(candidates), "selected_T": len(targets), "targets": targets,
        "scope": "baseline-selected fixed output pairs; not an oracle result",
    }


def summarize_oracle(panel, results, controls):
    wanted = {key(r): r for r in panel["targets"]}
    require(len(wanted) == len(panel["targets"]), "duplicate panel targets")
    require(len({key(r) for r in results}) == len(results), "duplicate oracle GT")
    require({key(r) for r in results} == set(wanted), "missing or unexpected oracle GT")
    require(controls.get("schema_version") == 1, "control report schema missing")
    require(controls.get("positive_control_passed") is True, "solver positive control failed/missing")
    require(controls.get("same_solver_config") is True, "controls used a different solver config")
    require(controls.get("native_zero_delta_passed") is True, "native decoder identity unverified")
    require(controls.get("known_coefficient_mask_control_passed") is True, "known coefficient mask control unverified")
    for field in ("solver_code_sha256", "oracle_config_sha256"):
        require(isinstance(controls.get(field), str) and len(controls[field]) == 64,
                f"missing control identity: {field}")
    statuses = Counter()
    recovered = 0
    by_area = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for r in results:
        target = wanted[key(r)]
        require(r["output_detection_id"] == target["output_detection_id"], "output identity changed")
        require(r["source_prediction_sha"] == panel["source_prediction_sha"], "prediction source changed")
        require(r.get("baseline_replay_equal") is True, "zero-delta replay mismatch")
        require(r.get("solver_code_sha256") == controls["solver_code_sha256"], "solver code changed")
        require(r.get("oracle_config_sha256") == controls["oracle_config_sha256"], "oracle config changed")
        require(r.get("baseline_mask_iou") == target["baseline_mask_iou"], "baseline IoU changed")
        status = r["solver_status"]
        require(status in ("converged", "budget_exhausted", "numerical_error"), "unknown solver status")
        statuses[status] += 1
        # Numerical errors remain in the denominator and block scientific adjudication.
        found = False
        if status != "numerical_error":
            found = finite(r["oracle_mask_iou"], 0, 1) >= .75
            finite(r["delta_norm"], 0)
            finite(r["gradient_inf_norm"], 0)
        area = finite(target["gt_area"], 0)
        band = "small" if area < 1024 else "medium" if area < 9216 else "large"
        by_area[band][0] += int(found)
        by_area[band][1] += 1
        recovered += int(found)
    n, failures = len(results), panel["all_task_failures"]
    ratio = lambda a, b: a / b if b else None
    if statuses["numerical_error"]:
        gate = "blocked_numerical_error"
    elif n < 200:
        gate = "blocked_insufficient_sample"
    elif recovered / n >= .20:
        gate = "opportunity_threshold_met"
    elif statuses["budget_exhausted"]:
        gate = "blocked_solver_budget"
    else:
        gate = "opportunity_threshold_not_met"
    return {
        "selected_T": n, "eligible_T": panel["eligible_T"],
        "all_task_failures": failures, "found_same_detection_mask75": recovered,
        "solver_status_counts": dict(statuses),
        "target_share_of_all_failures": ratio(panel["eligible_T"], failures),
        "found_fraction_of_selected_T": ratio(recovered, n),
        "observed_found_fraction_of_all_failures": ratio(recovered, failures),
        "by_gt_area": {k: {"found": v[0], "denominator": v[1]} for k, v in by_area.items()},
        "gate_candidate_status": gate,
        "scientific_status": "requires_run_evidence_review",
        "units": "fractions_0_to_1",
        "limits": [
            "Same-detection oracle IoU, not deployment recovery or COCO AP.",
            "If sampled, found/all failures is an observed subset lower bound, not an extrapolation.",
            "Budget exhaustion without enough found solutions is inconclusive, not proof of non-representability.",
        ],
    }


def compare_outputs(baseline, method, zero_delta=False):
    require(len(baseline) == len(method), "prediction row count changed")
    for i, (a, b) in enumerate(zip(baseline, method)):
        for field in ("image_id", "category_id", "score", "bbox"):
            require(field in a and a[field] == b.get(field), f"row {i}: changed {field}")
        finite(b["score"], 0, 1)
        require(len(b["bbox"]) == 4, f"row {i}: invalid box")
        for value in b["bbox"]:
            finite(value)
        require("segmentation" in a and "segmentation" in b, f"row {i}: mask missing")
        require(a["segmentation"]["size"] == b["segmentation"]["size"], f"row {i}: RLE size changed")
        if zero_delta:
            require(a["segmentation"] == b["segmentation"],
                    f"row {i}: zero-delta RLE changed (exact serialized RLE comparison)")
    return {"rows": len(baseline), "output_invariants": "passed",
            "zero_delta_exact_rle": "passed" if zero_delta else "not_checked"}



def check_run_identity(before, after, fit_ids, dev_ids, trainable_names, allowed_names):
    require(bool(before.get("tensors")) and bool(before.get("training_modes")), "empty base state manifest")
    require(before == after, "base parameters, buffers or module training modes changed")
    require(all(mode is False for mode in before["training_modes"].values()), "base model is not fully in eval mode")
    for group in (fit_ids, dev_ids):
        require(bool(group) and len(set(group)) == len(group), "empty/duplicate image IDs")
    require(not set(fit_ids) & set(dev_ids), "fit/dev image leakage")
    require(bool(allowed_names) and len(set(allowed_names)) == len(allowed_names), "empty/duplicate trainable whitelist")
    require(len(set(trainable_names)) == len(trainable_names) and set(trainable_names) == set(allowed_names),
            "trainable names differ from frozen whitelist")
    return {"base_state_and_modes": "unchanged_eval", "fit_dev_overlap": 0,
            "fit_images": len(fit_ids), "dev_images": len(dev_ids),
            "trainable_whitelist": "passed",
            "scope": "checks supplied manifests; runner must generate them from actual models"}


def fingerprint_base_state(model):
    """Runner hook: include ALL state_dict parameters/buffers and module train/eval modes."""
    result = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        result[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    return {"tensors": result, "training_modes": {name: bool(m.training) for name, m in model.named_modules()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--ledger", required=True)
    sel.add_argument("--limit", type=int)
    sel.add_argument("--out", required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--panel", required=True)
    summ.add_argument("--results", required=True)
    summ.add_argument("--controls", required=True)
    summ.add_argument("--out", required=True)
    comp = sub.add_parser("compare-outputs")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--method", required=True)
    comp.add_argument("--zero-delta", action="store_true")
    comp.add_argument("--out", required=True)
    state = sub.add_parser("check-run-identity")
    for name in ("before", "after", "fit-ids", "dev-ids", "trainable-names", "allowed-names", "out"):
        state.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.action == "select":
        result = make_panel(read_jsonl(args.ledger), sha256(args.ledger), args.limit)
    elif args.action == "summarize":
        result = summarize_oracle(read_json(args.panel), read_jsonl(args.results), read_json(args.controls))
        result["inputs_sha256"] = {k: sha256(getattr(args, k)) for k in ("panel", "results", "controls")}
    elif args.action == "check-run-identity":
        fields = ("before", "after", "fit_ids", "dev_ids", "trainable_names", "allowed_names")
        result = check_run_identity(*(read_json(getattr(args, k)) for k in fields))
        result["inputs_sha256"] = {k: sha256(getattr(args, k)) for k in fields}
    else:
        result = compare_outputs(read_json(args.baseline), read_json(args.method), args.zero_delta)
        result["inputs_sha256"] = {k: sha256(getattr(args, k)) for k in ("baseline", "method")}
    write_new(args.out, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
