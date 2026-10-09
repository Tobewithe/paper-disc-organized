"""Resolve all scorer input paths and read the frozen original association gate.

No GT decoding, matching, model inference, AP, Boundary or quality selection.
The original frozen evaluator's fixed_source function is called unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.util
import sys
import evaluate_target_comparison as evaluator
from target_scoring_common import ANN_SHA, ARMS, BASE_PRED_SHA, InputLock, check, jsonl, read_json, sha, write_json

EXPECTED_FIXED = {"INSTANCES.jsonl": "1a06dc886dc5d7f9db7059c7c65d2ba1299b4c29d5b641fe53f5b51194e9b58a",
    "paired/SUMMARY.json": "234c84be749861920362683b9f0bdbd0c52a5aeb16a96260e1d2c2b1794eb183",
    "final/SUMMARY.json": "9dac293ec5aace746a75a0299b76dd66e25f3fd3f82af0d0864e0fa6b33397c8"}

def main(args):
    plan = read_json(args.plan)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / "SUMMARY.json").exists(), "New input preflight Run required")
    lock = InputLock()
    for path in (args.plan, Path(__file__), Path(evaluator.__file__), Path(evaluator.__file__).with_name("target_scoring_common.py")):
        lock.add(path)
    root = Path(plan["study"])
    inference = root / "runs" / plan["upstream_runs"]["inference"]
    ids = read_json(inference / "SUMMARY.json")["image_ids"]
    check(len(ids) == 5000 and len(set(ids)) == 5000, "Complete source val scope required")
    fixed = Path(plan["fixed_source"])
    check(lock.add(fixed) == EXPECTED_FIXED["INSTANCES.jsonl"]
          and lock.add(fixed.parent / "SUMMARY.json") == EXPECTED_FIXED["paired/SUMMARY.json"]
          and lock.add(fixed.parent.parent / "SUMMARY.json") == EXPECTED_FIXED["final/SUMMARY.json"],
          "Actual laptop fixed-source three files differ from accepted original desktop bytes")
    grouped, receipt = evaluator.fixed_source(fixed, ids, False, lock)
    check(receipt["full_rows"] == receipt["selected_rows"] == 86600
          and receipt["full_baseline_success"] == 36266 and receipt["full_baseline_failure"] == 50334,
          "Original final8 frozen associations do not close")
    required = [inference / n for n in ("run.json", "SUMMARY.json", "COMPLETE.json", "INPUTS.json", "SOURCE_LOCK.json", "NATIVE_DECISIONS.jsonl", "BASELINE_PARITY_IMAGES.jsonl")]
    reference = Path(plan["reference_baseline"])
    required += [reference / "predictions.json", reference / "COMPLETE.json", reference.parent / "BASELINE_PARITY_IMAGES.jsonl",
                 Path(plan["reference_scoring"]) / "baseline/COCO_METRICS.json", root / "PROTOCOL.md", Path(plan["annotations"])]
    inputs = read_json(inference / "INPUTS.json")
    for arm in ARMS:
        required += [inference / arm / "predictions.json", inference / arm / "COMPLETE.json"]
        required += [inference / arm / "images" / f"{iid:012d}.json" for iid in ids]
        if arm != "baseline":
            required += [Path(inputs["model_runs"][arm]) / name for name in ("model.json", "SUMMARY.json")]
    required += [reference / "images" / f"{iid:012d}.json" for iid in ids]
    for value in read_json(inference / "SOURCE_LOCK.json")["files"].values():
        required.append(Path(value["path"]))
        if value.get("snapshot"):
            required.append(inference / value["snapshot"])
    vendor = Path(plan["boundary_vendor"])
    provenance = Path(plan["boundary_provenance"])
    check(lock.add(provenance) == plan["boundary_provenance_sha256"], "Boundary provenance changed")
    vendor_map = read_json(provenance)["files"]
    check(len(vendor_map) == 6, "Six exact Boundary sources required")
    for relative, expected in vendor_map.items():
        check(lock.add(vendor / relative) == expected, "Boundary source changed")
        required.append(vendor / relative)
    engineering = Path(plan["AP_engineering_run"])
    required += [engineering / name for name in ("run.json", "SUMMARY.json", "COMPLETE.json", "SOURCE_LOCK.json")]
    eng_summary = read_json(engineering / "SUMMARY.json")
    required += [engineering / relative for relative in eng_summary["artifacts"]]
    for path in required:
        check(path.is_file(), "Missing exact scorer/consumer input: " + str(path))
    check(lock.add(plan["annotations"]) == ANN_SHA and lock.add(reference / "predictions.json") == BASE_PRED_SHA,
          "Original annotations/native baseline source SHA differs")
    reference_metrics = read_json(Path(plan["reference_scoring"]) / "baseline/COCO_METRICS.json")
    check(set(reference_metrics["segm"]) == set(evaluator.METRICS), "Original accepted baseline12 metric schema differs")
    # Check installation paths without importing or running either COCO backend.
    for module in ("pycocotools.coco", "pycocotools.cocoeval", "pycocotools.mask", "pycocotools._mask", "cv2"):
        spec = importlib.util.find_spec(module)
        check(spec is not None and spec.origin and Path(spec.origin).is_file(), "Required scoring backend is absent: " + module)
    write_json(out / "SOURCE_LOCK.json", {"files": lock.finish(), "GT_rematched": False})
    summary = {"status": "input_path_preflight_complete", "passed": True, "fixed_source": receipt,
        "required_files_present_count": len(required), "expected_original_fixed_source_sha256": EXPECTED_FIXED,
        "plan_sha256": sha(args.plan), "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
        "GT_parsed_or_decoded": False, "GT_rematched": False, "AP_or_Boundary_computed": False, "GPU_or_model_execution": False,
        "scope": "Exact source positions/existence and SHA plus unchanged frozen evaluator fixed_source read gate; no new statistical or quality result"}
    write_json(out / "SUMMARY.json", summary)
    write_json(out / "COMPLETE.json", {"status": summary["status"], "summary_sha256": sha(out / "SUMMARY.json"), "source_lock_sha256": sha(out / "SOURCE_LOCK.json")})

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
