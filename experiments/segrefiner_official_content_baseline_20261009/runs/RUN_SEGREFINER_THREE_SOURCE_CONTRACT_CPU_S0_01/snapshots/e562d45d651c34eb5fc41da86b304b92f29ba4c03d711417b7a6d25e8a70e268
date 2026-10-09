"""Thin three-source wrapper around accepted frozen CPU metric kernels.

Scope/source validation is new. Pixel, fixed-pair, COCOeval and Boundary metric
kernels are reused, not represented as independent new implementations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from segrefiner_scoring_bridge import (ARMS, ANN_SHA, CORES, ScopeReader, check, load_cores,
    read, runner_gate, score_gate, seal, sha, validate_sources, verify_seal, write, validate_runner_record, load_pixel_checker)


def archive_sources(plan, out, lock):
    source = out / "source"
    source.mkdir(exist_ok=True)
    sources = {}
    paths = [Path(__file__), Path(__file__).with_name("segrefiner_scoring_bridge.py"), Path(plan["plan_path"])]
    paths += [Path(plan["cores"]) / n for n in CORES]
    for p in paths:
        digest = lock.add(p)
        sources[p.name] = digest
        shutil.copy2(p, source / p.name)
    return sources


def make_readers(io, roots, lock, ids, native_ids):
    return {arm: ScopeReader(io, path, lock, native_ids if arm != ARMS[2] else ids, ids) for arm, path in roots.items()}


def contract_engineering(plan, out):
    """Small actual contract Run; no model, COCO GT, full prediction parsing."""
    out.mkdir(parents=True, exist_ok=True)
    io, score, multi = load_cores(plan["cores"])
    lock = io.InputLock()
    sources = archive_sources(plan, out, lock)
    cases = {}
    for status in ("failed", "cancelled", "running"):
        record = {"source_kind": "runner_observed", "status": status, "return_code": 0,
                  "artifact_completeness": "complete", "finished_at": "synthetic"}
        try:
            validate_runner_record(record)
        except ValueError:
            cases["reject_" + status + "_even_with_complete_flags"] = True
        else:
            raise AssertionError("Non-success source passed literal runner gate")
    try:
        validate_runner_record({"source_kind": "launcher_pre_execution_failure_observed", "status": "completed",
                               "return_code": 0, "artifact_completeness": "complete", "finished_at": "synthetic"})
    except ValueError:
        cases["reject_posthoc_launcher_record_as_scientific_execution"] = True
    else:
        raise AssertionError("A launcher metadata record passed as scientific execution")
    fixture = out / "synthetic_source_view"
    (fixture / "images").mkdir(parents=True)
    write(fixture / "predictions.json", [])
    write(fixture / "COMPLETE.json", {"status": "prediction_complete", "image_count": 2,
          "predictions_sha256": sha(fixture / "predictions.json"), "synthetic_fixture_only": True})
    for iid in (1, 2):
        write(fixture / "images" / f"{iid:012d}.json", {"image_id": iid, "detections": [], "synthetic_fixture_only": True})
    reader = ScopeReader(io, fixture, lock, [1, 2], [1])
    check(reader.read(1, 0) == [], "True zero-candidate source did not retain zero output")
    reader.finish(1)
    cases["source_full_scope_retained_zero_candidate_prefix_valid"] = True
    reader = ScopeReader(io, fixture, lock, [1, 2], [2])
    reader.read(2, 0)
    try:
        reader.finish(1)
    except ValueError:
        cases["nonprefix_engineering_scope_rejected"] = True
    else:
        raise AssertionError("Nonprefix engineering source was accepted")
    original_seal = seal(fixture, ["predictions.json"])
    verify_seal(fixture, original_seal, lock)
    # Use a fresh guard lock so it checks the actual modified bytes rather than
    # an already memoized first observation. Only this explicit fixture changes.
    (fixture / "predictions.json").write_text("[ ]\n", encoding="utf-8")
    try:
        verify_seal(fixture, original_seal, io.InputLock())
    except ValueError:
        cases["modified_source_output_seal_rejected"] = True
    else:
        raise AssertionError("Modified source output seal was accepted")
    # The intentional fixture mutation is recorded, not described as a stable
    # scientific source input; all actual executable sources are rehashed.
    lock.files.pop(str((fixture / "predictions.json").resolve()), None)
    write(out / "SOURCE_LOCK.json", {"files": lock.finish(), "all_before_after_sha256_equal": True, "executed_source_sha256": sources})
    summary = {"status": "contract_engineering_complete", "passed": True, "checks": cases,
               "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "GPU_or_model_execution": False,
               "GT_parsed": False, "scientific_quality_metrics": None,
               "scope": "Explicit provenance and true-zero/prefix contracts; accepted metric cores loaded unchanged; no metric or model result"}
    write(out / "SUMMARY.json", summary)
    artifacts = seal(out, [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in ("run.json", "stdout.log", "stderr.log", ".run.claim")])
    write(out / "CONTRACT_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "artifacts": artifacts})


def identity_and_pairs(plan, out, io, score, lock, engineering=False):
    ids, roots, bridge = validate_sources(plan, io, lock, engineering)
    native = Path(plan["native_run"])
    native_ids = read(native / "baseline" / "COMPLETE.json")["fingerprint"]["image_ids"]
    readers = make_readers(io, roots, lock, ids, native_ids)
    fixed, fixed_source = score.load_fixed_source(Path(plan["fixed_source"]), lock, ids, engineering)
    check(engineering or fixed_source["selected_fixed_rows"] == 86600, "Formal original fixed86600 cohort required")
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    for module in (mask_utils, sys.modules[COCO.__module__], sys.modules[COCOeval.__module__]):
        lock.add(module.__file__)
    check(lock.add(plan["annotations"]) == ANN_SHA, "Original val GT annotation SHA differs")
    with contextlib.redirect_stdout(__import__("io").StringIO()):
        coco = COCO(plan["annotations"])
    totals = {arm: score.new_pair_acc() for arm in ARMS}
    gt_labels = {arm: Counter() for arm in ARMS}
    identity_counts = {arm: Counter(candidate_count=0, empty_mask_count=0) for arm in ARMS}
    decisions = iter(io.jsonl(native / "NATIVE_DECISIONS.jsonl"))
    diagnostics = iter(io.jsonl(Path(plan["seg_run"]) / "IMAGE_DIAGNOSTICS.jsonl"))
    counts = dict(native_rows=0, selected_supported_first64=0, model_valid_area512=0, tiny_or_empty_fallback=0, actual_denoiser_forward_calls=0)
    image_meta = {row["image_id"]: row for row in read(plan["image_meta"])["images"]}
    empty_chains = {arm: __import__("hashlib").sha256() for arm in ARMS}
    scoped_predictions = {arm: [] for arm in ARMS} if engineering else None
    with contextlib.ExitStack() as stack:
        identity = stack.enter_context((out / "IDENTITY_IMAGES.jsonl").open("x", encoding="utf-8"))
        pair_outputs = {}
        if not engineering:
            for arm in ARMS:
                directory = out / "paired" / arm
                directory.mkdir(parents=True)
                pair_outputs[arm] = {key: stack.enter_context((directory / name).open("x", encoding="utf-8"))
                    for key, name in (("instances", "INSTANCES.jsonl"), ("images", "IMAGES.jsonl"), ("gt", "GT_LABELS.jsonl"))}
        for position, iid in enumerate(ids):
            decision, diag = next(decisions, None), next(diagnostics, None)
            check(decision is not None and diag is not None and decision["image_id"] == diag["image_id"] == iid,
                  "Native/refiner per-image source sequence differs")
            count = decision["native_rows"]
            check(type(count) is int and count >= 0 and len(decision["decisions"]) == count and diag["native_rows"] == count,
                  "Explicit zero/nonzero normal count missing or inconsistent")
            rows = {arm: reader.read(iid, count) for arm, reader in readers.items()}
            original_image = native / "baseline" / "images" / f"{iid:012d}.json"
            check(lock.add(original_image) == diag["original_image_json_sha256"] == image_meta[iid]["baseline_image_json_sha256"],
                  "Historical baseline per-image source SHA changed")
            check(diag["image_sha256"] == image_meta[iid]["image_sha256"] == lock.add(image_meta[iid]["image_path"]),
                  "Actual RGB JPEG/source binding changed")
            positions, eligible_area = [], []
            image_counts = {arm: Counter(candidate_count=0, empty_mask_count=0) for arm in ARMS}
            for index, base in enumerate(rows["baseline"]):
                detail = decision["decisions"][index]
                check(detail["detection_index"] == index and detail["in_first64"] is (index < 64)
                      and detail["first64_proto_supported"] is (index < 64 and detail["proto_roi_supported"]),
                      "Original ordinal/first64/protoROI eligibility differs")
                area = io.rle_area(base["segmentation"], mask_utils)
                if detail["first64_proto_supported"]:
                    positions.append(index)
                    eligible_area.append(area)
                for arm in ARMS:
                    row = rows[arm][index]
                    check(all(row[k] == base[k] for k in io.IDENTITY_KEYS), "Three-source native class/box/score/raw identity changed")
                    check(row["segmentation"]["size"] == base["segmentation"]["size"], "Original RLE image grid changed")
                    if not detail["first64_proto_supported"] or (arm == ARMS[2] and area < 512):
                        check(row["segmentation"] == base["segmentation"], "Outside scope or official tiny/empty fallback changed")
                    empty = io.rle_area(row["segmentation"], mask_utils) == 0
                    image_counts[arm]["candidate_count"] += 1
                    image_counts[arm]["empty_mask_count"] += empty
                    identity_counts[arm].update(candidate_count=1, empty_mask_count=empty)
                    if empty:
                        empty_chains[arm].update(f"{iid}:{index}\n".encode())
                    if engineering:
                        scoped_predictions[arm].append({k: row[k] for k in io.COCO_KEYS})
            actual = diag["actual"]
            valid = sum(a >= 512 for a in eligible_area)
            check(actual["selected_supported_ordinals"] == positions and actual["model_valid_area512"] == valid
                  and actual["tiny_or_empty_fallback"] == len(positions) - valid
                  and actual["outside_scope_identity"] == count - len(positions)
                  and actual["actual_denoiser_forward_calls"] == 6 * ((valid + 7) // 8)
                  and actual["gt_used"] is False and actual["all_ordinal_count_preserved"] is True,
                  "Actual batch8 model/skip/tiny/outside zero-count contract differs")
            check(valid == 0 or actual["seed"] == 20261009 + iid, "Fixed per-image stochastic seed changed")
            rng = Path(plan["seg_run"]) / diag["rng_state_file"]
            check(lock.add(rng) == diag["rng_state_sha256"], "Saved RNG state bytes changed")
            counts["native_rows"] += count
            counts["selected_supported_first64"] += len(positions)
            counts["model_valid_area512"] += valid
            counts["tiny_or_empty_fallback"] += len(positions) - valid
            counts["actual_denoiser_forward_calls"] += actual["actual_denoiser_forward_calls"]
            identity.write(json.dumps({"image_id": iid, "candidate_count": count,
                                      "arm_counts": {a: dict(c) if count else {} for a, c in image_counts.items()},
                                      "zero_counter_legacy_representation_only_when_explicit_native_count_zero": count == 0,
                                      "all_identity_scope_tiny_empty_guards_passed": True,
                                      "all_arms_identity_exact": True, "official_baseline_rle_exact": True,
                                      "all_empty_ordinals_retained": True, "first64_domain_exact": True}) + "\n")
            if not engineering:
                score.compute_fixed_pairs(iid, rows, fixed.get(iid, []), coco, mask_utils, pair_outputs, totals, gt_labels)
            if (position + 1) % 100 == 0:
                print("SEG_CPU_IDENTITY_PAIRS", position + 1, len(ids), flush=True)
    check(next(diagnostics, None) is None, "Trailing SegRefiner diagnostics beyond declared scope")
    if not engineering:
        check(next(decisions, None) is None, "Trailing native image outside full5000 scope")
    for reader in readers.values():
        reader.finish(len(ids))
    source_summary = read(Path(plan["seg_run"]) / "SUMMARY.json")
    check(all(source_summary[name] == value for name, value in counts.items()), "Source inference totals differ from full-frame exported proof")
    provenance = out / "source_provenance"
    provenance.mkdir()
    for origin, name in ((fixed_source["instances_path"], "FIXED_INSTANCES_SOURCE.jsonl"),
                         (Path(fixed_source["directory"]) / "SUMMARY.json", "FIXED_PAIRED_SOURCE_SUMMARY.json"),
                         (Path(fixed_source["directory"]).parent / "SUMMARY.json", "FINAL8_SOURCE_SUMMARY.json")):
        shutil.copy2(origin, provenance / name)
    identity_summary = {"arm_counts": {arm: dict(value) for arm, value in identity_counts.items()}, "counts": counts,
                        "empty_ordinal_sha256": {arm: value.hexdigest() for arm, value in empty_chains.items()},
                        "all_normal_native_rows_and_empty_ordinals_preserved": True}
    return ids, roots, bridge, fixed_source, identity_summary, totals, gt_labels, coco, COCO, COCOeval


def run_score(plan, out, engineering=False):
    out.mkdir(parents=True, exist_ok=True)
    io, score, multi = load_cores(plan["cores"])
    lock = io.InputLock()
    sources = archive_sources(plan, out, lock)
    result = identity_and_pairs(plan, out, io, score, lock, engineering)
    ids, roots, bridge, fixed_source, identity, totals, labels, coco, COCO, COCOeval = result
    write(out / "SCORE_INPUTS.json", {"image_ids": ids, "arms": list(ARMS), "engineering": engineering,
                                      "bridge": bridge, "fixed_source": fixed_source, "source_sha256": sources})
    metrics, paired = {}, {}
    if not engineering:
        for arm in ARMS:
            values, seconds = score.coco_metrics(coco, roots[arm] / "predictions.json", ids, out / arm, "segm", COCO, COCOeval, np)
            metrics[arm] = {"segm": values, "predictions_sha256": sha(roots[arm] / "predictions.json"), "seconds": seconds}
            write(out / arm / "COCO_METRICS.json", metrics[arm])
            paired[arm] = {"statistics": score.pair_finish(totals[arm]), "unique_GT_association_labels": dict(labels[arm]),
                           "association_source_sha256": fixed_source["instances_sha256"], "confidence_intervals": None}
            write(out / "paired" / arm / "SUMMARY.json", paired[arm])
        baseline_expected = read(Path(plan["reference_scoring"]) / "baseline" / "COCO_METRICS.json")["segm"]
        lock.add(Path(plan["reference_scoring"]) / "baseline" / "COCO_METRICS.json")
        score.equal_value(metrics["baseline"]["segm"], baseline_expected, "accepted ordinary baseline12 parity")
        for arm in ARMS:
            for key, count in (("matched_detection_count", 86600), ("baseline_success_count", 36266), ("baseline_failure_count", 50334)):
                check(totals[arm][key] == count, "Original fixed final8 denominator changed")
    write(out / "SOURCE_LOCK.json", {"files": lock.finish(), "all_before_after_sha256_equal": True, "executed_source_sha256": sources})
    summary = {"status": "engineering_identity_verified_metrics_unknown" if engineering else "complete", "passed": True,
        "image_count": len(ids), "engineering": engineering, "arms": list(ARMS), "metrics": metrics,
        "fixed_paired": paired, "fixed_source": fixed_source, "identity": identity, "bridge": bridge,
        "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "GPU_or_model_execution": False,
        "association_rematched": False, "native_forward_replayed": False, "confidence_intervals": None,
        "limitations": ["Accepted metric kernels reused through explicit three-source wrapper", "Unknown actual checkpoint training manifest", "batch8 execution adaptation; one fixed stochastic realization"]}
    write(out / "SUMMARY.json", summary)
    artifacts = seal(out, [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in ("run.json", "stdout.log", "stderr.log", ".run.claim")])
    write(out / "SCORE_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "artifacts": artifacts})


def run_readout(plan, out, scoring):
    io, score, multi = load_cores(plan["cores"])
    def validation(args, lock):
        sources = archive_sources(plan, out, lock)
        ids, roots, bridge = validate_sources(plan, io, lock, False)
        scorer, complete = score_gate(scoring, "SCORE_COMPLETE.json", lock)
        check(scorer["status"] == "complete" and scorer["arms"] == list(ARMS), "Complete exact three-arm ordinary scorer required")
        fixed, fixed_source = score.load_fixed_source(Path(plan["fixed_source"]), lock, ids, False)
        check(fixed_source["instances_sha256"] == scorer["fixed_source"]["instances_sha256"], "Pixel scorer fixed association changed")
        roots_by_name = {arm: path / "predictions.json" for arm, path in roots.items()}
        original_json_array = multi.json_array
        def routed_json_array(path):
            path = Path(path)
            if path.name == "predictions.json" and path.parent.name in roots_by_name:
                path = roots_by_name[path.parent.name]
            return original_json_array(path)
        multi.json_array = routed_json_array
        write(out / "THREE_SOURCE_WRAPPER.json", {"bridge": bridge, "metric_kernels_reused": CORES, "source_sha256": sources,
                                                   "substitutions": ["ARMS", "source validation", "actual prediction path resolution"]})
        return ids, False, scorer, {arm: io.ArmReader(root, lock, ids) for arm, root in roots.items()}, fixed, fixed_source
    multi.validate_inputs = validation
    args = argparse.Namespace(run=Path(plan["native_run"]), scoring_run=scoring, source_paired=Path(plan["fixed_source"]),
        annotations=Path(plan["annotations"]), protocol=Path(plan["study"]) / "PROTOCOL.md", out_dir=out,
        boundary_vendor=Path(plan["boundary_vendor"]), boundary_provenance=Path(plan["boundary_provenance"]),
        boundary_provenance_sha256=plan["boundary_provenance_sha256"])
    multi.run_readout(args)
    summary = read(out / "SUMMARY.json")
    check(summary["status"] == "complete" and summary["boundary"]["status"] == "completed", "Pixels+Boundary not fully completed")
    # Keep the actual accepted-core receipt; add a wrapper seal that the driver
    # and checker consume, without rewriting core-created science summaries.
    artifacts = seal(out, [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in ("run.json", "stdout.log", "stderr.log", ".run.claim")])
    write(out / "WRAPPER_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "artifacts": artifacts})


def run_verify(plan, out, scoring, readout):
    out.mkdir(parents=True, exist_ok=True)
    io, score, multi = load_cores(plan["cores"])
    lock = io.InputLock()
    sources = archive_sources(plan, out, lock)
    scorer, _ = score_gate(scoring, "SCORE_COMPLETE.json", lock)
    pixels, _ = score_gate(readout, "WRAPPER_COMPLETE.json", lock)
    checker = load_pixel_checker(plan["cores"])
    proof_lock = checker.Fingerprint()
    pixel_proof = checker.verify_pixels(readout, scoring, read(readout / "READOUT_INPUTS.json"), pixels,
                                      read(readout / "PIXEL_SUMMARY.json"), scorer, proof_lock)
    boundary_proof = checker.verify_boundary(readout, pixels, scorer, read(readout / "SOURCE_LOCK.json"), proof_lock)
    proof_receipt = proof_lock.finish()
    lock.files.update(proof_receipt["files"])
    arrays = {}
    for arm in ARMS:
        actual = score.npz_metrics(scoring / arm / "COCO_segm_ACCUMULATED.npz", np)
        score.equal_value(actual, scorer["metrics"][arm]["segm"], arm + ".ordinary arrays")
        boundary = score.npz_metrics(readout / "boundary" / (arm + "_ACCUMULATED.npz"), np)
        score.equal_value(boundary, pixels["boundary"]["metrics"][arm], arm + ".boundary arrays")
        arrays[arm] = {"segm": actual, "boundary": boundary}
        rebuilt = score.new_pair_acc()
        for row in io.jsonl(scoring / "paired" / arm / "INSTANCES.jsonl"):
            check(row["damage"] is (row["baseline_mask_iou"] >= .75 and row["method_mask_iou"] < .75)
                  and row["repair"] is (row["baseline_mask_iou"] < .75 and row["method_mask_iou"] >= .75), "Saved fixed-pair outcome differs")
            score.pair_add(rebuilt, row)
        score.equal_value(score.pair_finish(rebuilt), scorer["fixed_paired"][arm]["statistics"], arm + ".paired reaggregation")
    write(out / "SOURCE_LOCK.json", {"files": lock.finish(), "all_before_after_sha256_equal": True, "executed_source_sha256": sources})
    report = {"status": "complete", "passed": True, "image_count": 5000, "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
              "arrays_reconstructed": arrays, "fixed_pair_rows_reaggregated": True, "producer_and_output_seals_verified": True,
              "pixel_candidates_images_and_macro_reconstructed": True,
              "pixel_proof": pixel_proof, "boundary_proof": boundary_proof,
              "GPU_or_model_execution": False, "GT_rematched": False,
              "scope": "Saved official accumulated arrays and fixed pair counts; accepted pixel core outputs sealed; no independent GPU/model or COCO matching replay"}
    write(out / "SUMMARY.json", report)
    write(out / "VERIFICATION.json", report)
    artifacts = seal(out, [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in ("run.json", "stdout.log", "stderr.log", ".run.claim")])
    write(out / "VERIFY_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "artifacts": artifacts})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--stage", choices=("contract", "engineering", "score", "readout", "verify"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scoring-run", type=Path)
    p.add_argument("--readout-run", type=Path)
    args = p.parse_args()
    plan = read(args.plan)
    plan["plan_path"] = str(args.plan.resolve())
    try:
        if args.stage == "contract":
            contract_engineering(plan, args.out_dir.resolve())
        elif args.stage in ("engineering", "score"):
            run_score(plan, args.out_dir.resolve(), args.stage == "engineering")
        elif args.stage == "readout":
            run_readout(plan, args.out_dir.resolve(), args.scoring_run.resolve())
        else:
            run_verify(plan, args.out_dir.resolve(), args.scoring_run.resolve(), args.readout_run.resolve())
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "WRAPPER_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "GPU_or_model_execution": False})
        raise
