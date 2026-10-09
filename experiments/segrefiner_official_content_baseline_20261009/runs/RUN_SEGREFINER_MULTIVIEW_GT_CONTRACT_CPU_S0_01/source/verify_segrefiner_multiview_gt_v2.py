"""Exercise the actual v2 multiview validation closure, hashing GT without parsing it.

Other source/scorer/readers and the metric loop are explicit small stubs. The
annotation gate and the accepted InputLock add/finish implementation are real.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import score_segrefiner_comparison_v2 as wrapper
from segrefiner_scoring_bridge import ANN_SHA, ARMS, CORES, check, read, seal, sha, write


def main(args):
    plan = read(args.plan)
    plan["plan_path"] = str(args.plan.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / "SUMMARY.json").exists(), "New GT contract Run required")
    io_path = Path(plan["cores"]) / "comparison_io.py"
    check(sha(io_path) == CORES[io_path.name], "Accepted InputLock source differs")
    spec = importlib.util.spec_from_file_location("accepted_GT_contract_io", io_path)
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    original = Path(plan["annotations"])
    check(original.is_file() and original.stat().st_size == 19987840 and sha(original) == ANN_SHA,
          "Actual full val annotation byte input required")
    source_lock = io.InputLock()
    sources = [Path(__file__), Path(wrapper.__file__), Path(wrapper.__file__).with_name("segrefiner_scoring_bridge.py"), args.plan, io_path, original]
    source_dir = out / "source"
    source_dir.mkdir()
    for path in sources:
        source_lock.add(path)
        if path != original:
            shutil.copy2(path, source_dir / path.name)
    fixture = out / "fixtures"
    fixture.mkdir()
    wrong = fixture / "wrong_annotations.json"
    wrong.write_bytes(b"{}\n")
    mutable = fixture / "same_GT_bytes_then_mutated.json"
    shutil.copy2(original, mutable)
    fixed_sha = "explicit_stub_fixed_cohort_identity"
    wrapper.validate_sources = lambda plan, io, lock, engineering: ([1], {a: out / "stub_sources" / a for a in ARMS}, {"explicit_contract_stub": True})
    wrapper.score_gate = lambda scoring, complete, lock: ({"status": "complete", "arms": list(ARMS), "fixed_source": {"instances_sha256": fixed_sha}}, {})
    cases = {}
    for name, annotation, should_pass, mutate in (("actual_val_GT_correct_SHA_locked", original, True, False),
             ("wrong_GT_SHA_rejected", wrong, False, False), ("GT_changed_after_validation_rejected_at_finish", mutable, False, True)):
        case_out = out / name
        case_out.mkdir()
        case_plan = dict(plan, annotations=str(annotation))
        observed = {"validation_called": False, "validation_completed": False, "finish_called": False}
        multi = SimpleNamespace(json_array=lambda path: iter(()))
        def stub_metric_loop(actual_args):
            lock = io.InputLock()
            observed["validation_called"] = True
            multi.validate_inputs(actual_args, lock)
            observed["validation_completed"] = True
            check(lock.files[str(annotation.resolve())]["sha256_before"] == ANN_SHA, "Actual validation omitted annotation InputLock.add")
            if mutate:
                with annotation.open("ab") as handle:
                    handle.write(b"\n")
            observed["finish_called"] = True
            receipt = lock.finish()
            write(case_out / "SOURCE_LOCK.json", {"files": receipt, "all_before_after_sha256_equal": True})
            write(case_out / "SUMMARY.json", {"status": "complete", "boundary": {"status": "completed"}, "contract_fixture_only": True})
        multi.run_readout = stub_metric_loop
        stub_io = SimpleNamespace(ArmReader=lambda root, lock, ids: {"explicit_stub": str(root)})
        stub_score = SimpleNamespace(load_fixed_source=lambda path, lock, ids, engineering: ({}, {"instances_sha256": fixed_sha}))
        wrapper.load_cores = lambda directory: (stub_io, stub_score, multi)
        try:
            wrapper.run_readout(case_plan, case_out, out / "stub_scoring")
        except ValueError as error:
            check(not should_pass, "Correct original GT validation failed")
            check(("Source bytes changed:" in str(error) and mutate and observed["validation_completed"] and observed["finish_called"])
                  or (str(error) == "Original val GT annotation SHA differs" and not mutate and not observed["validation_completed"]),
                  "Guard rejection was not the intended actual GT validation/finish gate")
            observed["rejection"] = str(error)
        else:
            check(should_pass and observed["finish_called"], "Wrong/changed GT was accepted")
            receipt = read(case_out / "SOURCE_LOCK.json")["files"][str(original.resolve())]
            check(receipt["sha256_before"] == receipt["sha256_after"] == ANN_SHA, "Correct GT before/after source receipt missing")
        cases[name] = observed
    write(out / "SOURCE_LOCK.json", {"files": source_lock.finish(), "all_before_after_sha256_equal": True})
    summary = {"status": "GT_validation_contract_complete", "passed": True, "checks": cases,
        "GT_parsed": False, "GT_decoded": False, "GPU_or_model_execution": False, "COCO_or_Boundary_metrics_computed": False,
        "actual_annotations": str(original.resolve()), "actual_annotations_size_bytes": original.stat().st_size, "actual_annotations_sha256": ANN_SHA,
        "scoring_v2_sha256": sha(wrapper.__file__), "plan_sha256": sha(args.plan), "source_lock_sha256": sha(out / "SOURCE_LOCK.json"),
        "scope": "Actual v2 run_readout.validation closure and accepted InputLock tested; unrelated sources and metric loop are explicit stubs; no quality result"}
    write(out / "SUMMARY.json", summary)
    artifacts = seal(out, [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in ("run.json", "stdout.log", "stderr.log", ".run.claim")])
    write(out / "CONTRACT_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"),
          "source_lock_sha256": sha(out / "SOURCE_LOCK.json"), "artifacts": artifacts})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
