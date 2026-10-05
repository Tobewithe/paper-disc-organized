"""Resumeable remote 7L sequence; every scientific stage is a Workbench run."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path("/root/coefficient_expected_benefit_20260927")
RUNNER = Path("/root/autodl-tmp/coefficient_predictability_20260924/runner.py")
PYTHON = Path("/root/miniconda3/bin/python")
STUDY = "STUDY_0d9e14b39d5e4fafba576b311b4321c1"
FULL_TARGET_ID = "RUN_9ad188b832fd9bc49c0aafc7a61de989"
BANK = Path("/root/autodl-tmp/coefficient_data_scaling_20260924/runs/RUN_a24f6c759f714fb9bfcd5a59bb04e6d3")
ANNOTATIONS = BANK / "conversion_input/instances_train2017.json"
DEV = Path("/root/autodl-tmp/coefficient_direction_predictability_20260925/runs/RUN_3b1968a2fb1b41bfb427c6e1d08ab93e/DATA.pt")
WEIGHTS = Path("/root/autodl-tmp/coefficient_predictability_20260924/yolo26m-seg.pt")
OFFICIAL = Path("/root/autodl-tmp/coefficient_predictability_20260924")
ORACLE = Path("/root/autodl-tmp/coefficient_finite_oracle_20260925")
STATE = ROOT / "ORCHESTRATION.json"


def save(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def wait_for(run_id):
    directory = ROOT / "runs" / run_id
    while True:
        path = directory / "run.json"
        status = "pending"
        if path.is_file():
            try:
                record = json.loads(path.read_text())
                status = record.get("status")
                if status == "completed":
                    assert record.get("return_code") == 0, (run_id, record)
                    assert (directory / "COMPLETE.json").is_file(), run_id
                    return directory
                if status in {"failed", "error", "cancelled"}:
                    raise RuntimeError(f"{run_id}: {status}, see {directory / 'stderr.log'}")
            except json.JSONDecodeError:
                pass
        print(json.dumps(dict(waiting=run_id, stage_status=status)),
              flush=True)
        time.sleep(30)


def run(state, name, script, arguments, inputs=()):
    if name in state["runs"]:
        run_id = state["runs"][name]
        return wait_for(run_id)
    run_id = "RUN_" + uuid.uuid4().hex
    directory = ROOT / "runs" / run_id
    command = [str(PYTHON), str(RUNNER), "--study", STUDY, "--run-id", run_id,
        "--output", str(directory), "--cwd", str(ROOT), "--snapshot", f"scripts/{script}",
        "--snapshot", "PROTOCOL.md", "--expect", f"runs/{run_id}/COMPLETE.json"]
    for source in (ROOT / "MANIFEST.json", *inputs):
        command.extend(("--input", str(source)))
    command.extend(("--", str(PYTHON), "-u", f"scripts/{script}", *map(str, arguments),
                    "--out", str(directory)))
    launched = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if launched.returncode:
        raise RuntimeError(f"Could not launch {name}: {launched.stdout}\n{launched.stderr}")
    state["runs"][name] = run_id
    save(state)
    print(json.dumps(dict(started=name, run_id=run_id, output=str(directory))), flush=True)
    return wait_for(run_id)


def main():
    state = json.loads(STATE.read_text()) if STATE.is_file() else dict(
        study_id=STUDY, full_target_run=FULL_TARGET_ID, runs={})
    assert state["study_id"] == STUDY and state["full_target_run"] == FULL_TARGET_ID
    target = wait_for(FULL_TARGET_ID)
    print(json.dumps(dict(target_complete=str(target))), flush=True)
    models = {}
    for fold in range(5):
        models[fold] = run(state, f"correction_fold_{fold}", "train_oof_correction.py", [
            "--manifest", ROOT / "MANIFEST.json", "--targets", target,
            "--dev-targets", DEV, "--fold", fold],
            inputs=(target / "COMPLETE.json", DEV))
    full_model = run(state, "correction_full_fit", "train_oof_correction.py", [
        "--manifest", ROOT / "MANIFEST.json", "--targets", target,
        "--dev-targets", DEV, "--fold", 5],
        inputs=(target / "COMPLETE.json", DEV))
    scores = []
    for fold in range(5):
        scores.append(run(state, f"oof_benefit_fold_{fold}", "score_oof_benefit.py", [
            "--manifest", ROOT / "MANIFEST.json", "--targets", target,
            "--bank", BANK, "--annotations", ANNOTATIONS, "--weights", WEIGHTS,
            "--official-source", OFFICIAL, "--checkpoint", models[fold] / "BEST.pt",
            "--fold", fold, "--shard-images", 100],
            inputs=(target / "COMPLETE.json", models[fold] / "BEST.pt")))
    gate = run(state, "benefit_gate", "train_benefit_gate.py", [
        "--manifest", ROOT / "MANIFEST.json", "--scores", *scores],
        inputs=tuple(p / "COMPLETE.json" for p in scores))
    labels = run(state, "test_label_conversion", "prepare_test_labels.py", [
        "--manifest", ROOT / "MANIFEST.json", "--annotations", ROOT / "sealed_instances_val2017.json",
        "--images", ROOT / "test_data/images/val2017", "--gate", gate],
        inputs=(gate / "COMPLETE.json", ROOT / "sealed_instances_val2017.json"))
    test = run(state, "independent_test_scoring", "score_independent_test.py", [
        "--manifest", ROOT / "MANIFEST.json", "--test-labels", labels, "--gate", gate,
        "--checkpoint", full_model / "BEST.pt", "--weights", WEIGHTS,
        "--official-source", OFFICIAL, "--shard-images", 100],
        inputs=(gate / "COMPLETE.json", labels / "COMPLETE.json", full_model / "BEST.pt"))
    evaluation = run(state, "locked_test_evaluation", "evaluate_benefit_gate.py", [
        "--manifest", ROOT / "MANIFEST.json", "--gate", gate,
        "--test-scores", test], inputs=(gate / "COMPLETE.json", test / "COMPLETE.json"))
    state["complete"] = str(evaluation)
    save(state)
    print(json.dumps(dict(all_complete=True, results=str(evaluation / "RESULTS.json"))), flush=True)


if __name__ == "__main__":
    main()
