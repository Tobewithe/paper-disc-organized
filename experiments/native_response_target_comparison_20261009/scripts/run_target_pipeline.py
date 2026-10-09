"""Finite registered engineering -> one20k extraction -> two fits -> native5k.

Every child uses the original workbench runner. Nonzero exit stops the chain;
no resume into an old Run, implicit retry, model selection or new parameter.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from target_common import dump, load_config, now, sha


def execute(c, root, name, script, extra, expected):
    out = root / "runs" / name
    if out.exists():
        raise FileExistsError("Child Run already exists: " + name)
    script_path = Path(__file__).with_name(script)
    command = [sys.executable, c["runner"], "--study", c["study_id"], "--output", str(out), "--run-id", name,
               "--cwd", str(root), "--input", c["protocol"], "--snapshot", str(script_path),
               "--snapshot", str(Path(__file__).with_name("target_common.py")),
               "--snapshot", str(Path(__file__).with_name("native_response.py")),
               "--scope", json.dumps(dict(environment="local_laptop_28358lan", stage=script,
                                          protocol_sha256=c["protocol_sha256"], no_yolo_training=True)),
               "--metrics", str(out / "SUMMARY.json")]
    for filename in expected:
        command += ["--expect", str(out / filename)]
    command += ["--", sys.executable, str(script_path), "--config", c["config_path"],
                "--protocol-sha256", c["protocol_sha256"], "--output", str(out)] + extra
    print("REGISTERED_STAGE " + name, flush=True)
    done = subprocess.run(command, check=False)
    if done.returncode:
        raise RuntimeError(f"Stage {name} returned {done.returncode}; chain stopped")
    metadata = json.loads((out / "run.json").read_text())
    summary = json.loads((out / "SUMMARY.json").read_text())
    if metadata.get("status") != "completed" or metadata.get("return_code") != 0 or metadata.get("artifact_completeness") != "complete" or not summary.get("passed"):
        raise RuntimeError("Runner/child completion gate failed: " + name)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--suffix", required=True, help="Fresh Run sequence, e.g.01; retries choose a new sequence")
    p.add_argument("--engineering-only", action="store_true")
    args = p.parse_args()
    c = load_config(args.config, args.protocol_sha256)
    c["config_path"] = str(Path(args.config).resolve())
    root, out = Path(c["root"]).resolve(), Path(args.output).resolve()
    if (out / "PIPELINE_INPUTS.json").exists():
        raise FileExistsError("Pipeline retry requires a new Run")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "PIPELINE_INPUTS.json", dict(config_sha256=sha(args.config), protocol_sha256=args.protocol_sha256,
                                           engineering_only=args.engineering_only, actual_environment="local_laptop_28358lan",
                                           started_at=now(), retry_automatically=False, stages=[]))
    began, stages = time.perf_counter(), {}
    try:
        for engineering in (True, False):
            if not engineering and args.engineering_only:
                break
            phase = "ENGINEERING" if engineering else "FORMAL"
            extra = ["--engineering"] if engineering else []
            extract = execute(c, root, f"RUN_NATIVE_TARGET_{phase}_EXTRACT_S0_{args.suffix}", "extract_targets.py", extra,
                              ["COMPLETE.json", "ROWS_PARTS.json", "IMAGES.jsonl", "SOURCE_LOCK.json", "SUMMARY.json"])
            fits = {}
            for target in ("I", "H"):
                fits[target] = execute(c, root, f"RUN_NATIVE_TARGET_{phase}_FIT_{target}_S0_{args.suffix}", "fit_target.py",
                                       ["--input", str(extract), "--target", target],
                                       ["COMPLETE.json", "model.json", "model.joblib", "SOURCE_LOCK.json", "SUMMARY.json"])
            inference = execute(c, root, f"RUN_NATIVE_TARGET_{phase}_INFERENCE_S0_{args.suffix}", "infer_targets.py",
                                ["--model-i", str(fits["I"]), "--model-h", str(fits["H"])] + extra,
                                ["COMPLETE.json", "BASELINE_PARITY.json", "BASELINE_PARITY_IMAGES.jsonl", "NATIVE_DECISIONS.jsonl", "SUMMARY.json"])
            stages[phase] = dict(extract=str(extract), fit_I=str(fits["I"]), fit_H=str(fits["H"]), inference=str(inference),
                                 extract_summary_sha256=sha(extract / "SUMMARY.json"),
                                 inference_summary_sha256=sha(inference / "SUMMARY.json"))
            dump(out / "PIPELINE_PROGRESS.json", dict(stages=stages, completed_at=now()))
            # The engineering hard gate considers only contracts, not AP/IoU quality.
            if engineering:
                ie = json.loads((inference / "SUMMARY.json").read_text())
                ee = json.loads((extract / "SUMMARY.json").read_text())
                if ie["image_count"] != 4 or ee["images"] != 4 or not ee["synthetic_contract"]["passed"]:
                    raise ValueError("Fixed engineering contract failed")
                dump(out / "ENGINEERING_GATE.json", dict(passed=True, source_runs=stages[phase], quality_used_for_decision=False,
                                                        gate="4train/4val; exact source/native/feature/model/identity/fallback integer contracts"))
        summary = dict(passed=True, status="pipeline_complete", stages=stages, protocol_sha256=args.protocol_sha256,
                       engineering_only=args.engineering_only, elapsed_seconds=time.perf_counter()-began,
                       completed_at=now(), metrics=None, metrics_status="independent scoring required")
        dump(out / "SUMMARY.json", summary)
        dump(out / "PIPELINE_COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), stages=stages))
    except Exception as exc:
        dump(out / "PIPELINE_FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), stages=stages, completed_at=now()))
        raise


if __name__ == "__main__":
    main()
