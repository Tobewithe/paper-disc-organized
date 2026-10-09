"""Explicitly authorized post-hoc records of four observed pre-runner failures.

No observed exit code or execution timestamp exists. Do not fabricate either,
and do not label these records runner_observed or scientific completion.
"""
import argparse
import json
from pathlib import Path
import shutil
from target_common import dump, now, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempts = [(1, 436, "register_lifecycle_engineering.ps1", None),
                (2, 26432, "register_lifecycle_engineering02.ps1", None),
                (3, 20532, "register_lifecycle_engineering03.ps1", "launch_20261009_054745_144.log"),
                (4, 23532, "register_lifecycle_engineering04.ps1", "launch_20261009_054929_945.log")]
    made = []
    for number, pid, script, logfile in attempts:
        name = f"RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_{number:02d}"
        run = root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        if (run / "run.json").exists():
            raise ValueError("Never replace an existing historical record")
        archive = run / "source"
        archive.mkdir(exist_ok=True)
        target = root / "scripts" / script
        shutil.copy2(target, archive / script)
        existing = run / "LAUNCH_RECEIPT.json"
        if not existing.exists():
            dump(existing, dict(status="launch_returned_but_no_runner_record_observed", wmi_return_value=0,
                 wmi_process_id=pid, source_run_json_available=False, child_process_exited_before_inspection=True,
                 stdout_stderr_available=False, expected_launch_log=logfile, actual_launch_log_available=False if logfile else None,
                 scientific_gpu_work_started=False, lifecycle_probe_started=None, failure_reason=None,
                 intended_target_source_sha256=sha(target), observation="WMI PID returned; subsequent process query empty and no Run artifacts observed"))
        dump(run / "run.json", dict(schema_version=2, run_id=name,
             study_id="STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009",
             source_kind="launcher_pre_execution_failure_observed", post_hoc=True, status="failed",
             return_code=None, created_at=None, started_at=None, finished_at=None, failure_reason=None,
             record_written_at=now(), host="YANGLEI", pid=pid, scientific_execution_started=False,
             cpu_lifecycle_probe_started=None, gpu_execution_started=False, gt_parsed=False,
             artifact_completeness="unknown", expected_outputs=[],
             launcher_observation=dict(wmi_return_value=0, process_exited_by_later_query=True,
                                       runner_record_observed=False, output_summary_observed=False),
             actual_exit_and_execution_time_unavailable=True, source_snapshot_limit="retained intended script; no claim that child executed it",
             receipt_sha256=sha(existing), intended_script_sha256=sha(target),
             note="Authorized post-hoc launcher-entry failure; not runner_observed, no scientific completion claimed"))
        files = sorted(p for p in run.rglob("*") if p.is_file() and p.name not in ("manifest.sha256", "transfer.json"))
        (run / "manifest.sha256").write_text("\n".join(sha(p) + "  " + p.relative_to(run).as_posix() for p in files) + "\n", encoding="utf-8")
        made.append(dict(run_id=name, run_json_sha256=sha(run / "run.json"), manifest_sha256=sha(run / "manifest.sha256")))
    dump(out / "SUMMARY.json", dict(passed=True, scope="preserve four real launcher-entry failures post-hoc",
                                    records=made, unknown_exit_and_times_preserved=True, scientific_sources_unchanged=True))


if __name__ == "__main__":
    main()
