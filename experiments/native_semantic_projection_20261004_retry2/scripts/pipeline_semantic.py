"""Single-use recorded smoke -> evaluation queue for zero-training flip replay.

No automatic retry/resume and no laptop or desktop execution. All stages use
independent pre-registered Runs and preserve their own terminal receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = "/root/native_semantic_projection_20261004_retry2"
STAGES = (("smoke", "evaluate_semantic.py"), ("evaluation", "evaluate_semantic.py"))


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_record(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return dict(pid=pid, live=False)
        os.kill(pid, 0)
        proc = Path("/proc") / str(pid)
        status = (proc / "stat").read_text().split(")", 1)[1].strip().split()[0]
        command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        return dict(pid=pid, live=status != "Z", command=command, process_state=status)
    except (OSError, ValueError, IndexError, TypeError):
        return dict(pid=pid, live=False)


def verify_code(root):
    ready = load(root / "CODE_READY.json")
    files = ready["files"]
    required = {"PROTOCOL.md", "RUN_CONFIG.json", "RUN_IDS.json", "SPLIT.json",
                *["scripts/" + entry for _, entry in STAGES], "scripts/pipeline_semantic.py"}
    if not required.issubset(files):
        raise AssertionError("CODE_READY does not freeze every stage/config/protocol/split")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise AssertionError("Frozen code/config differs: " + relative)
    return dict(path=str(root / "CODE_READY.json"), sha256=digest(root / "CODE_READY.json"), files=len(files))


def preflight(root, ids):
    observations = []
    state_path = root / "PIPELINE_STATUS.json"
    if state_path.exists():
        state = load(state_path)
        handles = [process_record(state.get(key)) for key in ("pid", "child_pid") if state.get(key)]
        raise RuntimeError(f"Prior pipeline state exists; no restart. status={state.get('status')}, handles={handles}")
    for stage, _ in STAGES:
        output = root / "runs" / ids["runs"][stage]
        if (output / "run.json").exists():
            old = load(output / "run.json")
            handle = process_record(old.get("pid"))
            raise RuntimeError(f"Existing {stage} Run is preserved, status={old.get('status')}, handle={handle}")
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Existing nonempty Run directory: {output}")
    # A process can exist before its first state write. Check actual handles,
    # not just a stale lock, and refuse a concurrent queue for this Study.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        if str(root / "scripts") in command and any(script in command for _, script in STAGES):
            observed = process_record(int(proc.name))
            observations.append(observed)
            if observed["live"]:
                raise RuntimeError(f"Existing actual Study process; no duplicate start: {observed}")
    return observations


def input_paths(root, cfg, ids, stage):
    cache = Path(cfg["cache"])
    paths = [root / "RUN_CONFIG.json", root / "RUN_IDS.json", root / "SPLIT.json", root / "CODE_READY.json",
             Path(cfg["weights"]), Path(cfg["source_root"]) / "runner.py", cache / "INDEX.json", cache / "CACHE_IDENTITY.json",
             Path(cfg["original_screen_config"]), Path(cfg["original_screen_results"]), Path(cfg["annotations_train"])]
    if stage == "evaluation":
        paths += [Path(cfg["smoke_run"]) / "COMPLETE.json"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if os.name == "nt" or str(root) != ROOT:
        raise RuntimeError("This queue is authorized only on its Linux GPU server root")
    cfg, ids = load(root / "RUN_CONFIG.json"), load(root / "RUN_IDS.json")
    if cfg["server_root"] != str(root) or set(ids["runs"]) != {stage for stage, _ in STAGES}:
        raise AssertionError("Study/Run registry differs from the fixed two-stage queue")
    if len(set(ids["runs"].values())) != len(STAGES):
        raise AssertionError("Every stage requires a distinct Run ID")
    ready = verify_code(root)
    observations = preflight(root, ids)
    claim = root / "PIPELINE.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, json.dumps(dict(pid=os.getpid(), started_at=now(), study_id=ids["study_id"])).encode())
    os.close(descriptor)
    state = dict(status="running", pid=os.getpid(), started_at=now(), study_id=ids["study_id"],
                 runs=ids["runs"], code_ready=ready, preflight_process_observations=observations,
                 stage_results={}, automatic_retry=False, automatic_resume=False, automatic_followup=False)
    state_path = root / "PIPELINE_STATUS.json"
    child = None
    try:
        save(state_path, state)
        for stage, script in STAGES:
            verify_code(root)
            output = root / "runs" / ids["runs"][stage]
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"Run appeared after preflight; refusing duplicate: {output}")
            inputs = input_paths(root, cfg, ids, stage)
            snapshots = [root / "PROTOCOL.md", root / "RUN_CONFIG.json", root / "RUN_IDS.json", root / "SPLIT.json",
                         root / "CODE_READY.json", *sorted((root / "scripts").glob("*.py"))]
            if (root / "REUSED_CODE.json").is_file():
                snapshots.append(root / "REUSED_CODE.json")
            command = [sys.executable, str(Path(cfg["source_root"]) / "runner.py"), "--study", ids["study_id"],
                       "--run-id", ids["runs"][stage], "--cwd", str(root), "--output", str(output)]
            for path in inputs:
                command += ["--input", str(path)]
            for path in snapshots:
                command += ["--snapshot", str(path)]
            command += ["--expect", str(output / "COMPLETE.json"), "--", sys.executable, "-u",
                        str(root / "scripts" / script), "--config", str(root / "RUN_CONFIG.json"), "--out", str(output)]
            if stage == 'smoke': command += ['--smoke']
            state.update(stage=stage, current_run=ids["runs"][stage], stage_started_at=now(), command=command)
            save(state_path, state)
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                env=dict(os.environ, PYTHONPATH=cfg["source_python"], OMP_NUM_THREADS=str(cfg.get("cpu_threads", 6)), PYTHONUTF8="1"))
            state["child_pid"] = child.pid
            save(state_path, state)
            code = child.wait()
            state.pop("child_pid", None)
            run_record = load(output / "run.json") if (output / "run.json").is_file() else {}
            receipt = load(output / "COMPLETE.json") if (output / "COMPLETE.json").is_file() else {}
            state["stage_results"][stage] = dict(run_id=ids["runs"][stage], exit_code=code,
                runner_status=run_record.get("status"), completed_at=now(), receipt=receipt)
            save(state_path, state)
            if code != 0 or run_record.get("status") != "completed":
                raise RuntimeError(f"{stage} failed with exit={code}; preserving Run, no retry")
            successful = receipt.get("passed") is True if stage == "smoke" else receipt.get("completed") is True
            if not successful:
                raise RuntimeError(f"{stage} exited zero without its required passing/completed receipt")
        state.update(status="completed", stage="finished", finished_at=now(),
                     interpretation="Registered experiment completed; scientific outcome and total Goal remain separate")
        save(state_path, state)
    except BaseException as exc:
        live = child is not None and child.poll() is None
        state.update(status="monitor_interrupted_child_live" if live else "failed", error=repr(exc),
                     traceback=traceback.format_exc(), observed_at=now())
        if live:
            state["child_process"] = process_record(child.pid)
        else:
            state["finished_at"] = now()
        save(state_path, state)
        raise


if __name__ == "__main__":
    main()
