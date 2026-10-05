"""Portable, standard-library-only run recorder; no server or GPU library needed.

Can also be copied to a remote host and invoked directly as a Python script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, payload: dict):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class RunRecorder:
    """Freeze inputs before execution; record failure artifacts as well as success."""
    def __init__(self, root: Path, run_id: str, study_id: str, command: list[str], cwd: Path,
                 inputs: list[str] | None = None, snapshot_paths: list[str] | None = None,
                 expected_outputs: list[str] | None = None, scope: dict | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "run.json"
        if self.path.exists():
            raise ValueError("运行记录已存在；重试或重新评估必须使用新的运行目录")
        claim = self.root / ".run.claim"
        try:
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError as exc:
            raise ValueError("此运行目录已被启动器占用，请使用新的运行目录") from exc
        self.cwd = Path(cwd).resolve()
        self.expected_outputs = expected_outputs or []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.payload = {"schema_version": 2, "run_id": run_id, "study_id": study_id,
                        "source_kind": "runner_observed", "command": command, "working_directory": str(self.cwd),
                        "status": "preparing", "created_at": _now(), "host": platform.node(),
                        "environment": {"recorder_python": sys.executable, "recorder_python_version": platform.python_version(), "platform": platform.platform()},
                        "inputs": [], "snapshots": [], "scope": scope or {}, "expected_outputs": self.expected_outputs,
                        "dependency_coverage": "explicit_paths_only", "locations": [{"host": platform.node(), "path": str(self.root)}]}
        for value in inputs or []:
            path = (self.cwd / value).resolve()
            self.payload["inputs"].append({"path": str(path), "exists": path.is_file(), "sha256": _hash(path) if path.is_file() else "", "size": path.stat().st_size if path.is_file() else None})
        snapshots = self.root / "snapshots"
        snapshots.mkdir(exist_ok=True)
        for value in snapshot_paths or []:
            path = (self.cwd / value).resolve()
            if not path.is_file():
                raise ValueError(f"快照文件不存在：{path}")
            data = path.read_bytes()
            revision = hashlib.sha256(data).hexdigest()
            target = snapshots / revision
            target.write_bytes(data)
            self.payload["snapshots"].append({"path": str(path), "revision": revision, "snapshot": target.relative_to(self.root).as_posix()})
        try:
            self.payload["git"] = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.cwd, text=True, stderr=subprocess.DEVNULL).strip(),
                                   "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=self.cwd, stderr=subprocess.DEVNULL).strip())}
        except (OSError, subprocess.CalledProcessError):
            self.payload["git"] = {"commit": None, "dirty": None}
        _write(self.path, self.payload)

    def start(self, pid: int):
        self.payload.update(status="running", pid=pid, started_at=_now(), last_heartbeat=_now())
        _write(self.path, self.payload)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop_event.wait(5):
            with self.lock:
                self.payload["last_heartbeat"] = _now()
                _write(self.path, self.payload)

    def finish(self, code: int | None, *, error: str = "", cancelled: bool = False):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        with self.lock:
            missing = [p for p in self.expected_outputs if not (self.cwd / p).exists()]
            self.payload.update(status="cancelled" if cancelled else "completed" if code == 0 else "failed",
                                return_code=code, finished_at=_now(), error=error,
                                artifact_completeness="missing" if missing else "complete" if self.expected_outputs else "unknown",
                                missing_outputs=missing)
            self.payload["artifacts"] = [{"path": str((self.cwd / p).resolve()), "exists": (self.cwd / p).exists(),
                                          "sha256": _hash(self.cwd / p) if (self.cwd / p).is_file() else ""} for p in self.expected_outputs]
            _write(self.path, self.payload)


def execute(args) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("需要 -- 后的执行命令")
    root = Path(args.output).resolve()
    recorder = RunRecorder(root, args.run_id or f"RUN_{uuid.uuid4().hex}", args.study, command, Path(args.cwd), args.input, args.snapshot, args.expect,
                           json.loads(args.scope) if args.scope else {})
    env = os.environ.copy()
    env["RESEARCH_RUN_ID"] = recorder.payload["run_id"]
    env["RESEARCH_RUN_DIRECTORY"] = str(root)
    # Python children report from inside the actual interpreter, not the launcher.
    if Path(command[0]).name.lower().startswith("python"):
        probe = "import sys,platform,importlib.metadata as m,json; names=['torch','numpy','ultralytics']; v={};\nfor n in names:\n try:v[n]=m.version(n)\n except m.PackageNotFoundError:pass\nprint(json.dumps({'python':sys.executable,'python_version':platform.python_version(),'packages':v}))"
        try:
            recorder.payload["environment"]["execution"] = json.loads(subprocess.check_output([command[0], "-c", probe], cwd=recorder.cwd, env=env, text=True, timeout=30))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            recorder.payload["environment"]["execution_probe_error"] = str(exc)
    process = None
    try:
        with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=recorder.cwd, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            recorder.start(process.pid)
            print(json.dumps({"run_id": recorder.payload["run_id"], "pid": process.pid, "record": str(recorder.path)}, ensure_ascii=False), flush=True)
            code = process.wait()
        if args.metrics:
            metrics_path = (recorder.cwd / args.metrics).resolve()
            try:
                if metrics_path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("指标文件超过 2 MiB，请提供汇总 JSON")
                data = metrics_path.read_bytes()
                with recorder.lock:
                    recorder.payload["metrics"] = json.loads(data)
                    recorder.payload["metrics_source"] = {"path": str(metrics_path), "sha256": hashlib.sha256(data).hexdigest()}
            except (OSError, ValueError) as exc:
                with recorder.lock:
                    recorder.payload["metrics_read_error"] = str(exc)
        recorder.finish(code)
        return code
    except KeyboardInterrupt:
        if process:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            process.wait(timeout=15)
        recorder.finish(process.returncode if process else None, cancelled=True)
        return 130
    except Exception as exc:
        recorder.finish(process.returncode if process else None, error=str(exc))
        raise


def parser():
    result = argparse.ArgumentParser(description="记录实际运行、输入与快照；无需启动工作台")
    result.add_argument("--study", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--run-id", default="")
    result.add_argument("--cwd", default=os.getcwd())
    result.add_argument("--input", action="append", default=[])
    result.add_argument("--snapshot", action="append", default=[])
    result.add_argument("--expect", action="append", default=[])
    result.add_argument("--scope", default="")
    result.add_argument("--metrics", default="", help="结束时读取的汇总 JSON；不自动判断结果")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


if __name__ == "__main__":
    raise SystemExit(execute(parser().parse_args()))
