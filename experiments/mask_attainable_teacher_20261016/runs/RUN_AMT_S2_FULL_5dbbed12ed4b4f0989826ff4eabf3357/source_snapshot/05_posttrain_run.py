"""Small registered-run lifecycle for the authorized laptop pilot."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/mask_readout_mechanism_v2_20261012"
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, sha256, require, write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temporary, path)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, key):
        return getattr(self.stream, key)


class RegisteredRun:
    def __init__(self, kind, stage, parent, code_files, wall_seconds, protocol="S2_EXECUTION_20261013.md"):
        self.run_id = "RUN_" + kind + "_" + uuid.uuid4().hex
        self.path = STUDY / "runs" / self.run_id
        self.lock = STUDY / ("." + stage + ".lock")
        self.parent = str(parent)
        self.code_files, self.wall_seconds, self.protocol = code_files, wall_seconds, protocol
        self.start = time.monotonic()
        self.meta = {"run_id": self.run_id, "study_id": "STUDY_MASK_READOUT_MECHANISM_V2_20261012",
                     "stage_id": stage, "kind": kind, "host": socket.gethostname(), "pid": os.getpid(),
                     "started_at": now(), "execution_status": "running", "plan_status": "freezing",
                     "scientific_status": "unadjudicated", "artifact_status": "partial", "transfer_status": "local_only",
                     "parent_run_id": Path(parent).name, "executor": {"environment": "laptop_windows",
                     "interpreter": sys.executable, "directory": str(ROOT), "host": socket.gethostname()},
                     "command": [sys.executable, "-B", *sys.argv], "wall_budget_seconds": wall_seconds, "exit_code": None}

    def __enter__(self):
        with self.lock.open("x", encoding="utf8") as stream:
            json.dump(self.meta, stream)
        self.path.mkdir(parents=True, exist_ok=False)
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.log = (self.path / "execution.log").open("x", encoding="utf8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, self.log), Tee(sys.stderr, self.log)
        self.update()
        print("RUN_DIRECTORY", self.path, flush=True)
        return self

    def freeze(self, contract):
        origin = read_json(STUDY / "S1_EXECUTION_SOURCE.json")
        self.meta["code_commit"] = origin["code_commit"]
        self.sources = {rel: sha256(ROOT / rel) for rel in self.code_files}
        self.meta["dirty_patch_sha256"] = sha256(STUDY / "S1_SOURCE_DIRTY.patch")
        shutil.copyfile(STUDY / "S1_SOURCE_DIRTY.patch", self.path / "dirty.patch")
        write_new(self.path / "SOURCE_MANIFEST.json", {"files": self.sources,
                  "code_commit": self.meta["code_commit"], "note": "actual uncommitted sources are snapshotted and hashed"})
        for i, relative in enumerate(self.sources):
            target = self.path / "source_snapshot" / (f"{i:02d}_" + Path(relative).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        shutil.copyfile(STUDY / self.protocol, self.path / "PROTOCOL.md")
        write_new(self.path / "EVAL_CONTRACT.json", contract)
        self.meta.update(plan_status="frozen", protocol_path=str(self.path / "PROTOCOL.md"),
                         protocol_sha256=sha256(self.path / "PROTOCOL.md"),
                         contract_sha256=sha256(self.path / "EVAL_CONTRACT.json"))
        self.update()

    def update(self, **fields):
        self.meta.update(fields)
        save(self.path / "run.json", self.meta)

    def check_budget(self):
        if time.monotonic() - self.start > self.wall_seconds:
            raise TimeoutError("frozen execution wall-clock budget exhausted")

    def finish(self, summary, status="passed", scientific="engineering_only", next_action=""):
        for relative, digest in self.sources.items():
            require(sha256(ROOT / relative) == digest, "source changed during run: " + relative)
        require(read_json(self.path / "CONFIG_AUDIT.json")["status"] == "passed",
                "post-experiment actual configuration audit is required before adjudication")
        write_new(self.path / "SUMMARY.json", {"run_id": self.run_id, **summary})
        write_new(self.path / "DECISION.json", {"status": status, "scientific_status": scientific,
                  "summary_sha256": sha256(self.path / "SUMMARY.json"), "next_action": next_action})
        self.update(execution_status="completed", scientific_status=scientific, artifact_status="complete", exit_code=0)

    def __exit__(self, exc_type, exc, tb):
        if exc:
            error = "".join(traceback.format_exception(exc_type, exc, tb))
            (self.path / "ERROR.txt").write_text(error, encoding="utf8")
            print(error, file=sys.stderr, flush=True)
            self.update(execution_status="error", scientific_status="blocked_engineering", exit_code=1, error=str(exc))
            if not (self.path / "DECISION.json").exists():
                write_new(self.path / "DECISION.json", {"status": "blocked", "reason": str(exc), "next_action": "Inspect retained failure"})
        elif self.meta["execution_status"] != "completed":
            self.update(execution_status="error", scientific_status="blocked_incomplete", exit_code=1)
        self.update(ended_at=now(), elapsed_seconds=time.monotonic() - self.start)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
        self.log.close()
        records = [{"relative_path": p.relative_to(self.path).as_posix(), "size_bytes": p.stat().st_size,
                    "sha256": sha256(p), "storage_scope": "execution_host", "availability": "present"}
                   for p in sorted(self.path.rglob("*")) if p.is_file() and p.name not in {"run.json", "ARTIFACT_MANIFEST.json"}]
        write_new(self.path / "ARTIFACT_MANIFEST.json", {"artifacts": records})
        self.lock.unlink()
        return False


def configure_native():
    from native_output_bridge import apply_runtime_policy, native_environment
    import importlib.metadata
    profile = read_json(ROOT / "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json")
    for name, version in profile["packages"].items():
        require(importlib.metadata.version(name) == version, "package drift: " + name)
    apply_runtime_policy(profile["runtime_policy"])
    environment = native_environment()
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), "native CUDA device required")
    return profile, environment


def tensor_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((list(item.shape), item.dtype)).encode())
        digest.update(item.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()
