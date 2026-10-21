import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import shutil
import sys
import traceback
import uuid

ROOT = Path(os.environ.get(
    "VOR_ROOT", "/srv/gpu-lab/users/a25023104/projects/video-occlusion-pretraining-v02"))
STUDY = "occlusion_representation_pretraining_20261019"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class Run:
    def __init__(self, stage, contract):
        self.stage, self.contract = stage, contract
        self.id = "RUN_VOR_" + stage.upper() + "_" + uuid.uuid4().hex[:12]
        self.path = ROOT / "experiments" / STUDY / "runs" / self.id

    def __enter__(self):
        self.lock = ROOT / "locks" / (self.stage + ".lock")
        self.lock.parent.mkdir(exist_ok=True)
        # No blind stale-lock removal. A crash requires explicit PID/job inspection.
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        self.path.mkdir(parents=True, exist_ok=False)
        source = []
        for p in sorted((ROOT / "shared/video_occlusion_pretraining").glob("*.py")):
            destination = self.path / "source" / p.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(p, destination)
            source.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
        save(self.path / "SOURCE_MANIFEST.json", source)
        self.meta = dict(study_id="STUDY_OCCLUSION_REPRESENTATION_PRETRAINING_20261019",
                         run_id=self.id, stage_id=self.stage, started_at=now(),
                         execution_status="running", scientific_status="unadjudicated",
                         executor=socket.gethostname(), pid=os.getpid(),
                         python=sys.executable, command=sys.argv,
                         slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                         training_steps_completed=0, exit_code=None)
        self.meta["source_manifest_sha256"] = sha(self.path / "SOURCE_MANIFEST.json")
        self.meta["dirty_source_snapshot_sha256"] = self.meta["source_manifest_sha256"]
        config = read(ROOT/"EXECUTION_CONFIG.json")
        self.meta["code_commit"] = config["source_commit"]
        self.meta["environment"] = {k: importlib.metadata.version(k)
                                    for k in ("torch","numpy","Pillow")}
        protocol = ROOT/"experiments"/STUDY/(
            "EXECUTION_E1_V02.md" if self.stage.startswith(("train_","evaluate_")) else "EXECUTION_E0_V02.md")
        if protocol.exists():
            self.meta["protocol_path"] = str(protocol)
            self.meta["protocol_sha256"] = sha(protocol)
            shutil.copy2(protocol,self.path/protocol.name)
        for name in ("EXECUTION_CONFIG.json","TRAIN_FREEZE.json","BACKBONE_DOWNLOAD.json"):
            if (ROOT/name).exists():
                shutil.copy2(ROOT/name,self.path/name)
        for name in ("CODE_MANIFEST.json", "EXECUTION_CONFIG.json"):
            if (ROOT / name).exists():
                self.meta[name + "_sha256"] = sha(ROOT / name)
        save(self.path / "EVAL_CONTRACT.json", self.contract)
        self.meta["contract_sha256"] = sha(self.path / "EVAL_CONTRACT.json")
        os.write(self.fd, json.dumps(self.meta).encode())
        self.update()
        save(ROOT / (self.stage + "_pointer.json"),
             dict(run_id=self.id, run_path=str(self.path), started_at=self.meta["started_at"]))
        print(json.dumps(self.meta), flush=True)
        return self

    def update(self, **kwargs):
        self.meta.update(kwargs)
        save(self.path / "run.json", self.meta)
        save(ROOT/"PIPELINE_STATUS.json",dict(updated_at=now(),run_id=self.id,
             stage=self.stage,run_path=str(self.path),slurm_job_id=self.meta["slurm_job_id"],
             execution_status=self.meta["execution_status"],
             training_steps_in_this_run=self.meta["training_steps_completed"],
             scientific_status=self.meta["scientific_status"],
             authority="run.json_on_execution_server_not_local_or_github_snapshot"))

    def __exit__(self, kind, error, tb):
        if error:
            (self.path / "ERROR.txt").write_text(
                "".join(traceback.format_exception(kind, error, tb)), encoding="utf-8")
        self.update(ended_at=now(), exit_code=1 if error else 0,
                    execution_status="failed" if error else "completed")
        rows = [dict(path=str(p.relative_to(self.path)), sha256=sha(p),
                     size_bytes=p.stat().st_size)
                for p in self.path.rglob("*") if p.is_file()]
        save(self.path / "ARTIFACT_MANIFEST.json", rows)
        os.close(self.fd)
        with contextlib.suppress(FileNotFoundError):
            self.lock.unlink()

