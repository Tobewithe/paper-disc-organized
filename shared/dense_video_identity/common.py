from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=encode, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def setup_paths(root: Path):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    (root / "runtime_config").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(root / "runtime_config")
    for p in (root / "deps", root / "vendor" / "ultralytics_8_4_100", root / "vendor" / "TrackEval-master"):
        sys.path.insert(0, str(p))


class Run:
    """Create records only at actual execution; preserve failure and exclusive stage lock."""
    def __init__(self, root: Path, stage: str, config: dict, parents=None):
        self.root, self.stage = root, stage
        self.id = f"RUN_{stage.upper()}_{dt.datetime.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
        self.path = root / "runs" / self.id
        self.config, self.parents = config, parents or []

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / f".{self.stage}.lock"
        with self.lock.open("x", encoding="utf-8") as f:
            json.dump(dict(pid=os.getpid(), host=platform.node(), run_id=self.id), f)
        self.path.mkdir(parents=True)
        write_json(self.path / "EVAL_CONTRACT.json", self.config)
        source_files = []
        for p in sorted(Path(__file__).parent.glob("*.py")):
            dest = self.path / "source" / p.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(p.read_bytes())
            source_files.append(dict(path=p.name, sha256=sha256(dest)))
        write_json(self.path / "SOURCE_SNAPSHOT.json", source_files)
        study_id = read_json(self.root / "study.json").get("study_id", "STUDY_DENSE_VIDEO_IDENTITY_20261018") if (self.root / "study.json").exists() else "STUDY_DENSE_VIDEO_IDENTITY_20261018"
        self.meta = dict(study_id=study_id, run_id=self.id,
                         stage_id=self.stage, started_at=now(), ended_at=None,
                         executor=dict(host=platform.node(), python=sys.executable, pid=os.getpid()),
                         command=subprocess.list2cmdline([sys.executable] + sys.argv),
                         parent_run_ids=self.parents, execution_status="running",
                         scientific_status="unadjudicated", exit_code=None,
                         source_snapshot_sha256=sha256(self.path / "SOURCE_SNAPSHOT.json"),
                         contract_sha256=sha256(self.path / "EVAL_CONTRACT.json"))
        for f in ["PROTOCOL.md", "METHOD_PROTOCOL.md", "TRAJECTORY_PROTOCOL.md", "PROTECTED_EXTENSION_PROTOCOL.md", "IMPLEMENTATION_CONTRACT.md", "ENGINEERING_CORRECTIONS.md", "input_manifest.json", "source_manifest.json", "config.json", "identity_pointer.json", "annotations/occlusion_labels.json"]:
            if (self.root / f).exists():
                self.meta[f + "_sha256"] = sha256(self.root / f)
                snapshot = self.path / "inputs" / f
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes((self.root / f).read_bytes())
        write_json(self.path / "run.json", self.meta)
        print("RUN_START", self.id, flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.meta.update(ended_at=now(), execution_status="failed" if exc else "completed", exit_code=1 if exc else 0)
        if exc:
            (self.path / "ERROR.txt").write_text("".join(traceback.format_exception(exc_type, exc, tb)), encoding="utf-8")
        write_json(self.path / "run.json", self.meta)
        if self.lock.exists() and read_json(self.lock).get("run_id") == self.id:
            self.lock.unlink()
        print("RUN_END", self.id, self.meta["execution_status"], flush=True)


def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(br - tl, 0).prod(-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    ba = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return inter / np.maximum(aa[:, None] + ba[None] - inter, 1e-8)


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_gt(root, seq):
    a = np.loadtxt(root / seq["gt"], delimiter=",", ndmin=2)
    a = a[a[:, 6] == 1]
    a[:, 4:6] += a[:, 2:4]
    # AnimalTrack files use one-based frames; pixel geometry is retained as published.
    return {f: a[a[:, 0] == f] for f in range(1, seq["frames"] + 1)}
