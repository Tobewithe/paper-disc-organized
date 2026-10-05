"""Run official-supervision smoke and fixed pilot; fail closed on any stage error."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
queue_id = "OFFICIAL_" + uuid.uuid4().hex
state_path = root / (queue_id + ".json")
state = dict(queue_id=queue_id, started_at=stamp, status="running", stages=[])


def save():
    state_path.write_text(json.dumps(state, indent=2))
    (root / "OFFICIAL_QUEUE_LATEST.json").write_text(json.dumps({"path": str(state_path), "queue_id": queue_id}))


def run_stage(name, stage, args, inputs):
    out = root / "runs" / ("RUN_" + uuid.uuid4().hex)
    entry = dict(stage=name, output=str(out), status="running")
    state["stages"].append(entry)
    save()
    cmd = [sys.executable, str(root / "runner.py"), "--study", "STUDY_67821036f497467aa3baf93c4fb9ee30",
           "--output", str(out), "--cwd", str(root), "--expect", str(out / "COMPLETE.json")]
    for f in ("scripts/official_pipeline.py", "scripts/feature_probes.py", "scripts/run_official_sequence.py", "OFFICIAL_PROTOCOL.md"):
        cmd += ["--snapshot", str(root / f)]
    for f in inputs:
        cmd += ["--input", str(f)]
    cmd += ["--", sys.executable, "-u", str(root / "scripts/official_pipeline.py"), stage, *map(str, args), "--out", str(out)]
    code = subprocess.call(cmd)
    entry.update(status="completed" if code == 0 else "failed", return_code=code)
    if code:
        state["status"] = "failed"
    save()
    if code:
        sys.exit(code)
    return out


split = json.loads((root / "SPLIT.json").read_text())
smoke_split = root / (queue_id + "_smoke_split.json")
smoke_split.write_text(json.dumps({"fit":split["fit"][:4], "dev":split["dev"][:3], "val":split["val"][:8]}))


def sequence(label, split_file, smoke=False):
    bank = run_stage(label+"_cache", "cache", ["--data", root/"data", "--weights", root/"yolo26m-seg.pt", "--split", split_file],
                     [split_file, root/"yolo26m-seg.pt"])
    ann = root / "data/annotations/instances_val2017.json"
    oracle = run_stage(label+"_oracle", "oracle", ["--bank",bank,"--annotations",ann,"--each-group",2 if smoke else 100,
                                                  "--iterations",10 if smoke else 100], [bank/"COMPLETE.json"])
    probe = run_stage(label+"_probes_s0", "probes", ["--bank",bank,"--annotations",ann,"--epochs",1 if smoke else 30,
        "--batch-images",2 if smoke else 8,"--seed",0], [bank/"COMPLETE.json"])
    analysis = run_stage(label+"_analysis", "analyze", ["--bank",bank,"--probe",probe,"--oracle",oracle],
        [oracle/"COMPLETE.json",probe/"COMPLETE.json"])
    return dict(bank=str(bank),oracle=str(oracle),probe=str(probe),analysis=str(analysis))


state["smoke"] = sequence("smoke", smoke_split, True)
state["pilot"] = sequence("pilot", root/"SPLIT.json")
state.update(status="completed", finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
