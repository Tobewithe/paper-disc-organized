"""Register and run strict CPU/synthetic checks on the intended baseline executor."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/official_gt_failure_rebaseline_20261013"
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require, sha256, write_new


def main():
    import psutil
    require(psutil.virtual_memory().available >= 2 * 1024**3, "not enough free RAM for engineering preflight")
    run = STUDY / "runs" / ("RUN_ENGINEERING_LAPTOP_" + uuid.uuid4().hex)
    run.mkdir(parents=True)
    meta = {"run_id": run.name, "study_id": read_json(STUDY / "study.json")["study_id"],
            "stage_id": "S0_engineering", "execution_status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "executor": {"environment": "laptop_windows", "host": socket.gethostname(), "interpreter": sys.executable},
            "code_commit": read_json(ROOT / "EXECUTION_SOURCE.json")["code_commit"],
            "protocol_path": str(STUDY / "PROTOCOL.md"), "protocol_sha256": sha256(STUDY / "PROTOCOL.md"),
            "commands": [], "scientific_status": "not_applicable_engineering_only", "transfer_status": "local_only"}
    write_new(run / "run.json", meta)
    commands = [
        [sys.executable, "-B", str(ROOT / "shared/evaluation/tests/test_posttrain_audit.py"), "--require-integration", "-v"],
        [sys.executable, "-B", str(ROOT / "shared/evaluation/tests/test_posttrain_completion.py"), "--require-integration", "-v"],
        [sys.executable, "-B", str(ROOT / "shared/evaluation/tests/test_native_baseline_receipt.py"), "-v"],
        [sys.executable, "-B", str(STUDY / "scripts/check_native_configuration.py")]]
    code = 0
    for i, command in enumerate(commands):
        with (run / f"check_{i}.log").open("x", encoding="utf8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true"})
        meta["commands"].append({"command": command, "exit_code": result.returncode, "log": f"check_{i}.log"})
        print("CHECK", i, result.returncode, flush=True)
        if result.returncode:
            code = result.returncode
            break
    meta.update(exit_code=code, ended_at=datetime.now(timezone.utc).isoformat(),
                execution_status="completed" if code == 0 else "failed", artifact_status="complete")
    (run / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    print("ENGINEERING_RUN", run, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
