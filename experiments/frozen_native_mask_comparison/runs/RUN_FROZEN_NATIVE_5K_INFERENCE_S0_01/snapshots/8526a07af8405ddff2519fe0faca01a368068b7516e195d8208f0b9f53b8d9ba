"""Execute a finite, predeclared sequence of separately recorded Runs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_sources(entries):
    for item in entries:
        path = Path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Locked source changed: {path}")


def check_requirements(requirements):
    for item in requirements:
        value = json.loads(Path(item["file"]).read_text(encoding="utf-8"))
        for key in item["key"].split("."):
            value = value[key]
        expected = item["equals"]
        if isinstance(expected, bool):
            passed = value is expected
        else:
            passed = value == expected
        if not passed:
            raise RuntimeError(f"Stage prerequisite failed: {item['file']} {item['key']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError("Pipeline status already exists; a retry needs a new Run")
    if not config.get("locked_sources") or not config.get("stages"):
        raise RuntimeError("A fixed source lock and finite stage list are required")
    status = {"started_at": now(), "status": "running", "stages": []}
    write_status(output, status)
    try:
        for stage in config["stages"]:
            check_sources(config["locked_sources"])
            check_requirements(stage.get("requires", []))
            record = {"run_id": stage["run_id"], "started_at": now(),
                      "status": "running", "command": stage["command"]}
            status["stages"].append(record)
            write_status(output, status)
            result = subprocess.run(stage["command"], cwd=stage["cwd"], check=False)
            record.update(return_code=result.returncode, finished_at=now(),
                          status="completed" if result.returncode == 0 else "failed")
            write_status(output, status)
            if result.returncode:
                raise RuntimeError(f"Stage failed: {stage['run_id']}")
            check_requirements(stage.get("ensures", []))
            check_sources(config["locked_sources"])
        status.update(status="completed", finished_at=now())
        write_status(output, status)
        return 0
    except Exception as exc:
        status.update(status="failed", error=str(exc), finished_at=now())
        write_status(output, status)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
