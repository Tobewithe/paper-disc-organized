"""Run the full-COCO coefficient generalization curve after bank completion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


SIZES = ((16, 200), (64, 80), (256, 40), (800, 30), (3200, 20), (10000, 15))


def write(path, content):
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def run_id():
    return "RUN_" + uuid.uuid4().hex


def main(a):
    state = {"bank": str(a.bank), "status": "waiting_for_bank", "runs": []}
    status = a.root / "FULL_CURVE_QUEUE.json"
    write(status, state)
    while not (a.bank / "COMPLETE.json").exists():
        if (a.bank / "run.json").exists():
            bank_state = json.loads((a.bank / "run.json").read_text())
            if bank_state["status"] in ("failed", "interrupted"):
                state["status"] = "bank_failed"
                write(status, state)
                raise RuntimeError("Official cache did not complete")
        time.sleep(30)
    audit = json.loads((a.bank / "QUANTIZATION_AUDIT.json").read_text())
    max_error = max(row["absolute_error"] for row in audit)
    assert max_error < 1e-3, max_error
    state.update(status="running", bank_max_loss_error=max_error)
    write(status, state)
    runner = a.source / "runner.py"
    def execute(label, script, arguments, inputs):
        rid = run_id()
        output = a.root / "runs" / rid
        command = [sys.executable, str(runner), "--study", str(a.root / "study.json"),
            "--output", str(output), "--run-id", rid, "--cwd", str(a.root)]
        for item in inputs:
            command += ["--input", str(item)]
        for item in (script, a.root / "PROTOCOL.md"):
            command += ["--snapshot", str(item)]
        command += ["--expect", str(output / "COMPLETE.json"), "--metrics",
                    str(output / "SUMMARY.json"), "--", sys.executable, "-u", str(script),
                    *arguments, "--out", str(output)]
        record = {"label": label, "run_id": rid, "path": str(output), "status": "running",
                  "started_at": datetime.now(timezone.utc).isoformat()}
        state["runs"].append(record)
        write(status, state)
        result = subprocess.run(command, cwd=a.root)
        record["status"] = "completed" if result.returncode == 0 else "failed"
        record["return_code"] = result.returncode
        write(status, state)
        if result.returncode:
            state["status"] = "failed"
            write(status, state)
            raise RuntimeError(f"{label} exited {result.returncode}")
        return output

    anchor = execute("new_16_anchor_oracle", a.root / "scripts" / "anchor_oracle.py",
        ["--source", str(a.source), "--bank", str(a.bank)],
        [a.bank / "COMPLETE.json", a.bank / "ORDER.json"])
    for n, epochs in SIZES:
        execute(f"h_only_n{n}", a.root / "scripts" / "scale_curve.py",
            ["--source", str(a.source), "--bank", str(a.eval_bank),
             "--train-bank", str(a.bank), "--order", str(a.bank / "ORDER.json"),
             "--anchor-oracle", str(anchor / "ROWS.json"), "--images", str(n),
             "--epochs", str(epochs), "--min-epochs", str(epochs)],
            [a.bank / "COMPLETE.json", a.bank / "ORDER.json", anchor / "ROWS.json"])
    state["status"] = "completed"
    write(status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("root", "source", "bank", "eval_bank"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(parser.parse_args())
