"""Preserve a new durable prefix before repairing only epoch callback metadata."""
import argparse
import subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
a=p.parse_args()
raise SystemExit(subprocess.run(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile',
    '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(Path(__file__).with_name('stop_observer_at_checkpoint.ps1')),
    '-Root', str(a.root), '-ControllerRun', a.run_id]).returncode)
