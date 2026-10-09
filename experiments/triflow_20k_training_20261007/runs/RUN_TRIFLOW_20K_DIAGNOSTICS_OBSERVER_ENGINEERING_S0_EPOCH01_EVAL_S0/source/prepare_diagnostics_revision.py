"""Run the strictly owned, checkpoint-preserving diagnostic-mode handoff."""
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
raise SystemExit(subprocess.run([
    'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
    '-File', str(Path(__file__).with_name('stop_for_diagnostics_revision.ps1')),
    '-Root', str(args.root), '-ControllerRun', args.run_id]).returncode)
