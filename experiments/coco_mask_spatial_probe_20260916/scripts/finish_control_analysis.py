"""Bounded continuation of one already-running evaluation, then its analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for k in ('root', 'runner', 'source', 'previous', 'raw', 'annotations'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--coordinate-control', action='store_true')
    a = p.parse_args()
    deadline = time.monotonic()+7200
    while time.monotonic() < deadline:
        record = json.loads((a.source/'run.json').read_text())
        if record['status'] == 'completed':
            assert (a.source/'COMPLETE.json').is_file()
            break
        if record['status'] in ('failed', 'cancelled'):
            raise RuntimeError('Upstream evaluation did not complete: '+record['status'])
        time.sleep(5)
    else:
        raise TimeoutError('Evaluation not complete within two hours')
    out = a.root/'runs'/a.run_id
    script = a.root/'scripts/analyze_strength_control.py'
    cmd = [sys.executable, '-X', 'utf8', str(a.runner), '--run-id', a.run_id,
           '--study', 'STUDY_82ea1e5264ca430a97dff79b8912f847', '--output', str(out),
           '--cwd', str(a.root), '--snapshot', str(script), '--snapshot', str(Path(__file__).resolve())]
    for f in [a.source/'RESULTS.json', a.source/'MATCHED_GT75.json', a.source/'SPLIT.json',
              a.previous/'RESULTS.json', a.previous/'MATCHED_GT75.json', a.raw, a.annotations]:
        cmd += ['--input', str(f)]
    cmd += ['--expect', str(out/'COMPLETE.json'), '--', sys.executable, '-X', 'utf8', '-u', str(script)]
    for k in ('source', 'previous', 'raw', 'annotations'):
        cmd += ['--'+k, str(getattr(a, k))]
    cmd += ['--out', str(out)]
    if a.coordinate_control:
        cmd += ['--coordinate-control']
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__': main()
