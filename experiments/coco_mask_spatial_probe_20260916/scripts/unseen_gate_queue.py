"""Run the preregistered fresh-image gate diagnostic on the remote host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads((a.root / 'UNSEEN_GATE_PROTOCOL.json').read_text())
    reference = a.root / 'inputs/runs/RUN_690c1580277a44679752ff727705f2e6'
    head = a.root / 'inputs/runs/RUN_8217274af2db49ffa6fb8d6725f8496b/epoch8.pt'
    source = a.root / 'runs' / spec['diagnostic_run']
    for key, script, args, inputs in [
        ('diagnostic_run', 'roi_transfer_diagnostic.py',
         ['--root', a.root, '--data', a.data, '--reference', reference, '--fresh-unseen'],
         [reference / 'SPLIT.json', a.root / 'yolo26m-seg.pt', head,
          a.root / 'inputs/runs/RUN_95530b30e7c04daf8acbf175b8b1204a/epoch8.pt',
          a.root / 'inputs/runs/RUN_6c39da21dae64ec9893479d0d28c7d33/epoch8.pt']),
        ('gate_run', 'benefit_gate_probe.py', ['--source', source, '--checkpoint', head],
         [source / 'diagnostic_bank.pt', source / 'instances.jsonl', source / 'SPLIT.json', head]),
    ]:
        rid = spec[key]
        out = a.root / 'runs' / rid
        if (out / 'run.json').exists():
            raise RuntimeError(f'Refusing to overwrite existing run {rid}')
        cmd = [sys.executable, str(a.root / 'runner.py'), '--run-id', rid,
               '--study', 'STUDY_82ea1e5264ca430a97dff79b8912f847',
               '--output', str(out), '--cwd', str(a.root)]
        for name in {script, 'component_seed_probe.py', 'learn_refinement.py',
                     'repair_refinement.py', 'unseen_gate_queue.py'}:
            cmd += ['--snapshot', str(a.root / 'scripts' / name)]
        cmd += ['--snapshot', str(a.root / 'UNSEEN_GATE_PROTOCOL.json')]
        for path in inputs:
            cmd += ['--input', str(path)]
        cmd += ['--expect', str(out / 'COMPLETE.json'), '--', sys.executable,
                '-u', str(a.root / 'scripts' / script), *map(str, args), '--out', str(out)]
        print(json.dumps({'starting': rid, 'script': script}), flush=True)
        subprocess.run(cmd, cwd=a.root, check=True)
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
