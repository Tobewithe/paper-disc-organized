"""Launch only the small authorized pilot after checking pinned environment/GPU."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from readout_input_probe import sha, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='readout_input_pilot_20260912')
    args=ap.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        raise ValueError('Unsafe run name')
    root=Path(__file__).resolve().parent
    if str(root)!='/root/autodl-tmp/coco_clean_20260911':
        raise RuntimeError('Launcher is remote-only; do not change local or pinned runtime')
    if Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/pytorch/bin/python').resolve():
        raise RuntimeError('Use remote pytorch Python')
    import ultralytics
    if ultralytics.__version__!='8.4.143':
        raise RuntimeError('Ultralytics runtime mismatch')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(query.splitlines()[0])>=500:
        raise RuntimeError('GPU0 is occupied; no run launched')
    run=root/'diagnostics'/args.name
    if run.exists():
        raise RuntimeError(f'Preserve prior run: {run}')
    run.mkdir()
    commands=[
        [sys.executable,str(root/'cache_readout_input_pilot.py'),'--out',str(run/'smoke_cache'),'--smoke'],
        [sys.executable,str(root/'cache_readout_input_pilot.py'),'--out',str(run/'cache')],
        [sys.executable,str(root/'train_readout_input_pilot.py'),'--cache',str(run/'cache'),'--out',str(run/'trained')],
        [sys.executable,str(root/'eval_readout_input_pilot.py'),'--cache',str(run/'cache'),'--trained',str(run/'trained'),'--out',str(run/'evaluation')],
        [sys.executable,str(root/'summarize_readout_input_pilot.py'),'--evaluation',str(run/'evaluation')],
    ]
    shell=root/'logs'/f'{args.name}.sh'
    log=root/'logs'/f'{args.name}.log'
    exitpath=root/'logs'/f'{args.name}.exit'
    lines=['#!/bin/bash','set -euo pipefail','export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4',
           f'exec > >(tee {shlex.quote(str(log))}) 2>&1',
           'trap '+shlex.quote('code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitpath)))+' EXIT']
    lines.extend(shlex.join(c) for c in commands)
    shell.write_text('\n'.join(lines)+'\n')
    names=['readout_input_probe.py','cache_readout_input_pilot.py','train_readout_input_pilot.py',
           'eval_readout_input_pilot.py','summarize_readout_input_pilot.py','readout_input_pilot_protocol.json']
    write_json(run/'LAUNCH.json',dict(status='LAUNCHING',name=args.name,commands=commands,
               source_hashes={n:sha(root/n) for n in names},log=str(log),exit=str(exitpath),
               scope='Small32/64 head-only pilot. Formal CCL queue unchanged.'))
    subprocess.run(['screen','-dmS',args.name,'bash',str(shell)],check=True)
    print(json.dumps(dict(screen=args.name,run=str(run),log=str(log),exit=str(exitpath))))


if __name__=='__main__':
    main()
