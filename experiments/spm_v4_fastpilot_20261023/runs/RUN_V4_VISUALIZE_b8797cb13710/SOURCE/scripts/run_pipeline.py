import argparse
from pathlib import Path
import platform
import sys

import torch
import ultralytics

from common import dump, read, run_context
from data import prepare_dataset, Store
from train import train_stage


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('stage', choices=['all', 'prepare', 'engineering', 'baseline', 'prototype', 'continued', 'adapter', 'spm', 'visualize'])
    args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(4)
    assert ultralytics.__version__ == '8.4.100'
    dump(root/'ENVIRONMENT.json', {'python': sys.version, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                                    'platform': platform.platform(), 'cuda': torch.version.cuda,
                                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    if args.stage in ['all', 'prepare'] and not (root/'PREPARE_POINTER.json').exists():
        with run_context(root, 'prepare') as (run, meta):
            prepare_dataset(root, run)
            dump(run/'DECISION.json', {'status': 'data_cached', 'next_action': 'train_real_only_baseline'})
    if args.stage == 'prepare':
        return
    store = Store(read(root/'DATASET.json')['rows'])
    stages = ['engineering', 'baseline', 'prototype', 'continued', 'adapter', 'spm', 'visualize'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if args.stage == 'all' and (root/(stage.upper()+'_POINTER.json')).exists():
            print('REUSE_COMPLETED_STAGE', stage, flush=True); continue
        if stage == 'engineering':
            from engineering import smoke
            smoke(root, store)
        elif stage == 'prototype':
            from prototype import train_prototypes
            train_prototypes(root, store)
        elif stage == 'visualize':
            from visualize import visualize
            visualize(root, store)
        else:
            train_stage(root, store, stage)


if __name__ == '__main__':
    main()
