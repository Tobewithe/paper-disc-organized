import argparse
from pathlib import Path
import platform
import sys
import torch
import ultralytics

from common import dump, read, run_context
from data import Store, prepare_real
from train_baseline import smoke, train

def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'train']); p.add_argument('--root', required=True)
    args = p.parse_args(); root = Path(args.root); torch.set_num_threads(4)
    assert ultralytics.__version__ == '8.4.100'
    if args.stage == 'prepare':
        with run_context(root, 'prepare_real') as (run, meta):
            prepare_real(root, run)
        return
    dump(root/'ENVIRONMENT.json', {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                                    'ultralytics': ultralytics.__version__, 'cuda': torch.version.cuda,
                                    'gpu': torch.cuda.get_device_name(), 'training_precision': 'bfloat16_autocast'})
    store = Store(read(root/'REAL_DATASET.json')['rows'])
    smoke(root, store); train(root, store)

if __name__ == '__main__':
    main()
