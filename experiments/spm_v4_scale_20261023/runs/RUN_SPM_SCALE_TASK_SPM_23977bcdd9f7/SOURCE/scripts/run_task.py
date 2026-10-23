import argparse
from pathlib import Path

import cv2
import torch
import ultralytics

from proto_train import train_prototypes
from train_method import smoke, train

def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['warm', 'final', 'smoke', 'continued', 'adapter', 'random_memory', 'task_spm'])
    p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(4); cv2.setNumThreads(1); assert ultralytics.__version__ == '8.4.100'
    if args.stage in ['warm', 'final']:
        train_prototypes(root, args.stage)
    elif args.stage == 'smoke':
        smoke(root)
    else:
        train(root, args.stage)

if __name__ == '__main__':
    main()
