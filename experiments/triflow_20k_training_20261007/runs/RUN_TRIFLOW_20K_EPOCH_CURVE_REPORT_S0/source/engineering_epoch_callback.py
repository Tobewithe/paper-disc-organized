"""Engineering sink for verifying observer boundaries, never a COCO evaluation."""
import argparse
import json
from pathlib import Path
from frozen_io import dump_json, sha256

p = argparse.ArgumentParser()
p.add_argument('--snapshot', type=Path, required=True)
p.add_argument('--expected-epoch', type=int, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
a, extra = p.parse_known_args()
import torch
snapshot = torch.load(a.snapshot, map_location='cpu', weights_only=False)
if snapshot['contract']['smoke_only'] is not True or snapshot['contract']['max_images'] is None:
    raise ValueError('Engineering callback is restricted to actual smoke data')
if snapshot['reason'] != 'epoch_boundary' or snapshot['state']['cursor'] != {
        'epoch': a.expected_epoch+1, 'image_position': 0, 'chunk_start': 0}:
    raise ValueError('Actual completed-epoch boundary required')
receipt = {'passed': True, 'engineering_only': True, 'scope': 'Observer checkpoint capture, not COCO AP',
           'epoch': a.expected_epoch, 'snapshot_sha256': sha256(a.snapshot),
           'head_state_sha256': snapshot['head_state_sha256'], 'optimizer_applied': snapshot['state']['applied'],
           'actual_epochs': snapshot['state']['epochs'], 'ignored_extra_arguments': extra,
           'actual_coCo_evaluation_performed': False}
dump_json(a.root / 'runs' / a.run_id / 'ENGINEERING_CALLBACK.json', receipt)
print(json.dumps({'engineering_callback_passed': True, 'epoch': a.expected_epoch}), flush=True)
