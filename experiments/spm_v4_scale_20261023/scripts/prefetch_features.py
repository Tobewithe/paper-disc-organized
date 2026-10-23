import argparse
from pathlib import Path
import shutil

import cv2
import torch

from common import dump, read, sha, state_hash
from data import Store
from feature_cache import signatures
from proto_data import extract
from task_context import task_context
from task_model import load


def main(root, stage):
    torch.manual_seed(0)
    with task_context(root, 'prefetch_'+stage, freeze='PREFETCH_FREEZE.json') as (run, meta):
        source = root/'runs/RUN_SPM_SCALE_BASELINE_162af9978b28/best.pt'
        digest = sha(source); snapshot = run/'Bstar_snapshot.pt'; shutil.copyfile(source, snapshot)
        assert sha(snapshot) == digest == sha(source), 'Source checkpoint changed during snapshot copy'
        encoder = load(snapshot).eval().requires_grad_(False); before = state_hash(encoder)
        dataset = 'WARM_DATASET.json' if stage == 'warm' else 'ALL_DATASET.json'
        domains = ['real', 'cutout'] if stage == 'warm' else ['synthetic']
        store = Store(read(root/dataset)['rows']); features = extract(root, store, encoder, run, domains)
        assert state_hash(encoder) == before
        receipt = {'upstream_checkpoint_sha256': digest, 'snapshot_source_run_id': source.parent.name,
                   'snapshot_selected_step': torch.load(snapshot, map_location='cpu', weights_only=False)['step'],
                   'dataset_sha256': sha(root/dataset), 'feature_sources': signatures(root),
                   'features_sha256': sha(run/'features.pt'), 'tokens': len(features['a']),
                   'source_episodes': len(features['episodes']), 'optimizer_steps': 0,
                   'scope': 'provisional_cache_only_final_Bstar_SHA_must_match_before_use'}
        dump(run/'CACHE_RECEIPT.json', receipt); dump(run/'SUMMARY.json', receipt)
        dump(run/'DECISION.json', {'status': 'provisional_features_ready_not_training',
                                   'next_action': 'reuse_only_if_final_selected_Bstar_and_all_inputs_match'})
        meta.update(optimizer_steps=0, upstream_checkpoint_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['warm', 'synthetic']); parser.add_argument('--root', required=True)
    args = parser.parse_args(); torch.set_num_threads(4); cv2.setNumThreads(1); main(Path(args.root), args.stage)
