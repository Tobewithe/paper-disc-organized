"""Verify real cache copy and reject mismatched backbone/data before formal training."""
import argparse
from pathlib import Path

import torch

from common import dump, read, sha
from feature_cache import reuse
from task_context import task_context


def main(root):
    parent = root/'runs'/read(root/'PREFETCH_SYNTHETIC_POINTER.json')['run_id']
    snapshot = parent/'Bstar_snapshot.pt'; receipt = read(parent/'CACHE_RECEIPT.json')
    with task_context(root, 'engineering_cache', freeze='CACHE_CHECK_FREEZE.json') as (run, meta):
        matched = run/'matching'; matched.mkdir()
        payload = reuse(root, matched, snapshot, 'synthetic', 'ALL_DATASET.json')
        assert payload is not None and sha(matched/'features.pt') == receipt['features_sha256']
        assert len(payload['a']) == receipt['tokens'] and len(payload['episodes']) == receipt['source_episodes']
        rows = {r['source_id']: r for r in read(root/'ALL_DATASET.json')['rows']}
        assert all(rows[e['source_id']]['split'] == 'train' and e['domain'] == 'synthetic' for e in payload['episodes'])
        initial = Path(read(root/'PILOT_CONFIG.json')['initial_weights'])
        other_backbone = run/'mismatching_backbone'; other_backbone.mkdir()
        assert reuse(root, other_backbone, initial, 'synthetic', 'ALL_DATASET.json') is None
        assert not (other_backbone/'features.pt').exists()
        other_data = run/'mismatching_data'; other_data.mkdir()
        assert reuse(root, other_data, snapshot, 'synthetic', 'WARM_DATASET.json') is None
        assert not (other_data/'features.pt').exists()
        summary = {'real_feature_cache_parent_run_id': parent.name, 'verified_feature_file_sha256': receipt['features_sha256'],
                   'real_tokens_loaded': len(payload['a']), 'train_only_source_episodes': len(payload['episodes']),
                   'matching_cache_copied_and_loaded': True, 'mismatching_backbone_rejected': True,
                   'mismatching_dataset_rejected': True, 'formal_training_steps': 0}
        dump(run/'SUMMARY.json', summary); dump(run/'DECISION.json', {'status': 'cache_engineering_passed',
                         'next_action': 'continue_frozen_training_with_optional_verified_cache_reuse'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    torch.set_num_threads(2); main(Path(parser.parse_args().root))
