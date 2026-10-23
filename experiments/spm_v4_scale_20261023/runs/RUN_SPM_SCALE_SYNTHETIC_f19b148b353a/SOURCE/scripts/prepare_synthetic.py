import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import dump, read, run_context, sha
from data_base import letterbox, save_sample

def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    with run_context(root, 'synthetic', contract='TASK_PROTOCOL_V2.md', freeze='DATA_FREEZE_SYN.json') as (run, meta):
        rows = read(root/'WARM_DATASET.json')['rows']; geometry = set(); entries = []
        for k in range(3):
            child = root/'render_plans'/f'shard_{k}'; folder = child/'runs'/read(child/'FORMAL_POINTER.json')['run_id']
            assert read(folder/'run.json')['execution_status'] == 'completed'
            entries.extend(read(folder/'DATA_MANIFEST.json')['rows'])
        assert len(entries) == 4096
        for i, row in enumerate(sorted(entries, key=lambda r: r['source_id'])):
            assert sha(row['rgb_path']) == row['rgb_sha256'] and sha(row['label_path']) == row['label_sha256']
            image = cv2.imread(row['rgb_path']); ids = np.load(row['label_path'])['instance_id']; geometry.add(row['group'])
            targets = [(ids == k).astype(np.uint8) for k in np.unique(ids) if k > 0 and (ids == k).sum() >= 16]
            image, targets, rect = letterbox(image, targets)
            cached = save_sample(root/'cache', f'synthetic_{i:05d}', image, targets,
                                 {'source_id': row['source_id'], 'domain': 'synthetic', 'group': row['group'], 'split': row['split'],
                                  'image_id': 100000+i, 'appearance_id': row['appearance_id'], 'source_sha256': row['rgb_sha256']}, rect)
            cached['ordinal'] = len(rows); rows.append(cached)
            if i % 200 == 0:
                print('SYNTHETIC_CACHE', i+1, 4096, flush=True)
        assert len(geometry) == 2048
        train = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'train'}
        dev = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'dev'}; assert not train & dev
        counts = dict(Counter(r['domain']+'/'+r['split'] for r in rows))
        dump(root/'ALL_DATASET.json', {'rows': rows, 'counts': counts, 'geometry_scenes': 2048,
                                      'warm_dataset_sha256': sha(root/'WARM_DATASET.json'), 'test_included': False})
        dump(run/'SUMMARY.json', {'counts': counts, 'geometry_scenes': 2048, 'dataset_sha256': sha(root/'ALL_DATASET.json')})
        dump(run/'DECISION.json', {'status': 'full_mixed_inputs_ready', 'next_action': 'task_prototype_mixed_and_real_adaptation'})

if __name__ == '__main__':
    main()
