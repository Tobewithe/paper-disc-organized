import argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cm

from common import dump, read, run_context, sha
from data import Store
from data_base import save_sample

def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    with run_context(root, 'cutout', contract='TASK_PROTOCOL_V2.md', freeze='DATA_FREEZE.json') as (run, meta):
        real = read(root/'REAL_DATASET.json')['rows']; train = [r for r in real if r['split'] == 'train']; store = Store(real)
        eligible = []
        for row in train:
            for i, encoded in enumerate(row['rles']):
                box = cm.toBbox(encoded); area = float(cm.area(encoded))
                if min(box[2:]) >= 12 and area/max(1., box[2]*box[3]) >= .48:
                    eligible.append((row, i))
        assert eligible
        rng = np.random.default_rng(23102311); rows = list(real)
        for j in range(4096):
            base = train[int(rng.integers(len(train)))]; donor, index = eligible[int(rng.integers(len(eligible)))]
            rgb = store.get(base)['rgb'].copy(); donor_rgb = store.get(donor)['rgb']
            mask = cm.decode(donor['rles'][index]).astype(np.uint8); ys, xs = np.where(mask > 0)
            x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
            patch, mask = donor_rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            scale = float(rng.uniform(.8, 1.2)); hh, ww = min(300, max(8, round(len(patch)*scale))), min(300, max(8, round(patch.shape[1]*scale)))
            patch = cv2.resize(patch, (ww, hh)); mask = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
            left, top, width, height = base['valid_rect']
            hh, ww = min(hh, height), min(ww, width); patch, mask = patch[:hh, :ww], mask[:hh, :ww]
            x = int(rng.integers(left, left+width-ww+1)); y = int(rng.integers(top, top+height-hh+1))
            pasted = np.zeros((640, 640), bool); pasted[y:y+hh, x:x+ww] = mask > 0
            rgb[y:y+hh, x:x+ww][mask > 0] = patch[mask > 0]
            targets = [np.where(pasted, 0, cm.decode(enc)).astype(np.uint8) for enc in base['rles']]
            targets = [m for m in targets if m.sum() >= 16]+[pasted.astype(np.uint8)]
            row = save_sample(root/'cache', f'cutout_{j:05d}', rgb, targets,
                              {'source_id': f'cutout/{j}', 'domain': 'cutout', 'group': f'cutout/{j}', 'split': 'train',
                               'image_id': 200000+j, 'base_source_id': base['source_id'], 'donor_source_id': donor['source_id'],
                               'donor_instance': index}, base['valid_rect'])
            row['ordinal'] = len(rows); rows.append(row)
            if j % 200 == 0:
                print('CUTOUT', j+1, 4096, flush=True)
        dump(root/'WARM_DATASET.json', {'rows': rows, 'real_dataset_sha256': sha(root/'REAL_DATASET.json'),
                                       'counts': {'real_train': 1272, 'real_dev': 432, 'cutout_train': 4096}, 'test_included': False})
        dump(run/'SUMMARY.json', {'cutout_images': 4096, 'eligible_donor_instances': len(eligible),
                                  'all_donors_train_only': True, 'warm_dataset_sha256': sha(root/'WARM_DATASET.json')})
        dump(run/'DECISION.json', {'status': 'train_only_cutout_complete', 'next_action': 'instance_task_prototype_warmup'})

if __name__ == '__main__':
    main()
