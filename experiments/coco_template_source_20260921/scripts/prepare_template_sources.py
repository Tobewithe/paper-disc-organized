"""Three prespecified template sources; no val data or val outcomes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shared_shape_head import bank_indices


def main():
    parser = argparse.ArgumentParser()
    for key in ['bank', 'split', 'learned-template', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, map_location='cpu', mmap=True, weights_only=False)
    split = json.loads(args.split.read_text())
    fit, _ = bank_indices(bank, split)
    learned = np.asarray(json.loads(args.learned_template.read_text())['field'], dtype=np.float32)
    assert learned.shape == (4, 4)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), indexing='ij')
    analytic = -(xx*xx+yy*yy)
    permutation = np.random.default_rng(20260921).permutation(16)
    shuffled = learned.ravel()[permutation].reshape(4, 4)
    sums, counts = {}, defaultdict(int)
    for first in range(0, len(fit), 128):
        indexes = fit[first:first+128]
        residual = bank['target'][indexes].float()-bank['base'][indexes].float().sigmoid()
        pooled = F.adaptive_avg_pool2d(residual[:, None], (4, 4))[:, 0]
        for index, field in zip(indexes, pooled):
            iid = bank['records'][index]['image_id']
            assert iid in split['fit_image_ids']
            sums[iid] = sums.get(iid, torch.zeros(4, 4))+field
            counts[iid] += 1
    residual_field = torch.stack([sums[iid]/counts[iid] for iid in sorted(sums)]).mean(0).numpy()
    fields = {'analytic_center': analytic, 'shuffled_learned': shuffled, 'fit_residual': residual_field}
    results = {}
    for name, field in fields.items():
        assert np.isfinite(field).all() and field.std() > 1e-8, name
        result = {'field': field.tolist(), 'source': name,
                  'normalization': 'Existing per-predicted-crop zero mean/unit std; signed amplitude.',
                  'val_used': False, 'extra_template_network_trained': False}
        if name == 'shuffled_learned':
            result.update(permutation_seed=20260921, permutation=permutation.tolist(),
                          learned_source=str(args.learned_template))
            assert np.array_equal(np.sort(field.ravel()), np.sort(learned.ravel()))
        if name == 'fit_residual':
            result.update(records=len(fit), images=len(sums),
                          definition='AdaptiveAvgPool4(target - sigmoid(base)); record mean per image, image mean.',
                          gt_usage='Only filtered fit800 bank labels. No selection200 or val labels.')
        (args.out/(name+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
        results[name] = result
    (args.out/'COMPLETE.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps({'templates': list(fields), 'fit_records': len(fit), 'fit_images': len(sums)}), flush=True)


if __name__ == '__main__':
    main()
