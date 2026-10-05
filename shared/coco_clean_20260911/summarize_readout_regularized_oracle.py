"""S030: paired image-cluster estimates, no model or regularization selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from readout_input_probe import sha, write_json


def read(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); run = a.run
    receipt = json.loads((run/'COMPLETE.json').read_text())
    if receipt['status'] != 'COMPLETE':
        raise RuntimeError('Incomplete run')
    names = ['pixels.csv', 'spatial.csv', 'solver.csv', 'targets.csv', 'witness.csv']
    for name in names:
        if sha(run/name) != receipt['hashes'][name]:
            raise RuntimeError('Changed input: '+name)
    targets = read(run/'targets.csv')
    meta = {int(r['annotation_id']): r for r in targets}
    arms = ['original', 'unconstrained_saved', 'unconstrained_norm_restored', 'ridge_0.01', 'ridge_0.1']
    pairs = [(m, 'original') for m in arms[1:]] + [('ridge_0.01', 'unconstrained_saved'), ('ridge_0.01', 'unconstrained_norm_restored')]
    means = []; contrasts = []
    for filename, metrics in [('spatial.csv', ['coco_iou', 'coverage', 'neighbor', 'background']),
                              ('pixels.csv', ['objective', 'bce', 'dice', 'native_iou'])]:
        rows = read(run/filename)
        domain = filename[:-4]
        values = {(r['split'], r.get('domain', 'fullmask'), r['mode'], int(r['annotation_id'])):
                  np.array([float(r[k]) if r[k] else np.nan for k in metrics]) for r in rows}
        for split in ['fit', 'transfer', 'combined']:
            tt = [r for r in targets if split == 'combined' or r['split'] == split]
            images = sorted({int(r['image_id']) for r in tt}); image_index = {v: k for k, v in enumerate(images)}
            draws = np.random.default_rng(20260912).multinomial(len(images), np.ones(len(images))/len(images), size=2000)
            aids = sorted(int(r['annotation_id']) for r in tt)
            ix = np.array([image_index[int(meta[t]['image_id'])] for t in aids])
            high = np.array([meta[t]['high'] == 'True' for t in aids])
            for sub in (['fullmask'] if domain == 'spatial' else ['train512', 'unused']):
                arrays = {m: np.array([values[meta[t]['split'], sub, m, t] for t in aids]) for m in arms}
                for group, mask in [('all', np.ones(len(aids), bool)), ('high', high), ('other', ~high)]:
                    if not mask.any(): continue
                    for mode, vals in arrays.items():
                        q = vals[mask]
                        means.append(dict(domain=domain, subdomain=sub, split=split, group=group, mode=mode,
                            targets=int(mask.sum()), images=len(set(ix[mask])),
                            valid={k:int(np.isfinite(q[:,j]).sum()) for j,k in enumerate(metrics)},
                            **{k:float(np.nanmean(q[:,j])) if np.isfinite(q[:,j]).any() else None for j,k in enumerate(metrics)}))
                    for left, right in pairs:
                        delta = arrays[left] - arrays[right]
                        for j, metric in enumerate(metrics):
                            good = mask & np.isfinite(delta[:,j])
                            if not good.any(): continue
                            counts = np.bincount(ix[good], minlength=len(images)).astype(float)
                            totals = np.bincount(ix[good], weights=delta[good,j], minlength=len(images))
                            den = np.einsum('bi,i->b', draws, counts, optimize=False)
                            num = np.einsum('bi,i->b', draws, totals, optimize=False)
                            boot = num[den>0]/den[den>0]
                            contrasts.append(dict(domain=domain, subdomain=sub, split=split, group=group,
                                comparison=left+'-'+right, metric=metric, targets=int(good.sum()),
                                delta=float(delta[good,j].mean()), ci95=np.quantile(boot,[.025,.975]).tolist()))
    solver = read(run/'solver.csv'); witness = read(run/'witness.csv')
    numerical = {}
    for mode in ['ridge_0.01', 'ridge_0.1']:
        q = [r for r in solver if r['mode'] == mode]
        numerical[mode] = dict(solves=len(q), hits_cap=sum(r['hit_cap']=='True' for r in q),
            median_norm_ratio=float(np.median([float(r['coeff_norm_ratio']) for r in q])),
            max_norm_ratio=max(float(r['coeff_norm_ratio']) for r in q),
            max_relative_delta=max(float(r['relative_delta_norm']) for r in q),
            all_objectives_nonincreasing=all(float(r['final_total'])<=float(r['initial_total'])+1e-7 for r in q),
            all_norm_bounds_pass=all(float(r['relative_delta_norm'])<=float(r['relative_delta_bound'])+1e-5 for r in q))
    numerical['unconstrained'] = dict(median_norm_ratio=float(np.median([float(r['unconstrained_norm_ratio']) for r in witness])),
        max_norm_ratio=max(float(r['unconstrained_norm_ratio']) for r in witness),
        norm_restored_mask_xor=sum(int(r['norm_restore_pixel_xor']) for r in witness))
    cohort = {split:dict(targets=sum(r['split']==split for r in targets),
        images=len({r['image_id'] for r in targets if r['split']==split}),
        high=sum(r['split']==split and r['high']=='True' for r in targets),
        empty_unused=sum(r['split']==split and int(r['unused_n'])==0 for r in targets)) for split in ['fit','transfer']}
    result = dict(cohort=cohort, means=means, contrasts=contrasts, numerical=numerical,
        sources={name:sha(run/name) for name in names},
        scope='Same160GT-assisted targets;both image splits fit their ownGT. No AP or cross-image method result. '
        'Paired2000image-cluster pointwise exploratory CIs;no multiplicity correction. '
        'Unused pixels excluded from solving but sameimage sameGTbox;not independent images. '
        'lambda .01 primary,.1 sensitivity chosen before execution;report all arms. '
        'Norm regularization depends on the fixed prototype basis;not a proposed novel method.')
    write_json(run/'ANALYSIS.json', result)
    print(json.dumps(dict(cohort=cohort,numerical=numerical,
        main=[r for r in contrasts if r['group']=='all' and r['comparison']=='ridge_0.01-original' and r['subdomain'] in ['fullmask','unused']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
