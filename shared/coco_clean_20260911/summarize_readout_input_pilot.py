"""Exploratory image-cluster paired intervals, preserving seed/draw dependence."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import stable_seed, write_json, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evaluation',type=Path,required=True)
    args=ap.parse_args()
    path=args.evaluation/'metrics.csv'
    with path.open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    # Averaging random layouts first prevents treating each layout as a new GT.
    by_target=defaultdict(list)
    for r in rows:
        key=(r['split'],r['density'],int(r['image_id']),int(r['annotation_id']),r['mode'],int(r['seed']))
        by_target[key].append(float(r['coco_iou']))
    seed_values={k:float(np.mean(v)) for k,v in by_target.items()}
    target_modes=defaultdict(list)
    for k,value in seed_values.items():
        target_modes[k[:-1]].append(value)
    averaged={k:float(np.mean(v)) for k,v in target_modes.items()}
    comparisons=[]
    for split in ['fit','transfer']:
        for density in ['all','high','other']:
            for left,right in [('linear','original'),('own','original'),('ordered','own'),
                               ('mean','own'),('ordered','mean'),('ordered','shuffled')]:
                deltas=defaultdict(list)
                for key,value in averaged.items():
                    s,d,image,aid,mode=key
                    if s!=split or mode!=left or (density!='all' and d!=density):
                        continue
                    other=(s,d,image,aid,right)
                    if other not in averaged:
                        raise RuntimeError('Unpaired target sets')
                    deltas[image].append(value-averaged[other])
                if not deltas:
                    continue
                ids=sorted(deltas)
                sums=np.array([sum(deltas[i]) for i in ids]);counts=np.array([len(deltas[i]) for i in ids])
                rng=np.random.default_rng(stable_seed('bootstrap',split,density,left,right))
                indices=rng.integers(len(ids),size=(2000,len(ids)))
                boots=sums[indices].sum(1)/counts[indices].sum(1)
                low,high=np.quantile(boots,[.025,.975])
                comparisons.append(dict(split=split,density=density,comparison=f'{left}-{right}',
                       targets=int(counts.sum()),images=len(ids),delta_iou=float(sums.sum()/counts.sum()),
                       ci95=[float(low),float(high)],bootstrap_images=2000,
                       scope='Pointwise exploratory image CI after averaging3 seeds/draws; not full training-randomness CI or multiplicity-adjusted confirmatory result'))
    by_seed=[]
    for split in ['fit','transfer']:
        for density in ['all','high','other']:
            modes=sorted({k[-2] for k in seed_values})
            for mode in modes:
                for seed in [-1,0,1,2]:
                    values=[v for k,v in seed_values.items() if k[0]==split and k[-2]==mode and k[-1]==seed and (density=='all' or k[1]==density)]
                    if values:
                        by_seed.append(dict(split=split,density=density,mode=mode,seed=seed,targets=len(values),mean_iou=float(np.mean(values))))
    write_json(args.evaluation/'PAIRED_ANALYSIS.json',dict(source_sha256=sha(path),comparisons=comparisons,
               per_seed=by_seed,interpretation='No AP, no Recall, no success decision on test thresholds. Small exploratory fixed-attribution pilot.'))


if __name__=='__main__':
    main()
