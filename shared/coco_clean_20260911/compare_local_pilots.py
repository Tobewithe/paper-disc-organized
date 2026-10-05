"""Image-cluster paired intervals, averaging three fixed training seeds.

Intervals capture sampling images, conditional on the three completed seeds;
they are not training-population intervals or multiplicity-adjusted claims.
"""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'diagnostics'
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    result=[]
    for name in ['local_coeff_eval_20260911','local_dice_eval_20260911']:
        folder=ROOT/name
        if not (folder/'COMPLETE.json').exists():continue
        spatial=read(folder/'spatial.csv');recovery=read(folder/'gt_recovery.csv')
        imageids=sorted({int(r['image_id']) for r in recovery});imageidx={v:k for k,v in enumerate(imageids)}
        rng=np.random.default_rng(20260911);draw=rng.integers(len(imageids),size=(2000,len(imageids)))
        for data,metrics in [(spatial,['coverage','same_neighbor','background','mask_iou']),(recovery,['mask_recovered75'])]:
            indexed={(r['arm'],int(r['target_annotation'])):r for r in data}
            initial=[r for r in data if r['arm']=='initial']
            for group in ['all','high','low']:
                cohort=[r for r in initial if group=='all' or (float(r['target_ici'])>.5+1e-10)==(group=='high')]
                for metric in metrics:
                    sums=np.zeros(len(imageids));counts=np.zeros(len(imageids))
                    for r in cohort:
                        aid=int(r['target_annotation']);ii=imageidx[int(r['image_id'])];d=[]
                        for seed in [0,1,2]:
                            values=[]
                            for mode in ['global','local']:
                                v=indexed[(f'{mode}_s{seed}',aid)][metric];values.append(float(v=='True') if metric=='mask_recovered75' else float(v))
                            d.append(values[1]-values[0])
                        sums[ii]+=np.mean(d);counts[ii]+=1
                    boot=sums[draw].sum(1)/counts[draw].sum(1).clip(1);lo,hi=np.quantile(boot,[.025,.975])
                    result.append(dict(experiment=name,group=group,metric=metric,contrast='local - equal-capacity global',seed_average=3,targets=len(cohort),mean_pp=100*sums.sum()/max(counts.sum(),1),ci_low_pp=100*lo,ci_high_pp=100*hi))
    out=ROOT/'LOCAL_PILOT_PAIRED_INTERVALS.json';out.write_text(json.dumps(dict(bootstrap='2,000 paired image-cluster draws, seeds averaged, pointwise exploratory intervals, no multiple-comparison adjustment',results=result),indent=2),encoding='utf-8')
    for r in result:
        if r['group']=='high':print(json.dumps(r))

if __name__=='__main__':main()
