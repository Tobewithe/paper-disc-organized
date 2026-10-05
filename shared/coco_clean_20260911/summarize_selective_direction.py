"""S039 paired local-probe statistics; no AP claim from fixed attribution."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    for name in ['pixels','full_masks','directions','witness']:
        if sha(run/f'{name}.csv')!=receipt['hashes'][f'{name}.csv']:raise RuntimeError('Changed result')
    pixels=read(run/'pixels.csv');full=read(run/'full_masks.csv');witness=read(run/'witness.csv');dirs=read(run/'directions.csv')
    meta={(int(r['image_id']),int(r['annotation_id'])):r for r in witness}
    summaries=[];contrasts=[];holdout=[];recovery=[]
    pixel_fields=['own_bce_change','same_bce_change','background_bce_change','own_positive_change',
        'same_positive_change','background_positive_change','constant_energy_fraction','logit_rms']
    full_fields=['iou_raw','iou_valid','coverage','neighbor','background']
    comparisons=[('same_projected','same'),('same_projected','background_projected'),('same_projected','constant_down'),('same_projected','total')]
    for split in ['fit','transfer']:
        images=sorted({int(r['image_id']) for r in witness if r['split']==split});index={iid:k for k,iid in enumerate(images)}
        draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
        def interval(values):
            vv=[(tid,v) for tid,v in values if v is not None and np.isfinite(v)]
            if not vv:return dict(n=0,mean=None,ci95=None)
            vals=np.array([v for _,v in vv]);ix=np.array([index[tid[0]] for tid,_ in vv])
            counts=np.bincount(ix,minlength=len(images)).astype(float);totals=np.bincount(ix,weights=vals,minlength=len(images))
            den=np.einsum('bi,i->b',draws,counts,optimize=False);num=np.einsum('bi,i->b',draws,totals,optimize=False)
            b=np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
            return dict(n=len(vals),images=len(set(ix)),mean=float(vals.mean()),ci95=np.nanquantile(b,[.025,.975]).tolist())
        for high in [True,False]:
            ws=[r for r in witness if r['split']==split and (r['high']=='True')==high]
            holdout.append(dict(split=split,high=high,targets=len(ws),
                zero_unused=sum(int(r['unused_unique'])==0 for r in ws),less_than32_unused=sum(int(r['unused_unique'])<32 for r in ws),
                unused_unique_quantiles=np.quantile([int(r['unused_unique']) for r in ws],[0,.25,.5,.75,1]).tolist(),
                own_nonempty=sum(int(r['unused_own'])>0 for r in ws),same_nonempty=sum(int(r['unused_same'])>0 for r in ws),
                background_nonempty=sum(int(r['unused_background'])>0 for r in ws)))
            for magnitude in [.001,.01]:
                for domain,source,metrics in [('direction',pixels,pixel_fields),('unused',pixels,pixel_fields),('full',full,full_fields)]:
                    values=defaultdict(list)
                    for r in source:
                        if r['split']!=split or (r['high']=='True')!=high or float(r['magnitude'])!=magnitude:continue
                        if domain!='full' and r['domain']!=domain:continue
                        tid=(int(r['image_id']),int(r['annotation_id']));arm=r['arm']
                        for metric in metrics:
                            v=(float(r[metric+'_after'])-float(r[metric+'_before'])) if domain=='full' and r[metric+'_after'] and r[metric+'_before'] else (
                                float(r[metric]) if domain!='full' and r.get(metric) else None)
                            if v is not None:values[arm,tid,metric].append(v)
                    means={key:float(np.mean(v)) for key,v in values.items()}
                    tids=[tid for tid,r in meta.items() if r['split']==split and (r['high']=='True')==high]
                    for arm in cfg['arms']:
                        summaries.append(dict(split=split,high=high,magnitude=magnitude,domain=domain,arm=arm,
                            metrics={metric:interval([(tid,means.get((arm,tid,metric))) for tid in tids]) for metric in metrics}))
                    for left,right in comparisons:
                        contrasts.append(dict(split=split,high=high,magnitude=magnitude,domain=domain,comparison=left+'-'+right,
                            metrics={metric:interval([(tid,means[left,tid,metric]-means[right,tid,metric]) for tid in tids
                                if (left,tid,metric) in means and (right,tid,metric) in means]) for metric in metrics}))
                for arm in cfg['arms']:
                    rr=[r for r in full if r['split']==split and (r['high']=='True')==high and float(r['magnitude'])==magnitude and r['arm']==arm]
                    seedrows=[]
                    for seed in range(3):
                        ss=[r for r in rr if int(r['seed'])==seed]
                        seedrows.append(dict(seed=seed,targets=len(ss),before=sum(float(r['iou_raw_before'])>=.75 for r in ss),
                            after=sum(float(r['iou_raw_after'])>=.75 for r in ss),
                            rescue=sum(float(r['iou_raw_before'])<.75<=float(r['iou_raw_after']) for r in ss),
                            harmed=sum(float(r['iou_raw_after'])<.75<=float(r['iou_raw_before']) for r in ss)))
                    recovery.append(dict(split=split,high=high,magnitude=magnitude,arm=arm,seeds=seedrows))
    result=dict(summaries=summaries,contrasts=contrasts,holdout=holdout,recovery=recovery,
        complete={k:v for k,v in receipt.items() if k!='hashes'},
        direction_limits=dict(max_relative_coefficient_step=max(float(r['relative_coefficient_step']) for r in dirs),
            inactive_rows=sum(r['active']!='True' for r in dirs),
            max_projected_own_firstorder=max(float(r['own_firstorder_after']) for r in dirs if r['projected']=='True'),
            constant_energy_roundoff_max=max(float(r['constant_energy_fraction']) for r in pixels if r.get('constant_energy_fraction'))),
        scope='Same128S038targets,three savedseeds,GTdirection/no training. Directionusesfirst512,unusedcoordinatesstrictlydisjointbutspatiallycorrelated. '
            'Three seeds averaged pertarget;2000imageclusterCI within eachsplit,pointwise/unadjusted. '
            'Zero/insufficientnegative directions kept unchanged. Full32targets/group includesinactive; regionalpixelmetricsuseownavailabledenominators. '
            'Fixed attributionIoU75 transitions are not officialRecall/AP. Projection is conventionalfirstorderprobe,notnovelmethod. '
            'Own firstorder condition is not finitebinarycoverage guarantee. No pickingbeststep orseed.',
        hashes={name:sha(run/f'{name}.csv') for name in ['pixels','full_masks','directions','witness']})
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(holdout=holdout,limits=result['direction_limits'],primary=[r for r in contrasts if r['split']=='transfer'
        and r['high'] and r['magnitude']==.01 and r['domain'] in ['unused','full']])),flush=True)


if __name__=='__main__':main()
