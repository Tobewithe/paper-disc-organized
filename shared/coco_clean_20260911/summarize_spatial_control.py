"""S040 image-cluster intervals after averaging the three saved seeds per target."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from readout_input_probe import sha,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();p=a.run
    receipt=json.loads((p/'COMPLETE.json').read_text());cfg=json.loads((p/'protocol.json').read_text())
    data={}
    for name in ['full_masks','pixels','fits','witness']:
        if sha(p/f'{name}.csv')!=receipt['hashes'][f'{name}.csv']:raise RuntimeError('Changed result')
        with (p/f'{name}.csv').open(encoding='utf-8') as f:data[name]=list(csv.DictReader(f))
    summaries=[];contrasts=[];recovery=[]
    for split in ['fit','transfer']:
        images=sorted({int(r['image_id']) for r in data['witness'] if r['split']==split});ix={i:k for k,i in enumerate(images)}
        draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
        def interval(vv):
            vv=[(tid,v) for tid,v in vv if v is not None and np.isfinite(v)]
            if not vv:return dict(n=0,mean=None,ci95=None)
            values=np.array([v for _,v in vv]);im=np.array([ix[t[0]] for t,_ in vv]);n=len(images)
            counts=np.bincount(im,minlength=n).astype(float);sums=np.bincount(im,weights=values,minlength=n)
            den=np.einsum('bi,i->b',draws,counts,optimize=False);num=np.einsum('bi,i->b',draws,sums,optimize=False)
            b=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
            return dict(n=len(values),images=len(set(im)),mean=float(values.mean()),ci95=np.nanquantile(b,[.025,.975]).tolist())
        for high in [True,False]:
            for domain in ['full','direction','unused']:
                metrics=['iou_raw','coverage','neighbor','background'] if domain=='full' else ['iou','own_bce_change','same_bce_change','background_bce_change']
                vals=defaultdict(list)
                for r in data['full_masks' if domain=='full' else 'pixels']:
                    if r['split']!=split or (r['high']=='True')!=high or (domain!='full' and r['domain']!=domain):continue
                    tid=(int(r['image_id']),int(r['annotation_id']))
                    for m in metrics:
                        if m.endswith('_change'):v=float(r[m]) if r.get(m) else None
                        else:v=float(r[m+'_after'])-float(r[m+'_before']) if r.get(m+'_after') and r.get(m+'_before') else None
                        if v is not None:vals[r['arm'],tid,m].append(v)
                means={k:float(np.mean(v)) for k,v in vals.items()}
                tids={(int(r['image_id']),int(r['annotation_id'])) for r in data['witness'] if r['split']==split and (r['high']=='True')==high}
                for arm in cfg['arms']:
                    summaries.append(dict(split=split,high=high,domain=domain,arm=arm,
                        metrics={m:interval([(t,means.get((arm,t,m))) for t in tids]) for m in metrics}))
                for right in ['global32','global_spatial_bias36','global_nonlinear128']:
                    left='spatial_coeff128'
                    contrasts.append(dict(split=split,high=high,domain=domain,comparison=left+'-'+right,
                        metrics={m:interval([(t,means[left,t,m]-means[right,t,m]) for t in tids if (left,t,m) in means and (right,t,m) in means]) for m in metrics}))
            for arm in cfg['arms']:
                seeds=[]
                for seed in range(3):
                    rr=[r for r in data['full_masks'] if r['split']==split and (r['high']=='True')==high and r['arm']==arm and int(r['seed'])==seed]
                    seeds.append(dict(seed=seed,n=len(rr),before=sum(float(r['iou_raw_before'])>=.75 for r in rr),after=sum(float(r['iou_raw_after'])>=.75 for r in rr),
                        rescue=sum(float(r['iou_raw_before'])<.75<=float(r['iou_raw_after']) for r in rr),harmed=sum(float(r['iou_raw_after'])<.75<=float(r['iou_raw_before']) for r in rr)))
                recovery.append(dict(split=split,high=high,arm=arm,seeds=seeds))
    result=dict(summaries=summaries,contrasts=contrasts,recovery=recovery,complete={k:v for k,v in receipt.items() if k!='hashes'},
        scope='GT-assisted regularized finite convex fits at original512 coordinates; unused deduplicated and excludes all512 coords. '
            'Same-image spatial dependence; fixed128 hash targets, not failures. Three seeds averaged pertarget,2000imageclusterCI,pointwiseunadjusted. '
            'RawfullIoU75 fixed attribution not official AP/Recall. Outputbasis and regularizationgeometry differ despite matched counts.',
        hashes={n:sha(p/f'{n}.csv') for n in data})
    write_json(p/'ANALYSIS.json',result)
    print(json.dumps(dict(complete=result['complete'],primary=[r for r in summaries if r['split']=='transfer' and r['high'] and r['domain']=='full'],
        contrasts=[r for r in contrasts if r['split']=='transfer' and r['high'] and r['domain'] in ['unused','full']])))


if __name__=='__main__':main()
