"""S036 paired image bootstrap of frozen full-val instance recovery, seed means first."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    cfg=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    receipt=json.loads((run/'COMPLETE.json').read_text(encoding='utf-8'))
    if receipt['images']!=5000:raise RuntimeError('Incomplete full val')
    task=read(run/'task_summary.csv');images=cfg['image_ids'];index={x:k for k,x in enumerate(images)}
    modes=['original','coefficient','constant','instance_bias','constant_fullbatch']
    values=defaultdict(list);meta={};sets={}
    for arm in cfg['arms']:
        rows=read(run/'evaluation'/f'{arm}_gt.csv')
        mode=arm if arm in ['original','constant_fullbatch'] else arm.rsplit('_s',1)[0]
        sets[arm]=set()
        for r in rows:
            aid=int(r['annotation_id']);sets[arm].add(aid)
            values[mode,aid].append([float(r['hit75']=='True'),float(r['hit90']=='True')])
            meta[aid]=(int(r['image_id']),float(r['ici']),r['group'])
    expected=next(iter(sets.values()))
    if any(s!=expected for s in sets.values()):raise RuntimeError('Different denominators')
    tids=sorted(meta);ix=np.array([index[meta[t][0]] for t in tids]);dens=np.array([meta[t][1] for t in tids])
    arrays={m:np.array([np.mean(values[m,t],axis=0) for t in tids]) for m in modes}
    masks={'all':np.ones(len(tids),bool),'low':dens<=1e-10,'middle':(dens>1e-10)&(dens<=.5+1e-10),
        'high':dens>.5+1e-10,'nonhigh':dens<=.5+1e-10}
    draws=np.random.default_rng(20260912).multinomial(5000,np.full(5000,1/5000),size=2000)
    def bootstrap(value,mask):
        count=np.bincount(ix[mask],minlength=5000).astype(float)
        total=np.bincount(ix[mask],weights=value[mask],minlength=5000)
        den=np.einsum('bi,i->b',draws,count,optimize=False)
        num=np.einsum('bi,i->b',draws,total,optimize=False)
        return np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
    comparisons=[('coefficient','original'),('instance_bias','original'),('constant','original'),
        ('constant_fullbatch','original'),('coefficient','instance_bias'),('instance_bias','constant'),
        ('instance_bias','constant_fullbatch')]
    contrasts=[]
    for left,right in comparisons:
        delta=arrays[left][:,0]-arrays[right][:,0]
        for group,mask in masks.items():
            if not mask.any():continue
            b=bootstrap(delta,mask)
            contrasts.append(dict(comparison=left+'-'+right,metric='r75',group=group,n=int(mask.sum()),
                delta_pp=float(delta[mask].mean()*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
        for other in ['nonhigh','low','middle']:
            b=bootstrap(delta,masks['high'])-bootstrap(delta,masks[other])
            contrasts.append(dict(comparison=left+'-'+right,metric='gap_narrowing',group=other+'-vs-high',
                delta_pp=float((delta[masks['high']].mean()-delta[masks[other]].mean())*100),
                ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
    means=[]
    fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_low','r75_middle','r75_high','r75_nonhigh',
        'r90_all','r90_low','r90_middle','r90_high','r90_nonhigh','gap','gap_low_high','gap90']
    for mode in modes:
        rows=[r for r in task if r['arm']==mode or r['arm'].rsplit('_s',1)[0]==mode]
        if len(rows)!=(1 if mode in ['original','constant_fullbatch'] else 3):raise RuntimeError('Wrong seed count')
        means.append(dict(mode=mode,seeds=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in fields},
            mask_ap_seed_sd=float(np.std([float(r['mask_ap']) for r in rows],ddof=1)) if len(rows)>1 else 0.))
    result=dict(images=5000,ordinary_gt=len(tids),groups={g:int(m.sum()) for g,m in masks.items()},means=means,
        seed_results=task,contrasts=contrasts,original_box=json.loads((run/'original_box_ap.json').read_text()),
        scope='Frozen final heads, new local forward, full previously exploredval2017. '
            '2000pairedimagebootstrap seedmeans first, pointwise95%CI, noAPCI or multiplicity adjustment. '
            'P90 is descriptive pooledPRcurve with one globaloperatingpoint perarm; no thresholdCI or deploymentcalibration. '
            'Not unbiased freshholdout afterprojectwide exploration; no densityspecific or causalclaim from aggregategain alone.')
    (run/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(means=means,primary=[r for r in contrasts if r['group'] in ['high','nonhigh-vs-high']]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
