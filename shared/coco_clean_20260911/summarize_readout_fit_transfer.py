"""Image-cluster summaries of saved-model fit and independent GT oracle."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    run=a.run;receipt=json.loads((run/'COMPLETE.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
    if receipt['status']!='COMPLETE':raise RuntimeError('Incomplete input')
    modes=['original','global','scalar','rich_neighbor','oracle'];means=[];contrasts=[]
    for domain,filename,metrics in [('spatial','spatial.csv',['coco_iou','coverage','neighbor','background']),
                                   ('pixels','pixels.csv',['objective','bce','dice','native_iou'])]:
        rows=read(run/filename);data=defaultdict(list);meta={}
        for r in rows:
            aid=int(r['annotation_id']);sub=r.get('domain','fullmask');key=r['split'],sub,r['mode'],aid
            data[key].append([float(r[m]) if r[m] else np.nan for m in metrics])
            meta[aid]=dict(image_id=int(r['image_id']),high=r['high']=='True',oracle=r['oracle_selected']=='True')
        values={k:np.mean(v,axis=0) for k,v in data.items()}
        for split in ['fit','transfer']:
            image_ids=protocol['selection'][split];ix={i:j for j,i in enumerate(image_ids)}
            draws=np.random.default_rng(20260912).multinomial(len(image_ids),np.ones(len(image_ids))/len(image_ids),size=2000)
            for sub in (['fullmask'] if domain=='spatial' else ['train512','unused']):
                for cohort in ['all','oracle_subset']:
                    aids=sorted({aid for s,d,m,aid in values if s==split and d==sub and m=='original' and (cohort=='all' or meta[aid]['oracle'])})
                    idx=np.array([ix[meta[i]['image_id']] for i in aids]);high=np.array([meta[i]['high'] for i in aids])
                    arms=modes if cohort=='oracle_subset' else modes[:-1]
                    arrays={m:np.array([values[split,sub,m,i] for i in aids]) for m in arms}
                    for group,mask in [('all',np.ones(len(aids),bool)),('high',high),('other',~high)]:
                        if not mask.any():continue
                        for mode in arms:
                            vals=arrays[mode][mask]
                            means.append(dict(domain=domain,subdomain=sub,split=split,cohort=cohort,group=group,mode=mode,
                               targets=int(mask.sum()),valid_targets={m:int(np.isfinite(vals[:,k]).sum()) for k,m in enumerate(metrics)},
                               **{m:float(np.nanmean(vals[:,k])) if np.isfinite(vals[:,k]).any() else None for k,m in enumerate(metrics)}))
                        for left,right in [('global','original'),('scalar','original'),('rich_neighbor','original')]+([('oracle','original'),('oracle','global'),('oracle','rich_neighbor')] if cohort=='oracle_subset' else []):
                            delta=arrays[left]-arrays[right]
                            for k,metric in enumerate(metrics):
                                ok=mask&np.isfinite(delta[:,k]);count=int(ok.sum())
                                if not count:continue
                                denom=np.bincount(idx[ok],minlength=len(image_ids)).astype(float)
                                total=np.bincount(idx[ok],weights=delta[ok,k],minlength=len(image_ids))
                                den=np.einsum('bi,i->b',draws,denom,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
                                good=den>0;boot=num[good]/den[good]
                                contrasts.append(dict(domain=domain,subdomain=sub,split=split,cohort=cohort,group=group,
                                    comparison=left+'-'+right,metric=metric,n=count,
                                    delta=float(delta[ok,k].mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    solver=read(run/'solver.csv');targets=read(run/'targets.csv')
    special=dict(oracle_solves=len(solver),hit_cap=sum(r['hit_cap']=='True' for r in solver),
        max_norm_ratio=max(float(r['coeff_norm_ratio']) for r in solver),
        median_norm_ratio=float(np.median([float(r['coeff_norm_ratio']) for r in solver])),
        empty_unused=sum(int(r['holdout_unique'])==0 for r in targets),
        median_unused=float(np.median([int(r['holdout_unique']) for r in targets])),
        no_positive_train512=sum(int(r['native_positive512'])==0 for r in targets))
    write_json(run/'ANALYSIS.json',dict(means=means,contrasts=contrasts,special=special,
        source_sha256={name:sha(run/name) for name in ['pixels.csv','spatial.csv','solver.csv','targets.csv']},
        scope='Post-hoc pointwise2000image-cluster intervals conditional on saved3seed means. '
        'No AP or official Recall;80fit+80transfer images,160GT-assistedoracles. '
        'Unused positions exclude all512 training positions, but same-image same-box and labels; not cross-image generalization. '
        'Train512 and unused smoothing/sample sizes differ: compare treatments within domain, not absolute objectives across domains. '
        'No multiplicity correction, dense oracle subsets can be small.'))
    print(json.dumps(dict(special=special,means=[r for r in means if r['group']=='all'])),flush=True)


if __name__=='__main__':main()
