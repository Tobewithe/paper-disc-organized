"""S032 all-seed task, density gap and paired image-cluster comparisons."""
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def mode(s):return s.rsplit('_s',1)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);a=ap.parse_args()
    run=a.run;receipt=json.loads((run/'COMPLETE.json').read_text())
    for name in ['task_summary.csv','gt_recovery.csv','spatial.csv']:
        if sha(run/name)!=receipt['hashes'][name]:raise RuntimeError('Changed results')
    task=read(run/'task_summary.csv');gt=read(run/'gt_recovery.csv');spatial=read(run/'spatial.csv')
    cfg=json.loads((run/'protocol.json').read_text());cache=Path(cfg['cache'])
    modes=['original']+cfg['modes'];ids=json.loads((cache/'selection.json').read_text())['transfer'];ii={x:k for k,x in enumerate(ids)}
    fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','pair75_high','gap']
    means=[]
    for m in modes:
        q=[r for r in task if mode(r['arm'])==m]
        if len(q)!=(1 if m=='original' else 3):raise RuntimeError('Seed count wrong')
        means.append(dict(mode=m,seeds=len(q),**{f:float(np.mean([float(r[f]) for r in q])) for f in fields},
            ap_seed_sd=float(np.std([float(r['mask_ap']) for r in q],ddof=1)) if len(q)>1 else 0.))
    predparity=[]
    for name in ['original_s-1_d0']+[f'native_overlap_s{s}_d0' for s in [0,1,2]]:
        oldname=name.replace('native_overlap','global')
        with gzip.open(run/'predictions'/f'{name}.json.gz','rt') as f:new=json.load(f)
        with gzip.open(a.prior/'predictions'/f'{oldname}.json.gz','rt') as f:old=json.load(f)
        predparity.append(dict(arm=name,exact=new==old,predictions=len(new)))
        if name=='original_s-1_d0' and new!=old:raise RuntimeError('Original prediction parity failed')
    draws=np.random.default_rng(20260912).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    pairs=[('native_overlap','original'),('native_independent','original'),('raw_coco','original'),
           ('native_independent','native_overlap'),('raw_coco','native_overlap'),('raw_coco','native_independent')]
    contrasts=[];groups_out=[]
    for domain,rows,metrics in [('task',gt,['hit75']),('spatial',spatial,['iou','coverage','neighbor','background'])]:
        buckets=defaultdict(list);meta={};sets=defaultdict(set)
        for r in rows:
            aid=int(r['annotation_id']);m=mode(r['arm']);sets[r['arm']].add(aid)
            buckets[m,aid].append([float(r[k]=='True') if k=='hit75' else float(r[k]) if r[k] else np.nan for k in metrics])
            meta[aid]=(int(r['image_id']),float(r['ici']))
        firstset=next(iter(sets.values()))
        if any(s!=firstset for s in sets.values()):raise RuntimeError('GTdenominator differs')
        aids=sorted(meta);ix=np.array([ii[meta[t][0]] for t in aids]);density=np.array([meta[t][1] for t in aids])
        groups={'all':np.ones(len(aids),bool),'low':density<=1e-10,'middle':(density>1e-10)&(density<=.5+1e-10),
                'high':density>.5+1e-10,'nonhigh':density<=.5+1e-10}
        arrays={m:np.array([np.mean(buckets[m,t],axis=0) for t in aids]) for m in modes}
        def boot(v,mask):
            good=mask&np.isfinite(v);denom=np.bincount(ix[good],minlength=len(ids)).astype(float)
            total=np.bincount(ix[good],weights=v[good],minlength=len(ids))
            den=np.einsum('bi,i->b',draws,denom,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
            return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        for m,vals in arrays.items():
            for group,mask in groups.items():groups_out.append(dict(domain=domain,mode=m,group=group,n=int(mask.sum()),
                **{metric:float(np.nanmean(vals[mask,k])) for k,metric in enumerate(metrics)}))
        for left,right in pairs:
            delta=arrays[left]-arrays[right]
            for k,metric in enumerate(metrics):
                for group,mask in groups.items():
                    b=boot(delta[:,k],mask)
                    contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric=metric,group=group,n=int(mask.sum()),
                        delta_pp=float(np.nanmean(delta[mask,k])*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
            if domain=='task':
                b=boot(delta[:,0],groups['high'])-boot(delta[:,0],groups['nonhigh'])
                contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric='gap_narrowing',group='high-vs-nonhigh',
                    delta_pp=float((delta[groups['high'],0].mean()-delta[groups['nonhigh'],0].mean())*100),
                    ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
    result=dict(means=means,seed_results=task,group_means=groups_out,contrasts=contrasts,prediction_parity=predparity,
        parameter_replay=json.loads((run/'prior_replay.json').read_text()),
        source_hashes={n:sha(run/n) for n in ['task_summary.csv','gt_recovery.csv','spatial.csv']},
        scope='Samefrozenhead3labels3seeds;allGTofficialmatching beforeICI. Existing300exploredtrain2017images notval. '
        '2000pairedimageclusterpointwiseCI,seedaveraged first,noAPCI or final fulltrainingclaim. '
        'Independent/raw labels are existing controls,notnovelmethods. Nativeoverlap exactprior replay measured. '
        'No GT enters inference, no selected epoch/seed/threshold;all15epochs retained.')
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(means=means,prediction_parity=predparity,
        primary=[r for r in contrasts if r['domain']=='task' and r['group'] in ['high','high-vs-nonhigh']]),ensure_ascii=False))


if __name__=='__main__':main()
