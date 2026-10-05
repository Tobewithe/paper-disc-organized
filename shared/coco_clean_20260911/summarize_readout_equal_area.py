"""S034: all arms/seed means, paired image uncertainty and actual area drift."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def mode(s):return s.rsplit('_s',1)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    if receipt['status']!='COMPLETE':raise RuntimeError('Incomplete experiment')
    filenames=['task_summary.csv','gt_recovery.csv','pair_recovery.csv','spatial.csv','area_witness.csv']
    for filename in filenames:
        if sha(run/filename)!=receipt['hashes'][filename]:raise RuntimeError('Changed '+filename)
    task=read(run/'task_summary.csv');gt=read(run/'gt_recovery.csv');spatial=read(run/'spatial.csv');pair=read(run/'pair_recovery.csv')
    modes=['original','learned','original_rank'];fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','pair75_high','gap']
    means=[]
    for m in modes:
        q=[r for r in task if mode(r['arm'])==m]
        if len(q)!=(1 if m=='original' else 3):raise RuntimeError('Seed count changed')
        means.append(dict(mode=m,seeds=len(q),**{f:float(np.mean([float(r[f]) for r in q])) for f in fields},
            mask_ap_seed_sd=float(np.std([float(r['mask_ap']) for r in q],ddof=1)) if len(q)>1 else 0.))
    images=cfg['images'];index={i:k for k,i in enumerate(images)}
    draws=np.random.default_rng(20260912).multinomial(len(images),np.ones(len(images))/len(images),size=2000)
    contrasts=[];group_means=[]
    for domain,rows,metrics in [('task',gt,['hit75']),('spatial',spatial,['iou','coverage','neighbor','background']),('pair',pair,['hit75'])]:
        buckets=defaultdict(list);meta={};sets=defaultdict(set)
        for r in rows:
            target=(int(r['annotation_a']),int(r['annotation_b'])) if domain=='pair' else int(r['annotation_id'])
            m=mode(r['arm']);sets[r['arm']].add(target)
            buckets[m,target].append([float(r[k]=='True') if k=='hit75' else float(r[k]) if r[k] else np.nan for k in metrics])
            meta[target]=(int(r['image_id']),float(r['ici']))
        expected=next(iter(sets.values()))
        if any(v!=expected for v in sets.values()):raise RuntimeError('Denominator differs')
        targets=sorted(meta);ix=np.array([index[meta[t][0]] for t in targets]);density=np.array([meta[t][1] for t in targets])
        groups={'all':np.ones(len(targets),bool),'low':density<=1e-10,'middle':(density>1e-10)&(density<=.5+1e-10),'high':density>.5+1e-10,'nonhigh':density<=.5+1e-10}
        arrays={m:np.array([np.mean(buckets[m,t],axis=0) for t in targets]) for m in modes}
        def boot(v,mask):
            good=mask&np.isfinite(v);count=np.bincount(ix[good],minlength=len(images)).astype(float)
            total=np.bincount(ix[good],weights=v[good],minlength=len(images))
            den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
            return np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
        for m,arr in arrays.items():
            for g,mask in groups.items():
                if not mask.any():continue
                group_means.append(dict(domain=domain,mode=m,group=g,n=int(mask.sum()),
                    **{k:float(np.nanmean(arr[mask,j])) for j,k in enumerate(metrics)}))
        for left,right in [('learned','original'),('original_rank','original'),('learned','original_rank')]:
            delta=arrays[left]-arrays[right]
            for j,k in enumerate(metrics):
                for group,mask in groups.items():
                    if not mask.any():continue
                    b=boot(delta[:,j],mask)
                    contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric=k,group=group,n=int(mask.sum()),
                        delta_pp=float(np.nanmean(delta[mask,j])*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
            if domain=='task':
                b=boot(delta[:,0],groups['high'])-boot(delta[:,0],groups['nonhigh'])
                contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric='gap_narrowing',group='high-vs-nonhigh',
                    delta_pp=float((delta[groups['high'],0].mean()-delta[groups['nonhigh'],0].mean())*100),
                    ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
    area=read(run/'area_witness.csv');area_summaries=[]
    for subset,q in [('all_predictions',area),('fixed_matched',[r for r in area if r['annotation_id']])]:
        if not q:continue
        drift=np.array([float(r['original_area_delta']) for r in q]);rel=np.array([float(r['original_area_delta_relative']) for r in q])
        area_summaries.append(dict(subset=subset,rows=len(q),unique_predictions=len({(r['image_id'],r['source_index']) for r in q}),
            input_area_all_exact=True,original_area_equal_fraction=float(np.mean(drift==0)),
            mean_original_area_delta=float(drift.mean()),mean_absolute_original_area_delta=float(np.abs(drift).mean()),max_absolute_original_area_delta=float(np.abs(drift).max()),
            mean_absolute_original_relative_delta=float(np.abs(rel).mean()),relative_absolute_quantiles=np.quantile(np.abs(rel),[.5,.9,.95,.99,1.]).tolist(),
            empty_learned_original_nonempty_control=sum(int(r['learned_original_area'])==0 and int(r['control_original_area'])>0 for r in q),
            mean_original_mask_iou=float(np.mean([float(r['original_mask_iou']) for r in q])),
            mean_input_xor=float(np.mean([int(r['input_mask_xor']) for r in q])),
            fraction_changed_input=float(np.mean([int(r['input_mask_xor'])>0 for r in q])),
            cutoff_tie_records=sum(r['boundary_tie']=='True' for r in q),
            empty_input_records=sum(int(r['k'])==0 for r in q)))
    result=dict(means=means,seed_results=task,group_means=group_means,contrasts=contrasts,area=area_summaries,
        exact_prediction_parity=json.loads((run/'prediction_parity.json').read_text()),
        source_hashes={n:sha(run/n) for n in filenames},
        scope='300exploredtrain2017,3savedseeds,no newtraining. AllordinaryGTtaskmatching beforeICI. '
        '2000pairedimageclusterpointwiseCI seedmean first,noAPCI/multiplicity/finalval claim. '
        'Kpredicted by learnedhead,notGT;controlstilldepends onthathead,not independentcheaper method. '
        'Exact640binaryarea only; originalresolutiondrift measured. Equal pointmeans do not establish equivalence.')
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(means=means,area=area_summaries,
        primary=[r for r in contrasts if r['domain']=='task' and r['group'] in ['high','high-vs-nonhigh']]),ensure_ascii=False))


if __name__=='__main__':main()
