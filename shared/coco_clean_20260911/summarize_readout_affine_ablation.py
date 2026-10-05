"""Frozen affine removal task/spatial means and paired image intervals."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text())
    if receipt['status']!='COMPLETE':raise RuntimeError('Incomplete source')
    ids=json.loads((run/'protocol.json').read_text())['images'];ix={i:j for j,i in enumerate(ids)}
    rows=read(run/'task_summary.csv');means=[];modes=['original','full','affine','remainder']
    for mode in modes:
        rr=[r for r in rows if r['arm'].rsplit('_s',1)[0]==mode]
        means.append(dict(mode=mode,seeds=len(rr),**{k:float(np.mean([float(r[k]) for r in rr])) for k in
                       ['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','pair75_high','gap']}))
    draws=np.random.default_rng(20260912).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    contrasts=[];groupmeans=[]
    for domain,path,metrics in [('task','gt_recovery.csv',['hit75']),('spatial','spatial.csv',['iou','coverage','neighbor','background'])]:
        rr=read(run/path);bucket=defaultdict(list);meta={};sets=defaultdict(set)
        for r in rr:
            mode=r['arm'].rsplit('_s',1)[0];aid=int(r['annotation_id']);sets[r['arm']].add(aid)
            bucket[mode,aid].append([float(r[m]=='True') if m=='hit75' else float(r[m]) if r[m] else np.nan for m in metrics])
            meta[aid]=int(r['image_id']),float(r['ici'])
        expected=next(iter(sets.values()))
        if any(v!=expected for v in sets.values()):raise RuntimeError('Denominator mismatch')
        aids=sorted(meta);idx=np.array([ix[meta[i][0]] for i in aids]);high=np.array([meta[i][1]>.5+1e-10 for i in aids])
        arrays={m:np.array([np.mean(bucket[m,i],0) for i in aids]) for m in modes}
        def boot(d,mask):
            valid=mask&np.isfinite(d);n=np.bincount(idx[valid],minlength=len(ids)).astype(float)
            s=np.bincount(idx[valid],weights=d[valid],minlength=len(ids))
            den=np.einsum('bi,i->b',draws,n,optimize=False);num=np.einsum('bi,i->b',draws,s,optimize=False)
            return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        for group,mask in [('all',np.ones(len(aids),bool)),('high',high),('nonhigh',~high)]:
            for mode in modes:
                groupmeans.append(dict(domain=domain,group=group,mode=mode,targets=int(mask.sum()),
                   **{m:float(np.nanmean(arrays[mode][mask,k])) for k,m in enumerate(metrics)}))
            for left,right in [('full','original'),('affine','original'),('remainder','original'),('remainder','full'),('full','affine')]:
                d=arrays[left]-arrays[right]
                for k,m in enumerate(metrics):
                    b=boot(d[:,k],mask)
                    contrasts.append(dict(domain=domain,comparison=left+'-'+right,group=group,metric=m,
                        delta_pp=float(np.nanmean(d[mask,k])*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
                if domain=='task' and group=='all':
                    b=boot(d[:,0],high)-boot(d[:,0],~high)
                    contrasts.append(dict(domain=domain,comparison=left+'-'+right,group='high-vs-nonhigh',metric='gap_narrowing',
                        delta_pp=float((d[high,0].mean()-d[~high,0].mean())*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
    write_json(run/'ANALYSIS.json',dict(means=means,groups=groupmeans,contrasts=contrasts,
        exact_prediction_replay=receipt['exact_prediction_replay'],
        source_hashes={p:sha(run/p) for p in ['task_summary.csv','gt_recovery.csv','spatial.csv']},
        scope='Fixedfit-prediction affineprojection, exploratory300alreadyusedtransfertrain2017. '
              'Three savedmodelseeds, pointwiseimagecluster2000intervals, noAPCI or matchedprecisionRecall. '
              'Original/fullpredictionJSON equality withS026 verified; remainder does not imply learnedfeatureabsence.'))
    print(json.dumps(dict(means=means,contrasts=[r for r in contrasts if r['domain']=='task' and r['group'] in ['high','high-vs-nonhigh']])),flush=True)


if __name__=='__main__':main()
