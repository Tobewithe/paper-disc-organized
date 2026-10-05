"""S035 all scalar controls with fixed task denominator and image-cluster CIs."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json
from readout_composition_control import comparison


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def mode(s):return s.rsplit('_s',1)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    files=['task_summary.csv','gt_recovery.csv','pair_recovery.csv','spatial.csv','bias_predictions.csv']
    for name in files:
        if sha(run/name)!=receipt['hashes'][name]:raise RuntimeError('Changed results')
    task=read(run/'task_summary.csv');gt=read(run/'gt_recovery.csv');spatial=read(run/'spatial.csv');pairs=read(run/'pair_recovery.csv')
    modes=['original','coefficient','constant','instance_bias','constant_fullbatch'];fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','pair75_high','gap']
    means=[]
    for m in modes:
        q=[r for r in task if mode(r['arm'])==m]
        if len(q)!=(1 if m in ['original','constant_fullbatch'] else 3):raise RuntimeError('Wrong seed count')
        means.append(dict(mode=m,seeds=len(q),**{f:float(np.mean([float(r[f]) for r in q])) for f in fields},
            ap_seed_sd=float(np.std([float(r['mask_ap']) for r in q],ddof=1)) if len(q)>1 else 0.))
    images=cfg['transfer_images'];index={x:k for k,x in enumerate(images)}
    draws=np.random.default_rng(20260912).multinomial(len(images),np.ones(len(images))/len(images),size=2000)
    comparisons=[('constant','original'),('constant_fullbatch','original'),('instance_bias','original'),('coefficient','original'),
                 ('instance_bias','constant'),('instance_bias','constant_fullbatch'),('instance_bias','coefficient'),
                 ('constant_fullbatch','constant'),('constant_fullbatch','coefficient')]
    contrasts=[];groupmeans=[];controlled=[]
    for domain,rows,metrics in [('task',gt,['hit75']),('pair',pairs,['hit75']),('spatial',spatial,['iou','coverage','neighbor','background'])]:
        values=defaultdict(list);meta={};sets=defaultdict(set);cats={}
        for r in rows:
            tid=(int(r['annotation_a']),int(r['annotation_b'])) if domain=='pair' else int(r['annotation_id'])
            m=mode(r['arm']);sets[r['arm']].add(tid)
            values[m,tid].append([float(r[k]=='True') if k=='hit75' else float(r[k]) if r[k] else np.nan for k in metrics])
            meta[tid]=(int(r['image_id']),float(r['ici']))
            if domain=='task':cats[tid]=(int(r['category_id']),r['area_bin'])
        expected=next(iter(sets.values()))
        if any(v!=expected for v in sets.values()):raise RuntimeError('Different GT denominators')
        tids=sorted(meta);ix=np.array([index[meta[t][0]] for t in tids]);dens=np.array([meta[t][1] for t in tids])
        arrays={m:np.array([np.mean(values[m,t],axis=0) for t in tids]) for m in modes}
        groups={'all':np.ones(len(tids),bool),'low':dens<=1e-10,'middle':(dens>1e-10)&(dens<=.5+1e-10),'high':dens>.5+1e-10,'nonhigh':dens<=.5+1e-10}
        def boot(v,mask):
            ok=mask&np.isfinite(v);count=np.bincount(ix[ok],minlength=len(images)).astype(float)
            total=np.bincount(ix[ok],weights=v[ok],minlength=len(images))
            den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
            return np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
        for m,arr in arrays.items():
            for g,mask in groups.items():
                if not mask.any():continue
                groupmeans.append(dict(domain=domain,mode=m,group=g,n=int(mask.sum()),**{k:float(np.nanmean(arr[mask,j])) for j,k in enumerate(metrics)}))
        for left,right in comparisons:
            delta=arrays[left]-arrays[right]
            for j,k in enumerate(metrics):
                for g,mask in groups.items():
                    if not mask.any():continue
                    b=boot(delta[:,j],mask)
                    contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric=k,group=g,n=int(mask.sum()),
                        delta_pp=float(np.nanmean(delta[mask,j])*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
            if domain=='task':
                b=boot(delta[:,0],groups['high'])-boot(delta[:,0],groups['nonhigh'])
                contrasts.append(dict(domain=domain,comparison=left+'-'+right,metric='gap_narrowing',group='high-vs-nonhigh',
                    delta_pp=float((delta[groups['high'],0].mean()-delta[groups['nonhigh'],0].mean())*100),
                    ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
                if (left,right) in [('instance_bias','original'),('constant_fullbatch','original'),('instance_bias','coefficient')]:
                    rr=[dict(image_id=meta[t][0],category_id=cats[t][0],size=cats[t][1],high=meta[t][1]>.5+1e-10,gain=float(delta[j,0])) for j,t in enumerate(tids)]
                    controlled.append(dict(comparison=left+'-'+right,metric='high_minus_nonhigh_gain',**comparison(rr,('category_id','size'),'gain',images,draws)))
    bias=read(run/'bias_predictions.csv');bias_summary=[]
    for m in ['constant','constant_fullbatch','instance_bias']:
        q=[r for r in bias if r['mode']==m];bb=np.array([float(r['bias']) for r in q])
        bias_summary.append(dict(mode=m,rows=len(q),mean=float(bb.mean()),quantiles=np.quantile(bb,[0,.01,.1,.5,.9,.99,1]).tolist(),
            positive_fraction=float(np.mean(bb>0)),unchanged_area_fraction=float(np.mean([int(r['original_input_area'])==int(r['predicted_input_area']) for r in q])),
            shrunk_fraction=float(np.mean([int(r['predicted_input_area'])<int(r['original_input_area']) for r in q]))))
    result=dict(means=means,seed_results=task,group_means=groupmeans,contrasts=contrasts,composition=controlled,bias_predictions=bias_summary,
        fit_metrics=json.loads((run/'fit_metrics.json').read_text()),fullbatch_reference=json.loads((run/'constant_fullbatch.json').read_text()),
        parity=json.loads((run/'prediction_parity.json').read_text()),
        source_hashes={name:sha(run/name) for name in files},
        scope='300exploredtrain2017images, allGTofficial matching;3newbiasseeds and3savedcoefficientseeds. '
            'Fullbatchconstant isone train-only stationary reference,not equalSGDbudget or guaranteedoptimum. '
            '2000pointwiseimageclusterCI seedmeans first;noAPCI/multiplicity/finalval/equivalence inference. '
            'Bias preserves within-instance logitranking but postresize binarygeometry can differ. '
            'Category-size commonstrata require>=3GTperdensity;min-countweights recomputedperdraw,coarsecontrol notcausality.')
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(means=means,reference=result['fullbatch_reference'],
        primary=[r for r in contrasts if r['domain']=='task' and r['group'] in ['high','high-vs-nonhigh']]),ensure_ascii=False))


if __name__=='__main__':main()
