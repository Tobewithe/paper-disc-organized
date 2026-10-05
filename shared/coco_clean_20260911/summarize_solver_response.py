"""S041 fixed final states, paired image-cluster inference and descriptive P90."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,contextlib,csv,gzip,io,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from readout_input_probe import sha,write_json
from eval_readout_input_pilot import ici
from eval_frozen_readouts_val_v2 import evaluate as precision_eval
from readout_composition_control import comparison
from run_rich_pixel_readout import csv_save


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def mode(s):return s.rsplit('_s',1)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    for name in ['task_summary.csv','gt_recovery.csv','spatial.csv','pair_recovery.csv','fit_metrics.json','teacher_fit.csv']:
        if sha(run/name)!=receipt['hashes'][name]:raise RuntimeError('Changed result')
    task=read(run/'task_summary.csv');gtrows=read(run/'gt_recovery.csv');spatial=read(run/'spatial.csv');pairs=read(run/'pair_recovery.csv')
    modes=['original','saved']+cfg['modes'];images=cfg['transfer_images'];ii={v:k for k,v in enumerate(images)}
    comparisons=[('solver_response','direct'),('solver_response','self_response'),('solver_response','saved'),
        ('direct','saved'),('self_response','saved'),('solver_response','original')]
    draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
    means=[];groups_out=[];contrasts=[];controlled=[]
    for m in modes:
        rr=[r for r in task if mode(r['arm'])==m]
        if len(rr)!=(1 if m=='original' else 3):raise RuntimeError('Seedcount')
        fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','gap','pair75_high']
        means.append(dict(mode=m,seeds=len(rr),**{k:float(np.mean([float(r[k]) for r in rr])) for k in fields},
            ap_seed_sd=float(np.std([float(r['mask_ap']) for r in rr],ddof=1)) if len(rr)>1 else 0.))
    for domain,rows,metrics in [('task',gtrows,['hit75']),('pair',pairs,['hit75']),('spatial',spatial,['iou','coverage','neighbor','background'])]:
        values=defaultdict(list);meta={};sets=defaultdict(set);cats={}
        for r in rows:
            tid=(int(r['annotation_a']),int(r['annotation_b'])) if domain=='pair' else int(r['annotation_id'])
            m=mode(r['arm']);sets[r['arm']].add(tid);meta[tid]=(int(r['image_id']),float(r['ici']))
            values[m,tid].append([float(r[k]=='True') if k=='hit75' else float(r[k]) if r[k] else np.nan for k in metrics])
            if domain=='task':cats[tid]=(int(r['category_id']),r['area_bin'])
        firstset=next(iter(sets.values()))
        if any(s!=firstset for s in sets.values()):raise RuntimeError('Denominatormismatch')
        tids=sorted(meta);ix=np.array([ii[meta[t][0]] for t in tids]);density=np.array([meta[t][1] for t in tids])
        groups={'all':np.ones(len(tids),bool),'low':density<=1e-10,'middle':(density>1e-10)&(density<=.5+1e-10),
            'high':density>.5+1e-10,'nonhigh':density<=.5+1e-10}
        arrays={m:np.array([np.mean(values[m,t],axis=0) for t in tids]) for m in modes}
        def boot(v,mask):
            good=mask&np.isfinite(v);denom=np.bincount(ix[good],minlength=len(images)).astype(float)
            total=np.bincount(ix[good],weights=v[good],minlength=len(images)).astype(float)
            den=np.einsum('bi,i->b',draws,denom,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
            return np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
        for m,arr in arrays.items():
            for g,mask in groups.items():
                if not mask.any():continue
                groups_out.append(dict(domain=domain,mode=m,group=g,n=int(mask.sum()),**{k:float(np.nanmean(arr[mask,j])) for j,k in enumerate(metrics)}))
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
                    delta_pp=float((delta[groups['high'],0].mean()-delta[groups['nonhigh'],0].mean())*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
                if right in ['direct','self_response']:
                    rr=[dict(image_id=meta[t][0],category_id=cats[t][0],size=cats[t][1],high=meta[t][1]>.5+1e-10,gain=float(delta[j,0])) for j,t in enumerate(tids)]
                    controlled.append(dict(comparison=left+'-'+right,**comparison(rr,('category_id','size'),'gain',images,draws)))
    tf=read(run/'teacher_fit.csv');teacher=[]
    for seed in range(3):
        rr=[r for r in tf if int(r['seed'])==seed]
        teacher.append(dict(seed=seed,targets=len(rr),nonconverged=sum(r['converged']!='True' for r in rr),
            objective_increased=sum(float(r['objective_after'])>float(r['objective_before'])+1e-9 for r in rr),
            norm_ratio_quantiles=np.quantile([float(r['coefficient_norm_ratio']) for r in rr],[0,.5,.95,1]).tolist(),
            logit_rms_quantiles=np.quantile([float(r['fit_delta_rms']) for r in rr],[0,.5,.95,1]).tolist(),
            bce_before=float(np.mean([float(r['bce_before']) for r in rr])),bce_after=float(np.mean([float(r['bce_after']) for r in rr]))))
    result=dict(means=means,seed_results=task,groups=groups_out,contrasts=contrasts,composition=controlled,teacher=teacher,
        fit_metrics=json.loads((run/'fit_metrics.json').read_text()),parity=json.loads((run/'prediction_parity.json').read_text()),
        scope='1200fit/300previouslyexploredtransfer train2017;same final15additional epochs/3seeds/inputs/head/optimizer start. '
            'GTteachers fitonly. Official allGTmatching beforeICI;seedmeans first,2000pointwiseimageclusterCI,nounadjustedAPCI. '
            'Historicalr75_low in task_summary is nonhigh, true low/middle/high in groups. No selectedseed/epoch/lambda. '
            'No causal/novelty/finalbenchmark claim. Teacher precomputation is extra compute, stated separately;shared SGD budgets equal.',
        source_hashes={n:sha(run/n) for n in ['task_summary.csv','gt_recovery.csv','spatial.csv','teacher_fit.csv']})
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(means=means,teacher=teacher,primary=[r for r in contrasts if r['domain']=='task' and r['group'] in ['high','high-vs-nonhigh']])),flush=True)
    # Supplementary metric, fixed operating curve definition from S036.
    cache=Path(cfg['cache']);subset=cache/'conversion_input/instances_probe.json';obj=json.loads(subset.read_text())
    gt=COCO();gt.dataset=dict(info=obj.get('info',{}),categories=obj['categories'],images=[r for r in obj['images'] if r['id'] in images],
        annotations=[r for r in obj['annotations'] if r['image_id'] in images])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    meta={r['id']:ici(r,[b for b in gt.imgToAnns[r['image_id']] if not b.get('iscrowd',0)]) for r in gt.anns.values() if not r.get('iscrowd',0)}
    p90=[];pr=[];base={r['arm']:r for r in task};basegt={(r['arm'],int(r['annotation_id'])):r['hit75']=='True' for r in gtrows}
    for row in task:
        arm=row['arm']
        with gzip.open(run/'predictions'/f'{arm}.json.gz','rt') as f:pp=json.load(f)
        r,rr,op=precision_eval(gt,meta,images,pp,arm)
        if abs(r['mask_ap']-float(base[arm]['mask_ap']))>1e-12 or any(v['hit75']!=basegt[arm,v['annotation_id']] for v in rr):raise RuntimeError('Precision evaluator replay')
        p90.append(r);pr.extend(rr);print(json.dumps(dict(stage='P90',arm=arm,high=r['r90_high'])),flush=True)
    csv_save(run/'precision_task.csv',p90);csv_save(run/'precision_gt.csv',pr)
    result['precision_means']=[dict(mode=m,**{k:float(np.mean([r[k] for r in p90 if mode(r['arm'])==m])) for k in
        ['r90_all','r90_low','r90_middle','r90_high','r90_nonhigh','gap90']}) for m in modes]
    result['precision_scope']=('One pooled microPrecision>=.9 point perarm, maskIoU.75; tiedscores handled asblocks, allGTordinary kept. '
        'No separate densitythreshold; descriptive PRcurve not deploymentcalibration. No thresholduncertainty interval.')
    write_json(run/'ANALYSIS.json',result)
    write_json(run/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',hashes={n:sha(run/n) for n in ['ANALYSIS.json','precision_task.csv','precision_gt.csv']},source_sha256=sha(__file__)))
    print(json.dumps(dict(precision_means=result['precision_means'])),flush=True)


if __name__=='__main__':main()
