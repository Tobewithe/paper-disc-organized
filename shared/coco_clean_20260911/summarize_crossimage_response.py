"""Official task outcomes and paired spatial checks of locked cross-image readouts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,contextlib,gzip,io,json,time
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from frozen_mechanism_probe import ROOT,sha,write_json
from three_region_probe import write_csv
from relative_ownership_experiment import read
from summarize_relative_ownership import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    receipt=json.loads((out/'EVALUATION_COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,digest in receipt['hashes'].items():assert sha(out/name)==digest,name
    lock=json.loads((out/'LOCKED_SETTINGS.json').read_text());assert sha(out/'LOCKED_SETTINGS.json')==receipt['lock_sha256']
    chosen=json.loads((out/'selection.json').read_text());ids=chosen['evaluation'];arms=['initial']+list(lock['models']);index={iid:j for j,iid in enumerate(ids)}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    meta={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')}
    predhash=json.loads((out/'PREDICTION_HASHES.json').read_text());task=[];allgt=[];allpairs=[];started=time.monotonic()
    for domain in ['normal','expand20']:
        for arm in arms:
            pp=[]
            for iid in ids:
                path=out/'predictions_by_image'/domain/arm/f'{iid}.json.gz';assert sha(path)==predhash[str(path.relative_to(out))]
                with gzip.open(path,'rt') as f:pp.extend(json.load(f))
            assert all(np.isfinite(p['score']) for p in pp)
            row,gg,pairs=evaluate(gt,meta,ids,pp,arm);row['domain']=domain;task.append(row)
            allgt.extend(dict(domain=domain,**r) for r in gg);allpairs.extend(dict(domain=domain,**r) for r in pairs)
            assert len(gg)==sum(sum(not x.get('iscrowd',0) for x in gt.imgToAnns[i]) for i in ids)
            print(json.dumps(row),flush=True)
            write_json(out/'task_progress.json',dict(completed=len(task),total=2*len(arms),seconds=round(time.monotonic()-started,1)))
    write_csv(out/'task_summary.csv',task);write_csv(out/'gt_recovery.csv',allgt);write_csv(out/'pair_recovery.csv',allpairs)
    spatial=read(out/'evaluation_spatial.csv');draw=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    metrics=['coverage','same_neighbor','neighbor','background','mask_iou','exclusive_same_neighbor','predicted_area_over_gt']
    contrasts=[];aggregate=[];density=[]
    for domain in ['normal','expand20']:
        for kind,records,key,fields in [('spatial',spatial,lambda r:int(r['target_annotation']),metrics),('task',allgt,lambda r:int(r['annotation_id']),['hit75']),
                                        ('pair',allpairs,lambda r:(int(r['annotation_a']),int(r['annotation_b'])),['hit75'])]:
            arrays={};base=None
            for arm in arms:
                rr=sorted([r for r in records if r['domain']==domain and r['arm']==arm],key=key)
                if base is None:base=rr;expected=[key(r) for r in rr]
                assert [key(r) for r in rr]==expected and len(set(expected))==len(expected)
                arrays[arm]=np.array([[float(r[m]) for m in fields] for r in rr])
            for family in ['joint','sham']:arrays[family+'_mean']=np.mean([arrays[f'{family}_s{s}'] for s in range(3)],axis=0)
            ii=np.array([index[int(r['image_id'])] for r in base]);ici=np.array([float(r['target_ici' if kind=='spatial' else 'ici']) for r in base])
            groups={'all':np.ones(len(base),bool),'high':ici>.5+1e-10,'low':ici<=.5+1e-10,'zero':ici==0,'gt1':ici>1}
            for group,mask in groups.items():
                if not mask.any():continue
                count=np.bincount(ii[mask],minlength=len(ids));den=draw@count;valid=den>0
                for arm,array in arrays.items():
                    aggregate.append(dict(domain=domain,kind=kind,group=group,arm=arm,n=int(mask.sum()),images=int((count>0).sum()),**{m:float(array[mask,j].mean()) for j,m in enumerate(fields)}))
                comparisons=[('joint_mean','initial'),('sham_mean','initial'),('joint_mean','sham_mean')]+[(f'{family}_s{s}','initial') for family in ['joint','sham'] for s in range(3)]
                for treatment,control in comparisons:
                    delta=arrays[treatment]-arrays[control]
                    for j,metric in enumerate(fields):
                        sums=np.bincount(ii[mask],weights=delta[mask,j],minlength=len(ids));boot=(draw@sums)[valid]/den[valid]*100
                        contrasts.append(dict(domain=domain,kind=kind,group=group,treatment=treatment,control=control,metric=metric,n=int(mask.sum()),images=int((count>0).sum()),
                            mean_pp=float(delta[mask,j].mean()*100),ci_low_pp=float(np.quantile(boot,.025)),ci_high_pp=float(np.quantile(boot,.975))))
            if kind=='task':
                strata={}
                for j,r in enumerate(base):strata.setdefault((r['category_id'],r['area_bin']),{'high':[],'low':[]})['high' if ici[j]>.5+1e-10 else 'low'].append(j)
                shared={k:v for k,v in strata.items() if v['high'] and v['low']};total=sum(len(v['high'])+len(v['low']) for v in shared.values())
                delta=(arrays['joint_mean']-arrays['initial'])[:,0]
                density.append(dict(domain=domain,common_strata=len(shared),common_gt=total,
                    **{g+'_standardized_r75_gain_pp':float(sum((len(v['high'])+len(v['low']))/total*delta[v[g]].mean() for v in shared.values())*100) if total else None for g in ['high','low']}))
    write_csv(out/'aggregate_summary.csv',aggregate)
    def lookup(kind,metric,control='initial'):
        return next(r for r in contrasts if r['domain']=='normal' and r['group']=='high' and r['treatment']=='joint_mean' and r['control']==control and r['kind']==kind and r['metric']==metric)
    cov=lookup('spatial','coverage');near=lookup('spatial','same_neighbor');bg=lookup('spatial','background');iou=lookup('spatial','mask_iou');sham=lookup('spatial','mask_iou','sham_mean')
    tests={'positive_iou_mean':iou['mean_pp']>0,'positive_iou_vs_sham_mean':sham['mean_pp']>0,'coverage_abs_mean_le025pp':abs(cov['mean_pp'])<=.25,
           'coverage_ci_lower_gt_minus05pp':cov['ci_low_pp']>-.5,'same_neighbor_mean_decrease':near['mean_pp']<0,'background_increase_le025pp':bg['mean_pp']<=.25,
           'high_r75_not_decrease':lookup('task','hit75')['mean_pp']>=0,'high_pair75_not_decrease':lookup('pair','hit75')['mean_pp']>=0}
    gate=dict(status='EXPLORATORY_CANDIDATE_ONLY' if all(tests.values()) else 'NO_GO',criteria=tests,
              limitation='Pointwise pilot criteria are not proof of equivalence or density specificity. No automatic end-to-end training; report per-seed and uncertainty.')
    write_json(out/'PAIRED_ANALYSIS.json',dict(scope='2000 paired image-cluster bootstrap, three observed sampling-seed predictions averaged per GT; conditional on fit/calibration data, pointwise exploratory intervals, reused val. AP point estimates only.',contrasts=contrasts,density_standardization=density,gate=gate))
    write_json(out/'TASK_COMPLETE.json',dict(status='COMPLETE',network_training=False,seconds=time.monotonic()-started,images=len(ids),source_sha256=sha(__file__),
        hashes={p.name:sha(p) for p in [out/'task_summary.csv',out/'gt_recovery.csv',out/'pair_recovery.csv',out/'aggregate_summary.csv',out/'PAIRED_ANALYSIS.json']}))
    print(json.dumps(gate),flush=True)
    for r in contrasts:
        if r['domain']=='normal' and r['group']=='high' and r['treatment']=='joint_mean' and r['metric'] in ['coverage','same_neighbor','background','mask_iou','hit75']:print(json.dumps(r),flush=True)


if __name__=='__main__':main()
