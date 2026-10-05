"""Locked gate, three-seed means, and paired image bootstrap for ownership pilot."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
from pathlib import Path
import numpy as np
from frozen_mechanism_probe import sha,write_json
from three_region_probe import write_csv
from relative_ownership_experiment import read

FAMILIES=['bce_dice','coverage_only','rank_only','rank_coverage','pair_bce_coverage','ccl_coverage']
METRICS=['coverage','same_neighbor','background','mask_iou']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    receipt=json.loads((out/'COMPLETE.json').read_text())
    for name,digest in receipt['hashes'].items():
        if sha(out/name)!=digest:raise RuntimeError(('result hash',name))
    ids=json.loads((out/'selection.json').read_text())['evaluation'];index={iid:j for j,iid in enumerate(ids)}
    arms=receipt['arms'];draw=np.random.default_rng(20260912).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    contrasts=[];aggregate=[];density=[]
    for kind,file,key,fields in [('spatial','spatial.csv',lambda r:int(r['target_annotation']),METRICS),('task','gt_recovery.csv',lambda r:int(r['annotation_id']),['hit75']),
                                  ('pair','pair_recovery.csv',lambda r:(int(r['annotation_a']),int(r['annotation_b'])),['hit75'])]:
        records=read(out/file);arrays={};base=None
        for arm in arms:
            rows=sorted([r for r in records if r['arm']==arm],key=key)
            if base is None:base=rows;expected=[key(r) for r in rows]
            if [key(r) for r in rows]!=expected or len(set(expected))!=len(expected):raise RuntimeError('Unpaired outcome cohorts')
            arrays[arm]=np.array([[float(r[m]) if m!='hit75' else float(str(r[m]).lower()=='true') for m in fields] for r in rows])
        for family in FAMILIES:arrays[family]=np.mean([arrays[f'{family}_s{s}'] for s in [0,1,2]],axis=0)
        ici=np.array([float(r['target_ici' if kind=='spatial' else 'ici']) for r in base]);ii=np.array([index[int(r['image_id'])] for r in base])
        comparisons=[(f,'bce_dice') for f in FAMILIES if f!='bce_dice']+[('rank_coverage','coverage_only'),('rank_coverage','rank_only'),('rank_coverage','pair_bce_coverage'),('rank_coverage','ccl_coverage')]+[(f,'initial') for f in FAMILIES]
        for group,mask in [('all',np.ones(len(base),bool)),('high',ici>.5+1e-10),('low',ici<=.5+1e-10)]:
            if not mask.any():continue
            count=np.bincount(ii[mask],minlength=len(ids));den=draw@count;valid=den>0
            for arm,values in arrays.items():aggregate.append(dict(kind=kind,group=group,arm=arm,n=int(mask.sum()),images=int((count>0).sum()),**dict(zip(fields,map(float,values[mask].mean(0))))))
            for treat,control in comparisons:
                delta=arrays[treat]-arrays[control]
                sums=np.stack([np.bincount(ii[mask],weights=delta[mask,j],minlength=len(ids)) for j in range(len(fields))],1);boot=(draw@sums)[valid]/den[valid,None]*100
                ci=np.quantile(boot,[.025,.975],axis=0)
                for j,m in enumerate(fields):contrasts.append(dict(kind=kind,group=group,treatment=treat,control=control,metric=m,n=int(mask.sum()),images=int((count>0).sum()),mean_pp=float(delta[mask,j].mean()*100),ci_low_pp=float(ci[0,j]),ci_high_pp=float(ci[1,j])))
        if kind=='task':
            strata={}
            for j,r in enumerate(base):strata.setdefault((r['category_id'],r['area_bin']),dict(high=[],low=[]))['high' if ici[j]>.5+1e-10 else 'low'].append(j)
            shared={k:v for k,v in strata.items() if v['high'] and v['low']};total=sum(len(v['high'])+len(v['low']) for v in shared.values())
            for f in FAMILIES:
                if f=='bce_dice':continue
                delta=(arrays[f]-arrays['bce_dice'])[:,0]
                density.append(dict(treatment=f,control='bce_dice',common_strata=len(shared),common_targets=total,
                    **{g+'_standardized_gain_pp':float(sum((len(v['high'])+len(v['low']))/total*delta[v[g]].mean() for v in shared.values())*100) if total else None for g in ['high','low']}))
    write_csv(out/'aggregate_summary.csv',aggregate)
    task=read(out/'task_summary.csv');taskmap={r['arm']:r for r in task};familytask=[]
    for family in FAMILIES:
        rows=[taskmap[f'{family}_s{s}'] for s in [0,1,2]]
        familytask.append(dict(family=family,**{m:float(np.mean([float(r[m]) for r in rows])) for m in ['mask_ap','mask_ap50','mask_ap75','r75_high','r75_low','pair75_high']},
            mask_ap_sd=float(np.std([float(r['mask_ap']) for r in rows],ddof=1))))
    write_csv(out/'family_task_summary.csv',familytask)
    def lookup(kind,metric,control='bce_dice',group='high'):
        return next(r for r in contrasts if r['kind']==kind and r['metric']==metric and r['treatment']=='rank_coverage' and r['control']==control and r['group']==group)
    cov=lookup('spatial','coverage');iou=lookup('spatial','mask_iou');r75=lookup('task','hit75');ft={r['family']:r for r in familytask}
    criteria=dict(high_r75_positive_ci=r75['ci_low_pp']>0,high_iou_positive_ci=iou['ci_low_pp']>0,high_pair_mean_positive=lookup('pair','hit75')['mean_pp']>0,
        same_neighbor_lower=lookup('spatial','same_neighbor')['mean_pp']<0,coverage_mean_ge_minus025=cov['mean_pp']>=-.25,coverage_ci_ge_minus05=cov['ci_low_pp']>=-.5,
        low_r75_ge_minus025=lookup('task','hit75',group='low')['mean_pp']>=-.25,ap_nondecrease=ft['rank_coverage']['mask_ap']>=ft['bce_dice']['mask_ap'],
        ranking_over_pair_bce_r75=lookup('task','hit75','pair_bce_coverage')['mean_pp']>0,ranking_over_pair_bce_iou=lookup('spatial','mask_iou','pair_bce_coverage')['mean_pp']>0)
    gate=dict(status='GO_EXPLORATORY_FULLVAL' if all(criteria.values()) else 'NO_GO',criteria=criteria,
        scope='Locked directional pilot criteria, pointwise intervals on reused val, not novelty confirmation or universal rejection.')
    write_json(out/'PAIRED_ANALYSIS.json',dict(scope='2000 paired image-cluster bootstrap, three trained head seeds averaged per target; fixed sampled train and observed seeds, pointwise intervals. No AP CI or multiple-testing adjustment.',contrasts=contrasts,density_standardization=density,gate=gate))
    write_json(out/'SUMMARY_COMPLETE.json',dict(status='COMPLETE',script_sha256=sha(__file__),hashes={p.name:sha(p) for p in [out/'aggregate_summary.csv',out/'family_task_summary.csv',out/'PAIRED_ANALYSIS.json']}))
    print(json.dumps(gate),flush=True)

if __name__=='__main__':main()
