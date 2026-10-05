"""Paired schedule comparison with explicit reused baseline/fixed-weight runs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from recording import atomic_json,now

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def draws(values,groups,seed=20260915):
    names=sorted(set(groups));index={g:i for i,g in enumerate(names)}
    sums=np.zeros(len(names));counts=np.zeros(len(names))
    for v,g in zip(values,groups):sums[index[g]]+=v;counts[index[g]]+=1
    choice=np.random.default_rng(seed).integers(0,len(names),(4000,len(names)))
    return sums[choice].sum(1)/counts[choice].sum(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());official=[];targets=[];sources=[]
    refs=[(Path(cfg['reference_remote_root']),r,True) for r in cfg['reference_evaluations']]
    refs += [(a.study,r,False) for r in cfg['runs'] if r['kind']=='evaluation']
    for root,spec,reused in refs:
        folder=root/'runs'/spec['run_id'];assert json.loads((folder/'COMPLETE.json').read_text())['status']=='complete'
        metrics=read(folder/'official_metrics.csv');panel=read(folder/'targeted_per_instance.csv')
        assert len(metrics)==4 and len(panel)==876
        official.extend(metrics);targets.extend(panel);sources.append(dict(run_id=spec['run_id'],arm=spec['arm'],checkpoint=spec['checkpoint'],reused=reused,path=str(folder)))
    write(a.out/'all_official_metrics.csv',official)
    omap={(r['checkpoint'],r['branch'],r['task'],r['arm']):r for r in official}
    tmap={(r['checkpoint'],r['branch'],r['arm'],r['annotation_id']):r for r in targets};assert len(tmap)==len(targets)
    ap=[];paired=[];interactions=[]
    keys=['raw_best_box_iou','raw_p3_best_box_iou','raw_center_error_norm','raw_same_class_box50','raw_correct_class_at_best',
          'final_confidence','final_box_iou','mask_iou','target_coverage','prediction_purity','boundary_f1',
          'final_box50','final_mask50','final_mask75','same_neighbor_leak_pred','background_leak_pred']
    for checkpoint in ['last','best']:
        for branch in cfg['evaluation_branches']:
            for reference in ['reg_only','baseline']:
                contrast='reg_scheduled minus '+reference
                for task in ['bbox','segm']:
                    b=omap[checkpoint,branch,task,reference];m=omap[checkpoint,branch,task,'reg_scheduled']
                    for key in b:
                        if key in ['arm','task','images','predictions','branch','checkpoint']:continue
                        bv=float(b[key]);mv=float(m[key]);ap.append(dict(checkpoint=checkpoint,branch=branch,task=task,contrast=contrast,metric=key,
                                                                     reference=bv,method=mv,delta_points=100*(mv-bv)))
                by_cohort={}
                for ci,cohort in enumerate(['raw_geometry_small','matched_small_control']):
                    items=[r for r in targets if r['checkpoint']==checkpoint and r['branch']==branch and r['arm']==reference and r['cohort']==cohort]
                    assert len(items)==(239 if ci==0 else 199)
                    ids=[r['annotation_id'] for r in items];groups=[r['image_id'] for r in items];by_cohort[cohort]={}
                    for key in keys:
                        b=np.array([float(tmap[checkpoint,branch,reference,i][key]) for i in ids]);m=np.array([float(tmap[checkpoint,branch,'reg_scheduled',i][key]) for i in ids])
                        sample=draws(m-b,groups,20260915+ci);lo,hi=np.quantile(sample,[.025,.975]);delta=float(np.mean(m-b))
                        paired.append(dict(checkpoint=checkpoint,branch=branch,cohort=cohort,contrast=contrast,metric=key,n=len(ids),reference=float(b.mean()),method=float(m.mean()),delta=delta,ci_low=float(lo),ci_high=float(hi)))
                        by_cohort[cohort][key]=(delta,sample)
                for key in ['raw_p3_best_box_iou','raw_center_error_norm','mask_iou']:
                    failure=by_cohort['raw_geometry_small'][key];control=by_cohort['matched_small_control'][key]
                    lo,hi=np.quantile(failure[1]-control[1],[.025,.975])
                    interactions.append(dict(checkpoint=checkpoint,branch=branch,contrast=contrast,metric=key,interaction=failure[0]-control[0],ci_low=float(lo),ci_high=float(hi)))
    write(a.out/'official_contrasts.csv',ap);write(a.out/'targeted_contrasts.csv',paired);write(a.out/'failure_control_interactions.csv',interactions)
    training=next(r for r in cfg['runs'] if r['kind']=='training')
    tr=a.study/'runs'/training['run_id'];receipt=json.loads((tr/'TRAINING_COMPLETE.json').read_text())
    ref=Path(cfg['reference_remote_root'])/'runs'/cfg['reference_training_run'];r1=read(ref/'results.csv')[0];n1=read(tr/'results.csv')[0]
    epoch1diff={key:float(n1[key])-float(r1[key]) for key in r1 if key not in ['epoch','time']}
    comparability=dict(first_epoch_native_metric_and_loss_differences=epoch1diff,first_epoch_auxiliary=receipt['first_epoch_reference'],
                       config_differences=receipt['config_differences'],observed_schedule=receipt['observed_schedule'],reference_reuse=sources)
    atomic_json(a.out/'comparability.json',comparability)
    text=['# Auxiliary schedule ablation: completed results','',
          'Primary: epoch 3 last.pt; secondary: native best.pt. One seed. AP point estimates; panel CIs are descriptive, without multiplicity correction.',
          'The same 20,000 train images and completed fixed-weight/baseline reference runs are used. GT-associated panel matching is not unique COCO recall.','',
          '| Checkpoint | Branch | Metric | Baseline | Fixed auxiliary | Scheduled auxiliary | Scheduled minus fixed |',
          '|---|---|---|---:|---:|---:|---:|']
    for checkpoint in ['last','best']:
        for branch in cfg['evaluation_branches']:
            for key in ['ap','ap75','aps','ar_small','ar75']:
                b,f,n=[100*float(omap[checkpoint,branch,'segm',arm][key]) for arm in ['baseline','reg_only','reg_scheduled']]
                text.append(f'| {checkpoint} | {branch} | Mask {key} | {b:.3f} | {f:.3f} | {n:.3f} | {n-f:+.3f} |')
    text += ['', '| Checkpoint | Branch | Contrast | Failure metric | Change x100 | 95% CI x100 |','|---|---|---|---|---:|---|']
    for row in paired:
        if row['cohort']=='raw_geometry_small' and row['metric'] in ['raw_p3_best_box_iou','final_box_iou','mask_iou','target_coverage','prediction_purity','boundary_f1','final_mask75']:
            text.append(f"| {row['checkpoint']} | {row['branch']} | {row['contrast']} | {row['metric']} | {100*row['delta']:+.3f} | [{100*row['ci_low']:+.3f}, {100*row['ci_high']:+.3f}] |")
    text += ['','Exact first-epoch rounded training/validation agreement: '+str(all(abs(v)<1e-12 for v in epoch1diff.values())),
             'First epoch should receive the same auxiliary coefficient; inspect comparability.json before causal interpretation.',
             'A positive schedule contrast supports a training-objective scaling effect for this budget; it does not alone distinguish temporal alignment from reduced total auxiliary exposure.']
    (a.out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',finished_at=now(),new_training_arms=1,reused_reference_arms=2,seed=0,primary='last',secondary='best',source_runs=sources,
                                         scientific_outcome='Tables generated; review all contrasts and comparability before claiming efficacy'))
    print('RESULTS_READY',a.out/'RESULTS.md',flush=True)

if __name__=='__main__':main()
