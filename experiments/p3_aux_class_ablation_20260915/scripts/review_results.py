"""Read completed paired experiments; verify scope and add interaction estimates."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import yaml

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean_draws(rows,metric):
    groups=sorted({r['image_id'] for r in rows});index={g:i for i,g in enumerate(groups)}
    sums=np.zeros(len(groups));counts=np.zeros(len(groups))
    for row in rows:sums[index[row['image_id']]]+=row[metric];counts[index[row['image_id']]]+=1
    draws=np.random.default_rng(20260915).integers(0,len(groups),(4000,len(groups)))
    return float(sums.sum()/counts.sum()),sums[draws].sum(1)/counts[draws].sum(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);cfg=json.loads((a.study/'protocol.json').read_text())
    training={r['arm']:a.study/'runs'/r['run_id'] for r in cfg['runs'] if r['kind']=='training'}
    arguments={arm:yaml.safe_load((path/'args.yaml').read_text()) for arm,path in training.items()}
    check={'training':{},'native_best_epoch':{},'configs_differ_only_in_output_names':True}
    for arm,path in training.items():
        receipt=json.loads((path/'TRAINING_COMPLETE.json').read_text());assert receipt['epochs']==3 and receipt['finite']
        check['training'][arm]=receipt
        records=read(path/'results.csv')
        # Every reported native box/mask AP peaks at epoch 1; no mixed-epoch table.
        check['native_best_epoch'][arm]={k:int(max(records,key=lambda r:float(r[k]))['epoch']) for k in ['metrics/mAP50-95(M)','metrics/mAP50-95(B)']}
        diff={k for k in arguments[arm] if arguments[arm][k]!=arguments['baseline'][k]}
        assert diff <= {'name','save_dir'},(arm,diff)
    for name in ['method_core.py','train_ablation.py']:
        assert (training['original']/'snapshots'/name).read_bytes()==(training['reg_only']/'snapshots'/name).read_bytes()
    metrics=[];targets=[]
    for run in cfg['runs']:
        if run['kind']!='evaluation':continue
        folder=a.study/'runs'/run['run_id'];complete=json.loads((folder/'COMPLETE.json').read_text())
        assert complete['full_images']==5000 and complete['panel_targets_per_branch']==438
        m=read(folder/'official_metrics.csv');t=read(folder/'targeted_per_instance.csv')
        assert len(m)==4 and len(t)==876
        metrics.extend(m);targets.extend(t)
    assert len(metrics)==24 and len(targets)==5256
    lookup={(r['checkpoint'],r['branch'],r['arm'],r['annotation_id']):r for r in targets}
    assert len(lookup)==len(targets)
    check['cohorts']={cohort:{'instances':len({r['annotation_id'] for r in targets if r['cohort']==cohort}),
                                  'images':len({r['image_id'] for r in targets if r['cohort']==cohort})} for cohort in {r['cohort'] for r in targets}}
    assert check['cohorts']['raw_geometry_small']['instances']==239
    assert check['cohorts']['matched_small_control']['instances']==199
    pairs=[('reg_only','original'),('original','baseline'),('reg_only','baseline')]
    interactions=[]
    for checkpoint in ['last','best']:
        for branch in ['one2many','one2one']:
            for method,reference in pairs:
                for metric in ['raw_p3_best_box_iou','raw_center_error_norm','mask_iou']:
                    estimates=[]
                    for cohort in ['raw_geometry_small','matched_small_control']:
                        items=[r for r in targets if r['checkpoint']==checkpoint and r['branch']==branch and r['arm']==reference and r['cohort']==cohort]
                        data=[{'image_id':r['image_id'],'delta':float(lookup[(checkpoint,branch,method,r['annotation_id'])][metric])-float(r[metric])} for r in items]
                        # Distinct cohort images; independent resampling streams by cohort.
                        groups=sorted({r['image_id'] for r in data});index={g:i for i,g in enumerate(groups)}
                        sums=np.zeros(len(groups));counts=np.zeros(len(groups))
                        for row in data:sums[index[row['image_id']]]+=row['delta'];counts[index[row['image_id']]]+=1
                        seed=20260915 if cohort=='raw_geometry_small' else 20260916
                        draws=np.random.default_rng(seed).integers(0,len(groups),(4000,len(groups)))
                        estimates.append((float(sums.sum()/counts.sum()),sums[draws].sum(1)/counts[draws].sum(1)))
                    values=estimates[0][1]-estimates[1][1];lo,hi=np.quantile(values,[.025,.975])
                    interactions.append(dict(checkpoint=checkpoint,branch=branch,contrast=method+' minus '+reference,metric=metric,
                                             failure_change=estimates[0][0],control_change=estimates[1][0],
                                             interaction=estimates[0][0]-estimates[1][0],ci_low=float(lo),ci_high=float(hi)))
    failure_images={r['image_id'] for r in targets if r['cohort']=='raw_geometry_small'}
    control_images={r['image_id'] for r in targets if r['cohort']=='matched_small_control'}
    assert failure_images.isdisjoint(control_images)
    write(a.out/'failure_control_interactions.csv',interactions)
    # Counts are diagnostic GT association, not COCO unique matching recall.
    counts=[]
    for checkpoint in ['last','best']:
        for branch in ['one2many','one2one']:
            for arm in training:
                rows=[r for r in targets if r['checkpoint']==checkpoint and r['branch']==branch and r['arm']==arm and r['cohort']=='raw_geometry_small']
                counts.append(dict(checkpoint=checkpoint,branch=branch,arm=arm,n=len(rows),
                                   **{metric:sum(int(r[metric]) for r in rows) for metric in ['raw_same_class_box50','final_box50','final_mask50','final_mask75']}))
    write(a.out/'target_success_counts.csv',counts)
    check['interpretation_limits']=['single seed','descriptive bootstrap without multiplicity correction','best supplementary, last primary','native AP uses converted GT; official tables use original COCO GT','prediction counts at conf .001 are not false-positive counts']
    (a.out/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',training=False,evaluation_rerun=False,official_rows=24,target_rows=5256,bootstrap=4000),indent=2))
    for row in interactions:
        if row['checkpoint']=='last' and row['contrast']=='reg_only minus baseline':print(json.dumps(row))

if __name__=='__main__':main()
