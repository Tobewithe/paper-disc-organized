"""Summarize matched branch changes and gate ownership without new inference."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def cluster_ci(values,groups):
    unique=sorted(set(groups));idx={g:i for i,g in enumerate(unique)}
    sums=np.zeros(len(unique));counts=np.zeros(len(unique))
    for v,g in zip(values,groups):sums[idx[g]]+=v;counts[idx[g]]+=1
    rng=np.random.default_rng(20260915);draw=rng.integers(0,len(unique),(4000,len(unique)))
    means=sums[draw].sum(1)/counts[draw].sum(1)
    return float(np.mean(values)),*map(float,np.quantile(means,[.025,.975]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=read(args.study/'runs/branch_full/targeted_per_instance.csv')
    by={(r['branch'],r['arm'],r['annotation_id']):r for r in records}
    assert len(by)==len(records)==438*4
    keys=['raw_p3_best_box_iou','raw_center_error_norm','final_box_iou','mask_iou','target_coverage','prediction_purity','boundary_f1','final_mask50','final_mask75']
    branch_rows=[]
    for cohort in sorted({r['cohort'] for r in records}):
        ids=sorted({r['annotation_id'] for r in records if r['cohort']==cohort});groups=[by[('one2many','baseline',i)]['image_id'] for i in ids]
        for key in keys:
            changes={}
            for branch in ['one2many','one2one']:
                b=np.array([float(by[(branch,'baseline',i)][key]) for i in ids]);m=np.array([float(by[(branch,'method',i)][key]) for i in ids]);changes[branch]=m-b
                delta,lo,hi=cluster_ci(m-b,groups);branch_rows.append(dict(cohort=cohort,metric=key,comparison=branch,n=len(ids),images=len(set(groups)),baseline=float(b.mean()),method=float(m.mean()),delta=delta,ci_low=lo,ci_high=hi))
            delta,lo,hi=cluster_ci(changes['one2one']-changes['one2many'],groups)
            branch_rows.append(dict(cohort=cohort,metric=key,comparison='one2one_change_minus_one2many_change',n=len(ids),images=len(set(groups)),baseline=None,method=None,delta=delta,ci_low=lo,ci_high=hi))
    write(args.out/'branch_cluster_bootstrap.csv',branch_rows)
    gate=read(args.study/'runs/gate_panel/gate_per_target.csv');images=read(args.study/'runs/gate_panel/image_draws.csv')
    assert len(images)==384 and len(gate)==337*3
    result=[];filters=[];gradients=[]
    for arm in ['initial','baseline','method']:
        for stratum in ['all','random_nonpilot','pilot_anchor']:
            chosen=[r for r in gate if r['arm']==arm and (stratum=='all' or r['stratum']==stratum)];selected=[r for r in chosen if r['gate']=='1'];n=len(selected)
            panel=[r for r in images if stratum=='all' or r['stratum']==stratum]
            s=dict(arm=arm,stratum=stratum,anchor_images=len(panel),selected=len(chosen),gated=n)
            conditions={
                'official_self':lambda r:r['owner']=='self',
                'official_background':lambda r:r['owner']=='background',
                'official_other':lambda r:r['owner'].startswith('other_'),
                'has_box50':lambda r:float(r['student_best_all_iou'])>=.5,
                'has_sameclass_box50':lambda r:float(r['student_best_same_class_iou'])>=.5,
                'edited_teacher_crosses50':lambda r:r['edit_needed_to_cross50']=='1',
                'positive_edit_delta_ge10':lambda r:float(r['teacher_edit_delta'])>=.1,
                'teacher_already_p3_box50':lambda r:float(r['teacher_original_best_p3_iou'])>=.5,
                'classification_direction_conflict':lambda r:r['class_grad_conflict']=='1',
                'no_neighbor':lambda r:int(r['neighbor_pixels'])==0,
                'mixup_present':lambda r:int(r['mixup_calls'])>0,
            }
            for key,condition in conditions.items():s[key]=sum(condition(r) for r in selected)
            result.append(s)
            for owner in ['all','self','background','other_same_class','other_different_class']:
                subset=[r for r in selected if owner=='all' or r['owner']==owner]
                cos=[float(r['box_grad_cosine']) for r in subset if r['box_grad_cosine']]
                gradients.append(dict(arm=arm,stratum=stratum,owner=owner,n=len(subset),class_conflict=sum(r['class_grad_conflict']=='1' for r in subset),box_cos_n=len(cos),box_cos_mean=float(np.mean(cos)) if cos else None,box_cos_negative=sum(x<0 for x in cos)))
            policies={
                'original_gate':lambda r:True,
                'self_assignment':lambda r:r['owner']=='self',
                'self_and_edit_response':lambda r:r['owner']=='self' and float(r['teacher_edit_delta'])>=.1,
                'all_candidate_geometric_failure':lambda r:float(r['student_best_all_iou'])<.5,
                'self_and_all_candidate_failure':lambda r:r['owner']=='self' and float(r['student_best_all_iou'])<.5,
                'self_edit_and_all_candidate_failure':lambda r:r['owner']=='self' and float(r['teacher_edit_delta'])>=.1 and float(r['student_best_all_iou'])<.5,
            }
            for policy,predicate in policies.items():filters.append(dict(arm=arm,stratum=stratum,policy=policy,retained=sum(predicate(r) for r in selected),original=n))
    write(args.out/'gate_accounting.csv',result);write(args.out/'gate_filter_counts.csv',filters);write(args.out/'output_gradient_directions.csv',gradients)
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',branch_instances=438,branch_images=len({r['image_id'] for r in records}),gate_anchors=384,gate_selected=337,branch_bootstrap='4000 paired image-cluster resamples, seed 20260915, conditional on the frozen panel',gate_filter_scope='counting hypothetical filters only, not training efficacy',official_AP_scope='single-run point estimates, no significance assertion'),indent=2))

if __name__=='__main__':main()
