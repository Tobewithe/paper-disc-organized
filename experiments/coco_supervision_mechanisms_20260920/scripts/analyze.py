"""Image-cluster uncertainty and matched comparisons; no confirmation-cohort access."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_estimate(rows, value, draws=2000):
    groups=defaultdict(list)
    for row in rows:
        v=value(row)
        if v is not None and np.isfinite(v):groups[row['image_id']].append(float(v))
    if not groups:return {'n':0}
    sums=np.array([sum(v) for v in groups.values()]); counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20260920)
    indices=rng.integers(0,len(groups),size=(draws,len(groups)))
    estimates=sums[indices].sum(1)/counts[indices].sum(1)
    return {'n':int(counts.sum()),'images':len(groups),'mean':float(sums.sum()/counts.sum()),
            'ci95':np.quantile(estimates,[.025,.975]).tolist()}


def matched_pairs(failures, successes, control_fill):
    pairs=[]
    for category in sorted({r['category_id'] for r in failures}):
        ff=[r for r in failures if r['category_id']==category]
        ss=[r for r in successes if r['category_id']==category]
        if not ff or not ss:continue
        area=np.abs(np.log(np.array([r['area'] for r in ff])[:,None]/np.array([r['area'] for r in ss])[None]))
        quality=np.abs(np.array([r['readout_probe']['box_iou'] for r in ff])[:,None]-np.array([r['readout_probe']['box_iou'] for r in ss])[None])
        fill=np.abs(np.array([r['training_label']['fill_fraction'] for r in ff])[:,None]-np.array([r['training_label']['fill_fraction'] for r in ss])[None])
        eligible=(area<=np.log(2))&(quality<=.1)
        if control_fill:eligible&=fill<=.1
        cost=area/np.log(2)+quality/.1+(fill/.1 if control_fill else 0)
        cost[~eligible]=1e6
        ii,jj=linear_sum_assignment(cost)
        pairs.extend((ff[i],ss[j]) for i,j in zip(ii,jj) if eligible[i,j])
    return pairs


def paired_estimate(pairs, value, draws=2000):
    # Resample source images, so a picture reused in several matched pairs remains clustered.
    if not pairs:return {'pairs':0}
    images=sorted({r['image_id'] for pair in pairs for r in pair}); index={v:k for k,v in enumerate(images)}
    fi=np.array([index[f['image_id']] for f,s in pairs]); si=np.array([index[s['image_id']] for f,s in pairs])
    vf=np.array([value(f) for f,s in pairs],dtype=float);vs=np.array([value(s) for f,s in pairs],dtype=float)
    rng=np.random.default_rng(20260920); values=[]
    for _ in range(draws):
        weights=np.bincount(rng.integers(len(images),size=len(images)),minlength=len(images))
        if weights[fi].sum() and weights[si].sum():
            values.append(float(np.average(vf,weights=weights[fi])-np.average(vs,weights=weights[si])))
    return {'pairs':len(pairs),'images':len(images),'failure_mean':float(vf.mean()),'success_mean':float(vs.mean()),
            'difference':float((vf-vs).mean()),'ci95':np.quantile(values,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(v) for v in (a.source/'instances.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if 'readout_probe' in r]
    failures=[r for r in eligible if r['geometry_state']=='box_good_mask_unavailable']
    successes=[r for r in eligible if r['geometry_state']=='joint_good']
    result={'total_gt':len(rows),'eligible_box75_witnesses':len(eligible),'failures':len(failures),'successes':len(successes)}
    result['groups']={}
    for name,group in [('failure',failures),('success',successes),('all_box75',eligible)]:
        data={}
        for metric,fn in {
            'semantic_coverage':lambda r:r['training_label']['semantic_self_coverage'],
            'nearest_instance_vanished':lambda r:r['training_label']['nearest_cells']==0,
            'fg_bg_cosine':lambda r:r['readout_probe']['fg_bg_cosine'],
            'joint_opposes_foreground':lambda r:r['readout_probe']['joint_dot_fg']<0,
            'joint_opposes_background':lambda r:r['readout_probe']['joint_dot_bg']<0,
            'negative_gradient_dot':lambda r:r['readout_probe']['fg_bg_cosine']<0,
            'outside_crop_fp_fraction':lambda r:r['readout_probe']['fp_outside_training_crop']/max(r['readout_probe']['baseline']['fp'],1),
        }.items():data[metric]=cluster_estimate(group,fn)
        if group:
            for arm in ['joint:.1','joint:.3','balanced:.1','balanced:.3']:
                # Serialized step names are '0.1'/'0.3'.
                arm=arm.replace(':.',':0.')
                data[arm]=cluster_estimate(group,lambda r:100*(r['readout_probe']['arms'][arm]['iou']-r['readout_probe']['baseline']['iou']) if arm in r['readout_probe']['arms'] else None)
            data['balanced_minus_joint_0.1']=cluster_estimate(group,lambda r:100*(r['readout_probe']['arms']['balanced:0.1']['iou']-r['readout_probe']['arms']['joint:0.1']['iou']) if all(k in r['readout_probe']['arms'] for k in ['balanced:0.1','joint:0.1']) else None)
            data['balanced_minus_joint_0.3']=cluster_estimate(group,lambda r:100*(r['readout_probe']['arms']['balanced:0.3']['iou']-r['readout_probe']['arms']['joint:0.3']['iou']) if all(k in r['readout_probe']['arms'] for k in ['balanced:0.3','joint:0.3']) else None)
            data['joint_loss_down_iou_down_0.3']=cluster_estimate(group,lambda r:float(
                r['readout_probe']['arms']['joint:0.3']['training_bce']<r['readout_probe']['training_bce'] and
                r['readout_probe']['arms']['joint:0.3']['iou']<r['readout_probe']['baseline']['iou']) if 'joint:0.3' in r['readout_probe']['arms'] else None)
        result['groups'][name]=data
    comparisons={};pair_ids={}
    for mode,control_fill in [('class_area_box',False),('class_area_box_fill',True)]:
        pairs=matched_pairs(failures,successes,control_fill)
        pair_ids[mode]=[[f['annotation_id'],s['annotation_id']] for f,s in pairs]
        comparisons[mode]={metric:paired_estimate(pairs,fn) for metric,fn in {
            'semantic_coverage':lambda r:r['training_label']['semantic_self_coverage'],
            'fg_bg_cosine':lambda r:r['readout_probe']['fg_bg_cosine'],
            'joint_opposes_foreground':lambda r:r['readout_probe']['joint_dot_fg']<0,
            'fill_fraction':lambda r:r['training_label']['fill_fraction']}.items()}
    result['matched_comparisons']=comparisons
    actual_failures=[r for r in failures if r['readout_probe']['actual_assignment_owner']['one2one']==r['annotation_id']]
    actual_successes=[r for r in successes if r['readout_probe']['actual_assignment_owner']['one2one']==r['annotation_id']]
    result['current_one2one_positive_witness_subset']={
        'failure_n':len(actual_failures),'success_n':len(actual_successes),
        'failure_joint_opposes_foreground':cluster_estimate(actual_failures,lambda r:r['readout_probe']['joint_dot_fg']<0),
        'success_joint_opposes_foreground':cluster_estimate(actual_successes,lambda r:r['readout_probe']['joint_dot_fg']<0)}
    own_pairs=matched_pairs(actual_failures,actual_successes,True)
    pair_ids['current_one2one_positive_class_area_box_fill']=[[f['annotation_id'],s['annotation_id']] for f,s in own_pairs]
    result['current_one2one_positive_witness_subset']['matched_joint_opposes_foreground']=paired_estimate(
        own_pairs,lambda r:r['readout_probe']['joint_dot_fg']<0)
    result['limits']=['Exploratory mechanism screening; original pretrained model saw train2017.',
        'Per-instance GT-assisted coefficient interventions are not a learned method or COCO AP.',
        'No shared-feature causal perturbation has been performed in this run.',
        'Image-cluster percentile bootstrap2000; matched control calipers class exact, area factor2, boxIoU0.1; fill0.1 in second set.',
        'No multiple-comparison correction; independent confirmation required before positive mechanism claim.']
    (a.out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (a.out/'matched_ids.json').write_text(json.dumps(pair_ids))
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','matched_comparisons']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
