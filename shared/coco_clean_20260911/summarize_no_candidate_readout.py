"""Summaries for S015, preserving all 317 statuses and conditional fit denominators."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from candidate_lineage_probe import read,need
from frozen_mechanism_probe import sha,write_json


def summarize(out):
    receipt=json.loads((out/'COMPLETE.json').read_text());need(receipt['status']=='COMPLETE','Incomplete')
    for name,digest in receipt['hashes'].items():need(sha(out/name)==digest,name)
    targets=read(out/'targets.csv');metrics=read(out/'metrics.csv');opt=read(out/'optimization.csv')
    ids=json.loads((out/'protocol.json').read_text())['images'];ix={iid:j for j,iid in enumerate(ids)}
    rng=np.random.default_rng(20260912);w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    base={int(r['annotation_id']):r for r in metrics if r['arm']=='original'}
    lookup={(r['arm'],int(r['annotation_id'])):r for r in metrics}
    ok={int(r['annotation_id']) for r in targets if r['fit_status']=='ok'}
    stage=Counter(r['stage'] for r in targets);status=Counter(r['fit_status'] for r in targets)
    groups=[];paired=[]
    for group in ['all_eligible','original_final_match','unretained_raw_anchor','crop_support_ge75','crop_support_lt75','small','medium_large']:
        rr=[r for r in targets if int(r['annotation_id']) in ok]
        if group=='original_final_match':rr=[r for r in rr if r['anchor']=='fixed_original_bbox50_match']
        if group=='unretained_raw_anchor':rr=[r for r in rr if r['anchor']=='GT_best_geometry_postconf_candidate']
        if group=='crop_support_ge75':rr=[r for r in rr if float(r['crop_coverage'])>=.75]
        if group=='crop_support_lt75':rr=[r for r in rr if float(r['crop_coverage'])<.75]
        if group=='small':rr=[r for r in rr if float(r['area'])<1024]
        if group=='medium_large':rr=[r for r in rr if float(r['area'])>=1024]
        aids=[int(r['annotation_id']) for r in rr]
        if not aids:continue
        for arm in ['original','threshold_oracle','free_coefficient','free_coefficient_bias']:
            rows=[lookup[(arm,aid)] for aid in aids]
            groups.append(dict(group=group,arm=arm,targets=len(rows),
                recovered75=sum(float(r['mask_iou'])>=.75 for r in rows),
                recovered90=sum(float(r['mask_iou'])>=.9 for r in rows),
                **{m:100*float(np.mean([float(r[m]) for r in rows])) for m in ['mask_iou','input_iou','coverage','same_neighbor','background','other_category']}))
        for arm,ref in [('threshold_oracle','original'),('free_coefficient','original'),('free_coefficient','threshold_oracle'),('free_coefficient_bias','free_coefficient')]:
            for field in ['mask_iou','coverage','same_neighbor','background']:
                num=np.zeros(len(ids));den=np.zeros(len(ids))
                for aid in aids:
                    r=lookup[(arm,aid)];b=lookup[(ref,aid)];j=ix[int(r['image_id'])]
                    num[j]+=float(r[field])-float(b[field]);den[j]+=1
                bd=w@den;values=(w@num)[bd>0]/bd[bd>0]
                paired.append(dict(group=group,arm=arm,reference=ref,metric=field,n=len(aids),
                    difference_pp=100*float(num.sum()/den.sum()),ci95_pp=(100*np.quantile(values,[.025,.975])).tolist(),valid_resamples=len(values)))
    partitions=[]
    for aid in sorted(ok):
        b=lookup[('original',aid)];t=lookup[('threshold_oracle',aid)];c=lookup[('free_coefficient',aid)];cb=lookup[('free_coefficient_bias',aid)]
        flags=[float(r['mask_iou'])>=.75 for r in [t,c,cb]]
        partitions.append(dict(annotation_id=aid,threshold_recovered=flags[0],coefficient_recovered=flags[1],bias_recovered=flags[2]))
    optimization=[]
    for arm in ['free_coefficient','free_coefficient_bias']:
        for phase in ['bce','bce_dice']:
            rr=[r for r in opt if r['arm']==arm and r['phase']==phase]
            optimization.append(dict(arm=arm,phase=phase,n=len(rr),
                median_iterations=float(np.median([int(r['iterations']) for r in rr])) if rr else None,
                hit_iter_limit=sum(int(r['iterations'])>=json.loads((out/'protocol.json').read_text())['optimizer']['max_iter_per_stage'] for r in rr),
                gradient_max_median=float(np.median([float(r['gradient_max']) for r in rr])) if rr else None,
                gradient_max_p95=float(np.quantile([float(r['gradient_max']) for r in rr],.95)) if rr else None))
    validanchors=[r for r in targets if r.get('crop_coverage','')!='']
    results=dict(cohort_targets=len(targets),images=len(ids),stages=dict(stage),fit_status=dict(status),
        valid_anchor_targets=len(validanchors),crop_support_below75=sum(float(r['crop_coverage'])<.75 for r in validanchors),
        no_good_postconf_candidate_threshold=.75,groups=groups,paired=paired,optimization=optimization,
        recovery_partition=dict(eligible=len(ok),threshold=sum(r['threshold_recovered'] for r in partitions),
            coefficient=sum(r['coefficient_recovered'] for r in partitions),bias=sum(r['bias_recovered'] for r in partitions),
            coeff_success_threshold_fail=sum(r['coefficient_recovered'] and not r['threshold_recovered'] for r in partitions),
            threshold_success_coeff_fail=sum(r['threshold_recovered'] and not r['coefficient_recovered'] for r in partitions),
            bias_success_coeff_fail=sum(r['bias_recovered'] and not r['coefficient_recovered'] for r in partitions),
            all_three_fail=sum(not any([r['threshold_recovered'],r['coefficient_recovered'],r['bias_recovered']]) for r in partitions)),
        interpretation='Per-instance achieved same-image oracle pixel quality, not official AP/R75 or certified expressivity limit. Low crowd-domain baseline can differ from cohort COCO IoU; all conditional denominators shown.',
        statistical_scope='2000 image-cluster paired exploratory resamples; no multiplicity or training-seed inference.',script_sha256=sha(__file__))
    write_json(out/'ANALYSIS.json',results)
    write_json(out/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',analysis_sha256=sha(out/'ANALYSIS.json'),script_sha256=sha(__file__)))
    print(json.dumps({k:v for k,v in results.items() if k not in ['paired','groups']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.out)
