"""Preregistered finite-population summaries; no torch or data mutation."""
from collections import defaultdict
import math
import numpy as np

def area_group(area):
    return 'S' if area<1024 else 'M' if area<9216 else 'L'

def iou_group(value):
    return 'lt50' if value<.5 else '50to75' if value<.75 else 'ge75'

def image_values(rows,field):
    groups=defaultdict(list)
    for r in rows:
        groups[r['image_id']].append(r[field]-r['baseline_iou'])
    return [math.fsum(groups[i])/len(groups[i]) for i in sorted(groups)]

def effect(rows,field,ci=True):
    values=image_values(rows,field)
    if not values:
        return dict(pairs=0,images=0,mean=None,ci95=None,repaired=0,repair_fraction=None)
    a=np.array(values,dtype=np.float64)
    interval=None
    if ci and len(a)>1:
        rng=np.random.default_rng(1729)
        draws=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(axis=1)
        interval=[float(v) for v in np.quantile(draws,[.025,.975],method='linear')]
    repaired=sum(r[field]>=.75 for r in rows if r['population']=='T')
    targets=sum(r['population']=='T' for r in rows)
    return dict(pairs=len(rows),images=len(values),mean=float(a.mean()),ci95=interval,repaired=repaired,repair_denominator=targets,repair_fraction=repaired/targets if targets else None)

def damage(tasks,arm):
    success=[r for r in tasks if r['baseline_success']]
    count=sum(not r[arm+'_success'] for r in success)
    failed=[r for r in tasks if not r['baseline_success']]
    recovered=sum(r[arm+'_success'] for r in failed)
    return dict(population_N=len(success),damaged_k=count,fraction=count/len(success) if success else None,recovered_k=recovered,original_failed_N=len(failed),scope='complete split finite-population census; not original S2 sample upper bound')

def summaries(rows,tasks):
    result={}
    for arm in ('U','G'):
        field=arm+'_iou'
        target=[r for r in rows if r['population']=='T']
        on=[r for r in target if r['gate_enabled']]
        main=dict(target=effect(target,field),all_bbox75=effect(rows,field),gate_on_target=effect(on,field),damage=damage(tasks,arm))
        groups={}
        for name in ('native_area_group','category_id','native_iou_group'):
            groups[name]={}
            for value in sorted({r[name] for r in rows},key=str):
                subset=[r for r in rows if r[name]==value]
                groups[name][str(value)]=dict(target=effect([r for r in subset if r['population']=='T'],field,False),all_bbox75=effect(subset,field,False))
        main['strata']=groups
        main['damage_strata']={name:{str(v):damage([r for r in tasks if r[name]==v],arm) for v in sorted({r[name] for r in tasks if r['baseline_success']},key=str)} for name in ('success_native_area_group','category_id')}
        t,o,d=main['target'],main['all_bbox75'],main['damage']
        main['fit_thresholds']=dict(target_effect=t['mean'] is not None and t['mean']>=.005 and t['ci95'] is not None and t['ci95'][0]>0,all_bbox75_guardrail=o['mean'] is not None and o['mean']>=-.001,success_damage_census=d['fraction'] is not None and d['fraction']<=.01)
        result[arm]=main
    return result

def standardized_gap(train,dev):
    train=[r for r in train if r['population']=='T' and r['gate_enabled']]
    dev=[r for r in dev if r['population']=='T' and r['gate_enabled']]
    key=lambda r:(r['category_id'],r['native_iou_group'])
    by=[]
    for population in (train,dev):
        groups=defaultdict(list)
        for r in population:
            groups[key(r)].append(r)
        by.append(groups)
    shared=sorted(k for k in set(by[0])&set(by[1]) if all(len({r['image_id'] for r in g[k]})>=10 for g in by))
    kept=[r for r in dev if key(r) in shared]
    coverage=len(kept)/len(dev) if dev else 0.
    receipt=dict(dev_gate_target_pairs=len(dev),shared_dev_pairs=len(kept),shared_dev_images=len({r['image_id'] for r in kept}),shared_coverage=coverage,shared_strata=[list(k) for k in shared],excluded_dev_pairs=len(dev)-len(kept),min_images_per_split_per_stratum=10,weighting='fixed dev target frequency across shared class x native IoU strata; pair-weighted, not S2 image macro',resamples=2000,seed=1730,valid_resamples=0,mean=None,ci95=None,sufficient_common_support=False)
    if not shared:
        return receipt
    weights=np.array([len(by[1][k])/len(kept) for k in shared],dtype=np.float64)
    matrices=[]
    for population in (train,dev):
        ids=sorted({r['image_id'] for r in population if key(r) in shared})
        idmap={v:i for i,v in enumerate(ids)}
        strata={v:i for i,v in enumerate(shared)}
        sums=np.zeros((len(ids),len(shared)),dtype=np.float64)
        counts=np.zeros_like(sums)
        for r in population:
            if key(r) in strata:
                i,j=idmap[r['image_id']],strata[key(r)]
                sums[i,j]+=r['G_iou']-r['baseline_iou']
                counts[i,j]+=1
        matrices.append((sums,counts))
    means=[float(weights@(s.sum(axis=0)/c.sum(axis=0))) for s,c in matrices]
    receipt.update(train_standardized_mean=means[0],dev_standardized_mean=means[1],mean=means[0]-means[1])
    rng=np.random.default_rng(1730)
    differences=[]
    for _ in range(2000):
        vals=[]
        for sums,counts in matrices:
            multiplicity=np.bincount(rng.integers(0,len(sums),size=len(sums)),minlength=len(sums))
            denominator=multiplicity@counts
            vals.append(float(weights@((multiplicity@sums)/denominator)) if np.all(denominator>0) else None)
        if all(v is not None for v in vals):
            differences.append(vals[0]-vals[1])
    receipt['valid_resamples']=len(differences)
    if differences:
        receipt['ci95']=[float(v) for v in np.quantile(differences,[.025,.975],method='linear')]
    receipt['sufficient_common_support']=coverage>=.5 and receipt['shared_dev_images']>=30 and len(differences)>=1900
    return receipt

def adjudicate(train,gap):
    passed=all(train['G']['fit_thresholds'].values())
    strong_gap=gap['sufficient_common_support'] and gap['mean']>=.005 and gap['ci95'][0]>0
    if not passed:
        finding='repair_requirements_already_unmet_on_fit'
        direction='attainable_conditional_repair_with_success_protection'
        action='Freeze a train-only attainable-mask learning comparison; do not expand data or shrink architecture based on an unproved overfitting claim.'
    elif strong_gap:
        finding='fit_supported_and_substantial_standardized_generalization_gap'
        direction='controlled_data_quantity_comparison'
        action='Freeze a data-quantity-only comparison with distribution, objective, architecture and evaluation held fixed and training exposure explicitly controlled.'
    elif not gap['sufficient_common_support']:
        finding='fit_supported_but_comparable_generalization_evidence_insufficient'
        direction='attainable_conditional_repair_strength'
        action='Freeze an attainable conditional-repair comparison using the established finite-oracle opportunity; do not claim small-data causality or relax this audit support criteria.'
    else:
        finding='fit_supported_without_substantial_generalization_gap_evidence'
        direction='attainable_conditional_repair_strength'
        action='Freeze an attainable conditional-repair comparison; the present evidence does not prioritize data expansion or parameter reduction.'
    return dict(finding=finding,priority_direction=direction,next_action=action,gated_fit_requirements_met=passed,substantial_standardized_gap=bool(strong_gap),new_training_authorized_by_this_decision=False,prior_S2_S3_decisions_unchanged=True,small_sample_causal_claim='not_identified_by_observational_train_dev_comparison')

def build_summary(train,dev,train_tasks,dev_tasks,coverage,losses,scope):
    fit_summary=summaries(train,train_tasks)
    dev_summary=summaries(dev,dev_tasks)
    gap=standardized_gap(train,dev) if scope=='full' else None
    decision=adjudicate(fit_summary,gap) if scope=='full' else dict(finding='engineering_only',priority_direction=None,next_action='Run authorized full frozen audit after smoke verification')
    loss_summary={arm:{k:math.fsum(r[arm][k] for r in losses)/len(losses) for k in ('total_loss','gt_loss','preservation_loss')} for arm in ('zero','U')}
    loss_summary['delta']={k:loss_summary['U'][k]-loss_summary['zero'][k] for k in loss_summary['zero']}
    counters=('outputs','gate_outputs','joined','gate_joined','valid_supervised','gate_valid_supervised','preserved','gate_preserved')
    coverage_summary={k:sum(r[k] for r in coverage) for k in counters}
    coverage_summary.update(images=len(coverage),images_with_gate_supervision=sum(r['gate_valid_supervised']>0 for r in coverage),images_without_supervision=sum(r['valid_supervised']==0 for r in coverage),gate_supervised_categories=sorted({v for r in coverage for v in r['gate_supervised_categories']}))
    for numerator,denominator in (('gate_outputs','outputs'),('gate_joined','joined'),('gate_valid_supervised','valid_supervised')):
        coverage_summary[numerator+'_fraction']=coverage_summary[numerator]/coverage_summary[denominator] if coverage_summary[denominator] else None
    return dict(scope=scope,train=fit_summary,dev=dev_summary,standardized_gate_target_gap=gap,training_coverage=coverage_summary,fixed_checkpoint_loss=loss_summary,decision=decision,new_training_steps=0,base_model_forward_calls=0,oracle_optimization_calls=0)
