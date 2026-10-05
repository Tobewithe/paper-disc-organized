"""Fit prediction-only gates on train2017 and score frozen gates from RLE banks."""
from __future__ import annotations
import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)


def features(rows,mode):
    a=np.array([float(r['baseline_area']) for r in rows])
    columns=[np.log1p(a)]
    if mode in ('shape','response'):
        columns.extend([np.log1p([float(r['mask_elongation']) for r in rows]),
            np.array([float(r['mask_extent']) for r in rows]),
            np.log1p([float(r['grid_compactness']) for r in rows])])
    if mode=='response':columns.append(np.array([float(r['removed_fraction']) for r in rows]))
    return np.column_stack(columns).astype(np.float64)


def evaluate(coco,records,ids,out,name):
    # loadRes mutates dictionaries: pass fresh canonical dictionaries each time.
    items=[{k:r[k] for k in ('image_id','category_id','score','segmentation')} for r in records if r['image_id'] in ids]
    with contextlib.redirect_stdout(io.StringIO()) as logs:
        result=coco.loadRes(items)
        ev=COCOeval(coco,result,'segm');ev.params.imgIds=sorted(ids)
        ev.evaluate();ev.accumulate();ev.summarize()
    values=[float(v) for v in ev.stats]
    (out/f'cocoeval_{name}.log').write_text(logs.getvalue(),encoding='utf-8')
    del result,ev,items;gc.collect()
    return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['fit','evaluate'],required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--split')
    p.add_argument('--model-run')
    p.add_argument('--global-bank',help='Optional source of a previously exported global-threshold comparator')
    args=p.parse_args()
    source,out=Path(args.input),Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():raise RuntimeError('Use a new Run')
    started=time.perf_counter()
    coco=COCO(args.annotations)
    candidates=list(csv.DictReader((source/'candidate_records.csv').open(newline='',encoding='utf-8')))
    keys=[(int(r['image_id']),int(r['candidate_index'])) for r in candidates]
    lookup={key:i for i,key in enumerate(keys)}
    assert len(lookup)==len(keys)
    modes=['area','shape','response']
    fit_info={}
    if args.phase=='fit':
        split=json.loads(Path(args.split).read_text())
        fit_ids=set(split['fit_ids']);eval_ids=set(split['selection_ids'])
        assert not fit_ids&eval_ids
        target=[];indices=[];target_images=[]
        for r in csv.DictReader((source/'instance_records.csv').open(newline='',encoding='utf-8')):
            if r['variant']!='smooth_gated':continue
            key=(int(r['image_id']),int(r['candidate_index']))
            if key[0] not in fit_ids:continue
            i=lookup[key]
            # Preserve baseline for newly empty outputs, matching boundary-only deployment.
            delta=float(r['iou'])-float(r['baseline_iou']) if float(candidates[i]['smooth_area'])>0 else 0.
            indices.append(i);target.append(delta);target_images.append(key[0])
        target=np.array(target);indices=np.array(indices)
        model_params=dict(loss='squared_error',learning_rate=.05,max_iter=100,max_leaf_nodes=7,
            min_samples_leaf=80,l2_regularization=1.,early_stopping=False,random_state=20260915)
        for mode in modes:
            x=features(candidates,mode)[indices]
            model=HistGradientBoostingRegressor(**model_params).fit(x,target)
            joblib.dump(model,out/f'{mode}.joblib')
            fit_info[mode]=dict(training_rows=len(target),training_images=len(set(target_images)),
                fit_mse=float(mean_squared_error(target,model.predict(x))),features=x.shape[1])
        models={mode:joblib.load(out/f'{mode}.joblib') for mode in modes}
        atomic(out/'frozen_models.json',dict(parameters=model_params,fit_ids=sorted(fit_ids),selection_ids=sorted(eval_ids),
            target='continuous IoU delta of fixed smooth boundary-only calibration for all baseline matched ordinary GT',
            gate='apply smooth only if predicted IoU delta > 0; restore baseline when smooth mask is empty',
            inputs={'area':['log1p(baseline predicted area)'],
                'shape':['log1p(area)','log1p(mask bbox elongation)','mask bbox extent','log1p(input grid compactness)'],
                'response':['all shape inputs','fraction of predicted foreground removed by smooth calibration']},
            model_files={mode:dict(path=f'{mode}.joblib',sha256=hashlib.sha256((out/f'{mode}.joblib').read_bytes()).hexdigest()) for mode in modes}))
    else:
        model_root=Path(args.model_run)
        frozen=json.loads((model_root/'frozen_models.json').read_text())
        eval_ids=set(json.loads((source/'image_ids.json').read_text()))
        assert not eval_ids&(set(frozen['fit_ids'])|set(frozen['selection_ids']))
        models={mode:joblib.load(model_root/f'{mode}.joblib') for mode in modes}
    baseline=json.loads((source/'predictions_official_zero.json').read_text())
    smooth=json.loads((source/'predictions_smooth_gated.json').read_text())
    smooth_map={(r['image_id'],r['candidate_index']):r for r in smooth}
    basemap={(r['image_id'],r['candidate_index']):r for r in baseline}
    assert len(basemap)==len(baseline)
    assert smooth_map.keys() <= basemap.keys()
    assert all((r['category_id'],r['score']) == (basemap[k]['category_id'],basemap[k]['score']) for k,r in smooth_map.items())
    base_subset=[r for r in baseline if r['image_id'] in eval_ids]
    metrics={};selection={};predictions={}
    variants={'official_zero':base_subset,'smooth_gated':[r for r in smooth if r['image_id'] in eval_ids],
        'boundary_only':[smooth_map.get((r['image_id'],r['candidate_index']),r) for r in base_subset]}
    for mode in modes:
        predicted=models[mode].predict(features(candidates,mode))
        decisions={key:float(predicted[i]) for i,key in enumerate(keys)}
        predictions[mode]=decisions
        output=[];applied=0
        for r in base_subset:
            key=(r['image_id'],r['candidate_index'])
            use=decisions[key]>0 and key in smooth_map
            output.append(smooth_map[key] if use else r);applied+=int(use)
        variants['risk_'+mode]=output
        selection[mode]=dict(calibrated=applied,kept_baseline=len(output)-applied,total=len(output))
    for variant,records in variants.items():
        atomic(out/'progress.json',dict(stage='cocoeval',variant=variant,images=len(eval_ids)))
        metrics[variant]=evaluate(coco,records,eval_ids,out,variant)
        # Keep only genuine COCO predictions plus explicit candidate identity; atomic export.
        atomic(out/f'predictions_{variant}.json',records)
        atomic(out/'metrics_partial.json',metrics)
        print(json.dumps(dict(variant=variant,ap=metrics[variant][0],images=len(eval_ids))),flush=True)
    selected_global=None
    if args.phase=='fit':
        for variant in ('global_0.25','global_0.5','official_global','global_1.0'):
            records=json.loads((source/f'predictions_{variant}.json').read_text())
            metrics[variant]=evaluate(coco,records,eval_ids,out,variant)
            del records;gc.collect()
            atomic(out/'metrics_partial.json',metrics)
        order=['official_zero','global_0.25','global_0.5','official_global','global_1.0']
        selected_global=max(order,key=lambda k:metrics[k][0])
        atomic(out/'selection.json',dict(selected_global=selected_global,
            criterion='highest COCO Mask AP on 500 train2017 selection images; ties choose smaller threshold',
            risk_gates='fixed decision >0; no gate-threshold selection or validation-set fitting',
            selected_gate=max(['risk_area','risk_shape','risk_response'],key=lambda k:metrics[k][0])))
    elif args.model_run:
        selected_global=json.loads((Path(args.model_run)/'selection.json').read_text())['selected_global']
        comparator=source/f'predictions_{selected_global}.json'
        if not comparator.exists() and args.global_bank:
            comparator=Path(args.global_bank)/f'predictions_{selected_global}.json'
        if comparator.exists():
            records=json.loads(comparator.read_text());metrics['selected_global']=evaluate(coco,records,eval_ids,out,'selected_global');del records
    totals=sum(not a.get('iscrowd',0) for iid in eval_ids for a in coco.imgToAnns[iid])
    sorted_ids=sorted(eval_ids);image_position={iid:i for i,iid in enumerate(sorted_ids)}
    per_image_total=np.array([sum(not a.get('iscrowd',0) for a in coco.imgToAnns[iid]) for iid in sorted_ids])
    gains={mode:np.zeros(len(sorted_ids),dtype=np.int64) for mode in [*modes,'fixed_smooth']}
    decision_rows=[]
    outcomes={key:dict(matched=0,base_success=0,after_success=0,repaired=0,damaged=0,mean_iou_delta=0.) for key in modes}
    with (source/'instance_records.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['variant']!='smooth_gated' or int(r['image_id']) not in eval_ids:continue
            key=(int(r['image_id']),int(r['candidate_index']))
            b=float(r['baseline_iou'])
            image_i=image_position[key[0]]
            fixed=float(r['iou']) if key in smooth_map else b
            gains['fixed_smooth'][image_i]+=int(fixed>=.75)-int(b>=.75)
            for mode in modes:
                a=float(r['iou']) if predictions[mode][key]>0 and key in smooth_map else b
                s=outcomes[mode];s['matched']+=1;s['base_success']+=int(b>=.75);s['after_success']+=int(a>=.75)
                s['repaired']+=int(b<.75<=a);s['damaged']+=int(a<.75<=b);s['mean_iou_delta']+=a-b
                gains[mode][image_i]+=int(a>=.75)-int(b>=.75)
                decision_rows.append(dict(image_id=key[0],candidate_index=key[1],annotation_id=int(r['annotation_id']),
                    mode=mode,predicted_gain=predictions[mode][key],baseline_iou=b,fixed_smooth_iou=fixed,
                    gated_iou=a,calibrated=int(predictions[mode][key]>0 and key in smooth_map)))
    for s in outcomes.values():
        s['net']=s['repaired']-s['damaged'];s['r75_delta_pp']=100*s['net']/totals
        s['mean_iou_delta']/=max(s['matched'],1)
    with (out/'decisions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(decision_rows[0]));w.writeheader();w.writerows(decision_rows)
    rng=np.random.default_rng(20260915)
    samples={key:[] for key in gains}
    for _ in range(2000):
        draw=rng.integers(0,len(sorted_ids),size=len(sorted_ids))
        denominator=per_image_total[draw].sum()
        for key,gain in gains.items():samples[key].append(100*gain[draw].sum()/denominator)
    comparisons={}
    for mode in modes:
        for control in ('area','fixed_smooth'):
            if mode==control:continue
            sample=np.array(samples[mode])-samples[control]
            comparisons[mode+'_vs_'+control]=dict(delta_r75_pp=100*float((gains[mode]-gains[control]).sum())/totals,
                paired_image_bootstrap_95ci_pp=[float(v) for v in np.quantile(sample,[.025,.975])])
    result=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),study_id='STUDY_8fb3468ebb704682a2225ebed0e16206',
        phase=args.phase,source_run=source.name,model_run=args.model_run,images=len(eval_ids),ordinary_gt=totals,
        metrics=metrics,delta_ap_points={k:100*(v[0]-metrics['official_zero'][0]) for k,v in metrics.items()},
        fit=fit_info,selection=selection,fixed_slot_outcomes=outcomes,selected_global=selected_global,
        selected_global_evaluated=(args.phase=='fit' or 'selected_global' in metrics),
        paired_r75_comparisons=comparisons,
        limitations=['train2017 previously seen by COCO-pretrained backbone','existing val2017 used in earlier exploration',
            'risk model fitted only on matched GT candidates, applied to every baseline nonempty prediction',
            'single small calibration split; no new architecture claim; no AP confidence interval'],elapsed_seconds=time.perf_counter()-started)
    atomic(out/'SUMMARY.json',result);atomic(out/'progress.json',dict(stage='completed'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
