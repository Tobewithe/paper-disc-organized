"""Freeze final B/C checkpoints, sample the success panel once, and adjudicate S2."""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import json
from pathlib import Path
import sys
import time
import subprocess
from posttrain_run import ROOT, STUDY, RegisteredRun, configure_native, read_json, require, sha256, write_new
from posttrain_audit import read_jsonl, population, compare_outputs
from s1_geometry import full_rle_iou

BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
WEIGHT=ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt'
CODE_FILES=['experiments/mask_readout_mechanism_v2_20261012/scripts/'+n for n in (
    'run_s2_evaluation.py','test_s2_evaluation.py','audit_s2_completed.py','adapter_model.py','adapter_data.py','adapter_primitives.py','s1_geometry.py','posttrain_run.py')]+[
    'experiments/mask_readout_mechanism_v2_20261012/S2_EXECUTION_20261013.md',
    'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
    'shared/evaluation/native_output_bridge.py','shared/evaluation/native_source_pins.json',
    'shared/evaluation/posttrain_audit.py','shared/evaluation/damage_acceptance.py',
    'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']


def image_effects(rows, field='delta_iou', target_only=True):
    groups=defaultdict(list)
    for row in rows:
        if not target_only or row['population']=='T':
            groups[row['image_id']].append(row[field])
    require(groups,'empty effect population')
    return {image_id:sum(values)/len(values) for image_id,values in sorted(groups.items())}


def bootstrap(values):
    import numpy as np
    array=np.array(list(values.values()),dtype=np.float64)
    require(len(array)>1 and np.isfinite(array).all(),'invalid effect sample')
    rng=np.random.default_rng(1729)
    draws=array[rng.integers(0,len(array),size=(2000,len(array)))].mean(axis=1)
    return {'mean':float(array.mean()),'ci95':[float(v) for v in np.quantile(draws,[.025,.975],method='linear')],
            'images':len(array),'resamples':2000,'seed':1729,'unit':'IoU_fraction_0_to_1','method':'paired_image_macro_percentile_linear'}


def export_and_measure(run,arm,checkpoint,source,baseline,dev_ids,coco):
    import torch
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native,encode_rle
    model=CoefficientAdapter(arm).cuda()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(saved['arm']==arm and saved['completed_epochs']==3,'wrong arm or nonfinal checkpoint')
    model.load_state_dict(saved['state_dict'],strict=True)
    del saved
    model.eval()
    frozen_hash=sha256(checkpoint)
    annotations_by_image=defaultdict(list)
    for row in baseline:
        if row['image_id'] in dev_ids and not row['is_ignored'] and row['bbox75_match_output_id'] is not None:
            annotations_by_image[row['image_id']].append(row)
    output=run.path/(arm+'_predictions.json')
    effects,global_indices=[],[]
    seen=0
    total_norm,total_rows=0.,0
    started=time.monotonic()
    with output.open('x',encoding='utf8') as stream, torch.inference_mode():
        stream.write('[')
        comma=False
        for image_id,record in source.records.items():
            run.check_budget()
            cache,original=source.load(image_id)
            delta=model(cache,'eval:0')
            require(torch.isfinite(delta).all().item(),'nonfinite method delta')
            decoded=decode_native(cache,cache['c0'].to(delta)+delta)
            rows=[{**row,'segmentation':encode_rle(decoded[i])} for i,row in enumerate(original)]
            compare_outputs(original,rows)
            mapping={index:rank for rank,index in enumerate(cache['prediction_indices'])}
            for row in annotations_by_image[image_id]:
                index=int(row['bbox75_match_output_id'].rsplit(':',1)[1])
                rank=mapping[index]
                gt_rle=coco.annToRLE(coco.anns[row['annotation_id']])
                baseline_iou=full_rle_iou(original[rank]['segmentation'],gt_rle)
                require(abs(baseline_iou-row['mask_iou_of_bbox_match'])<1e-12,'baseline fixed-pair IoU mismatch')
                actual=full_rle_iou(rows[rank]['segmentation'],gt_rle)
                effects.append({'image_id':image_id,'annotation_id':row['annotation_id'],'population':population(row),
                    'baseline_output_detection_id':row['bbox75_match_output_id'],'baseline_iou':baseline_iou,
                    'method_iou':actual,'delta_iou':actual-baseline_iou,'same_output_reaches_075':actual>=.75})
            for row in rows:
                if comma:stream.write(',')
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False))
                comma=True
            global_indices.extend(cache['prediction_indices'])
            total_norm+=float(delta.norm(dim=1).sum())
            total_rows+=len(rows)
            seen+=1
            if seen%50==0 or seen==len(source.records):
                stream.flush()
                run.update(phase='method_export',arm=arm,completed_images=seen,total_images=len(source.records))
                print('EVAL_EXPORT',arm,seen,'/',len(source.records),'seconds',round(time.monotonic()-started,1),flush=True)
            del cache,decoded,delta,rows,original
        stream.write(']\n')
    require(sha256(checkpoint)==frozen_hash,'checkpoint mutated in evaluation')
    require(global_indices==sorted(global_indices),'output order differs from native JSON')
    write_new(run.path/(arm+'_ROW_MAPPING.json'),{'source_prediction_sha256':source.manifest['predictions_sha256'],
              'original_global_indices':global_indices})
    with (run.path/(arm+'_FIXED_PAIR_EFFECTS.jsonl')).open('x',encoding='utf8') as stream:
        for row in effects:stream.write(json.dumps(row,allow_nan=False)+'\n')
    del model
    torch.cuda.empty_cache()
    return output,effects,global_indices,{'images':seen,'outputs':total_rows,
        'mean_delta_l2':total_norm/total_rows if total_rows else None,'export_seconds':time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b-run',type=Path,required=True)
    parser.add_argument('--c-run',type=Path,required=True)
    parser.add_argument('--cache-run',type=Path,required=True)
    parser.add_argument('--retry-of',type=Path,help='Recover an engineering failure using its exact already-drawn panel')
    args=parser.parse_args()
    for path in (args.b_run,args.c_run,args.cache_run):
        require(read_json(path/'CONFIG_AUDIT.json')['status']=='passed','prerequisite configuration audit missing/failed')
        require(read_json(path/'DECISION.json')['status']=='passed','prerequisite run did not pass')
    with RegisteredRun('S2_EVAL','S2',args.b_run,CODE_FILES,3600) as run:
        profile,environment=configure_native()
        require(sha256(GT)==profile['annotations_sha256'] and sha256(WEIGHT)==profile['model_sha256'],'frozen model/GT changed')
        for parent in (args.b_run,args.c_run):
            sources=read_json(parent/'SOURCE_MANIFEST.json')['files']
            for relative in CODE_FILES:
                if relative in sources:require(sha256(ROOT/relative)==sources[relative],'trained architecture/runtime source changed')
        split=read_json(args.cache_run/'SPLIT_CONTRACT.json')
        checkpoints={arm:{'run_id':path.name,'path':str(path/'last.pt'),'sha256':sha256(path/'last.pt'),
                        'training_contract_sha256':sha256(path/'EVAL_CONTRACT.json')}
                     for arm,path in [('B',args.b_run),('C',args.c_run)]}
        for arm,path in [('B',args.b_run),('C',args.c_run)]:
            contract=read_json(path/'EVAL_CONTRACT.json')
            require(contract['arm']==arm and contract['scope']=='full' and contract['expected_optimizer_steps']==1500,'wrong training contract')
            require(contract['fit_dev_split_sha256']==sha256(args.cache_run/'SPLIT_CONTRACT.json'),'different training split')
            require(sha256(path/'last.pt')==read_json(path/'SUMMARY.json')['checkpoint_sha256'],'checkpoint no longer equals completed training output')
        evaluation={'stage':'S2','primary_arm':'B','control_arm':'C','checkpoints':checkpoints,
                    'native_profile_sha256':sha256(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'),
                    'source_baseline_run':BASE.name,'native_cache_manifest_sha256':sha256(BASE/'trace/TRACE_MANIFEST.json'),
                    'ledger_sha256':sha256(BASE/'ledger/PER_GT_FAILURE.jsonl'),
                    'dev_ids':split['dev_ids'],'dev_T':split['dev_T'],'evaluation_phase':'eval:0',
                    'same_full_image_native_coefficient_matrix':True,'fixed_boxes_classes_scores_order':True,
                    'bootstrap':{'resamples':2000,'seed':1729,'CI':[.025,.975],'quantile':'linear','unit':'IoU_0_to_1'},
                    'gates':{'target_delta_min':.005,'target_ci_lower_strict_min':0.,'all_bbox75_delta_min':-.001,
                             'damage_one_sided95_upper_max':.01},
                    'damage_sample_n':1024,'damage_sample_timing':'after_this_contract_before_any_method_evaluation',
                    'task_evaluator':'pinned pycocotools COCOeval; full prediction JSON incl bbox; S0 dual backend parity already verified',
                    'maxDets':[1,10,100],'useCats':1,'environment':environment,'wall_seconds':3600}
        run.freeze(evaluation)
        from audit_s2_completed import audit
        audit(args.b_run)
        audit(args.c_run)
        training_gt_sha=sha256(args.cache_run/'dataset/training_annotations.json')
        require(training_gt_sha==read_json(args.cache_run/'EVAL_CONTRACT.json')['gt_subset_sha256'],
                'retained training GT differs from native-cache frozen annotations')
        for name in ('IMAGE_ORDERS.json','TRAINING_JOIN.json'):
            require(sha256(args.b_run/name)==sha256(args.c_run/name),'actual B/C samples or GT joins differ')
        write_new(run.path/'PAIRED_CONFIG_AUDIT.json',{'status':'passed',
                  'B_independent_audit_sha256':sha256(args.b_run/'INDEPENDENT_CONFIG_AUDIT.json'),
                  'C_independent_audit_sha256':sha256(args.c_run/'INDEPENDENT_CONFIG_AUDIT.json'),
                  'retained_training_GT_sha256':training_gt_sha,'training_GT_matches_pre_cache_contract':True,
                  'B_C_actual_image_order_and_GT_joins_identical':True})
        with (run.path/'evaluation_tests.log').open('x',encoding='utf8') as stream:
            tested=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('test_s2_evaluation.py'))],stdout=stream,stderr=subprocess.STDOUT)
        require(tested.returncode==0,'evaluation behavioral tests failed')
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        from damage_acceptance import select_damage_panel,score_damage
        # No resampling or retry to obtain a favorable damage sample.
        prior_panels=[p for p in (STUDY/'runs').glob('RUN_S2_EVAL_*/DAMAGE_PANEL.json')
                      if p.parent!=run.path and read_json(p)['checkpoint_sha256']==checkpoints['B']['sha256']]
        if args.retry_of is not None:
            require(read_json(args.retry_of/'run.json')['execution_status']=='error','retry requires retained engineering failure')
            require(sha256(args.retry_of/'EVAL_CONTRACT.json')==sha256(run.path/'EVAL_CONTRACT.json'),'retry changed frozen inference/evaluation contract')
            panel=read_json(args.retry_of/'DAMAGE_PANEL.json')
            require(all(read_json(p)==panel for p in prior_panels),'inconsistent prior damage sample')
            run.update(retry_of=args.retry_of.name,damage_panel_reused=True)
        else:
            require(not prior_panels,'damage draw for this checkpoint already exists; recover its exact panel')
            panel=select_damage_panel(baseline,evaluation['ledger_sha256'],checkpoints['B']['sha256'],sha256(run.path/'EVAL_CONTRACT.json'),1024)
        write_new(run.path/'DAMAGE_PANEL.json',panel)
        ids=sorted(set(split['dev_ids'])|set(panel['image_ids']))
        write_new(run.path/'EVALUATION_IMAGES.json',ids)
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        source=FrozenCache(BASE/'trace',ids)
        coco=COCO(str(GT))
        sys.path.insert(0,str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
        from build_native_ledger import evaluate_ledger
        effects_by_arm,results={},{}
        baseline_keys={(r['image_id'],r['annotation_id']):r for r in baseline}
        expected_T={(r['image_id'],r['annotation_id']) for r in baseline if r['image_id'] in set(split['dev_ids']) and population(r)=='T'}
        require(len(expected_T)==split['dev_T'] and len(expected_T)>=200,'dev population changed')
        for arm in ('B','C'):
            prediction_path,effects,global_indices,export_stats=export_and_measure(run,arm,Path(checkpoints[arm]['path']),source,baseline,set(split['dev_ids']),coco)
            require({(r['image_id'],r['annotation_id']) for r in effects if r['population']=='T'}==expected_T,'target effect coverage mismatch')
            target=bootstrap(image_effects(effects,target_only=True))
            overall=bootstrap(image_effects(effects,target_only=False))
            effects_by_arm[arm]=effects
            run.update(phase='COCO_task_ledger',arm=arm)
            with (run.path/(arm+'_ledger.log')).open('x',encoding='utf8') as stream,contextlib.redirect_stdout(stream):
                method,ap=evaluate_ledger(GT,read_json(prediction_path),ids,sha256(prediction_path))
            require({(r['image_id'],r['annotation_id']) for r in method}=={k for k,r in baseline_keys.items() if r['image_id'] in set(ids)},'method GT coverage changed')
            for row in method:
                previous=baseline_keys[(row['image_id'],row['annotation_id'])]
                require(row['is_ignored']==previous['is_ignored'],'GT ignore changed')
                oid=row['bbox75_match_output_id']
                mapped=None if oid is None else source.manifest['predictions_sha256']+':'+str(global_indices[int(oid.rsplit(':',1)[1])])
                require(mapped==previous['bbox75_match_output_id'],'bbox matching detection identity changed')
            damage=score_damage(panel,baseline,[r for r in method if r['image_id'] in set(panel['image_ids'])])
            with (run.path/(arm+'_GT_LEDGER.jsonl')).open('x',encoding='utf8') as stream:
                for row in method:stream.write(json.dumps(row,allow_nan=False)+'\n')
            results[arm]={'target':target,'all_bbox75':overall,'damage':damage,'export':export_stats,
                          'secondary_AP_on_union_dev_and_damage_images':ap,'prediction_sha256':sha256(prediction_path)}
            write_new(run.path/(arm+'_RESULTS.json'),results[arm])
            print('S2_EFFECT',arm,json.dumps({'target':target,'all_bbox75':overall,'damage':damage}),flush=True)
            run.check_budget()
        b_images=image_effects(effects_by_arm['B']);c_images=image_effects(effects_by_arm['C'])
        require(set(b_images)==set(c_images),'B/C target images differ')
        contrast=bootstrap({i:b_images[i]-c_images[i] for i in b_images})
        main=results['B']
        gates={'target_effect':main['target']['mean']>=.005 and main['target']['ci95'][0]>0,
               'all_bbox75_guardrail':main['all_bbox75']['mean']>=-.001,
               'success_damage':main['damage']['gate']=='damage_guardrail_passed'}
        status='passed' if all(gates.values()) else ('blocked' if gates['target_effect'] and gates['all_bbox75_guardrail'] and main['damage']['gate']=='blocked_insufficient_precision' else 'failed')
        source.verify()
        for info in checkpoints.values():require(sha256(info['path'])==info['sha256'],'checkpoint mutated')
        require(sha256(GT)==profile['annotations_sha256'] and sha256(WEIGHT)==profile['model_sha256'],'frozen GT/weight mutated')
        from native_output_bridge import native_environment
        after=native_environment()
        require(all(after[k]==v for k,v in profile['runtime_policy'].items()),'evaluation numerical policy drift')
        write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','scope':'S2 B/C paired evaluation',
            'checks':{'checkpoint_and_architecture_identity':True,'same_native_cached_geometry_P_c0_F':True,
                      'boxes_classes_scores_order_unchanged':True,'bbox_COCO_matching_unchanged':True,
                      'GT_and_ignore_policy_unchanged':True,'complete_dev_and_damage_image_coverage':True,
                      'bootstrap_and_damage_sampling_contract_unchanged':True,'native_runtime_policy_unchanged':True,
                      'frozen_cache_checkpoint_GT_weight_hashes_unchanged':True},
            'contract_sha256':sha256(run.path/'EVAL_CONTRACT.json'),'damage_panel_sha256':sha256(run.path/'DAMAGE_PANEL.json'),
            'intentional_changes':['B or C coefficient residual; segmentation RLE only']})
        summary={'gates':gates,'B':results['B'],'C':results['C'],'B_minus_C_target':contrast,
                 'spatial_order_increment_supported':contrast['ci95'][0]>0,'target_GT':len(expected_T),
                 'scientific_scope':'fixed native configuration COCO val2017 development; not independent confirmation'}
        next_action='Execute S3 full 5000 using exactly this B checkpoint' if status=='passed' else 'STOP current S2 configuration; retain negative/insufficient evidence; no S3 or scale-up'
        run.finish(summary,status=status,scientific='S2_'+status,next_action=next_action)
        print('S2_DECISION',status,json.dumps(gates),flush=True)


if __name__=='__main__':
    main()
