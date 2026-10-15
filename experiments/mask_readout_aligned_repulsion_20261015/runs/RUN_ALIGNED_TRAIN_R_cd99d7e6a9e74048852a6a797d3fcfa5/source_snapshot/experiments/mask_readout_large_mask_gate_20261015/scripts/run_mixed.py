"""Frozen binary mask selection and original S2 evaluation, with no model execution."""
import argparse
from collections import Counter
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_large_mask_gate_20261015'
PARENT=ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
REPLAY=ROOT/'experiments/mask_readout_benefit_audit_20261014/runs/RUN_BENEFIT_AUDIT_FULL_44b006cece39486aa502ca64398166d4'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
NATIVE=BASE/'trace/native/predictions.json'
CORRECTED=PARENT/'B_predictions.json'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0,str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
from posttrain_run import RegisteredRun,read_json,write_new,sha256,require
from posttrain_audit import read_jsonl,compare_outputs
from s1_geometry import iter_prediction_array,full_rle_iou
from run_s2_evaluation import image_effects,bootstrap
from build_native_ledger import evaluate_ledger
from damage_acceptance import score_damage
from area_gate import AREA_MIN,mix_row,adjudicate

CODE=['experiments/mask_readout_large_mask_gate_20261015/'+p for p in
      ('PROTOCOL.md','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
       'scripts/area_gate.py','scripts/test_area_gate.py','scripts/run_mixed.py','scripts/launch_mixed.ps1')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/'+p for p in
         ('posttrain_run.py','run_s2_evaluation.py','s1_geometry.py')]
CODE += ['experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
         'shared/evaluation/posttrain_audit.py','shared/evaluation/damage_acceptance.py']


def write_lines(path,rows):
    with path.open('x',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def rle_hash(rle):
    return hashlib.sha256(json.dumps(rle,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_inputs():
    expected=read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for section in ('files','frozen_sources'):
        for rel,digest in expected[section].items():
            require(sha256(ROOT/rel)==digest,'input/source drift: '+rel)
    source=read_json(PARENT/'EVAL_CONTRACT.json')
    profile=read_json(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json')
    packages={p:importlib.metadata.version(p) for p in profile['packages']}
    require(packages==profile['packages'],'package versions differ from baseline profile')
    require(importlib.metadata.version('ultralytics')=='8.4.100','wrong ultralytics overlay')
    actual={}
    for name,pin in source['external_evaluator_sources'].items():
        mod=importlib.import_module(name)
        actual[name]=dict(path=str(Path(mod.__file__).resolve()),sha256=sha256(mod.__file__))
        require(actual[name]==pin,'external evaluator path/hash differs: '+name)
    native_sources={}
    for name,digest in source['environment']['source_sha256'].items():
        mod=importlib.import_module(name)
        native_sources[name]=dict(path=str(Path(mod.__file__).resolve()),sha256=sha256(mod.__file__))
        require(native_sources[name]['sha256']==digest,'native source changed: '+name)
    import torch
    require(not torch.cuda.is_initialized(),'unexpected CUDA initialization')
    require(sys.version==source['environment']['python_version'],'Python differs from source evaluation')
    for run in (PARENT,REPLAY):
        require(read_json(run/'run.json')['execution_status']=='completed','source Run incomplete')
        require(read_json(run/'CONFIG_AUDIT.json')['status']=='passed','source config audit failed')
    panel=read_json(PARENT/'DAMAGE_PANEL.json')
    require(panel['inference_contract_sha256']==sha256(PARENT/'EVAL_CONTRACT.json'),'original panel binding invalid')
    require(panel['checkpoint_sha256']==source['checkpoints']['B']['sha256'],'checkpoint-panel mismatch')
    trace=read_json(BASE/'trace/TRACE_MANIFEST.json')
    require(trace['predictions_sha256']==sha256(NATIVE),'native prediction binding mismatch')
    require(read_json(PARENT/'SUMMARY.json')['B']['prediction_sha256']==sha256(CORRECTED),'corrected prediction binding mismatch')
    environment=dict(python=sys.version,platform=platform.platform(),packages=packages,
                     scipy=importlib.metadata.version('scipy'),psutil=importlib.metadata.version('psutil'),
                     external_evaluator_sources=actual,native_sources=native_sources,
                     compute='CPU serialized RLE selection and pinned COCOeval; no model forward')
    return expected,source,panel,environment


class MixedRun(RegisteredRun):
    def __init__(self,scope):
        super().__init__('LARGE_MASK_'+scope.upper(),'LARGE_MASK_GATE',PARENT,CODE,600 if scope=='smoke' else 1800)
        self.path=STUDY/'runs'/self.run_id
        self.lock=STUDY/'.LARGE_MASK_GATE.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_LARGE_MASK_GATE_20261015',scope=scope,
                         new_training_steps=0,model_forward_calls=0,decoder_calls=0,
                         baseline_run_id=BASE.name,damage_panel_reused_after_outcomes_seen=True)

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'another stage has an unresolved lock')
        return super().__enter__()

    def freeze(self,contract):
        context=read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources={rel:sha256(ROOT/rel) for rel in CODE}
        self.meta.update(code_commit=context['code_commit'],dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources,code_commit=context['code_commit']))
        for rel in self.sources:
            dst=self.path/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/rel,dst)
        shutil.copyfile(STUDY/'PROTOCOL.md',self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))


def export_mixed(run,ids):
    selected=set(ids)
    mapping=read_json(PARENT/'B_ROW_MAPPING.json')
    indices=mapping['original_global_indices']
    require(len(indices)==185621 and indices==sorted(set(indices)),'source row mapping changed')
    corrected=iter(iter_prediction_array(CORRECTED))
    cursor=0;count=0;used=[];choice={};coverage=Counter();per_image=Counter()
    paths=[run.path/'NATIVE_SUBSET_predictions.json',run.path/'MIXED_predictions.json',run.path/'GATE_ROWS.jsonl']
    with paths[0].open('x',encoding='utf8') as native_stream,paths[1].open('x',encoding='utf8') as mixed_stream,paths[2].open('x',encoding='utf8') as gate_stream:
        native_stream.write('[');mixed_stream.write('[')
        for global_index,native in enumerate(iter_prediction_array(NATIVE)):
            if cursor>=len(indices):break
            if global_index!=indices[cursor]:continue
            method=next(corrected,None)
            require(method is not None,'corrected predictions truncated')
            require(native['image_id']==method['image_id'],'source image/order mismatch')
            if native['image_id'] in selected:
                output,gate=mix_row(native,method)
                if count:native_stream.write(',');mixed_stream.write(',')
                native_stream.write(json.dumps(native,separators=(',',':'),allow_nan=False))
                mixed_stream.write(json.dumps(output,separators=(',',':'),allow_nan=False))
                area=gate['native_mask_area']
                gate.update(image_id=native['image_id'],global_native_index=global_index,mixed_index=count,
                            native_output_id=mapping['source_prediction_sha256']+':'+str(global_index),
                            native_rle_sha256=rle_hash(native['segmentation']),
                            corrected_rle_sha256=rle_hash(method['segmentation']),mixed_rle_sha256=rle_hash(output['segmentation']))
                gate_stream.write(json.dumps(gate)+'\n')
                used.append(global_index);choice[global_index]=(area,gate['apply_delta'])
                coverage['outputs']+=1;coverage['selected_outputs']+=int(gate['apply_delta'])
                coverage['changed_masks']+=int(gate['mask_changed'])
                coverage['empty_native_masks']+=int(area==0)
                coverage['exact_boundary_masks']+=int(area==AREA_MIN)
                coverage['native_S' if area<1024 else 'native_M' if area<9216 else 'native_L']+=1
                per_image[native['image_id']]+=int(gate['apply_delta']);count+=1
            cursor+=1
            if cursor%25000==0:
                run.check_budget();print('MIXING',cursor,len(indices),'written',count,flush=True)
        native_stream.write(']\n');mixed_stream.write(']\n')
    require(cursor==len(indices) and next(corrected,None) is None,'source prediction mapping not fully consumed')
    require(set(per_image)==selected,'missing evaluation image output rows')
    if run.meta['scope']=='full':require(used==indices,'full mixed output denominator/order changed')
    coverage.update(images=len(ids),images_with_selected_outputs=sum(v>0 for v in per_image.values()))
    result=dict(coverage);result['selected_output_fraction']=coverage['selected_outputs']/count
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=mapping['source_prediction_sha256'],original_global_indices=used))
    write_new(run.path/'GATE_COVERAGE.json',result)
    return choice,used,result


def fixed_effects(run,ids,choice,indices):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(sys.stdout):coco=COCO(str(GT))
    native=read_json(run.path/'NATIVE_SUBSET_predictions.json')
    mixed=read_json(run.path/'MIXED_predictions.json')
    compare_outputs(native,mixed)
    position={g:i for i,g in enumerate(indices)}
    old=[r for r in read_jsonl(PARENT/'B_FIXED_PAIR_EFFECTS.jsonl') if r['image_id'] in set(ids)]
    effects=[]
    for row in old:
        global_index=int(row['baseline_output_detection_id'].rsplit(':',1)[1])
        local=position[global_index];area,apply=choice[global_index]
        gt=coco.annToRLE(coco.anns[row['annotation_id']])
        before=full_rle_iou(native[local]['segmentation'],gt)
        after=full_rle_iou(mixed[local]['segmentation'],gt)
        require(abs(before-row['baseline_iou'])<1e-12,'original fixed pair IoU changed')
        expected=row['method_iou'] if apply else before
        require(abs(after-expected)<1e-12,'mixed fixed endpoint differs from selected source')
        effects.append(dict(image_id=row['image_id'],annotation_id=row['annotation_id'],population=row['population'],
            baseline_output_detection_id=row['baseline_output_detection_id'],baseline_iou=before,method_iou=after,
            delta_iou=after-before,same_output_reaches_075=after>=.75,native_mask_area=area,apply_delta=apply))
    require(len(effects)==len({(r['image_id'],r['annotation_id']) for r in effects}),'fixed pairs duplicate')
    write_lines(run.path/'FIXED_PAIR_EFFECTS.jsonl',effects)
    del native,mixed,coco;gc.collect()
    return effects


def validate_ledger(ledger,baseline,ids,indices,source_sha):
    wanted={(r['image_id'],r['annotation_id']):r for r in baseline if r['image_id'] in set(ids)}
    require(len(ledger)==len(wanted) and {(r['image_id'],r['annotation_id']) for r in ledger}==set(wanted),'GT coverage changed')
    for row in ledger:
        old=wanted[row['image_id'],row['annotation_id']]
        for key in ('is_ignored','gt_category_id','gt_area','task_box75'):
            require(row[key]==old[key],'GT/bbox policy changed: '+key)
        oid=row['bbox75_match_output_id']
        mapped=None if oid is None else source_sha+':'+str(indices[int(oid.rsplit(':',1)[1])])
        require(mapped==old['bbox75_match_output_id'],'bbox match identity changed')
    return wanted


def execute(scope,smoke_id):
    with MixedRun(scope) as run:
        expected,source,panel,environment=verify_inputs()
        original_ids=read_json(PARENT/'EVALUATION_IMAGES.json')
        ids=original_ids if scope=='full' else original_ids[:8]
        require(len(original_ids)==1314 and len(source['dev_ids'])==512 and len(panel['sample'])==1024,'source denominator changed')
        contract=dict(source_inference_evaluation_contract=source,scope=scope,image_ids=ids,
                      source_contract_sha256=sha256(PARENT/'EVAL_CONTRACT.json'),checkpoint_sha256=source['checkpoints']['B']['sha256'],
                      native_profile_sha256=source['native_profile_sha256'],area_min=AREA_MIN,area_source='native_original_image_compressed_RLE_foreground_pixels',
                      gate_input_excludes_GT=True,full_prediction_set=True,new_training_steps=0,model_forward_calls=0,
                      damage_panel_sha256=sha256(PARENT/'DAMAGE_PANEL.json'),damage_panel_reused_after_outcomes_seen=True,
                      fresh_95_coverage_guarantee=False,original_gates=source['gates'],bootstrap=source['bootstrap'],
                      original_denominators=dict(dev_images=512,all_pairs=2925,target_pairs=655,target_images=257,all_pair_images=508,
                                                  success_GT=1024,success_population=22957,evaluation_images=1314,outputs=185621),
                      execution_environment=environment,input_hashes=expected,wall_seconds=run.wall_seconds)
        if scope=='full':
            smoke=STUDY/'runs'/smoke_id
            require(read_json(smoke/'run.json')['execution_status']=='completed','successful smoke required')
            previous=read_json(smoke/'EVAL_CONTRACT.json')
            require(previous['execution_environment']==environment and previous['input_hashes']==expected,'smoke input/environment mismatch')
            for rel,digest in read_json(smoke/'SOURCE_MANIFEST.json')['files'].items():require(sha256(ROOT/rel)==digest,'source changed after smoke')
        run.freeze(contract)
        shutil.copyfile(PARENT/'DAMAGE_PANEL.json',run.path/'DAMAGE_PANEL.json')
        write_new(run.path/'PANEL_REUSE.json',dict(original_panel_sha256=sha256(PARENT/'DAMAGE_PANEL.json'),
            original_inference_contract_sha256=panel['inference_contract_sha256'],current_contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),
            actual_selection='exact existing sample, no new draw',outcomes_previously_seen=True,
            scope='nominal original-formula development screen; no renewed post-selection 95% coverage claim'))
        write_new(run.path/'EVALUATION_IMAGES.json',ids)
        run.update(smoke_run_id=smoke_id or None,phase='GT_free_prediction_mix')
        choice,indices,coverage=export_mixed(run,ids)
        effects=fixed_effects(run,ids,choice,indices)
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        source_sha=read_json(PARENT/'B_ROW_MAPPING.json')['source_prediction_sha256']
        if scope=='smoke':
            with (run.path/'native_smoke_ledger.log').open('x',encoding='utf8') as stream,contextlib.redirect_stdout(stream):
                native_ledger,_=evaluate_ledger(GT,read_json(run.path/'NATIVE_SUBSET_predictions.json'),ids,sha256(run.path/'NATIVE_SUBSET_predictions.json'))
            old=validate_ledger(native_ledger,baseline,ids,indices,source_sha)
            for row in native_ledger:
                previous=old[row['image_id'],row['annotation_id']]
                for field in ('task_mask75','task_mask50','task_box75'):require(row[field]==previous[field],'native smoke task parity failed')
                oid=row['segm75_match_output_id']
                mapped=None if oid is None else source_sha+':'+str(indices[int(oid.rsplit(':',1)[1])])
                require(mapped==previous['segm75_match_output_id'],'native smoke segm identity parity failed')
            write_new(run.path/'NATIVE_SMOKE_PARITY.json',dict(status='passed',images=len(ids),GT=len(native_ledger)))
        run.update(phase='complete_mixed_COCO_bbox_and_segm')
        print('COCO_MATCHING',len(ids),'images',coverage['outputs'],'outputs',flush=True)
        with (run.path/'mixed_ledger.log').open('x',encoding='utf8') as stream,contextlib.redirect_stdout(stream):
            ledger,ap=evaluate_ledger(GT,read_json(run.path/'MIXED_predictions.json'),ids,sha256(run.path/'MIXED_predictions.json'))
        wanted=validate_ledger(ledger,baseline,ids,indices,source_sha)
        write_lines(run.path/'MIXED_GT_LEDGER.jsonl',ledger)
        write_lines(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl',wanted.values())
        if scope=='full':
            require(len(effects)==2925 and sum(r['population']=='T' for r in effects)==655,'effect denominator changed')
            oldkeys={(r['image_id'],r['annotation_id'],r['baseline_output_detection_id']) for r in read_jsonl(PARENT/'B_FIXED_PAIR_EFFECTS.jsonl')}
            require({(r['image_id'],r['annotation_id'],r['baseline_output_detection_id']) for r in effects}==oldkeys,'fixed pair set changed')
            target=bootstrap(image_effects(effects));overall=bootstrap(image_effects(effects,target_only=False))
            require(target['images']==257 and overall['images']==508,'image macro denominator changed')
            damage=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in set(panel['image_ids'])])
            damage['coverage_interpretation']='nominal original formula on reused outcome-exposed panel; no fresh 95% guarantee'
            status,gates=adjudicate(target,overall,damage)
            keyed={(r['image_id'],r['annotation_id']):r for r in ledger}
            target_rows=[r for r in effects if r['population']=='T']
            dev_rows=effects
            sample_rows=[keyed[r['image_id'],r['annotation_id']] for r in panel['sample']]
            damaged=[dict(image_id=r['image_id'],annotation_id=r['annotation_id'],task_mask75=r['task_mask75'],
                         baseline_segm_output_id=wanted[r['image_id'],r['annotation_id']]['segm75_match_output_id'])
                     for r in sample_rows if r['task_mask75']=='failure']
            write_lines(run.path/'DAMAGED_PANEL_GT.jsonl',damaged)
            coverage.update(target_selected=sum(r['apply_delta'] for r in target_rows),target_total=655,
                            dev_selected=sum(r['apply_delta'] for r in dev_rows),dev_total=2925,
                            target_task_repaired=sum(keyed[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in target_rows),
                            target_fixed_detection_reaches075=sum(r['same_output_reaches_075'] for r in target_rows))
            success_selected=0
            for r in panel['sample']:
                oid=wanted[r['image_id'],r['annotation_id']]['segm75_match_output_id']
                success_selected+=int(choice[int(oid.rsplit(':',1)[1])][1])
            coverage.update(success_fixed_detection_selected=success_selected,success_total=1024)
            next_action=('Freeze/execute S3 full5000 with this checkpoint and fixed9216 gate; no automatic S3 or training' if status=='passed' else
                         'Close this fixed area strategy; freeze a native-input-geometry GT-objective-only protocol; no threshold scan or training')
            summary=dict(gates=gates,target=target,all_bbox75=overall,damage=damage,coverage=coverage,
                         secondary_AP_on_union_dev_and_damage_images=ap,target_GT=655,
                         scientific_scope='Outcome-exposed COCO development sample; original S2 numerical screen, not independent confirmation')
        else:
            status='smoke_passed';gates=None;next_action='Run the authorized frozen full mixed-output evaluation'
            summary=dict(coverage=coverage,fixed_pairs=len(effects),GT=len(ledger),scientific_scope='engineering_only')
        run.check_budget()
        again,again_source,again_panel,again_env=verify_inputs()
        require(again==expected and again_source==source and again_panel==panel and again_env==environment,'post-eval configuration drift')
        run.check_budget()
        require(sha256(run.path/'DAMAGE_PANEL.json')==sha256(PARENT/'DAMAGE_PANEL.json'),'panel copy changed')
        import torch,psutil
        require(not torch.cuda.is_initialized(),'unexpected model CUDA execution')
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
        require(peak<=4*1024**3,'memory budget exceeded')
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=1024**3,'artifact budget exceeded')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',checkpoint_sha256=contract['checkpoint_sha256'],
            contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),damage_panel_sha256=sha256(run.path/'DAMAGE_PANEL.json'),peak_rss_bytes=peak,
            checks=dict(all_input_and_source_hashes_before_after=True,pinned_evaluator_versions_paths_and_hashes=True,
                gate_uses_native_original_mask_only=True,all_nonmask_fields_and_order_preserved=True,full_same_image_prediction_and_GT_coverage=True,
                same_bbox_match_identity=True,independent_segm_COCO_rematching=True,selected_source_fixed_pair_IoU_equal=True,
                original_denominators_and_gate_numbers=True,existing_panel_bytes_unchanged=True,panel_reuse_limitation_explicit=True,
                no_model_forward_decoder_or_training=True),intentional_changes=['apply existing residual only when original native mask area >=9216']))
        run.finish(summary,status=status,scientific=('S2_development_'+status) if scope=='full' else 'engineering_only',next_action=next_action)
        print('MIXED_DECISION',json.dumps(dict(status=status,gates=gates,summary=summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('smoke','full'))
    parser.add_argument('--smoke-run',default='');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        files,_,_,env=verify_inputs();print(json.dumps(dict(status='passed',inputs=len(files['files']),sources=len(files['frozen_sources']),environment=env)))
    elif args.scope:execute(args.scope,args.smoke_run)
    else:parser.error('scope or preflight required')
