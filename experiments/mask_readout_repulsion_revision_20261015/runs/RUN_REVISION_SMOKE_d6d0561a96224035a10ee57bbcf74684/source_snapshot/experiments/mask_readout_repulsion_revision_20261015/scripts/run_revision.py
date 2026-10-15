"""Registered zero-training experiment 1 with native decoder and unchanged S2 evaluators."""
import argparse
from collections import Counter
import contextlib
import gc
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_repulsion_revision_20261015'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
PARENT=ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0,str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
sys.path.insert(0,str(ROOT/'experiments/mask_readout_large_mask_gate_20261015/scripts'))
from posttrain_run import RegisteredRun,configure_native,read_json,write_new,sha256,require
from posttrain_audit import read_jsonl,compare_outputs
from adapter_data import FrozenCache
from native_output_bridge import decode_native,encode_rle,native_environment
from s1_geometry import full_rle_iou
from run_s2_evaluation import image_effects,bootstrap
from build_native_ledger import evaluate_ledger
from damage_acceptance import score_damage
from run_mixed import validate_ledger
from area_gate import adjudicate
from revision import update,PARAMETERS

CODE=['experiments/mask_readout_repulsion_revision_20261015/'+p for p in
      ('PROTOCOL.md','REVIEW_SOURCE.txt','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
       'scripts/revision.py','scripts/test_revision.py','scripts/run_revision.py','scripts/launch_revision.ps1','scripts/verify_revision.py')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/'+p for p in
         ('posttrain_run.py','run_s2_evaluation.py','s1_geometry.py','adapter_data.py')]
CODE += ['shared/evaluation/'+p for p in ('native_output_bridge.py','posttrain_audit.py','damage_acceptance.py')]
CODE += ['experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
         'experiments/mask_readout_large_mask_gate_20261015/scripts/run_mixed.py',
         'experiments/mask_readout_large_mask_gate_20261015/scripts/area_gate.py']


def jl(path,rows):
    with path.open('x',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def inspect():
    profile,environment=configure_native()
    expected=read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for group in ('files','frozen_sources'):
        for rel,h in expected[group].items():require(sha256(ROOT/rel)==h,'input/source drift: '+rel)
    source=read_json(PARENT/'EVAL_CONTRACT.json')
    require(environment==source['environment'],'native numeric environment differs')
    for name,pin in source['external_evaluator_sources'].items():
        module=importlib.import_module(name)
        require(dict(path=str(Path(module.__file__).resolve()),sha256=sha256(module.__file__))==pin,'evaluator source differs: '+name)
    panel=read_json(PARENT/'DAMAGE_PANEL.json')
    require(panel['inference_contract_sha256']==sha256(PARENT/'EVAL_CONTRACT.json'),'historical panel binding changed')
    require(read_json(PARENT/'run.json')['execution_status']=='completed','parent evaluation incomplete')
    return expected,source,panel,environment


class RevisionRun(RegisteredRun):
    def __init__(self,scope):
        super().__init__('REVISION_'+scope.upper(),'E1_R2',BASE,CODE,900 if scope=='smoke' else 5400)
        self.path=STUDY/'runs'/self.run_id;self.lock=STUDY/'.E1_R2.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_REPULSION_REVISION_20261015',scope=scope,
                         new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,method_id='E1-R2-A-FG_and_B-LOCAL')

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'unresolved concurrent experiment lock')
        return super().__enter__()

    def freeze(self,contract):
        origin=read_json(STUDY/'SOURCE_CONTEXT.json');self.sources={rel:sha256(ROOT/rel) for rel in CODE}
        self.meta.update(code_commit=origin['code_commit'],dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources,code_commit=origin['code_commit']))
        for rel in self.sources:
            target=self.path/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
        shutil.copyfile(STUDY/'PROTOCOL.md',self.path/'PROTOCOL.md');write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))


CODE += ['experiments/mask_readout_subspace_repulsion_20261015/scripts/'+p for p in ('repulsion.py','verify_repulsion.py')]
OLD=ROOT/'experiments/mask_readout_subspace_repulsion_20261015/runs/RUN_REPULSION_FULL_8b2e19a348d1476fbf05b64aabb10c7f'


def iter_jsonl(path):
    with path.open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def export(run,source):
    import torch
    indices=[];actions={};stats={a:Counter() for a in ('A','B')};start=time.monotonic()
    torch.cuda.reset_peak_memory_stats();old_rows=iter_jsonl(OLD/'RULE_ROWS.jsonl');old_row=next(old_rows,None)
    with contextlib.ExitStack() as stack,torch.inference_mode():
        outs={a:stack.enter_context((run.path/f'{a}_predictions.json').open('x',encoding='utf8')) for a in stats}
        rules=stack.enter_context((run.path/'RULE_ROWS.jsonl').open('x',encoding='utf8'))
        metadata=stack.enter_context((run.path/'IMAGE_METADATA.jsonl').open('x',encoding='utf8'))
        for f in outs.values():f.write('[')
        for number,image_id in enumerate(source.records,1):
            run.check_budget();cache,original=source.load(image_id);require(cache['proto'].shape[0]==32,'prototype channel changed')
            native=decode_native(cache,device='cuda');native_rles=[encode_rle(m) for m in native]
            require(native_rles==[r['segmentation'] for r in original],'native RLE mismatch');del native;torch.cuda.empty_cache()
            coefficients,proof=update(cache,'cuda')
            for arm in stats:
                masks=decode_native(cache,coefficients[arm],device='cuda');rles=[encode_rle(m) for m in masks]
                rows=[dict(r,segmentation=m) for r,m in zip(original,rles)];compare_outputs(original,rows)
                for rank,(row,r) in enumerate(zip(rows,proof)):
                    changed=row['segmentation']!=native_rles[rank]
                    if not r['applied']:require(not changed,'unmodified row mask differs')
                    if indices or rank:outs[arm].write(',')
                    outs[arm].write(json.dumps(row,separators=(',',':'),allow_nan=False))
                    r['arms'][arm]['mask_changed']=changed
                    stats[arm]['outputs']+=1;stats[arm]['selected_coefficients']+=int(r['applied']);stats[arm]['changed_masks']+=int(changed)
                stats[arm]['images']+=1;stats[arm]['images_with_updates']+=int(any(r['applied'] for r in proof))
                del masks,rles,rows;torch.cuda.empty_cache()
            for rank,(r,idx) in enumerate(zip(proof,cache['prediction_indices'])):
                while old_row is not None and old_row['global_native_index']<idx:old_row=next(old_rows,None)
                require(old_row is not None and old_row['global_native_index']==idx,'old E1 mapping differs')
                require(old_row['neighbor_rank']==r['neighbor_rank'] and old_row['omega_pixels']==r['omega_pixels'] and old_row['applied']==r['old_applied'],'old E1 gate replay differs')
                for key in ('numerator','denominator','alpha'):
                    require(abs(old_row[key]-r['old_'+key])<=5e-5*(1+abs(old_row[key])),'old E1 numerical replay differs')
                r.update(image_id=image_id,global_native_index=idx,method_index=len(indices),raw_id=cache['raw_ids'][rank],
                         neighbor_raw_id=None if r['neighbor_rank']<0 else cache['raw_ids'][r['neighbor_rank']])
                rules.write(json.dumps(r,allow_nan=False)+'\n');indices.append(idx)
                actions[idx]=dict(applied=r['applied'],neighbor_rank=r['neighbor_rank'],conflict_pixels=r['conflict_pixels'])
            metadata.write(json.dumps(dict(image_id=image_id,cache_relative_path=source.records[image_id]['cache'],cache_sha256=source.records[image_id]['cache_sha256'],
                prediction_indices=cache['prediction_indices'],boxes=cache['boxes_input'].tolist(),scores=cache['scores'].tolist(),classes=cache['classes'].tolist(),
                raw_ids=cache['raw_ids'],input_hw=cache['input_hw'],proto_hw=list(cache['proto'].shape[-2:])),allow_nan=False)+'\n')
            del cache,original,native_rles,coefficients,proof;gc.collect();torch.cuda.empty_cache()
            if number%50==0 or number==len(source.records):print('EXPORT',number,len(source.records),{a:dict(v) for a,v in stats.items()},flush=True)
        for f in outs.values():f.write(']\n')
    expected=read_json(PARENT/'B_ROW_MAPPING.json')['original_global_indices']
    require(indices==sorted(indices),'prediction order changed')
    if run.meta['scope']=='full':require(indices==expected and len(indices)==185621,'output denominator changed')
    resources=dict(export_seconds=time.monotonic()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    require(resources['cuda_peak_reserved_bytes']<=6.5*1024**3,'GPU memory exceeded')
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=source.manifest['predictions_sha256'],original_global_indices=indices))
    return indices,actions,{a:dict(v) for a,v in stats.items()},resources


def evaluate_arm(run,arm,source,ids,indices,actions,panel,baseline):
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    coco=COCO(str(GT));prediction_path=run.path/f'{arm}_predictions.json';predictions=read_json(prediction_path)
    old_predictions=read_json(OLD/'REPULSION_predictions.json');old_indices=read_json(OLD/'ROW_MAPPING.json')['original_global_indices']
    old_position={idx:i for i,idx in enumerate(old_indices)};position={idx:i for i,idx in enumerate(indices)};effects=[]
    def confusion(pred,truth):
        tp=int(masks.area(masks.merge([pred,truth],intersect=True)))
        return dict(tp=tp,fp=int(masks.area(pred))-tp,fn=int(masks.area(truth))-tp)
    def changed_pixels(before,after,truth):
        common=int(masks.area(masks.merge([before,after],intersect=True)))
        common_tp=int(masks.area(masks.merge([before,after,truth],intersect=True)))
        old=confusion(before,truth);new=confusion(after,truth)
        removed=old['tp']+old['fp']-common;added=new['tp']+new['fp']-common
        removed_tp=old['tp']-common_tp;added_tp=new['tp']-common_tp
        return dict(removed_true=removed_tp,removed_false=removed-removed_tp,added_true=added_tp,added_false=added-added_tp)
    for r in read_jsonl(PARENT/'B_FIXED_PAIR_EFFECTS.jsonl'):
        if r['image_id'] not in set(ids):continue
        idx=int(r['baseline_output_detection_id'].rsplit(':',1)[1]);truth=coco.annToRLE(coco.anns[r['annotation_id']])
        native=source.predictions[idx]['segmentation'];new=predictions[position[idx]]['segmentation'];old=old_predictions[old_position[idx]]['segmentation']
        before=full_rle_iou(native,truth);after=full_rle_iou(new,truth)
        require(abs(before-r['baseline_iou'])<1e-12,'fixed pair changed')
        effects.append(dict(image_id=r['image_id'],annotation_id=r['annotation_id'],population=r['population'],baseline_output_detection_id=r['baseline_output_detection_id'],
            baseline_iou=before,method_iou=after,delta_iou=after-before,same_output_reaches_075=after>=.75,**actions[idx],
            confusion=dict(native=confusion(native,truth),old_E1=confusion(old,truth),method=confusion(new,truth)),
            changed_pixels=dict(old_E1=changed_pixels(native,old,truth),method=changed_pixels(native,new,truth))))
    jl(run.path/f'{arm}_FIXED_PAIR_EFFECTS.jsonl',effects)
    del coco,old_predictions;gc.collect();run.check_budget()
    with (run.path/f'{arm}_evaluation.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
        ledger,ap=evaluate_ledger(GT,predictions,ids,sha256(prediction_path))
    wanted=validate_ledger(ledger,baseline,ids,indices,source.manifest['predictions_sha256'])
    jl(run.path/f'{arm}_GT_LEDGER.jsonl',ledger)
    if arm=='A':jl(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl',wanted.values())
    if run.meta['scope']=='smoke':return dict(status='smoke_passed',fixed_pairs=len(effects),GT=len(ledger))
    require(len(effects)==2925 and sum(r['population']=='T' for r in effects)==655,'fixed population changed')
    target=bootstrap(image_effects(effects));overall=bootstrap(image_effects(effects,target_only=False))
    require(target['images']==257 and overall['images']==508,'image denominator changed')
    damage=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in set(panel['image_ids'])])
    damage['coverage_interpretation']='nominal original formula on exposed development panel'
    status,gates=adjudicate(target,overall,damage);keyed={(r['image_id'],r['annotation_id']):r for r in ledger}
    targets=[r for r in effects if r['population']=='T']
    return dict(status=status,gates=gates,target=target,all_bbox75=overall,damage=damage,secondary_subset_AP=ap,
        dev_selected=sum(r['applied'] for r in effects),target_selected=sum(r['applied'] for r in targets),
        target_task_repaired=sum(keyed[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in targets),
        target_fixed_detection_reaches075=sum(r['same_output_reaches_075'] for r in targets))


def execute(scope,smoke_id):
    with RevisionRun(scope) as run:
        expected,source_contract,panel,environment=inspect()
        original_ids=read_json(PARENT/'EVALUATION_IMAGES.json');ids=original_ids if scope=='full' else original_ids[:8]
        require(len(original_ids)==1314 and len(panel['sample'])==1024,'input population changed')
        contract=dict(method_id='E1-R2-A-FG_and_B-LOCAL',scope=scope,image_ids=ids,method_parameters=PARAMETERS,
            source_contract=source_contract,original_contract_sha256=sha256(PARENT/'EVAL_CONTRACT.json'),native_profile_sha256=source_contract['native_profile_sha256'],
            input_hashes=expected,environment=environment,new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,
            original_gates=source_contract['gates'],bootstrap=source_contract['bootstrap'],damage_panel_sha256=sha256(PARENT/'DAMAGE_PANEL.json'),
            damage_panel_reused=True,fresh_independent_95_coverage_guarantee=False,wall_seconds=run.wall_seconds)
        if scope=='full':
            smoke=STUDY/'runs'/smoke_id;require(read_json(smoke/'run.json')['execution_status']=='completed','successful smoke required')
            old=read_json(smoke/'EVAL_CONTRACT.json');require(old['input_hashes']==expected and old['environment']==environment,'smoke configuration differs')
            for rel,h in read_json(smoke/'SOURCE_MANIFEST.json')['files'].items():require(sha256(ROOT/rel)==h,'source changed after smoke')
        run.freeze(contract);shutil.copyfile(PARENT/'DAMAGE_PANEL.json',run.path/'DAMAGE_PANEL.json')
        write_new(run.path/'PANEL_REUSE.json',dict(original_panel_sha256=contract['damage_panel_sha256'],original_contract_sha256=panel['inference_contract_sha256'],
            new_contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),reuse_after_outcomes_seen=True,scope='development screen'))
        write_new(run.path/'EVALUATION_IMAGES.json',ids);run.update(phase='cache_verification',smoke_run_id=smoke_id or None)
        cache=FrozenCache(BASE/'trace',ids);run.update(phase='native_replay_two_arm_export')
        indices,actions,coverage,resources=export(run,cache);baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl');arms={}
        for arm in ('A','B'):
            run.update(phase=arm+'_fixed_pairs_and_complete_COCO');print('EVALUATING',arm,flush=True)
            arms[arm]=evaluate_arm(run,arm,cache,ids,indices,actions,panel,baseline);arms[arm]['coverage']=coverage[arm];gc.collect()
        cache.verify();again=inspect();require(again==(expected,source_contract,panel,environment),'configuration changed after evaluation')
        require(sha256(run.path/'DAMAGE_PANEL.json')==contract['damage_panel_sha256'],'panel bytes changed')
        import psutil
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss);require(peak<=4*1024**3,'CPU budget exceeded');run.check_budget()
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=1.5*1024**3,'artifact budget exceeded')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',peak_rss_bytes=peak,cache_images_verified_before_after=len(ids),
            native_zero_delta_replayed_rows=len(indices),new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,
            contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),checks=dict(inputs_sources_environment_unchanged=True,native_RLE_exact=True,
            all_nonmask_fields_and_order_equal=True,unselected_rows_identical=True,independent_bbox_segm_matching=True,same_bbox_match_identity=True,
            original_denominators_and_gate=True,panel_bytes_unchanged=True,old_E1_trigger_statistics_replayed=True)))
        selected=next((a for a in ('B','A') if arms[a]['status']=='passed'),None) if scope=='full' else None
        status=('passed' if selected else 'failed') if scope=='full' else 'smoke_passed'
        next_action=('Keep selected E1 candidate for full utility acceptance; do not auto-start' if selected else
                     'Prioritize evidence-supported selective coefficient repair; close analytic neighbor-ownership repulsion; no E2 training') if scope=='full' else 'Execute frozen full E1-R2'
        run.finish(dict(arms=arms,selected_candidate=selected,resources=resources,scientific_scope='outcome-exposed development screen'),
            status=status,scientific='E1_R2_development_'+status if scope=='full' else 'engineering_only',next_action=next_action)
        print('DECISION',status,json.dumps(arms),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('smoke','full'));p.add_argument('--smoke-run',default='');p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if a.preflight:
        expected,_,_,env=inspect();print(json.dumps(dict(status='passed',inputs=len(expected['files']),sources=len(expected['frozen_sources']),environment=env)))
    elif a.scope:execute(a.scope,a.smoke_run)
    else:p.error('scope or preflight required')
