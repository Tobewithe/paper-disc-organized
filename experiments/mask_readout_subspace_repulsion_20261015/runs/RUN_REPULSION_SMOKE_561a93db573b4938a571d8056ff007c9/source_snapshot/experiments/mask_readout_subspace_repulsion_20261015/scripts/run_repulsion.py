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
STUDY=ROOT/'experiments/mask_readout_subspace_repulsion_20261015'
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
from repulsion import update_coefficients

CODE=['experiments/mask_readout_subspace_repulsion_20261015/'+p for p in
      ('PROTOCOL.md','USER_PROPOSAL.md','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
       'scripts/repulsion.py','scripts/test_repulsion.py','scripts/run_repulsion.py','scripts/launch_repulsion.ps1')]
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


class RepulsionRun(RegisteredRun):
    def __init__(self,scope):
        super().__init__('REPULSION_'+scope.upper(),'E1_REPULSION',BASE,CODE,900 if scope=='smoke' else 3600)
        self.path=STUDY/'runs'/self.run_id;self.lock=STUDY/'.E1_REPULSION.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_SUBSPACE_REPULSION_20261015',scope=scope,
                         new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,method_id='E1-R1-v1')

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


def export(run,source):
    import torch
    source_indices=[];stats=Counter();by_index={};started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (run.path/'REPULSION_predictions.json').open('x',encoding='utf8') as out,(run.path/'RULE_ROWS.jsonl').open('x',encoding='utf8') as rules,(run.path/'IMAGE_METADATA.jsonl').open('x',encoding='utf8') as metadata,torch.inference_mode():
        out.write('[');count=0
        for number,image_id in enumerate(source.records,1):
            run.check_budget();cache,original=source.load(image_id)
            native_masks=decode_native(cache,device='cuda');native_rles=[encode_rle(m) for m in native_masks]
            require(native_rles==[r['segmentation'] for r in original],'native zero-delta RLE mismatch')
            del native_masks;torch.cuda.empty_cache()
            coefficients,proof=update_coefficients(cache,'cuda')
            masks=decode_native(cache,coefficients,device='cuda');rles=[encode_rle(m) for m in masks]
            rows=[dict(r,segmentation=m) for r,m in zip(original,rles)];compare_outputs(original,rows)
            for rank,(row,record,index) in enumerate(zip(rows,proof,cache['prediction_indices'])):
                unchanged=row['segmentation']==native_rles[rank]
                if not record['applied']:require(unchanged,'unchanged coefficient mask differs')
                if count:out.write(',')
                out.write(json.dumps(row,separators=(',',':'),allow_nan=False))
                record.update(image_id=image_id,global_native_index=index,method_index=count,mask_changed=not unchanged,
                    raw_id=cache['raw_ids'][rank],neighbor_raw_id=None if record['neighbor_rank']<0 else cache['raw_ids'][record['neighbor_rank']])
                rules.write(json.dumps(record,allow_nan=False)+'\n')
                source_indices.append(index);by_index[index]=dict(applied=record['applied'],neighbor_rank=record['neighbor_rank'],omega_pixels=record['omega_pixels'])
                count+=1;stats['outputs']+=1;stats['selected_coefficients']+=int(record['applied']);stats['changed_masks']+=int(not unchanged)
                stats['no_eligible_neighbor']+=int(record['neighbor_rank']<0)
                stats['empty_region_with_neighbor']+=int(record['neighbor_rank']>=0 and record['omega_pixels']==0)
            metadata.write(json.dumps(dict(image_id=image_id,cache_relative_path=source.records[image_id]['cache'],
                cache_sha256=source.records[image_id]['cache_sha256'],prediction_indices=cache['prediction_indices'],
                boxes=cache['boxes_input'].tolist(),scores=cache['scores'].tolist(),classes=cache['classes'].tolist(),
                raw_ids=cache['raw_ids'],input_hw=cache['input_hw'],proto_hw=list(cache['proto'].shape[-2:])),allow_nan=False)+'\n')
            stats['images']+=1;stats['images_with_updates']+=int(any(r['applied'] for r in proof))
            del cache,coefficients,masks,rles,rows,native_rles,proof,original;gc.collect();torch.cuda.empty_cache()
            if number%50==0 or number==len(source.records):print('EXPORT',number,len(source.records),dict(stats),flush=True)
    with (run.path/'REPULSION_predictions.json').open('a',encoding='utf8') as out:out.write(']\n')
    require(source_indices==sorted(source_indices),'source order changed')
    expected=read_json(PARENT/'B_ROW_MAPPING.json')['original_global_indices']
    if run.meta['scope']=='full':require(source_indices==expected and len(source_indices)==185621,'full output denominator changed')
    stats.update(export_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    require(stats['cuda_peak_reserved_bytes']<=6.5*1024**3,'GPU budget exceeded')
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=source.manifest['predictions_sha256'],original_global_indices=source_indices))
    return source_indices,by_index,dict(stats)


def fixed_pairs(run,source,ids,indices,actions):
    from pycocotools.coco import COCO
    coco=COCO(str(GT));predictions=read_json(run.path/'REPULSION_predictions.json');position={g:i for i,g in enumerate(indices)}
    effects=[]
    for old in read_jsonl(PARENT/'B_FIXED_PAIR_EFFECTS.jsonl'):
        if old['image_id'] not in set(ids):continue
        index=int(old['baseline_output_detection_id'].rsplit(':',1)[1]);gt=coco.annToRLE(coco.anns[old['annotation_id']])
        before=full_rle_iou(source.predictions[index]['segmentation'],gt)
        after=full_rle_iou(predictions[position[index]]['segmentation'],gt)
        require(abs(before-old['baseline_iou'])<1e-12,'baseline fixed pair changed')
        effects.append(dict(image_id=old['image_id'],annotation_id=old['annotation_id'],population=old['population'],
            baseline_output_detection_id=old['baseline_output_detection_id'],baseline_iou=before,method_iou=after,
            delta_iou=after-before,same_output_reaches_075=after>=.75,**actions[index]))
    jl(run.path/'FIXED_PAIR_EFFECTS.jsonl',effects)
    del coco,predictions;gc.collect();return effects


def execute(scope,smoke_id):
    with RepulsionRun(scope) as run:
        expected,source_contract,panel,environment=inspect()
        original_ids=read_json(PARENT/'EVALUATION_IMAGES.json');ids=original_ids if scope=='full' else original_ids[:8]
        require(len(original_ids)==1314 and len(panel['sample'])==1024,'original denominator drift')
        contract=dict(method_id='E1-R1-v1',scope=scope,image_ids=ids,coefficient_origin='native c0',source_contract=source_contract,
            original_contract_sha256=sha256(PARENT/'EVAL_CONTRACT.json'),native_profile_sha256=source_contract['native_profile_sha256'],
            method_parameters=dict(neighbor='Top1 higher priority same-class different raw',box_iou_strict_min=.1,core_side_fraction=.5,
                neighbor_side='strictly closer to neighbor center',epsilon=1e-8,strength=1.,positive_correlation_only=True,simultaneous_native_neighbors=True),
            input_hashes=expected,environment=environment,new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,
            original_gates=source_contract['gates'],bootstrap=source_contract['bootstrap'],damage_panel_sha256=sha256(PARENT/'DAMAGE_PANEL.json'),
            damage_panel_reused=True,fresh_independent_95_coverage_guarantee=False,wall_seconds=run.wall_seconds)
        if scope=='full':
            smoke=STUDY/'runs'/smoke_id;require(read_json(smoke/'run.json')['execution_status']=='completed','successful smoke required')
            old=read_json(smoke/'EVAL_CONTRACT.json');require(old['input_hashes']==expected and old['environment']==environment,'smoke configuration differs')
            for rel,h in read_json(smoke/'SOURCE_MANIFEST.json')['files'].items():require(sha256(ROOT/rel)==h,'code changed after smoke')
        run.freeze(contract);shutil.copyfile(PARENT/'DAMAGE_PANEL.json',run.path/'DAMAGE_PANEL.json')
        write_new(run.path/'PANEL_REUSE.json',dict(original_panel_sha256=contract['damage_panel_sha256'],
            original_contract_sha256=panel['inference_contract_sha256'],new_contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),
            reuse_after_outcomes_seen=True,scope='nominal original-formula development screen'))
        write_new(run.path/'EVALUATION_IMAGES.json',ids);run.update(phase='cache_verification',smoke_run_id=smoke_id or None)
        cache=FrozenCache(BASE/'trace',ids);run.update(phase='native_replay_and_analytic_export')
        indices,actions,coverage=export(run,cache);effects=fixed_pairs(run,cache,ids,indices,actions)
        run.update(phase='complete_prediction_COCO_evaluation');print('COCO_MATCHING',len(ids),len(indices),flush=True)
        with (run.path/'evaluation.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
            ledger,ap=evaluate_ledger(GT,read_json(run.path/'REPULSION_predictions.json'),ids,sha256(run.path/'REPULSION_predictions.json'))
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        wanted=validate_ledger(ledger,baseline,ids,indices,cache.manifest['predictions_sha256'])
        jl(run.path/'METHOD_GT_LEDGER.jsonl',ledger);jl(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl',wanted.values())
        if scope=='full':
            require(len(effects)==2925 and sum(r['population']=='T' for r in effects)==655,'fixed-pair denominator drift')
            target=bootstrap(image_effects(effects));overall=bootstrap(image_effects(effects,target_only=False))
            require(target['images']==257 and overall['images']==508,'image denominator drift')
            damage=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in set(panel['image_ids'])])
            damage['coverage_interpretation']='nominal bound on reused outcome-exposed panel, not fresh independent confidence'
            status,gates=adjudicate(target,overall,damage);keyed={(r['image_id'],r['annotation_id']):r for r in ledger}
            targets=[r for r in effects if r['population']=='T']
            coverage.update(dev_selected=sum(r['applied'] for r in effects),dev_total=2925,target_selected=sum(r['applied'] for r in targets),target_total=655,
                target_task_repaired=sum(keyed[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in targets),
                target_fixed_detection_reaches075=sum(r['same_output_reaches_075'] for r in targets))
            summary=dict(gates=gates,target=target,all_bbox75=overall,damage=damage,coverage=coverage,secondary_subset_AP=ap,
                         scientific_scope='reused development data; E1-R1-v1 only; no fresh independent confirmation')
            scientific='E1_development_'+status
            next_action='Freeze experiment 2 selective directional adapter protocol, correcting norm/zero-boundary claims; no automatic training'
        else:
            status='smoke_passed';scientific='engineering_only';next_action='Execute authorized frozen full E1 evaluation'
            summary=dict(coverage=coverage,fixed_pairs=len(effects),GT=len(ledger),scientific_scope='engineering_only')
        run.check_budget();cache.verify()
        again,again_source,again_panel,again_env=inspect()
        require((again,again_source,again_panel,again_env)==(expected,source_contract,panel,environment),'post-run configuration changed')
        require(sha256(run.path/'DAMAGE_PANEL.json')==contract['damage_panel_sha256'],'panel mutated')
        import psutil
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
        require(peak<=4*1024**3,'CPU memory budget exceeded');run.check_budget()
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=1024**3,'artifact budget exceeded')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',peak_rss_bytes=peak,cache_images_verified_before_after=len(ids),
            native_zero_delta_replayed_rows=len(indices),new_training_steps=0,base_model_forward_calls=0,adapter_forward_calls=0,
            contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),checks=dict(inputs_sources_environment_unchanged=True,
            native_RLE_exact=True,all_nonmask_fields_and_order_equal=True,unchanged_coefficients_exact_mask_identity=True,
            independent_bbox_segm_matching=True,same_bbox_match_identity=True,original_denominators_and_gate=True,panel_bytes_unchanged=True)))
        run.finish(summary,status=status,scientific=scientific,next_action=next_action)
        print('DECISION',json.dumps(dict(status=status,summary=summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('smoke','full'));parser.add_argument('--smoke-run',default='');parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args()
    if a.preflight:
        expected,_,_,env=inspect();print(json.dumps(dict(status='passed',inputs=len(expected['files']),sources=len(expected['frozen_sources']),environment=env)))
    elif a.scope:execute(a.scope,a.smoke_run)
    else:parser.error('scope or preflight required')
