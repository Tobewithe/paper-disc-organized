"""Three registered finite stages: contract/panel, original-input F, fixed-checkpoint evaluation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import contextlib
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

from rfc_runtime import *
from rfc_model import RegionFAdapter, parameter_receipt
from rfc_inputs import spatial_indices
from rfc_controls import native_area_gate, gated_delta, mix_native_rows, match_functional_amplitude
from rfc_training import Sidecars
from rfc_eval_stats import GROUPS, summarize
from posttrain_audit import read_jsonl, population, compare_outputs
from adapter_data import FrozenCache
from s1_geometry import iter_prediction_array, full_rle_iou
from native_output_bridge import decode_native, encode_rle
from damage_acceptance import select_damage_panel, score_damage

GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
GPRED=ROOT/'experiments/mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d/predictions_mixed.json'
GPRED_SHA='15d8cc2884777854a7120099e4ee218b8750ee59f63d2b3fcc4571fa2a191590'


class EvalRun(Run):
    def freeze(self,contract):
        context=read_json(STUDY/'EVALUATION_SOURCE.json'); self.sources=context['files']
        for i,(rel,digest) in enumerate(self.sources.items()):
            require(sha256(ROOT/rel)==digest,'evaluation source drift: '+rel)
            target=self.path/'source_snapshot'/(f'{i:03d}_'+Path(rel).name)
            target.parent.mkdir(exist_ok=True); shutil.copyfile(ROOT/rel,target)
        shutil.copyfile(STUDY/'EVALUATION_PROTOCOL.md',self.path/'PROTOCOL.md')
        shutil.copyfile(STUDY/'EXECUTION_DIRTY.patch',self.path/'dirty.patch')
        write_new(self.path/'SOURCE_MANIFEST.json',context)
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',code_commit=context['code_commit'],
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))


def training_audit(data):
    pair=read_json(STUDY/'TRAINING_PAIR_FREEZE.json'); receipts={}
    for arm in ('A','Q'):
        path=pointer('TRAIN_'+arm); info=pair[arm]; summary=read_json(path/'SUMMARY.json')
        require(path.name==info['run_id'],'checkpoint run changed')
        require(sha256(path/'last.pt')==info['checkpoint_sha256']==summary['checkpoint_sha256'],'checkpoint changed')
        for item in read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']:
            require(sha256(path/item['relative_path'])==item['sha256'],'training artifact changed')
        require(read_json(path/'SOURCE_MANIFEST.json')['files']==read_json(STUDY/'EXECUTION_SOURCE.json')['files'],'trained source manifest changed')
        require(summary['optimizer_steps']==1500 and summary['images_seen']==6000 and summary['epochs']==3,'training budget changed')
        require(len(summary['parameters_with_nonzero_gradient'])==17,'ineffective parameter path')
        for name in ('IMAGE_ORDERS.json','TRAINING_JOIN.json','initial.pt'):
            reference=(B if name=='initial.pt' else G)/name
            require(sha256(path/name)==sha256(reference),'actual original training input changed: '+name)
        contract=read_json(path/'EVAL_CONTRACT.json')
        require_no_drift(contract['environment'],data['environment'],'training vs evaluation environment')
        require_no_drift(contract['optimizer'],data['spec']['training']['optimizer'],'original optimizer')
        model=RegionFAdapter(arm)
        saved=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        require(saved['arm']==arm and saved['completed_epochs']==3 and saved['optimizer_steps']==1500,'not final checkpoint')
        model.load_state_dict(saved['state_dict'],strict=True)
        require(tensor_state_hash(model)==summary['final_state_sha256'],'checkpoint tensors do not match training')
        require(parameter_receipt(model)['parameters']==167872,'capacity changed')
        receipts[arm]={'run_id':path.name,'checkpoint_sha256':sha256(path/'last.pt'),
                       'tensor_state_sha256':tensor_state_hash(model),'optimizer_steps':1500,
                       'training_contract_sha256':sha256(path/'EVAL_CONTRACT.json')}
        del model,saved
    require(sha256(G/'last.pt')==data['spec']['inference']['G_checkpoint_sha256'],'G checkpoint changed')
    require(sha256(GPRED)==GPRED_SHA,'G full predictions changed')
    require(sha256(GT)==data['profile']['annotations_sha256'],'validation GT changed')
    return receipts


def freeze_stage():
    require(not list((STUDY/'runs').glob('RUN_RFC_EVAL_FREEZE_*/DAMAGE_PANEL.json')),'damage panel already drawn; never redraw')
    with EvalRun('EVAL_FREEZE',pointer('TRAIN_Q'),1800) as run:
        data=inspect_inputs(); resources=initial_resources(); checkpoints=training_audit(data)
        contract={'stage':'RFC-v1 evaluation','checkpoints':checkpoints,'G_predictions_sha256':GPRED_SHA,
                  'design_sha256':sha256(STUDY/'DESIGN_SPEC.json'),
                  'source_manifest_sha256':sha256(STUDY/'EVALUATION_SOURCE.json'),
                  'training_pair_freeze_sha256':sha256(STUDY/'TRAINING_PAIR_FREEZE.json'),
                  'ledger_sha256':sha256(BASE/'ledger/PER_GT_FAILURE.jsonl'),
                  'native_manifest_sha256':sha256(BASE/'trace/TRACE_MANIFEST.json'),
                  'data_split_sha256':sha256(CACHE/'SPLIT_CONTRACT.json'),
                  'annotations_sha256':sha256(GT),'environment':data['environment'],
                  'dev_ids':data['split']['dev_ids'],'additional_optimizer_steps':0,
                  'scientific_spec':data['spec'],'resources':resources}
        run.freeze(contract)
        with (run.path/'evaluation_tests.log').open('x',encoding='utf8') as log:
            tested=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('test_evaluation.py'))],stdout=log,stderr=subprocess.STDOUT)
        require(tested.returncode==0,'evaluation wiring tests failed')
        write_new(run.path/'TRAINING_COMPLETION_VERIFICATION.json',{'status':'passed','checkpoints':checkpoints,
                  'all_training_artifacts_hashed':True,'original_orders_GT_joins_initialization_exact':True})
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        panel=select_damage_panel(baseline,contract['ledger_sha256'],checkpoints['A']['checkpoint_sha256'],
                                  sha256(run.path/'EVAL_CONTRACT.json'),1024)
        require(panel['population_size']==22957 and panel['sample_size']==1024,'wrong damage population')
        write_new(run.path/'DAMAGE_PANEL.json',panel)
        ids=sorted(set(data['split']['dev_ids'])|set(panel['image_ids']))
        require(len(ids)<=1536,'evaluation image budget changed')
        write_new(run.path/'EVALUATION_IMAGES.json',ids)
        audit(run,data,paired_training_audit=True,source_and_checkpoints_frozen_before_single_random_draw=True)
        run.finish({'images':len(ids),'damage_sample':1024,'damage_images':len(panel['image_ids']),
                    'checkpoints':checkpoints},scientific='evaluation_inputs_frozen_not_evaluated',
                   next_action='original batch replay for fixed evaluation images')
    set_pointer('EVAL_FREEZE',run)


def input_stage():
    from rfc_native_replay import export_split
    parent=pointer('EVAL_FREEZE'); training_export=pointer('INPUT_FULL')
    spent=read_json(training_export/'run.json')['elapsed_seconds']
    require(not list((STUDY/'runs').glob('RUN_RFC_EVAL_INPUT_*/SUMMARY.json')),'evaluation F already exported')
    with EvalRun('EVAL_INPUT',parent,10800-spent) as run:
        data=inspect_inputs(); resources=initial_resources()
        ids=read_json(parent/'EVALUATION_IMAGES.json')
        run.freeze({'freeze_run':parent.name,'inference_contract_sha256':sha256(parent/'EVAL_CONTRACT.json'),
                    'damage_panel_sha256':sha256(parent/'DAMAGE_PANEL.json'),
                    'evaluation_images_sha256':sha256(parent/'EVALUATION_IMAGES.json'),
                    'prior_formal_F_export_seconds':spent,'remaining_F_seconds':10800-spent,
                    'scientific_evaluation':False,'resources':resources,'environment':data['environment']})
        run.F_existing=all_F_bytes(); run.F_written=0
        result=export_split(run,data,'dev',ids)
        require(result['context_images']+2000<=7000,'formal base forward budget exceeded')
        write_new(run.path/'REGION_MANIFEST.json',{'dev':result})
        audit(run,data,tensor_identity_exact=True,zero_delta_RLE_exact=True,original_native_batch_members=True)
        run.finish({'images':len(ids),'context_images':result['context_images'],
                    'formal_base_forwards_total':2000+result['context_images'],
                    'prior_F_seconds':spent},scientific='evaluation_input_replay_verified',next_action='fixed checkpoint evaluation')
    set_pointer('EVAL_INPUT',run)


def jsonl(path,rows):
    with path.open('x',encoding='utf8') as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')


class PredictionWriter:
    def __init__(self,path):
        self.stream=path.open('x',encoding='utf8'); self.stream.write('['); self.indices=[]
    def append(self,rows,indices):
        require(len(rows)==len(indices),'row mapping length changed')
        for row,index in zip(rows,indices):
            if self.indices:self.stream.write(',')
            self.stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)); self.indices.append(index)
    def close(self):
        self.stream.write(']\n'); self.stream.close()
        require(self.indices==sorted(set(self.indices)),'prediction order or uniqueness changed')


def prediction_stage(run,data,freeze,input_run,source,baseline,ids):
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    dev=set(data['split']['dev_ids']); regions=Sidecars(input_run,'dev')
    require(set(regions.records)==set(ids),'F evaluation coverage mismatch')
    g_rows={i:r for i,r in enumerate(iter_prediction_array(GPRED)) if i in source.predictions}
    require(set(g_rows)==set(source.predictions),'G row coverage changed')
    coco=COCO(str(GT)); pairs=defaultdict(list)
    for r in baseline:
        if r['image_id'] in dev and not r['is_ignored'] and r['bbox75_match_output_id'] is not None:
            pairs[r['image_id']].append(r)
    models={}
    for arm in ('A','Q'):
        model=RegionFAdapter(arm).cuda()
        model.load_state_dict(torch.load(pointer('TRAIN_'+arm)/'last.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        models[arm]=model.eval()
    writers={g:PredictionWriter(run.path/(g+'_predictions.json')) for g in GROUPS}
    effects=[]; layout=[]; amplitudes=[]; image_receipts=[]
    coefficient_dir=run.path/'coefficients'; coefficient_dir.mkdir()
    with torch.inference_mode():
        for number,(iid,record) in enumerate(source.records.items(),1):
            run.check_budget(); cache,native=source.load(iid)
            gate=native_area_gate(torch.tensor([int(masks.area(r['segmentation'])) for r in native],dtype=torch.int64))
            inputs=regions.get(cache,'eval:0',gate)
            delta={g:gated_delta(models[g](inputs,'eval:0'),gate) for g in ('A','Q')}
            if iid in dev:
                delta['J']=gated_delta(models['A'](inputs,'eval:0',intervention='J'),gate)
                for other in ('Q','J'):
                    first,second,receipt=match_functional_amplitude(cache,delta['A'],delta[other],gate)
                    delta['Aeq'+other]=first; delta[other+'eqA']=second
                    amplitudes.append({'image_id':iid,'pair':'A-'+other,**receipt})
                for rank,enabled in enumerate(gate.tolist()):
                    if enabled:
                        permutation=spatial_indices(iid,rank,False); grid=inputs.regions[rank]
                        layout.append({'image_id':iid,'output_rank':rank,'gate_on':True,
                          'fixed_positions':sum(i==j for i,j in enumerate(permutation)),
                          'constant_F':bool(torch.equal(grid,grid[:,0:1,0:1].expand_as(grid))),
                          'permutation_changes_F':not torch.equal(grid.flatten(1),grid.flatten(1)[:,permutation])})
            torch.save({'image_id':iid,'prediction_indices':cache['prediction_indices'],'gate':gate,
                        'delta':{g:d.cpu() for g,d in delta.items()}},coefficient_dir/f'{iid}.pt')
            outputs={'N':native,'G':[g_rows[j] for j in cache['prediction_indices']]}
            compare_outputs(native,outputs['G'])
            require(all(a['segmentation']==b['segmentation'] for a,b,on in zip(native,outputs['G'],gate.tolist()) if not on),'G gate-off RLE changed')
            for g,d in delta.items():
                decoded=decode_native(cache,cache['c0'].to(d)+d)
                outputs[g]=mix_native_rows(native,[encode_rle(decoded[j]) for j in range(len(native))],gate)
                compare_outputs(native,outputs[g]); del decoded
            for g,rows in outputs.items(): writers[g].append(rows,cache['prediction_indices'])
            if iid in dev:
                mapping={index:rank for rank,index in enumerate(cache['prediction_indices'])}
                for row in pairs[iid]:
                    index=int(row['bbox75_match_output_id'].rsplit(':',1)[1]); rank=mapping[index]
                    gt=coco.annToRLE(coco.anns[row['annotation_id']])
                    values={g:full_rle_iou(outputs[g][rank]['segmentation'],gt) for g in GROUPS}
                    require(abs(values['N']-row['mask_iou_of_bbox_match'])<1e-12,'native fixed-pair IoU drift')
                    effect={'image_id':iid,'annotation_id':row['annotation_id'],'population':population(row),
                            'baseline_output_detection_id':row['bbox75_match_output_id'],
                            'native_gate':bool(gate[rank]),'IoU':values}
                    effect.update({g+'_delta':v-values['N'] for g,v in values.items()}); effects.append(effect)
            image_receipts.append({'image_id':iid,'rows':len(native),'gate_on':int(gate.sum()),
                                   'nonmask_identity_exact':True,'gate_off_RLE_exact':True})
            if number%25==0 or number==len(source.records):
                run.update(phase='fixed_checkpoint_predictions',images_done=number,total_images=len(ids))
                print('RFC_EVAL_IMAGES',number,len(ids),flush=True)
            del cache,native,outputs,inputs,delta
    for g,w in writers.items():
        w.close(); write_new(run.path/(g+'_ROW_MAPPING.json'),{'source_prediction_sha256':source.manifest['predictions_sha256'],
                                                             'original_global_indices':w.indices})
    effects.sort(key=lambda r:(r['image_id'],r['annotation_id']))
    jsonl(run.path/'FIXED_PAIR_EFFECTS.jsonl',effects)
    jsonl(run.path/'AMPLITUDE_CONTROLS.jsonl',amplitudes); jsonl(run.path/'LAYOUT_RECEIPTS.jsonl',layout)
    write_new(run.path/'OUTPUT_IDENTITY.json',image_receipts)
    regions.verify(); source.verify()
    del models,g_rows,coco; gc.collect(); torch.cuda.empty_cache()
    return effects


def ledger_stage(run,data,source,baseline,ids,panel):
    sys.path.insert(0,str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
    from build_native_ledger import evaluate_ledger
    baseline_keys={(r['image_id'],r['annotation_id']):r for r in baseline}
    damage={}; receipts={}; dev=set(data['split']['dev_ids']); panel_images=set(panel['image_ids'])
    for g in GROUPS:
        selected=ids if g in ('N','G','A','Q') else sorted(dev)
        path=run.path/(g+'_predictions.json'); indices=read_json(run.path/(g+'_ROW_MAPPING.json'))['original_global_indices']
        run.update(phase='COCO_complete_image_matching',arm=g); run.check_budget()
        with (run.path/(g+'_ledger.log')).open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
            ledger,ap=evaluate_ledger(GT,read_json(path),selected,sha256(path))
        expected={k for k,r in baseline_keys.items() if r['image_id'] in set(selected)}
        require({(r['image_id'],r['annotation_id']) for r in ledger}==expected,'GT coverage changed')
        for row in ledger:
            old=baseline_keys[(row['image_id'],row['annotation_id'])]
            require(row['is_ignored']==old['is_ignored'] and row['gt_category_id']==old['gt_category_id'],'GT policy changed')
            oid=row['bbox75_match_output_id']
            mapped=None if oid is None else source.manifest['predictions_sha256']+':'+str(indices[int(oid.rsplit(':',1)[1])])
            require(mapped==old['bbox75_match_output_id'],'bbox matched detection changed')
            if g=='N': require(row['task_mask75']==old['task_mask75'],'native mask matching changed')
        if g in ('N','G','A','Q'):
            damage[g]=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in panel_images])
        transitions=defaultdict(int)
        for row in ledger:
            if row['image_id'] not in dev or row['is_ignored']: continue
            old=baseline_keys[(row['image_id'],row['annotation_id'])]
            transitions[old['task_mask75']+'->'+row['task_mask75']]+=1
        receipts[g]={'images':len(selected),'GT':len(ledger),'predictions':len(indices),
                     'bbox_identity_exact':True,'dev_task_transitions':dict(transitions),
                     'incidental_AP_from_ledger_not_gate':ap,'AP_image_scope':'dev_plus_new_damage' if g in ('N','G','A','Q') else 'dev512',
                     'prediction_sha256':sha256(path)}
        jsonl(run.path/(g+'_GT_LEDGER.jsonl'),ledger); write_new(run.path/(g+'_LEDGER_RECEIPT.json'),receipts[g])
        print('RFC_LEDGER',g,json.dumps(damage.get(g,{})),flush=True)
        del ledger; gc.collect(); run.check_budget()
    return damage,receipts


def evaluate_stage():
    freeze=pointer('EVAL_FREEZE'); input_run=pointer('EVAL_INPUT')
    require(not list((STUDY/'runs').glob('RUN_RFC_EVAL_FULL_*/SUMMARY.json')),'scientific evaluation already completed')
    with EvalRun('EVAL_FULL',input_run,14400) as run:
        data=inspect_inputs(); resources=initial_resources(); training_audit(data)
        ids=read_json(freeze/'EVALUATION_IMAGES.json'); panel=read_json(freeze/'DAMAGE_PANEL.json')
        frozen=read_json(freeze/'EVAL_CONTRACT.json')
        require(frozen['source_manifest_sha256']==sha256(STUDY/'EVALUATION_SOURCE.json'),'source changed since panel draw')
        require(panel['inference_contract_sha256']==sha256(freeze/'EVAL_CONTRACT.json'),'panel/contract identity changed')
        require(panel['baseline_ledger_sha256']==sha256(BASE/'ledger/PER_GT_FAILURE.jsonl'),'baseline ledger changed')
        run.freeze({'freeze_run':freeze.name,'inference_contract_sha256':sha256(freeze/'EVAL_CONTRACT.json'),
                    'damage_panel_sha256':sha256(freeze/'DAMAGE_PANEL.json'),
                    'region_input_run':input_run.name,'region_manifest_sha256':sha256(input_run/'REGION_MANIFEST.json'),
                    'evaluation_ids':ids,'additional_optimizer_steps':0,'resources':resources,'environment':data['environment']})
        shutil.copyfile(freeze/'DAMAGE_PANEL.json',run.path/'DAMAGE_PANEL.json')
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        source=FrozenCache(BASE/'trace',ids)
        effects=prediction_stage(run,data,freeze,input_run,source,baseline,ids)
        damage,ledgers=ledger_stage(run,data,source,baseline,ids,panel)
        summary=summarize(effects,damage); summary['ledger_receipts']=ledgers
        source.verify(); training_audit(data)
        audit(run,data,original_frozen_checkpoints_unchanged=True,full_nonmask_and_bbox_identity=True,
              fixed_pair_denominators_and_G_endpoints_exact=True,GT_independent_forward_and_controls=True,
              fresh_single_damage_draw=True,original_decoder_and_environment=True,additional_optimizer_steps=0)
        run.finish(summary,status=summary['utility_status'],scientific='RFC_v1_utility_'+summary['utility_status'],
                   next_action=summary['next_action'])
        print('RFC_FINAL',json.dumps(summary),flush=True)
    set_pointer('EVAL_FULL',run)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',required=True,choices=('freeze','input','evaluate'))
    stage=parser.parse_args().stage
    {'freeze':freeze_stage,'input':input_stage,'evaluate':evaluate_stage}[stage]()
