"""Additive S2 execution; original AMT training freeze is immutable."""
import contextlib
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import common as amt
from common import ROOT, STUDY, CACHE, read_json, write_new, sha256, require, now, save
from posttrain_run import RegisteredRun, tensor_state_hash
from posttrain_audit import read_jsonl, population, compare_outputs
from run_s2_evaluation import BASE, GT, image_effects, bootstrap
from s1_geometry import iter_prediction_array, full_rle_iou
sys.path.append(str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
sys.path.append(str(ROOT/'experiments/mask_readout_large_mask_gate_20261015/scripts'))
from build_native_ledger import evaluate_ledger
from damage_acceptance import select_damage_panel, score_damage
from area_gate import mix_row, adjudicate

TRAIN=STUDY/'runs/RUN_AMT_TRAIN_e10144e6d3d94803824aef1c636d831f'
OLD=ROOT/'experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'
S3=ROOT/'experiments/mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d'
NATIVE=BASE/'trace/native/predictions.json'
CONTROL=S3/'predictions_mixed.json'
CHECKPOINT=TRAIN/'last.pt'
CHECKPOINT_SHA='b39df4f2720bf60f3ffe64f79ce426d2dc55ee28188c0068e6b1aa3061cf79b7'
LOCAL=('s2_common.py','evaluate_s2.py','verify_s2.py','test_s2.py','s2_pipeline.py','launch_s2.ps1')
CODE=list(dict.fromkeys(amt.CODE+[
    'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
    'experiments/mask_readout_large_mask_gate_20261015/scripts/area_gate.py',
    'experiments/mask_readout_mechanism_v2_20261012/scripts/test_s2_evaluation.py',
    'experiments/mask_readout_large_mask_gate_20261015/scripts/test_area_gate.py',
    (STUDY/'EVALUATION_EXECUTION.md').relative_to(ROOT).as_posix(),
    (STUDY/'EVALUATION_SOURCE_CONTEXT.json').relative_to(ROOT).as_posix(),
    (STUDY/'EVALUATION_SOURCE_DIRTY.patch').relative_to(ROOT).as_posix()
]+[(STUDY/'scripts'/n).relative_to(ROOT).as_posix() for n in LOCAL]))


def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')


def inputs():
    data=amt.inspect_inputs()
    files=dict(data['inputs_sha256'])
    paths=[CHECKPOINT,TRAIN/'EVAL_CONTRACT.json',TRAIN/'SUMMARY.json',
        TRAIN/'CONFIG_AUDIT.json',TRAIN/'INDEPENDENT_CONFIG_AUDIT.json',TRAIN/'INDEPENDENT_AMT_AUDIT.json',
        TRAIN/'IMAGE_ORDERS.json',TRAIN/'TRAINING_JOIN.json',STUDY/'FREEZE_MANIFEST.json',
        BASE/'ledger/PER_GT_FAILURE.jsonl',BASE/'trace/TRACE_MANIFEST.json',NATIVE,GT,
        OLD/'FIXED_PAIR_EFFECTS.jsonl',OLD/'SUMMARY.json',OLD/'CONFIG_AUDIT.json',
        S3/'CONFIG_AUDIT.json',S3/'SUMMARY.json',S3/'ROW_MAPPING.json',CONTROL]
    files.update({p.relative_to(ROOT).as_posix():sha256(p) for p in paths})
    require(sha256(CHECKPOINT)==CHECKPOINT_SHA,'wrong final AMT checkpoint')
    for name in ('CONFIG_AUDIT.json','INDEPENDENT_CONFIG_AUDIT.json','INDEPENDENT_AMT_AUDIT.json'):
        require(read_json(TRAIN/name)['status']=='passed','training audit failed: '+name)
    require(read_json(TRAIN/'run.json')['execution_status']=='completed','training incomplete')
    require(read_json(TRAIN/'SUMMARY.json')['checkpoint_sha256']==CHECKPOINT_SHA,'training checkpoint binding')
    for p in (OLD,S3):
        require(read_json(p/'run.json')['execution_status']=='completed','control Run incomplete')
        require(read_json(p/'CONFIG_AUDIT.json')['status']=='passed','control configuration unverified')
    require(sha256(CONTROL)=='15d8cc2884777854a7120099e4ee218b8750ee59f63d2b3fcc4571fa2a191590','wrong old G full predictions')
    require(sha256(NATIVE)=='a8c3e7d0af9db4684653c9b7b412f734d6105dc1e1c98a20c0782cf8d2fd3151','native JSON drift')
    require(sha256(BASE/'trace/TRACE_MANIFEST.json')=='c4c2b11eb0f6c9628c7ab3494b74b08d6e0835b46426413fbdafb2b2119abfab','S0 trace drift')
    require(sha256(BASE/'ledger/PER_GT_FAILURE.jsonl')=='8e563c9a3f32e6a3289b6649685c8160f557d74f19b35209b9f712f79c2e5884','S0 ledger drift')
    frozen=read_json(STUDY/'S2_SOURCE_FREEZE.json')
    require({p:sha256(ROOT/p) for p in CODE}==frozen['files'],'S2 source freeze drift')
    data['inputs_sha256']=files
    return data


def check_contract(data):
    c=read_json(STUDY/'EVALUATION_CONTRACT.json')
    require(c['inputs_sha256']==data['inputs_sha256'],'evaluation inputs drift')
    require(c['environment']==data['environment'],'evaluation environment drift')
    require(c['external_evaluator_sources']==data['external_evaluator_sources'],'evaluator drift')
    require(c['source_freeze_sha256']==sha256(STUDY/'S2_SOURCE_FREEZE.json'),'source manifest drift')
    panel=read_json(STUDY/'S2_DAMAGE_PANEL.json')
    require(panel['checkpoint_sha256']==CHECKPOINT_SHA,'panel checkpoint drift')
    require(panel['inference_contract_sha256']==sha256(STUDY/'EVALUATION_CONTRACT.json'),'panel contract drift')
    receipt=read_json(STUDY/'S2_PANEL_RECEIPT.json')
    require(receipt['panel_sha256']==sha256(STUDY/'S2_DAMAGE_PANEL.json'),'panel bytes drift')
    return c,panel


def validate_ledger(rows,baseline,ids,indices,source_sha,native=False):
    wanted={(r['image_id'],r['annotation_id']):r for r in baseline if r['image_id'] in set(ids)}
    require(len(rows)==len(wanted) and {(r['image_id'],r['annotation_id']) for r in rows}==set(wanted),'GT denominator drift')
    def remap(oid):return None if oid is None else source_sha+':'+str(indices[int(oid.rsplit(':',1)[1])])
    for row in rows:
        old=wanted[row['image_id'],row['annotation_id']]
        for k in ('is_ignored','gt_category_id','gt_area','task_box75'):
            require(row[k]==old[k],'GT/bbox drift '+k)
        require(remap(row['bbox75_match_output_id'])==old['bbox75_match_output_id'],'bbox detection identity drift')
        if native:
            for k in ('task_mask75','task_mask50'):require(row[k]==old[k],'native ledger replay drift')
            require(remap(row['segm75_match_output_id'])==old['segm75_match_output_id'],'native segm detection identity drift')
    return wanted


def effects(rows,indices,template,coco):
    positions={i:j for j,i in enumerate(indices)}
    out=[]
    for old in template:
        j=positions[int(old['baseline_output_detection_id'].rsplit(':',1)[1])]
        require(rows[j]['image_id']==old['image_id'],'fixed pair image drift')
        actual=full_rle_iou(rows[j]['segmentation'],coco.annToRLE(coco.anns[old['annotation_id']]))
        out.append({k:old[k] for k in ('image_id','annotation_id','population','baseline_output_detection_id','baseline_iou')})
        out[-1].update(method_iou=actual,delta_iou=actual-old['baseline_iou'],same_output_reaches_075=actual>=.75)
    return out


def result(effect_rows,ledger,panel,baseline):
    key={(r['image_id'],r['annotation_id']):r for r in ledger}
    t=[r for r in effect_rows if r['population']=='T']
    return dict(target=bootstrap(image_effects(effect_rows)),
        all_bbox75=bootstrap(image_effects(effect_rows,target_only=False)),
        damage=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in set(panel['image_ids'])]),
        target_GT=len(t),all_bbox75_GT=len(effect_rows),
        target_fixed_detection_repaired=sum(r['same_output_reaches_075'] for r in t),
        target_task_repaired=sum(key[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in t))


def decide(primary,contrast):
    status,gates=adjudicate(primary['target'],primary['all_bbox75'],primary['damage'])
    gates['improvement_over_original_G']=contrast['ci95'][0]>0
    if not gates['improvement_over_original_G']:status='failed'
    return status,gates


class EvalRun(RegisteredRun):
    def __init__(self,scope):
        super().__init__('AMT_S2_'+scope.upper(),'AMT_S2',TRAIN,CODE,600 if scope=='smoke' else 3600)
        self.path=STUDY/'runs'/self.run_id
        self.lock=STUDY/'.AMT_S2.lock'
        self.meta.update(study_id='STUDY_MASK_ATTAINABLE_TEACHER_20261016',scope=scope,
            new_training_steps=0,base_model_forward_calls=0,baseline_run_id=BASE.name)
    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'another stage lock exists')
        require(not [p for p in ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard') if p.parent!=STUDY],'another execution owner')
        return super().__enter__()
    def freeze(self,contract):
        self.sources=read_json(STUDY/'S2_SOURCE_FREEZE.json')['files']
        origin=read_json(STUDY/'EVALUATION_SOURCE_CONTEXT.json')
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources,code_commit=origin['code_commit']))
        mapping={}
        for i,rel in enumerate(self.sources):
            dst=self.path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)
            dst.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/rel,dst)
            mapping[rel]=dst.relative_to(self.path).as_posix()
        write_new(self.path/'SOURCE_SNAPSHOT_MAP.json',mapping)
        for src,dst in [('EVALUATION_CONTRACT.json','EVAL_CONTRACT.json'),('PROTOCOL.md','PROTOCOL.md'),
            ('EVALUATION_EXECUTION.md','EVALUATION_EXECUTION.md'),('S2_DAMAGE_PANEL.json','DAMAGE_PANEL.json'),
            ('S2_PANEL_RECEIPT.json','PANEL_RECEIPT.json')]:shutil.copyfile(STUDY/src,self.path/dst)
        self.update(plan_status='frozen',code_commit=origin['code_commit'],
            dirty_patch_sha256=sha256(STUDY/'EVALUATION_SOURCE_DIRTY.patch'),
            protocol_path=str(self.path/'PROTOCOL.md'),protocol_sha256=sha256(self.path/'PROTOCOL.md'),
            contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),
            damage_panel_sha256=sha256(self.path/'DAMAGE_PANEL.json'))
