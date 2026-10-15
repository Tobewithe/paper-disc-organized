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
STUDY=ROOT/'experiments/mask_readout_aligned_repulsion_20261015'
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
from posttrain_run import tensor_state_hash
import run_s2_training as original_training

CODE=['experiments/mask_readout_aligned_repulsion_20261015/'+p for p in
    ('PROTOCOL.md','INPUT_EXPECTATIONS.json','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch',
     'scripts/common.py','scripts/aligned_loss.py','scripts/test_aligned.py','scripts/run_training.py',
     'scripts/run_evaluation.py','scripts/compare_arms.py','scripts/verify_evaluation.py','scripts/audit_training.py','scripts/launch.ps1')]
CODE += original_training.CODE_FILES
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/run_s2_evaluation.py',
    'experiments/mask_readout_subspace_repulsion_20261015/scripts/repulsion.py',
    'experiments/mask_readout_subspace_repulsion_20261015/scripts/verify_repulsion.py',
    'experiments/mask_readout_large_mask_gate_20261015/scripts/run_mixed.py',
    'experiments/mask_readout_large_mask_gate_20261015/scripts/area_gate.py',
    'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
    'shared/evaluation/damage_acceptance.py','shared/evaluation/posttrain_audit.py']
B_RUN=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2'
CACHE_RUN=ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
TRAIN_GT=CACHE_RUN/'dataset/training_annotations.json'


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


class ExperimentRun(RegisteredRun):
    def __init__(self,kind,scope,parent):
        budget=900 if scope=='smoke' else (7200 if kind.startswith('TRAIN') else 3600)
        super().__init__('ALIGNED_'+kind,'E4',parent,list(dict.fromkeys(CODE)),budget)
        self.path=STUDY/'runs'/self.run_id;self.lock=STUDY/'.E4.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_ALIGNED_REPULSION_20261015',scope=scope,
            method_id='E4-ALIGNED-v1',base_model_forward_calls=0,optimization_started=False)

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'unresolved concurrent experiment lock')
        return super().__enter__()

    def freeze(self,contract):
        origin=read_json(STUDY/'SOURCE_CONTEXT.json');self.sources={rel:sha256(ROOT/rel) for rel in self.code_files}
        self.meta.update(code_commit=origin['code_commit'],dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources,code_commit=origin['code_commit']))
        for rel in self.sources:
            target=self.path/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
        shutil.copyfile(STUDY/'PROTOCOL.md',self.path/'PROTOCOL.md');write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_path=str(self.path/'PROTOCOL.md'),protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))

