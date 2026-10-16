"""Immutable input identities and registered zero-training lifecycle."""
from pathlib import Path
import importlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_soft_fit_audit_20261016'
PARENT = ROOT/'experiments/mask_readout_mechanism_v2_20261012'
CACHE = PARENT/'runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
TRAIN = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
DEV = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
GATE = ROOT/'experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'
FIT = ROOT/'experiments/mask_readout_fit_diagnostic_20261014/runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926'
BASE = ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
GT = CACHE/'dataset/training_annotations.json'
VAL_GT = ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
PROFILE = ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'
WEIGHT = ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt'
CKPT_SHA = '15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce'
for directory in (PARENT/'scripts', ROOT/'shared/evaluation', ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts', ROOT/'experiments/mask_readout_fit_diagnostic_20261014/scripts', ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/scripts', ROOT/'experiments/mask_readout_large_mask_gate_20261015/scripts'):
    sys.path.append(str(directory))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now, save, tensor_state_hash
from posttrain_audit import read_jsonl

CODE = [STUDY.relative_to(ROOT).as_posix()+'/'+x for x in ('PROTOCOL.md','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch','PREPARATION.json','INPUT_EXPECTATIONS.json','scripts/common.py','scripts/prepare.py','scripts/run_audit.py','scripts/audit_stats.py','scripts/verify.py','scripts/test_statistics.py','scripts/launch.ps1')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/'+x for x in ('posttrain_run.py','adapter_model.py','adapter_data.py','adapter_primitives.py','s1_geometry.py','run_s2_evaluation.py')]
CODE += ['shared/evaluation/posttrain_audit.py','shared/evaluation/native_output_bridge.py','shared/evaluation/native_source_pins.json','experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py','experiments/mask_readout_fit_diagnostic_20261014/scripts/run_fit_diagnostic.py','experiments/mask_readout_soft_iou_pilot_20261014/scripts/soft_iou_loss.py','experiments/mask_readout_large_mask_gate_20261015/scripts/area_gate.py',PROFILE.relative_to(ROOT).as_posix()]

def input_paths():
    paths = [TRAIN/'last.pt', PROFILE, WEIGHT, GT, VAL_GT, CACHE/'trace/TRACE_MANIFEST.json', CACHE/'trace/NATIVE_ARGS.json', CACHE/'SPLIT_CONTRACT.json', BASE/'trace/NATIVE_ARGS.json', FIT/'BASELINE_GT_LEDGER.jsonl', FIT/'ROW_MAPPING.json', FIT/'EVAL_CONTRACT.json']
    paths += [TRAIN/x for x in ('EVAL_CONTRACT.json','SUMMARY.json','TRAINING_JOIN.json','IMAGE_ORDERS.json')]
    paths += [DEV/x for x in ('EVAL_CONTRACT.json','SUMMARY.json','B_predictions.json','B_GT_LEDGER.jsonl','BASELINE_SUBSET_GT_LEDGER.jsonl','B_FIXED_PAIR_EFFECTS.jsonl','B_ROW_MAPPING.json')]
    paths += [GATE/x for x in ('EVAL_CONTRACT.json','SUMMARY.json','MIXED_predictions.json','MIXED_GT_LEDGER.jsonl','FIXED_PAIR_EFFECTS.jsonl','ROW_MAPPING.json')]
    paths += [Path(read_json(CACHE/'trace/TRACE_MANIFEST.json')['predictions_path']),BASE/'trace/TRACE_MANIFEST.json',BASE/'trace/native/predictions.json']
    paths += [p/x for p in (TRAIN,DEV,GATE,FIT,CACHE) for x in ('run.json','CONFIG_AUDIT.json','ARTIFACT_MANIFEST.json','SOURCE_MANIFEST.json')]
    return list(dict.fromkeys(paths))

def inspect_inputs(frozen=True):
    profile, environment = configure_native()
    contract = read_json(TRAIN/'EVAL_CONTRACT.json')
    require(environment == contract['environment'] == read_json(DEV/'EVAL_CONTRACT.json')['environment'], 'actual native runtime changed')
    require(sha256(TRAIN/'last.pt') == CKPT_SHA == read_json(TRAIN/'SUMMARY.json')['checkpoint_sha256'], 'wrong checkpoint')
    require(sha256(GT)=='51477c03144651f01f941991661da85df0df17b6048a148feca0b7dad6bce26c','train annotation drift')
    require(sha256(CACHE/'trace/TRACE_MANIFEST.json')=='fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d'==contract['source_trace_sha256'],'cache drift')
    require(sha256(PROFILE)=='ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45'==contract['profile_sha256'],'profile drift')
    require(sha256(WEIGHT)==profile['model_sha256'] and sha256(VAL_GT)==profile['annotations_sha256'],'weight/validation GT drift')
    require(sha256(BASE/'trace/native/predictions.json')==read_json(BASE/'trace/TRACE_MANIFEST.json')['predictions_sha256']=='a8c3e7d0af9db4684653c9b7b412f734d6105dc1e1c98a20c0782cf8d2fd3151','original validation predictions drift')
    require(contract['epochs']==3 and contract['expected_optimizer_steps']==1500 and contract['parameters']==167872,'training identity changed')
    split=read_json(CACHE/'SPLIT_CONTRACT.json')
    require(sha256(CACHE/'SPLIT_CONTRACT.json')==contract['fit_dev_split_sha256'],'split changed')
    require(len(split['fit_ids'])==len(set(split['fit_ids']))==2000 and len(split['dev_ids'])==512 and not set(split['fit_ids'])&set(split['dev_ids']),'split populations changed')
    require(split['dev_ids']==read_json(DEV/'EVAL_CONTRACT.json')['dev_ids'],'dev identities changed')
    external={}
    for name,pin in read_json(DEV/'EVAL_CONTRACT.json')['external_evaluator_sources'].items():
        module=importlib.import_module(name)
        external[name]=dict(path=str(Path(module.__file__).resolve()),sha256=sha256(module.__file__))
        require(external[name]==pin,'evaluator drift: '+name)
    inputs={p.relative_to(ROOT).as_posix():sha256(p) for p in input_paths()}
    verified_artifacts=[]
    sources={}
    for parent in (TRAIN,DEV,GATE,FIT,CACHE):
        require(read_json(parent/'run.json')['execution_status']=='completed','source Run incomplete')
        require(read_json(parent/'CONFIG_AUDIT.json')['status']=='passed','source config failed')
        for entry in read_json(parent/'ARTIFACT_MANIFEST.json')['artifacts']:
            rel=(parent/entry['relative_path']).relative_to(ROOT).as_posix()
            if rel in inputs:
                require(inputs[rel]==entry['sha256'],'source artifact drift: '+rel)
                verified_artifacts.append(rel)
        for rel,digest in read_json(parent/'SOURCE_MANIFEST.json')['files'].items():
            if rel in CODE and rel!=STUDY.relative_to(ROOT).as_posix()+'/PROTOCOL.md':
                require(sha256(ROOT/rel)==digest,'historical computation source drift: '+rel)
                sources[rel]=digest
    native=read_json(CACHE/'trace/NATIVE_ARGS.json')
    original=read_json(BASE/'trace/NATIVE_ARGS.json')
    differences={k:dict(fit=native.get(k),val=original.get(k)) for k in set(native)|set(original) if native.get(k)!=original.get(k)}
    require(set(differences)<={'data','project'},'native preprocessing drift')
    if frozen:
        require(inputs==read_json(STUDY/'INPUT_EXPECTATIONS.json')['files'],'frozen inputs changed')
    return dict(profile=profile,environment=environment,split=split,files=inputs,external_sources=external,native_argument_differences=differences,verified_source_artifacts=verified_artifacts,matched_original_sources=sources)

class AuditRun(RegisteredRun):
    def __init__(self,scope):
        super().__init__('SOFT_FIT_'+scope.upper(),'SOFT_FIT_AUDIT',TRAIN,CODE,900 if scope=='smoke' else 3600)
        self.path=STUDY/'runs'/self.run_id
        self.lock=STUDY/'.SOFT_FIT.lock'
        self.meta.update(study_id='STUDY_MASK_SOFT_FIT_AUDIT_20261016',scope=scope,new_training_steps=0,base_model_forward_calls=0,oracle_optimization_calls=0,baseline_run_id=CACHE.name)
    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'another experiment lock exists')
        require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')),'another execution owner exists')
        return super().__enter__()
    def freeze(self,contract):
        frozen=read_json(STUDY/'FREEZE_MANIFEST.json')
        self.sources={rel:sha256(ROOT/rel) for rel in CODE}
        require(self.sources==frozen['files'],'code/protocol changed after freeze')
        origin=read_json(STUDY/'SOURCE_CONTEXT.json')
        self.meta.update(code_commit=origin['code_commit'],dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources,code_commit=origin['code_commit'],frozen_at=frozen['frozen_at']))
        mapping={}
        for i,rel in enumerate(self.sources):
            p=self.path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)
            p.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel,p)
            mapping[rel]=p.relative_to(self.path).as_posix()
        write_new(self.path/'SOURCE_SNAPSHOT_MAP.json',mapping)
        shutil.copyfile(STUDY/'PROTOCOL.md',self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_sha256=sha256(self.path/'PROTOCOL.md'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))

def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')
