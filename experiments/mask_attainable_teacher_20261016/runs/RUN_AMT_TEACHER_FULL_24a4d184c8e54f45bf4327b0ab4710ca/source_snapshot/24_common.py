"""AMT-v1 input pins and registered stages; never mutate earlier studies."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_attainable_teacher_20261016'
PARENT = ROOT/'experiments/mask_readout_mechanism_v2_20261012'
SOFT = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014'
for directory in (PARENT/'scripts', SOFT/'scripts', ROOT/'shared/evaluation'):
    sys.path.append(str(directory))
import pilot_common as previous
from posttrain_run import RegisteredRun, read_json, write_new, sha256, require, now, save, tensor_state_hash
from posttrain_audit import read_jsonl
from finite_coefficient_solver import DEFAULTS

CACHE = previous.CACHE_RUN
INITIAL = previous.B_RUN
CONTROL = SOFT/'runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
FIT = ROOT/'experiments/mask_readout_fit_diagnostic_20261014/runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926'
TRAIN_GT = previous.TRAIN_GT
SPEC = dict(name='AMT_v1_conditional_hard_functional_target', solver=dict(DEFAULTS),
    minimum_native_gain=.01, minimum_converged_fraction=.95, minimum_solved=200,
    minimum_accepted=200, minimum_accepted_fraction=.20, minimum_accepted_mean_gain=.05,
    minimum_gate_on_accepted=64, minimum_gate_on_images=32, inference_area=9216,
    eligibility='ordinary_original_training_join; task_mask75_failure; paired_mask_iou_lt_0.75; nonempty_ROI',
    target='full_same_image_FP32_CUDA_coefficients_matmul_P_gt_zero_binary_proto_mask',
    fallback='original_projected_GT_for_every_nonaccepted_join; no_filter_or_reweight',
    protection='original_all_output_BCE_weight_0.1_and_fixed_native_mask_area_gate')
LOCAL = ('common.py','prepare.py','teacher.py','targets.py','training.py','verify_teacher.py','verify_training.py','test_targets.py','pipeline.py','launch.ps1')
CODE = list(dict.fromkeys(previous.original_training.CODE_FILES + [
    (PARENT/'scripts'/n).relative_to(ROOT).as_posix() for n in
    ('finite_coefficient_solver.py','audit_s2_completed.py','run_s2_evaluation.py')] + [
    (SOFT/'scripts'/n).relative_to(ROOT).as_posix() for n in ('pilot_common.py','soft_iou_loss.py')] + [
    (SOFT/'INPUT_EXPECTATIONS.json').relative_to(ROOT).as_posix(),
    'shared/evaluation/damage_acceptance.py','shared/evaluation/DAMAGE_ACCEPTANCE.md'] + [
    (STUDY/x).relative_to(ROOT).as_posix() for x in ('PROTOCOL.md','SOURCE_CONTEXT.json','SOURCE_DIRTY.patch')]+[
    (STUDY/'scripts'/x).relative_to(ROOT).as_posix() for x in LOCAL]))

def inspect_inputs(frozen=True):
    data = previous.inspect_inputs()
    extra = [INITIAL/'initial.pt', CONTROL/'last.pt', CONTROL/'EVAL_CONTRACT.json',
        CONTROL/'SOURCE_MANIFEST.json', CONTROL/'CONFIG_AUDIT.json', CONTROL/'SUMMARY.json',
        FIT/'BASELINE_GT_LEDGER.jsonl', FIT/'ROW_MAPPING.json', FIT/'CONFIG_AUDIT.json',
        INITIAL/'TRAINING_JOIN.json', INITIAL/'IMAGE_ORDERS.json']
    files = dict(data['inputs_sha256'])
    files.update({p.relative_to(ROOT).as_posix():sha256(p) for p in extra})
    require(sha256(INITIAL/'initial.pt')=='ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd','initial checkpoint drift')
    require(sha256(CONTROL/'last.pt')=='15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce','reference checkpoint drift')
    require(sha256(FIT/'BASELINE_GT_LEDGER.jsonl')=='e5af5d1ff459342617069f6a6b34d255e61da43a696e5a50424bf77c34dace8d','training ledger drift')
    require(sha256(CACHE/'trace/TRACE_MANIFEST.json')=='fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d','train cache drift')
    require(read_json(CONTROL/'CONFIG_AUDIT.json')['status']=='passed','reference config audit failed')
    reference = read_json(CONTROL/'EVAL_CONTRACT.json')
    for k,v in data['original_contract'].items():
        require(reference[k]==v,'original/soft control configuration differs: '+k)
    for rel,digest in read_json(CONTROL/'SOURCE_MANIFEST.json')['files'].items():
        if rel in CODE:
            require(sha256(ROOT/rel)==digest,'reference implementation drift: '+rel)
    data['inputs_sha256']=files
    if frozen:
        freeze=read_json(STUDY/'FREEZE_MANIFEST.json')
        require(sha256(STUDY/'INPUT_EXPECTATIONS.json')==freeze['input_manifest_sha256'],'input manifest bytes changed after freeze')
        require(sha256(STUDY/'PREPARATION.json')==freeze['preparation_sha256'],'preparation receipt changed after freeze')
        require(files==read_json(STUDY/'INPUT_EXPECTATIONS.json')['files'],'frozen inputs drift')
        require({p:sha256(ROOT/p) for p in CODE}==freeze['files'],'frozen source drift')
    return data

def verify_after(data):
    previous.verify_after(data)

def teacher_gate(rows):
    attempted=[r for r in rows if r['attempted']]
    accepted=[r for r in rows if r['accepted']]
    gated=[r for r in accepted if r['native_area']>=SPEC['inference_area']]
    converged=sum(r['solver_status']=='converged' for r in attempted)
    n=len(attempted)
    mean_gain=sum(r['oracle_iou']-r['baseline_iou'] for r in accepted)/len(accepted) if accepted else 0.
    checks=dict(minimum_solved=n>=SPEC['minimum_solved'],
        convergence=converged/max(1,n)>=SPEC['minimum_converged_fraction'],
        minimum_accepted=len(accepted)>=SPEC['minimum_accepted'],
        accepted_fraction=len(accepted)/max(1,n)>=SPEC['minimum_accepted_fraction'],
        mean_native_gain=mean_gain>=SPEC['minimum_accepted_mean_gain'],
        gate_on_count=len(gated)>=SPEC['minimum_gate_on_accepted'],
        gate_on_images=len({r['image_id'] for r in gated})>=SPEC['minimum_gate_on_images'])
    return dict(checks=checks,passed=all(checks.values()),attempted=n,converged=converged,
        accepted=len(accepted),accepted_fraction=len(accepted)/max(1,n),mean_accepted_native_gain=mean_gain,
        gate_on_accepted=len(gated),gate_on_images=len({r['image_id'] for r in gated}),
        accepted_mask75_repaired=sum(r['oracle_iou']>=.75 for r in accepted))

class Run(RegisteredRun):
    def __init__(self,kind,parent,budget):
        super().__init__('AMT_'+kind,'AMT',parent,CODE,budget)
        self.path=STUDY/'runs'/self.run_id
        self.lock=STUDY/'.AMT.lock'
        self.meta.update(study_id='STUDY_MASK_ATTAINABLE_TEACHER_20261016',base_model_forward_calls=0)
    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')),'another experimental lock exists')
        owners=[p for p in ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard') if p.parent!=STUDY]
        require(not owners,'another execution owner exists')
        return super().__enter__()
    def freeze(self,contract):
        frozen=read_json(STUDY/'FREEZE_MANIFEST.json')
        self.sources={rel:sha256(ROOT/rel) for rel in CODE}
        require(self.sources==frozen['files'],'source changed after freeze')
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
        self.update(plan_status='frozen',protocol_sha256=sha256(self.path/'PROTOCOL.md'),
            contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))
