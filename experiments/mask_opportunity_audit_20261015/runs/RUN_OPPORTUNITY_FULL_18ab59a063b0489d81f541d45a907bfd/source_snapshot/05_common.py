"""Fixed sources and lifecycle for the S1 opportunity realization audit."""
from pathlib import Path
import importlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_opportunity_audit_20261015'
BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1 = ROOT / 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
TRAIN = ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
OLD = ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
S3 = ROOT / 'experiments/mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d'
GT = ROOT / 'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
WEIGHT = ROOT / 'assets/models/coco_clean_20260911/yolo26m-seg.pt'
PROFILE = ROOT / 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'
sys.path.insert(0, str(ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now, save, tensor_state_hash
from posttrain_audit import read_jsonl

CODE = ['experiments/mask_opportunity_audit_20261015/' + p for p in (
    'PROTOCOL.md', 'INPUT_EXPECTATIONS.json', 'SOURCE_CONTEXT.json', 'SOURCE_DIRTY.patch',
    'PREPARATION.json', 'scripts/common.py', 'scripts/prepare.py', 'scripts/run_audit.py',
    'scripts/verify.py', 'scripts/launch.ps1')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/' + p for p in (
    'posttrain_run.py', 'adapter_model.py', 'adapter_data.py', 'adapter_primitives.py', 's1_geometry.py')]
CODE += ['shared/evaluation/posttrain_audit.py', 'shared/evaluation/native_output_bridge.py',
         'shared/evaluation/native_source_pins.json', 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']


def inspect_inputs():
    expected = read_json(STUDY / 'INPUT_EXPECTATIONS.json')
    for relative, digest in expected['files'].items():
        require(sha256(ROOT / relative) == digest, 'input drift: ' + relative)
    profile, environment = configure_native()
    contract = read_json(OLD / 'EVAL_CONTRACT.json')
    require(environment == contract['environment'], 'actual native runtime changed')
    for name, pin in contract['external_evaluator_sources'].items():
        module = importlib.import_module(name)
        actual = dict(path=str(Path(module.__file__).resolve()), sha256=sha256(module.__file__))
        require(actual == pin, 'external evaluator drift: ' + name)
    for parent in (BASE, S1, TRAIN, OLD, S3):
        require(read_json(parent / 'run.json')['execution_status'] == 'completed', 'incomplete source Run')
    require(sha256(TRAIN / 'last.pt') == contract['checkpoints']['B']['sha256'], 'wrong adapter')
    return expected, profile, environment


class AuditRun(RegisteredRun):
    def __init__(self, scope, parent):
        super().__init__('OPPORTUNITY_' + scope.upper(), 'S1_S3_DIAGNOSTIC', parent, CODE,
                         900 if scope == 'smoke' else 1800)
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.OPPORTUNITY.lock'
        self.meta.update(study_id='STUDY_MASK_OPPORTUNITY_AUDIT_20261015', scope=scope,
                         base_model_forward_calls=0, new_training_steps=0, oracle_optimization_calls=0,
                         baseline_run_id=BASE.name, source_S1_run_id=S1.name, checkpoint_run_id=TRAIN.name)

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')), 'experiment lock exists')
        require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')), 'execution owner exists')
        return super().__enter__()

    def freeze(self, contract):
        frozen = read_json(STUDY / 'FREEZE_MANIFEST.json')
        self.sources = {rel: sha256(ROOT / rel) for rel in CODE}
        require(self.sources == frozen['files'], 'protocol/code contract changed after freeze')
        origin = read_json(STUDY / 'SOURCE_CONTEXT.json')
        self.meta.update(code_commit=origin['code_commit'], dirty_patch_sha256=sha256(STUDY / 'SOURCE_DIRTY.patch'))
        write_new(self.path / 'SOURCE_MANIFEST.json', dict(files=self.sources, code_commit=origin['code_commit'], frozen_at=frozen['frozen_at']))
        snapshots = {}
        for i, rel in enumerate(self.sources):
            path = self.path / 'source_snapshot' / (f'{i:02d}_' + Path(rel).name)
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / rel, path)
            snapshots[rel] = path.relative_to(self.path).as_posix()
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', snapshots)
        shutil.copyfile(STUDY / 'PROTOCOL.md', self.path / 'PROTOCOL.md')
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path / 'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path / 'PROTOCOL.md'),
                    input_manifest_sha256=sha256(STUDY / 'INPUT_EXPECTATIONS.json'),
                    contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'))


def jsonl(path, rows):
    with path.open('x', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


def summarize(rows):
    from collections import Counter
    positives = [r for r in rows if r['oracle_found']]
    counts = {key: sum(r['decomposition'] == key for r in positives) for key in ('L', 'G', 'R')}
    require(sum(counts.values()) == len(positives), 'decomposition incomplete')
    if counts['L'] > counts['G']:
        direction = 'conditional_repair_learning'
    elif counts['G'] > counts['L']:
        direction = 'damage_constrained_benefit_selector'
    elif counts['L']:
        direction = 'joint_conditional_repair_and_selection'
    else:
        direction = 'coverage_and_COCO_task_conversion'
    def stats(part):
        n = len(part)
        return dict(n=n, oracle_found=sum(r['oracle_found'] for r in part),
                    ungated_repaired=sum(r['ungated_repaired'] for r in part),
                    gated_repaired=sum(r['gated_repaired'] for r in part),
                    gate_enabled=sum(r['gate_enabled'] for r in part),
                    improved=sum(r['ungated_iou'] > r['native_iou'] for r in part),
                    worsened=sum(r['ungated_iou'] < r['native_iou'] for r in part),
                    unchanged=sum(r['ungated_iou'] == r['native_iou'] for r in part),
                    mean_ungated_delta=sum(r['ungated_iou']-r['native_iou'] for r in part)/n if n else None,
                    mean_gated_delta=sum(r['gated_iou']-r['native_iou'] for r in part)/n if n else None,
                    L=sum(r['decomposition']=='L' for r in part),
                    G=sum(r['decomposition']=='G' for r in part), R=sum(r['decomposition']=='R' for r in part))
    return dict(pairs=len(rows), images=len({r['image_id'] for r in rows}),
                oracle_positive=len(positives), oracle_unfound=len(rows)-len(positives), decomposition=counts,
                priority_direction=direction, scope='exact exposed S1 panel; not population inference or AP attribution',
                all=stats(rows), oracle_found_group=stats(positives),
                oracle_unfound_group=stats([r for r in rows if not r['oracle_found']]),
                L_group=stats([r for r in rows if r['decomposition']=='L']),
                native_area_groups={k:stats([r for r in rows if r['native_area_group']==k]) for k in ('S','M','L')},
                L_gate_enabled=sum(r['decomposition']=='L' and r['gate_enabled'] for r in rows),
                L_gate_disabled=sum(r['decomposition']=='L' and not r['gate_enabled'] for r in rows),
                s3_task_success=sum(r['s3_task_success'] for r in rows),
                s3_task_same_fixed_detection=sum(r['s3_task_success'] and r['s3_method_dt_id']==r['global_index']+1 for r in rows),
                ungated_COCO_task_status='not_recomputed', new_training_steps=0, base_model_forward_calls=0)
