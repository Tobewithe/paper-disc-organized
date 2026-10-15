"""Frozen inputs and run lifecycle for the original area-gated S3 candidate."""
from pathlib import Path
import importlib
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_gate_s3r_20261015'
BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
PARENT = ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
TRAIN = ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
GATE = ROOT / 'experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'
GT = ROOT / 'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
WEIGHT = ROOT / 'assets/models/coco_clean_20260911/yolo26m-seg.pt'
NATIVE = BASE / 'trace/native/predictions.json'
PROFILE = ROOT / 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'
sys.path.insert(0, str(ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0, str(ROOT / 'experiments/official_gt_failure_rebaseline_20261013/scripts'))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require
from posttrain_audit import read_jsonl

CODE = ['experiments/mask_gate_s3r_20261015/' + p for p in (
    'PROTOCOL.md', 'RECOVERY.md', 'INPUT_EXPECTATIONS.json', 'PREPARATION.json', 'SOURCE_CONTEXT.json', 'SOURCE_DIRTY.patch',
    'scripts/common.py', 'scripts/prepare.py', 'scripts/coco_bootstrap.py', 'scripts/run_s3.py',
    'scripts/verify_s3.py', 'scripts/test_bootstrap.py', 'scripts/launch.ps1')]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/' + p for p in (
    'posttrain_run.py', 'adapter_model.py', 'adapter_data.py', 'adapter_primitives.py', 's1_geometry.py')]
CODE += ['experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
         'shared/evaluation/posttrain_audit.py', 'shared/evaluation/native_output_bridge.py',
         'shared/evaluation/native_source_pins.json', 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']


def inspect_inputs():
    expected = read_json(STUDY / 'INPUT_EXPECTATIONS.json')
    for rel, digest in expected['files'].items():
        require(sha256(ROOT / rel) == digest, 'input drift: ' + rel)
    profile, environment = configure_native()
    source = read_json(PARENT / 'EVAL_CONTRACT.json')
    require(environment == source['environment'], 'actual native runtime changed')
    for name, pin in source['external_evaluator_sources'].items():
        module = importlib.import_module(name)
        actual = dict(path=str(Path(module.__file__).resolve()), sha256=sha256(module.__file__))
        require(actual == pin, 'external evaluator source/path changed: ' + name)
    require(read_json(GATE / 'DECISION.json')['status'] == 'passed', 'area-gate S2 prerequisite failed')
    for parent in (BASE, PARENT, TRAIN, GATE):
        require(read_json(parent / 'run.json')['execution_status'] == 'completed', 'incomplete prerequisite')
    require(sha256(TRAIN / 'last.pt') == source['checkpoints']['B']['sha256'], 'wrong checkpoint')
    return expected, environment


class S3Run(RegisteredRun):
    def __init__(self, scope, parent):
        super().__init__('S3_' + scope.upper(), 'S3', parent, CODE,
                         3600 if scope == 'smoke' else 21600)
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.S3.lock'
        self.meta.update(study_id='STUDY_MASK_GATE_S3R_20261015', scope=scope,
                         base_model_forward_calls=0, new_training_steps=0,
                         baseline_run_id=BASE.name, candidate_checkpoint_run=TRAIN.name)

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')), 'unresolved experiment lock')
        return super().__enter__()

    def freeze(self, contract):
        origin = read_json(STUDY / 'SOURCE_CONTEXT.json')
        freeze = read_json(STUDY / 'FREEZE_MANIFEST.json')
        self.sources = {rel: sha256(ROOT / rel) for rel in CODE}
        require(self.sources == freeze['files'], 'frozen execution code/input contract changed')
        self.meta.update(code_commit=origin['code_commit'], dirty_patch_sha256=sha256(STUDY / 'SOURCE_DIRTY.patch'))
        write_new(self.path / 'SOURCE_MANIFEST.json', dict(files=self.sources, code_commit=origin['code_commit']))
        snapshots = {}
        for i, rel in enumerate(self.sources):
            target = self.path / 'source_snapshot' / (f'{i:02d}_' + Path(rel).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
            snapshots[rel] = target.relative_to(self.path).as_posix()
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', snapshots)
        shutil.copyfile(STUDY / 'PROTOCOL.md', self.path / 'PROTOCOL.md')
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path / 'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path / 'PROTOCOL.md'),
                    contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY / 'INPUT_EXPECTATIONS.json'))


def jsonl(path, rows):
    with path.open('x', encoding='utf8') as stream:
        import json
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
