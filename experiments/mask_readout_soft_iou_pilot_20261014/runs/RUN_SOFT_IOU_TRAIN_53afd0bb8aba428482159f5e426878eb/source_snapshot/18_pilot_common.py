"""Study-local lifecycle; original implementation and inputs remain immutable."""
import importlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014'
PARENT = ROOT/'experiments/mask_readout_mechanism_v2_20261012'
sys.path.insert(0, str(PARENT/'scripts'))
from posttrain_run import RegisteredRun, configure_native, tensor_state_hash, read_json, require, sha256, write_new
import run_s2_training as original_training
import run_s2_evaluation as original_evaluation

B_RUN = PARENT/'runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2'
CACHE_RUN = PARENT/'runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
DEV_RUN = PARENT/'runs/RUN_S2_EVAL_a24a564568c943eab362784abfbe2c54'
BASE = original_evaluation.BASE
GT = original_evaluation.GT
TRAIN_GT = CACHE_RUN/'dataset/training_annotations.json'
WEIGHT = original_evaluation.WEIGHT
PROFILE = ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'
LOCAL_FILES = ['pilot_common.py', 'soft_iou_loss.py', 'run_training.py', 'run_evaluation.py',
               'test_soft_iou.py', 'collect_evidence.py', 'launch_pilot.ps1']
CODE_FILES = list(dict.fromkeys(original_training.CODE_FILES + original_evaluation.CODE_FILES +
    ['experiments/mask_readout_soft_iou_pilot_20261014/scripts/'+n for n in LOCAL_FILES] +
    ['experiments/mask_readout_soft_iou_pilot_20261014/'+n for n in
     ('PROTOCOL.md', 'INPUT_EXPECTATIONS.json', 'SOURCE_CONTEXT.json', 'SOURCE_DIRTY.patch')] +
    ['experiments/mask_readout_fit_diagnostic_20261014/scripts/collect_diagnostic_evidence.py',
     'shared/evaluation/DAMAGE_ACCEPTANCE.md']))


def external_sources():
    return {name: {'path': str(Path(importlib.import_module(name).__file__).resolve()),
                   'sha256': sha256(importlib.import_module(name).__file__)}
            for name in ('pycocotools.coco', 'pycocotools.cocoeval', 'pycocotools.mask', 'pycocotools._mask')}


class PilotRun(RegisteredRun):
    def __init__(self, kind, parent, budget):
        super().__init__('SOFT_IOU_'+kind, 'S2_SOFT_IOU', parent, CODE_FILES, budget)
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.S2_SOFT_IOU.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_SOFT_IOU_PILOT_20261014',
                         baseline_run_id=BASE.name, optimization_started=False)

    def __enter__(self):
        other = [p for p in ROOT.glob('experiments/*/.*.lock') if p != self.lock]
        require(not other, 'another experimental stage has an unresolved lock: '+str(other))
        return super().__enter__()

    def freeze(self, contract):
        context = read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources = {rel: sha256(ROOT/rel) for rel in self.code_files}
        self.meta.update(code_commit=context['code_commit'], dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json', {'files': self.sources, 'code_commit': context['code_commit'],
                                                   'actual_sources_snapshotted': True})
        shutil.copyfile(STUDY/'SOURCE_DIRTY.patch', self.path/'dirty.patch')
        for i, rel in enumerate(self.sources):
            target = self.path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel, target)
        shutil.copyfile(STUDY/'PROTOCOL.md', self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'), contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))


def inspect_inputs():
    profile, environment = configure_native()
    expected = read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for rel, digest in expected['files'].items():
        require(sha256(ROOT/rel) == digest, 'frozen input changed: '+rel)
    training = read_json(B_RUN/'EVAL_CONTRACT.json')
    require(environment == training['environment'], 'runtime differs from original B training')
    external = external_sources()
    require(external == expected['external_evaluator_sources'], 'external evaluation code drift')
    require(sha256(WEIGHT) == profile['model_sha256'] and sha256(GT) == profile['annotations_sha256'], 'model/GT drift')
    for parent in (B_RUN, CACHE_RUN, DEV_RUN):
        require(read_json(parent/'run.json')['execution_status'] == 'completed', 'incomplete prerequisite')
        require(read_json(parent/'CONFIG_AUDIT.json')['status'] == 'passed', 'unverified prerequisite')
    require(read_json(DEV_RUN/'DECISION.json')['status'] == 'failed', 'old BCE S2 decision changed')
    for parent in (B_RUN, DEV_RUN):
        for rel, digest in read_json(parent/'SOURCE_MANIFEST.json')['files'].items():
            actual = sha256(ROOT/rel)
            if actual != digest:
                reviewed = expected['reviewed_historical_auditor_change']
                require(rel == reviewed['path'] and digest == reviewed['historical_sha256'] and
                        actual == reviewed['current_sha256'], 'original implementation drift: '+rel)
    split = read_json(CACHE_RUN/'SPLIT_CONTRACT.json')
    require(len(split['fit_ids']) == len(set(split['fit_ids'])) == 2000, 'fit population changed')
    require(len(split['dev_ids']) == 512 and split['dev_T'] == 655, 'dev population changed')
    require(not set(split['fit_ids']) & set(split['dev_ids']), 'fit/dev overlap')
    require(read_json(B_RUN/'IMAGE_ORDERS.json') == original_training.orders(split['fit_ids'], 3), 'original orders changed')
    require(sha256(TRAIN_GT) == read_json(CACHE_RUN/'EVAL_CONTRACT.json')['gt_subset_sha256'], 'training GT drift')
    train_native = read_json(CACHE_RUN/'trace/NATIVE_ARGS.json')
    base_native = read_json(BASE/'trace/NATIVE_ARGS.json')
    differences = {k: {'fit': train_native.get(k), 'S0': base_native.get(k)}
                   for k in set(train_native)|set(base_native) if train_native.get(k) != base_native.get(k)}
    require(set(differences) <= {'data', 'project'}, 'native inference arguments drift')
    return {'profile': profile, 'environment': environment, 'original_contract': training, 'split': split,
            'external_evaluator_sources': external, 'inputs_sha256': expected['files'],
            'native_argument_differences': differences}


def verify_after(data):
    for rel, digest in data['inputs_sha256'].items():
        require(sha256(ROOT/rel) == digest, 'input mutated during run: '+rel)
    from native_output_bridge import native_environment
    require(native_environment() == data['environment'], 'numerical environment changed')
    require(external_sources() == data['external_evaluator_sources'], 'external source changed')


def jsonl(path, rows):
    import json
    with path.open('x', encoding='utf8') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False)+'\n')
