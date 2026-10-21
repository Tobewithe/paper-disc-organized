"""CLD execution identities and registered lifecycle; no automatic next stage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_direction_learnability_20261016'
PARENT = ROOT / 'experiments/mask_readout_mechanism_v2_20261012'
CACHE = PARENT / 'runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
INITIAL = PARENT / 'runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2'
TEACHER = ROOT / 'experiments/mask_attainable_teacher_20261016/runs/RUN_AMT_TEACHER_FULL_24a4d184c8e54f45bf4327b0ab4710ca'
FIT = ROOT / 'experiments/mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770'
GT = CACHE / 'dataset/training_annotations.json'
for path in (PARENT / 'scripts', ROOT / 'shared/evaluation', ROOT / 'experiments/official_gt_failure_rebaseline_20261013/scripts'):
    sys.path.append(str(path))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now, save, tensor_state_hash
from posttrain_audit import read_jsonl


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def science_check():
    frozen = read_json(STUDY / 'SCIENTIFIC_FREEZE.json')
    for item in frozen['files']:
        require(sha256(STUDY / item['path']) == item['sha256'], 'scientific freeze drift: ' + item['path'])
    return frozen


def source_check():
    science_check()
    freeze = read_json(STUDY / 'EXECUTION_FREEZE.json')
    for rel, digest in freeze['files'].items():
        require(sha256(ROOT / rel) == digest, 'execution source drift: ' + rel)
    return freeze


def inspect_inputs():
    frozen = source_check()
    records = read_json(STUDY / 'INPUT_MANIFEST.json')
    for item in records['files']:
        require(sha256(ROOT / item['path']) == item['sha256'], 'input drift: ' + item['path'])
    profile, environment = configure_native()
    inherited = read_json(STUDY / 'INHERITED_CONTRACT.json')
    require(environment == inherited['source_environment'], 'effective runtime differs from frozen cache')
    require(sha256(GT) == records['evaluation_annotations']['sha256'], 'effective train2000 annotation drift')
    require(sha256(Path(records['source_annotations']['executor_path'])) == records['source_annotations']['sha256'], 'full training source annotation drift')
    weight = ROOT / 'assets/models/coco_clean_20260911/yolo26m-seg.pt'
    require(sha256(weight) == profile['model_sha256'], 'base weight bytes changed')
    trace = read_json(CACHE / 'trace/TRACE_MANIFEST.json')
    require(trace['native_args'] == inherited['actual_cached_native_args'], 'cached native geometry contract changed')
    require(sha256(Path(trace['predictions_path'])) == trace['predictions_sha256'], 'native predictions changed')
    split = read_json(STUDY / 'SPLIT.json')
    require(len(split['fit_image_ids']) == 1600 and len(split['heldout_image_ids']) == 400, 'split size drift')
    require(not set(split['fit_image_ids']) & set(split['heldout_image_ids']), 'fit/heldout leakage')
    maps = {x['image_id']: x['recipient_rank_to_donor_rank'] for x in read_json(STUDY / 'CONDITION_MAP.json')['images']}
    for item in trace['images']:
        m = maps[item['image_id']]
        require(sorted(m) == list(range(len(item['prediction_indices']))), 'condition map is not bijective')
        require(len(m) < 2 or all(i != j for i, j in enumerate(m)), 'condition map has an unintended fixed point')
    return dict(environment=environment, profile=profile, split=split, maps=maps, inherited=inherited, execution_freeze=frozen)


def teacher_rows(image_ids):
    """Consume only selected train images; heldout targets never enter fitting."""
    selected = set(image_ids)
    result = {i: [] for i in image_ids}
    for row in read_jsonl(TEACHER / 'TEACHER_ROWS.jsonl'):
        if row['image_id'] in selected and row['accepted']:
            result[row['image_id']].append(row)
    for values in result.values():
        values.sort(key=lambda r: r['rank'])
    return result


def resource_check():
    import psutil
    other = [p.info for p in psutil.process_iter(['pid', 'name', 'cmdline'])
             if p.pid != os.getpid() and 'python' in (p.info['name'] or '').lower()]
    require(not other, 'another Python process is running: ' + str(other))
    require(not list(ROOT.glob('experiments/*/.*.lock')), 'unresolved experiment lock')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')), 'another execution owner')
    ram, disk = psutil.virtual_memory().available, shutil.disk_usage(ROOT).free
    require(ram >= 4 * 1024**3 and disk >= 8 * 1024**3, 'insufficient memory/disk headroom')
    return dict(available_ram_bytes=ram, free_disk_bytes=disk, other_python_processes=other)


class Run(RegisteredRun):
    def __init__(self, kind, parent, budget):
        freeze = source_check()
        super().__init__('CLD_' + kind, 'CLD', parent, list(freeze['files']), budget)
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.CLD.lock'
        self.meta.update(study_id='STUDY_MASK_DIRECTION_LEARNABILITY_20261016', base_model_forward_calls=0,
                         oracle_optimization_calls=0, adapter_forward_calls=0, optimizer_steps=0)

    def __enter__(self):
        self.resources = resource_check()
        return super().__enter__()

    def freeze(self, contract):
        freeze = source_check()
        contract = dict(contract)
        authorization = read_json(STUDY / 'OPERATION_AUTHORIZATION.json')
        require(authorization['study_id'] == self.meta['study_id'], 'operation authorization belongs to another Study')
        contract['operation_authorization'] = authorization
        contract['operation_authorization_sha256'] = sha256(STUDY / 'OPERATION_AUTHORIZATION.json')
        self.sources = freeze['files']
        origin = read_json(STUDY / 'EXECUTION_CONTEXT.json')
        write_new(self.path / 'SOURCE_MANIFEST.json', freeze)
        mapping = {}
        for index, relative in enumerate(self.sources):
            target = self.path / 'source_snapshot' / (f'{index:02d}_' + Path(relative).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            mapping[relative] = target.relative_to(self.path).as_posix()
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', mapping)
        shutil.copyfile(STUDY / 'PROTOCOL.md', self.path / 'PROTOCOL.md')
        shutil.copyfile(STUDY / 'EXECUTION_CONTEXT.json', self.path / 'EXECUTION_CONTEXT.json')
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        write_new(self.path / 'RESOURCES.json', self.resources)
        self.update(code_commit=origin['code_commit'], dirty_patch_sha256=origin['tracked_scoped_diff_sha256'],
                    protocol_sha256=sha256(self.path / 'PROTOCOL.md'), contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY / 'INPUT_MANIFEST.json'), plan_status='frozen',
                    execution_freeze_sha256=sha256(STUDY / 'EXECUTION_FREEZE.json'))


def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def check_completed(run, independent=True):
    require(run.resolve().parent == (STUDY / 'runs').resolve(), 'Run belongs to another Study')
    require(read_json(run / 'run.json')['execution_status'] == 'completed', 'dependency incomplete')
    require(read_json(run / 'CONFIG_AUDIT.json')['status'] == 'passed', 'dependency configuration failed')
    if independent:
        require(read_json(run / 'INDEPENDENT.json')['status'] == 'passed', 'dependency independent verification missing')
    require(read_json(run / 'SOURCE_MANIFEST.json')['files'] == source_check()['files'], 'dependency source identity differs')
