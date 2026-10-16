"""New Study ownership, immutable inputs and source freezing; old Runs are read-only."""
from pathlib import Path
import json
import os
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_repair_preservation_20261016'
BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1 = ROOT / 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
AUDIT = ROOT / 'experiments/mask_opportunity_audit_20261015/runs/RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd'
GT = ROOT / 'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
sys.path.insert(0, str(ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now, save
from posttrain_audit import read_jsonl

OWN = ['PROTOCOL.md', 'CONFIG.json', 'INPUT_EXPECTATIONS.json', 'SOURCE_CONTEXT.json', 'SOURCE_DIRTY.patch',
       'scripts/objectives.py', 'scripts/analysis_stats.py', 'scripts/native_geometry.py',
       'scripts/mechanism_common.py', 'scripts/prepare.py', 'scripts/run_mechanism.py',
       'scripts/verify.py', 'scripts/test_mechanism.py', 'scripts/launch.ps1']
CODE = ['experiments/mask_repair_preservation_20261016/' + p for p in OWN]
CODE += ['experiments/mask_readout_mechanism_v2_20261012/scripts/' + p for p in (
    'posttrain_run.py', 'adapter_data.py', 's1_geometry.py')]
CODE += ['shared/evaluation/' + p for p in ('posttrain_audit.py', 'native_output_bridge.py', 'native_source_pins.json')]
CODE += ['paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']


def inspect_inputs():
    cfg = read_json(STUDY / 'CONFIG.json')
    require(socket.gethostname().upper() == cfg['executor']['host'], 'authorized laptop required')
    require(str(ROOT).lower() == cfg['executor']['root'].lower(), 'unexpected executor checkout')
    require(Path(sys.executable).resolve() == Path(cfg['executor']['python']).resolve(), 'wrong interpreter')
    expected = read_json(STUDY / 'INPUT_EXPECTATIONS.json')
    for relative, digest in expected['files'].items():
        require(sha256(ROOT / relative) == digest, 'input changed: ' + relative)
    profile, environment = configure_native()
    require(environment == read_json(AUDIT / 'EVAL_CONTRACT.json')['environment'], 'native runtime changed')
    return cfg, expected, profile, environment


def ownership_check():
    import psutil
    require(not list(ROOT.glob('experiments/*/.*.lock')), 'existing experiment lock; do not remove')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')), 'existing execution owner; do not remove')
    others = [p.info for p in psutil.process_iter(['pid', 'name', 'cmdline'])
              if p.info['pid'] != os.getpid() and 'python' in (p.info['name'] or '').lower()]
    require(not others, 'other Python task present; leave it unchanged: ' + str(others))


def resources(run):
    import psutil
    cfg = read_json(STUDY / 'CONFIG.json')['budget']
    run.check_budget()
    require(psutil.Process().memory_info().rss < cfg['rss_bytes'], 'RSS budget exceeded')
    require(psutil.virtual_memory().available >= cfg['min_free_ram_bytes'], 'RAM reserve exhausted')
    require(shutil.disk_usage(ROOT).free >= cfg['min_free_disk_bytes'], 'disk reserve exhausted')
    require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()) <= cfg['output_bytes'], 'artifact budget exceeded')


class MechanismRun(RegisteredRun):
    def __init__(self, scope, parent):
        cfg = read_json(STUDY / 'CONFIG.json')
        super().__init__('REPAIR_KEEP_' + scope.upper(), 'FINITE_MECHANISM', parent, CODE,
                         cfg['budget'][scope + '_seconds'], protocol='PROTOCOL.md')
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.FINITE_MECHANISM.lock'
        self.meta.update(study_id='STUDY_MASK_REPAIR_PRESERVATION_20261016', scope=scope,
            base_model_forward_calls=0, new_training_steps=0, oracle_optimization_calls=0,
            baseline_run_id=BASE.name, source_S1_run_id=S1.name)

    def __enter__(self):
        ownership_check()
        return super().__enter__()

    def freeze(self, contract):
        frozen = read_json(STUDY / 'FREEZE_MANIFEST.json')
        self.sources = {rel: sha256(ROOT / rel) for rel in CODE}
        require(self.sources == frozen['files'], 'frozen code/protocol drift')
        require(sha256(STUDY / 'PREPARATION.json') == frozen['preparation_sha256'], 'preparation drift')
        origin = read_json(STUDY / 'SOURCE_CONTEXT.json')
        self.meta.update(code_commit=origin['code_commit'], source_context_sha256=sha256(STUDY / 'SOURCE_CONTEXT.json'),
                         dirty_patch_sha256=sha256(STUDY / 'SOURCE_DIRTY.patch'))
        write_new(self.path / 'SOURCE_MANIFEST.json', frozen)
        mapping = {}
        for i, rel in enumerate(self.sources):
            dest = self.path / 'source_snapshot' / (f'{i:02d}_' + Path(rel).name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
            mapping[rel] = dest.relative_to(self.path).as_posix()
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', mapping)
        for name in ('PROTOCOL.md', 'CONFIG.json', 'PREPARATION.json', 'INPUT_EXPECTATIONS.json'):
            shutil.copyfile(STUDY / name, self.path / name)
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path / 'PROTOCOL.md'),
            protocol_sha256=sha256(self.path / 'PROTOCOL.md'), input_manifest_sha256=sha256(STUDY / 'INPUT_EXPECTATIONS.json'),
            contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'))


def jsonl(path, rows):
    with path.open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
