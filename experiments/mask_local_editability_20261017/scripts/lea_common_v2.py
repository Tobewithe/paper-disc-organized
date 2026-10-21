"""Own-study registration and immutable source/input contracts."""
from pathlib import Path
import json, os, shutil, socket, sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_local_editability_20261017'
BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1 = ROOT / 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
PREV = ROOT / 'experiments/mask_repair_preservation_20261016'
CFULL = PREV / 'runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01'
CSMOKE = PREV / 'runs/RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7'
RAY = ROOT / 'experiments/mask_adapter_ray_audit_20261016'
S3 = ROOT / 'experiments/mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d'
GT = ROOT / 'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
for rel in ('experiments/mask_readout_mechanism_v2_20261012/scripts',
            'experiments/mask_repair_preservation_20261016/scripts',
            'shared/evaluation', 'experiments/official_gt_failure_rebaseline_20261013/scripts'):
    sys.path.insert(0, str(ROOT / rel))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, sha256, require, now, save
from posttrain_audit import read_jsonl


def jsonl(path, rows):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def check_ownership():
    import psutil
    require(not list(ROOT.glob('experiments/*/.*.lock')), 'Existing stage lock; inspect ownership')
    owners = list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard'))
    allowed = {os.getpid()}
    if owners:
        require(owners == [STUDY / '.AUTONOMOUS_OWNER.guard'], 'Other autonomous owner')
        owner = read_json(owners[0])
        require(owner['study_id'] == 'STUDY_MASK_LOCAL_EDITABILITY_20261017', 'Owner identity changed')
        process = psutil.Process(owner['pid'])
        command = ' '.join(process.cmdline()).replace('\\', '/')
        require(any('mask_local_editability_20261017/scripts/' + name in command for name in ('controller.py', 'controller_v2.py')), 'Owner process identity changed')
        allowed.add(owner['pid'])
    others = [p.info for p in psutil.process_iter(['pid', 'name', 'cmdline'])
              if p.info['pid'] not in allowed and 'python' in (p.info['name'] or '').lower()]
    require(not others, 'Other Python task present: ' + str(others))


def verify_files(mapping):
    for rel, digest in mapping.items():
        path = ROOT / rel
        require(path.is_file() and sha256(path) == digest, 'Frozen input/source changed: ' + rel)


def inspect_inputs():
    cfg = read_json(STUDY / 'CONFIG.json')
    require(socket.gethostname().upper() == cfg['executor']['host'], 'Unauthorized host')
    require(ROOT.resolve() == Path(cfg['executor']['root']).resolve(), 'Executor root changed')
    require(Path(sys.executable).resolve() == Path(cfg['executor']['python']).resolve(), 'Interpreter changed')
    frozen = read_json(STUDY / 'EXECUTION_FREEZE_V2.json')
    verify_files(frozen['files'])
    manifest = read_json(STUDY / 'INPUT_MANIFEST.json')
    verify_files(manifest['files'])
    profile, env = configure_native()
    require(env == manifest['environment'], 'Runtime differs from frozen native contract')
    return cfg, manifest, profile, env


def resource_check(run):
    import psutil, torch
    cfg = run.cfg['budget']
    run.check_budget()
    require(psutil.Process().memory_info().rss <= cfg['rss_bytes'], 'RSS budget exceeded')
    require(psutil.virtual_memory().available >= cfg['min_ram_bytes'], 'RAM reserve exhausted')
    require(shutil.disk_usage(ROOT).free >= cfg['min_disk_bytes'], 'Disk reserve exhausted')
    require(torch.cuda.memory_reserved() <= cfg['gpu_reserved_bytes'], 'GPU reserve budget exceeded')


class Run(RegisteredRun):
    def __init__(self, scope, parent):
        self.cfg = read_json(STUDY / 'CONFIG.json')
        self.frozen = read_json(STUDY / 'EXECUTION_FREEZE_V2.json')
        super().__init__('LEA_' + ('SMOKE_RECOVERY' if scope == 'smoke' else scope.upper()), 'LOCAL_EDITABILITY', parent,
                         list(self.frozen['files']), self.cfg['budget'][scope + '_seconds'], 'PROTOCOL.md')
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.LOCAL_EDITABILITY.lock'
        self.meta.update(study_id=self.cfg['study_id'], scope=scope, new_training_steps=0,
                         base_model_forward_calls=0, adapter_forward_calls=0, oracle_optimization_calls=0,
                         baseline_run_id=BASE.name, existing_method_run_id=S3.name)

    def __enter__(self):
        check_ownership()
        return super().__enter__()

    def freeze(self, contract):
        verify_files(self.frozen['files'])
        self.sources = self.frozen['files']
        origin = read_json(STUDY / 'SOURCE_CONTEXT.json')
        self.meta.update(code_commit=origin['code_commit'], dirty_patch_sha256=sha256(STUDY / 'PREEXISTING_TRACKED_DIFF.patch'))
        write_new(self.path / 'SOURCE_MANIFEST.json', self.frozen)
        mapping = {}
        for i, rel in enumerate(self.sources):
            dst = self.path / 'source_snapshot' / (str(i).zfill(3) + '_' + Path(rel).name)
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / rel, dst)
            mapping[rel] = dst.relative_to(self.path).as_posix()
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', mapping)
        for name in ('PROTOCOL.md', 'CONFIG.json', 'INPUT_MANIFEST.json', 'PREPARATION.json'):
            shutil.copyfile(STUDY / name, self.path / name)
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path / 'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path / 'PROTOCOL.md'),
                    input_manifest_sha256=sha256(self.path / 'INPUT_MANIFEST.json'),
                    contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'))
