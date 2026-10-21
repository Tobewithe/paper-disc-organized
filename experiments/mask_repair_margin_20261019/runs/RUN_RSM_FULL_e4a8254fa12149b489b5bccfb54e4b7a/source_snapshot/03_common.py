"""Own Study lifecycle; source caches and other sessions are read-only."""
from pathlib import Path
import json, os, shutil, socket, sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]
PARENT=ROOT/'experiments/mask_readout_mechanism_v2_20261012'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
S1=PARENT/'runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949'
PRIOR=ROOT/'experiments/mask_repair_preservation_20261016'
C_FULL=PRIOR/'runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01'
C_SMOKE=PRIOR/'runs/RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7'
OLD=ROOT/'experiments/mask_opportunity_audit_20261015/runs/RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd'
FIT=ROOT/'experiments/mask_coefficient_error_transfer_20261017/runs/RUN_CET_FIT_82cb342bda694f4da095bfcb83c9980e'
RAY=ROOT/'experiments/mask_adapter_ray_audit_20261016/runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b'
GT=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'

# Mirror the actual original laptop environment before torch/ultralytics import.
if socket.gethostname().upper()=='YANGLEI':
    old=json.loads((RAY/'EVAL_CONTRACT.json').read_text(encoding='utf8'))['environment']['execution_env']
    for key,value in old.items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key]=value
    sys.path.insert(0,str(ROOT/'_site'))
os.environ['OPENBLAS_NUM_THREADS']='1'
for path in (PARENT/'scripts', ROOT/'shared/evaluation', PRIOR/'scripts'):
    sys.path.append(str(path))
from posttrain_run import RegisteredRun, read_json, write_new, save, sha256, require, now, configure_native


def lines(path):
    with Path(path).open(encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]


def science_files():
    own=[STUDY/x for x in ['PROTOCOL.md','CONFIG.json','EXECUTION_CONTEXT.json']]
    own+=sorted((STUDY/'scripts').glob('*.py'))
    deps=[PARENT/'scripts'/x for x in ['posttrain_run.py','adapter_data.py','s1_geometry.py']]
    deps += [ROOT/'shared/evaluation'/x for x in ['posttrain_audit.py','native_output_bridge.py','native_source_pins.json']]
    deps += [PRIOR/'scripts/native_geometry.py',ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in own+deps}


def check_freeze():
    frozen=read_json(STUDY/'FREEZE_MANIFEST.json')
    require(science_files()==frozen['files'],'frozen science source changed')
    require(sha256(STUDY/'INPUT_MANIFEST.json')==frozen['input_manifest_sha256'],'input manifest changed')
    require(sha256(STUDY/'PREPARATION.json')==frozen['preparation_sha256'],'preparation changed')
    return frozen


def inputs_check():
    data=read_json(STUDY/'INPUT_MANIFEST.json')
    for r in data['files']:
        require(sha256(ROOT/r['path'])==r['sha256'],'input changed: '+r['path'])
    return data


def resource_check(run=None):
    import psutil
    cfg=read_json(STUDY/'CONFIG.json')['budget']
    if run:run.check_budget()
    ram=psutil.virtual_memory().available
    require(ram>=cfg['min_free_ram_bytes' if run else 'min_start_ram_bytes'],'insufficient RAM reserve')
    require(psutil.Process().memory_info().rss<=cfg['rss_bytes'],'RSS limit')
    require(shutil.disk_usage(ROOT).free>=cfg['min_free_disk_bytes'],'disk reserve')
    if 'torch' in sys.modules:
        import torch
        require(torch.cuda.memory_allocated()<=cfg['gpu_bytes'],'GPU allocation limit')
    return {'at':now(),'available_ram_bytes':ram,'disk_free_bytes':shutil.disk_usage(ROOT).free,
            'other_python_processes':[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.pid!=os.getpid() and 'python' in (p.info['name'] or '').lower()]}


class MarginRun(RegisteredRun):
    def __init__(self,scope,parent='none'):
        cfg=read_json(STUDY/'CONFIG.json');f=check_freeze()
        super().__init__('RSM_'+scope.upper(),'RSM',parent,list(f['files']),cfg['budget'][scope+'_seconds'],protocol='PROTOCOL.md')
        self.path=STUDY/'runs'/self.run_id; self.lock=STUDY/'.RSM.lock'
        self.meta.update(study_id=cfg['study_id'],scope=scope,new_training_steps=0,base_model_forward_calls=0,
                         adapter_forward_calls=0,G_budget_reference_status=cfg['G_budget']['status'])
    def __enter__(self):
        require(socket.gethostname().upper()=='YANGLEI','wrong executor')
        require(Path(sys.executable).resolve()==Path('C:/Users/28358/anaconda3/envs/pytorch/python.exe').resolve(),'wrong Python')
        self.resources=resource_check()
        return super().__enter__()
    def freeze(self,contract):
        f=check_freeze(); self.sources=f['files'];context=read_json(STUDY/'EXECUTION_CONTEXT.json')
        self.meta.update(code_commit=context['code_commit'],dirty_patch_sha256=context['tracked_diff_sha256'])
        mapping={}
        for index,rel in enumerate(self.sources):
            dst=self.path/'source_snapshot'/f'{index:02d}_{Path(rel).name}';dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel,dst);mapping[rel]=dst.relative_to(self.path).as_posix()
        write_new(self.path/'SOURCE_SNAPSHOT_MAP.json',mapping)
        write_new(self.path/'SOURCE_MANIFEST.json',f)
        for name in ['PROTOCOL.md','CONFIG.json','INPUT_MANIFEST.json','PREPARATION.json']:
            shutil.copyfile(STUDY/name,self.path/name)
        write_new(self.path/'RESOURCES.json',self.resources);write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_sha256=sha256(STUDY/'PROTOCOL.md'),
                    input_manifest_sha256=f['input_manifest_sha256'],contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))
