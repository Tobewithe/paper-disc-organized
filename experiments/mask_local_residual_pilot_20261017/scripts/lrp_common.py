"""LRP-v1 immutable inputs and registered runs; no scheduler or goal."""
from pathlib import Path
import json, os, shutil, socket, sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_local_residual_pilot_20261017'
CLD=ROOT/'experiments/mask_direction_learnability_20261016'
LEA=ROOT/'experiments/mask_local_editability_20261017'
for rel in ('experiments/mask_direction_learnability_20261016/scripts',
            'experiments/mask_readout_mechanism_v2_20261012/scripts',
            'experiments/mask_repair_preservation_20261016/scripts',
            'experiments/mask_local_editability_20261017/scripts',
            'experiments/official_gt_failure_rebaseline_20261013/scripts','shared/evaluation'):
    sys.path.append(str(ROOT/rel))
from posttrain_run import RegisteredRun, configure_native, read_json, write_new, save, sha256, require, now, tensor_state_hash
from posttrain_audit import read_jsonl
from cld_common import CACHE, GT, FIT, inspect_inputs as inherited_inputs

def jsonl(path, rows):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def source_check():
    freeze=read_json(STUDY/'EXECUTION_FREEZE.json')
    for rel,digest in freeze['files'].items():
        require(sha256(ROOT/rel)==digest,'source drift: '+rel)
    return freeze

def inspect_inputs():
    source_check()
    require(socket.gethostname().upper()=='YANGLEI','wrong executor host')
    require(str(ROOT).lower()=='d:\\coco_wire\\baseline_native_20261013','wrong executor root')
    result=inherited_inputs()
    lea=LEA/'runs/RUN_LEA_FULL_b383a8db922c451cbc77f2069d385cd7'
    require(read_json(lea/'INDEPENDENT_LAPTOP.json')['status']=='passed','LEA verification incomplete')
    require(read_json(lea/'SUMMARY.json')['decision']=='NEGATIVE_CURRENT_REGION','LEA evidence changed')
    return result

def resources():
    import psutil,torch
    require(psutil.Process().memory_info().rss<6*1024**3,'RSS cap')
    require(psutil.virtual_memory().available>=.75*1024**3,'RAM reserve')
    require(shutil.disk_usage(ROOT).free>=4*1024**3,'disk reserve')
    require(torch.cuda.memory_reserved()<=6*1024**3,'GPU cap')

class Run(RegisteredRun):
    def __init__(self,kind,parent='',budget=7200):
        frozen=source_check()
        super().__init__('LRP_'+kind,'LRP',parent,list(frozen['files']),budget,'PROTOCOL.md')
        self.path=STUDY/'runs'/self.run_id
        self.lock=STUDY/'.LRP.lock'
        self.meta.update(study_id='STUDY_MASK_LOCAL_RESIDUAL_PILOT_20261017',base_model_forward_calls=0,oracle_optimization_calls=0,optimizer_steps=0)
    def __enter__(self):
        import psutil
        require(not list(ROOT.glob('experiments/*/.*.lock')),'unresolved stage lock')
        owners=list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard'))
        require(not owners,'another autonomous owner')
        allowed={os.getpid()}
        state=STUDY/'PROCESS_STATUS.json'
        if state.exists():
            data=read_json(state)
            if data.get('status')=='running':
                p=psutil.Process(data['pid'])
                require('lrp_controller.py' in ' '.join(p.cmdline()),'controller identity changed')
                allowed.add(p.pid)
        other=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.pid not in allowed and 'python' in (p.info['name'] or '').lower()]
        require(not other,'other Python process: '+str(other))
        resources()
        return super().__enter__()
    def freeze(self,contract):
        self.sources=source_check()['files']
        write_new(self.path/'SOURCE_MANIFEST.json',dict(files=self.sources))
        mapping={}
        for i,rel in enumerate(self.sources):
            dst=self.path/'source_snapshot'/f'{i:03d}_{Path(rel).name}'
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel,dst)
            mapping[rel]=dst.relative_to(self.path).as_posix()
        write_new(self.path/'SOURCE_SNAPSHOT_MAP.json',mapping)
        for name in ('PROTOCOL.md','CONFIG.json','AUTHORIZATION.json','SOURCE_CONTEXT.json'):
            shutil.copyfile(STUDY/name,self.path/name)
        write_new(self.path/'EVAL_CONTRACT.json',contract)
        self.update(plan_status='frozen',protocol_sha256=sha256(self.path/'PROTOCOL.md'),contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),code_commit=read_json(STUDY/'SOURCE_CONTEXT.json')['code_commit'])
    def check_budget(self):
        super().check_budget()
        resources()

def completed(path):
    require(path.resolve().parent==(STUDY/'runs').resolve(),'foreign dependency')
    require(read_json(path/'run.json')['execution_status']=='completed','incomplete dependency')
    require(read_json(path/'CONFIG_AUDIT.json')['status']=='passed','dependency config invalid')

def config_audit(run,**values):
    inspect_inputs()
    write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',base_model_forward_calls=0,**values))
