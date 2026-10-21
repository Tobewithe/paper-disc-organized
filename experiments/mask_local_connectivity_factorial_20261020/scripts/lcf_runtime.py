"""Pinned imports, finite stage ownership, sources and actual configuration."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
import uuid

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
VENDOR=ROOT/'_site'
sys.path.insert(0,str(VENDOR))
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_CONFIG_DIR']=str(STUDY/'preflight/ultralytics_config')
(STUDY/'preflight/ultralytics_config').mkdir(parents=True,exist_ok=True)
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for p in (ROOT/'shared/evaluation',ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts',
          ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/scripts',
          ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts',
          ROOT/'experiments/mask_gate_s3r_20261015/scripts',ROOT/'experiments/mask_repair_preservation_20261016/scripts'):
    sys.path.append(str(p))
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    t.write_text(json.dumps(v,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def state_hash(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous();h.update(n.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def environment():
    from posttrain_run import configure_native
    import torch,ultralytics
    profile,env=configure_native()
    require(Path(ultralytics.__file__).resolve().is_relative_to(VENDOR.resolve()),'Ultralytics source is not pinned private vendor')
    require(ultralytics.__version__=='8.4.100','wrong Ultralytics')
    env.update(executable=sys.executable,ultralytics_file=ultralytics.__file__,gpu=torch.cuda.get_device_name(0))
    return profile,env

def frozen_check():
    frozen=read(STUDY/'FREEZE_MANIFEST.json')
    require(sha(STUDY/'PROTOCOL.md')==frozen['protocol_sha256'],'protocol drift')
    require(sha(STUDY/'CONFIG.json')==frozen['config_sha256'],'config drift')
    for row in frozen['files']:
        p=ROOT/row['path'];require(sha(p)==row['sha256'],'frozen source/input drift: '+row['path'])
    for row in frozen.get('input_files',[]):
        require(sha(row['path'])==row['sha256'],'frozen external input drift: '+row['path'])
    return frozen

class Run:
    def __init__(self,stage):
        self.stage=stage;self.path=STUDY/'runs'/('RUN_LCF_'+stage.upper()+'_'+uuid.uuid4().hex)
        self.config=read(STUDY/'CONFIG.json');self.start=time.monotonic();self.lock=STUDY/'.execution.lock'
        self.meta={'run_id':self.path.name,'study_id':self.config['study_id'],'stage':stage,'pid':os.getpid(),
                   'host':socket.gethostname(),'started_at':now(),'execution_status':'running','scientific_status':'unadjudicated',
                   'optimizer_steps':0,'command':[sys.executable,*sys.argv],'config_sha256':sha(STUDY/'CONFIG.json')}
    def __enter__(self):
        with self.lock.open('x',encoding='utf-8') as f:json.dump(self.meta,f)
        self.path.mkdir(parents=True)
        try:
            frozen=frozen_check()
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise
        write(self.path/'SOURCE_MANIFEST.json',frozen)
        for row in frozen['files']:
            target=self.path/'source_snapshot'/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/row['path'],target)
        self.meta.update(protocol_sha256=frozen['protocol_sha256'],code_commit=frozen['code_commit'],freeze_manifest_sha256=sha(STUDY/'FREEZE_MANIFEST.json'))
        write(self.path/'EVAL_CONTRACT.json',self.config)
        self.update(phase='initializing')
        return self
    def update(self,**kw):
        self.meta.update(kw);self.meta.update(updated_at=now(),elapsed_seconds=time.monotonic()-self.start)
        write(self.path/'run.json',self.meta);write(STUDY/'EXECUTION_STATUS.json',self.meta)
        s=read(STUDY/'study.json');s.update(execution_status='running' if self.meta['execution_status']=='running' else self.meta['execution_status'],
             current_phase=self.meta.get('phase'),active_run_id=self.path.name if self.meta['execution_status']=='running' else None)
        write(STUDY/'study.json',s)
    def check(self):
        import psutil
        limit=self.config['hours'].get(self.stage,24)*3600
        require(time.monotonic()-self.start<=limit,'frozen stage time exceeded')
        require(psutil.Process().memory_info().rss<=8*2**30,'RSS exceeds 8 GiB')
        require(psutil.virtual_memory().available>=.75*2**30,'available RAM below reserve')
        require(shutil.disk_usage(STUDY).free>=12*2**30,'disk reserve below 12 GiB')
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_initialized():
                require(torch.cuda.max_memory_allocated()<=6*2**30,'CUDA allocated exceeds 6 GiB')
                require(torch.cuda.max_memory_reserved()<=6*2**30,'CUDA reserved exceeds 6 GiB')
    def finish(self,summary,science='engineering_only'):
        self.check();frozen_check();write(self.path/'SUMMARY.json',summary)
        write(self.path/'CONFIGURATION_REVIEW.json',{'status':'passed','frozen_sources_and_config_exact':True,
              'protocol_sha256':self.meta['protocol_sha256'],'runtime_environment':read(self.path/'ENVIRONMENT.json') if (self.path/'ENVIRONMENT.json').exists() else None})
        self.update(execution_status='completed',phase='completed',finished_at=now(),exit_code=0,scientific_status=science)
        write(self.path/'ARTIFACT_MANIFEST.json',{p.relative_to(self.path).as_posix():sha(p) for p in self.path.rglob('*') if p.is_file()})
        write(STUDY/(self.stage.upper()+'_POINTER.json'),{'run_id':self.path.name,'summary_sha256':sha(self.path/'SUMMARY.json')})
    def __exit__(self,kind,value,tb):
        if kind:
            (self.path/'ERROR.txt').write_text(''.join(traceback.format_exception(kind,value,tb)),encoding='utf-8')
            self.update(execution_status='failed',phase='failed',error=str(value),finished_at=now(),exit_code=1,scientific_status='engineering_failure')
        if self.lock.exists() and read(self.lock).get('run_id')==self.path.name:self.lock.unlink()
        return False

def pointer(stage):
    d=read(STUDY/(stage.upper()+'_POINTER.json'));p=STUDY/'runs'/d['run_id']
    require(read(p/'run.json')['execution_status']=='completed','prerequisite incomplete: '+stage)
    require(sha(p/'SUMMARY.json')==d['summary_sha256'],'prerequisite summary drift')
    return p

def optimizer(model,config):
    import torch
    spec=dict(config['optimizer']);spec['betas']=tuple(spec['betas'])
    return torch.optim.AdamW(model.parameters(),**spec)

def save_model(path,model,opt,steps,epochs):
    import torch
    require(not path.exists(),'checkpoint overwrite')
    torch.save({'arm':model.ssr_arm,'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict() if opt else None,
                'optimizer_steps':steps,'completed_epochs':epochs,'seed':0,'config_sha256':sha(STUDY/'CONFIG.json')},path)
