import datetime,hashlib,json,os,pathlib,platform,shutil,sys,traceback,uuid
ROOT=pathlib.Path(os.environ.get('OCC_ROOT','/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2'))
STUDY=ROOT/'experiments/dense_video_occlusion_transfer_20261019'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,j):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(j,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
class Run:
 def __init__(self,stage,contract=None):self.stage=stage;self.contract=contract or {}
 def __enter__(self):
  self.id='RUN_OCC_'+self.stage.upper()+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8]
  self.path=STUDY/'runs'/self.id;self.path.mkdir(parents=True,exist_ok=False)
  self.lock=STUDY/(self.stage+'.lock')
  with self.lock.open('x') as f:f.write(self.id)
  sources=[]
  for p in sorted((ROOT/'shared/dense_video_occlusion').glob('*.py')):
   q=self.path/'source'/p.name;q.parent.mkdir(exist_ok=True);shutil.copy2(p,q);sources.append({'path':p.name,'sha256':sha(p)})
  save(self.path/'SOURCE_MANIFEST.json',sources)
  inputs={}
  for p in [STUDY/'PROTOCOL.md',STUDY/'IMPLEMENTATION_CONTRACT.md',ROOT/'assets/PACKAGE_MANIFEST.json',ROOT/'data/sequences.json',ROOT/'data/MANIFEST.json',ROOT/'scenes.json']:
   if p.exists():q=self.path/'inputs'/p.name;q.parent.mkdir(exist_ok=True);shutil.copy2(p,q);inputs[str(p.relative_to(ROOT))]=sha(p)
  save(self.path/'EVAL_CONTRACT.json',self.contract)
  self.meta={'study_id':'STUDY_DENSE_VIDEO_OCCLUSION_TRANSFER_20261019','run_id':self.id,'stage_id':self.stage,'started_at':now(),'execution_status':'running','executor':platform.node(),'python':sys.executable,'pid':os.getpid(),'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271','slurm_job_id':os.environ.get('SLURM_JOB_ID'),'command':sys.argv,'inputs':inputs,'source_manifest_sha256':sha(self.path/'SOURCE_MANIFEST.json'),'contract_sha256':sha(self.path/'EVAL_CONTRACT.json'),'training_steps':0}
  save(self.path/'run.json',self.meta);print('RUN_START',self.id,flush=True);return self
 def __exit__(self,typ,exc,tb):
  self.meta.update(execution_status='failed' if exc else 'completed',exit_code=1 if exc else 0,ended_at=now())
  if exc:(self.path/'ERROR.txt').write_text(''.join(traceback.format_exception(typ,exc,tb)))
  save(self.path/'run.json',self.meta)
  if self.lock.exists() and self.lock.read_text()==self.id:self.lock.unlink()
  if not exc:save(ROOT/(self.stage+'_pointer.json'),{'run_id':self.id,'path':str(self.path)})
  print('RUN_END',self.id,self.meta['execution_status'],flush=True)
