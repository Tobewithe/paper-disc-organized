"""Read-only export of frozen laptop inputs; no training or new tracker execution."""
import datetime,hashlib,json,pathlib,shutil,tarfile,traceback,uuid,gzip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

R=pathlib.Path('D:/thesis-video-occ-transfer-20261019')
OLD=pathlib.Path('D:/thesis-video-v4-20261019')

def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,j):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2),encoding='utf-8')
class Head(nn.Module):
 def __init__(self,ck):
  super().__init__();self.net=nn.Sequential(nn.Linear(ck['dim'],ck['config']['hidden']),nn.ReLU(),nn.Linear(ck['config']['hidden'],ck['config']['embedding']))
 def forward(self,x):return F.normalize(self.net(x),dim=-1)

def main():
 R.mkdir(exist_ok=True);rid='RUN_LEGACY_EXPORT_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8]
 run=R/'runs'/rid;run.mkdir(parents=True)
 meta={'run_id':rid,'study_id':'STUDY_DENSE_VIDEO_OCCLUSION_TRANSFER_20261019','stage_id':'legacy_input_export','executor':'28358lan source data export only','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_status':'running','training_steps':0,'source_sha256':sha(__file__)};save(run/'run.json',meta)
 try:
  data=R/'data';data.mkdir(exist_ok=False)
  manifest=read(OLD/'input_manifest.json');pointer=read(OLD/'identity_pointer.json');cp=pathlib.Path(pointer['checkpoint']);assert sha(cp)==pointer['sha256']
  ck=torch.load(cp,map_location='cpu',weights_only=False);model=Head(ck);model.load_state_dict(ck['state']);model.eval();torch.set_num_threads(4)
  er=pathlib.Path(read(OLD/'decision_evaluation_pointer.json')['path']);rows=[]
  for seq in manifest['sequences']:
   name=seq['name'];dest=data/name;dest.mkdir();cache=pathlib.Path(seq['cache_path']);rec=read(cache/'detections.json');raw=np.load(cache/'features.npz')['x']
   x=np.clip((raw-ck['mean'])/ck['std'],-8,8).astype(np.float32)
   with torch.inference_mode():z=np.concatenate([model(torch.from_numpy(x[i:i+2048])).numpy() for i in range(0,len(x),2048)])
   np.savez_compressed(dest/'features64.npz',z=z);save(dest/'detections.json',rec)
   shutil.copy2(seq['gt'],dest/'gt.txt');sources={}
   for arm,outname in [('bytetrack','native'),('botsort','reference'),('confirm','confirm')]:
    opts=[er/arm/name/'tracks.json']
    if seq.get('v3_run'):opts.append(pathlib.Path(seq['v3_run'])/arm/name/'tracks.json')
    p=next((p for p in opts if p.exists()),None);assert p is not None,(name,arm,opts)
    with gzip.open(dest/(outname+'.json.gz'),'wt',encoding='utf-8') as f:json.dump(read(p),f)
    sources[outname]={'source':str(p),'sha256':sha(p)}
   rows.append({**{k:seq[k] for k in ['name','frames','fps','width','height','video_sha256','gt_sha256','decision_split']},'original_video':seq['video'],'cache_features_sha256':sha(cache/'features.npz'),'identity_checkpoint_sha256':pointer['sha256'],'sources':sources})
   print('EXPORTED',name,len(rec),len(z),flush=True)
  save(data/'manifest.json',{'sequences':rows,'identity':pointer,'training_steps':0})
  vendor=pathlib.Path('D:/thesis-video-20261018/vendor/TrackEval-master/trackeval');assert vendor.exists();shutil.copytree(vendor,data/'vendor/trackeval',ignore=shutil.ignore_patterns('__pycache__'))
  save(data/'config.json',read(OLD/'config.json'))
  hashes=[{'path':str(p.relative_to(data)).replace('\\','/'),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(data.rglob('*')) if p.is_file()]
  save(data/'MANIFEST.json',hashes)
  with tarfile.open(R/'real-inputs.tar.gz','w:gz',compresslevel=1) as tf:tf.add(data,arcname='data')
  save(R/'EXPORT_RECEIPT.json',{'run_id':rid,'sha256':sha(R/'real-inputs.tar.gz'),'size_bytes':(R/'real-inputs.tar.gz').stat().st_size,'files':len(hashes),'sequences':len(rows)})
  meta.update(execution_status='completed',exit_code=0)
 except Exception:
  (run/'ERROR.txt').write_text(traceback.format_exc());meta.update(execution_status='failed',exit_code=1);raise
 finally:
  meta['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(run/'run.json',meta)

if __name__=='__main__':main()
