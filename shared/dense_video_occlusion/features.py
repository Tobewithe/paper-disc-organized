"""RGB-only causal observations. Geometry labels are targets, never hidden inputs."""
import collections,gzip,json,subprocess,time
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18
from torchvision.ops import roi_align
from .common import ROOT,Run,read,save,sha

def load_tracks(name,kind='native'):
 with gzip.open(ROOT/'data'/name/(kind+'.json.gz'),'rt') as f:return json.load(f)

def overlap(b,others):
 b=np.asarray(b);area=max(1.,np.prod(b[2:]-b[:2]));value=0.
 for c in others:
  value=max(value,float(np.prod(np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])))/area))
 return value

class Extractor:
 def __init__(self):
  assert torch.cuda.is_available();torch.set_num_threads(4)
  p=ROOT/'assets/resnet18-f37072fd.pth'
  assert sha(p)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
  m=resnet18(weights=None);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
  self.m=torch.nn.Sequential(*list(m.children())[:6]).eval().cuda();self.m.requires_grad_(False)
  self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 @torch.inference_mode()
 def __call__(self,rgb,boxes):
  h,w=rgb.shape[:2]
  if not boxes:return np.empty((0,256),np.float32)
  t=torch.from_numpy(rgb.copy()).cuda().permute(2,0,1)[None].float()/255
  f=self.m((t-self.mean)/self.std)
  b=np.asarray(boxes,np.float32);b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
  b[:,2:]=np.maximum(b[:,2:],b[:,:2]+1e-2)
  center=(b[:,:2]+b[:,2:])/2;half=(b[:,2:]-b[:,:2])*2
  c=np.c_[center-half,center+half];c[:,[0,2]]=np.clip(c[:,[0,2]],0,w);c[:,[1,3]]=np.clip(c[:,[1,3]],0,h)
  bs=torch.from_numpy(np.r_[b,c]).cuda();features=roi_align(f,[bs],output_size=3,spatial_scale=1/8,sampling_ratio=2,aligned=True).mean((-1,-2))
  own,context=features.chunk(2);return torch.cat([own,context],-1).cpu().numpy()

def stream(extractor,frames,observations,width,height,drop_seed=None):
 states={};rows=[];rng=np.random.default_rng(drop_seed) if drop_seed is not None else None
 for frame,rgb in frames:
  observed={int(k):dict(v) for k,v in observations[frame].items()}
  if rng is not None:
   for k in list(observed):
    if rng.random()<.1:del observed[k];continue
    b=np.asarray(observed[k]['box'],float);noise=rng.normal(0,.015,4)*np.tile(np.maximum(1,b[2:]-b[:2]),2)
    q=b+noise;q[2:]=np.maximum(q[2:],q[:2]+1);observed[k]['box']=q.tolist()
  for k,o in observed.items():
   if k not in states:states[k]={'last':frame,'box':np.asarray(o['box'],float),'velocity':np.zeros(4),'anchor':None}
  keys=sorted(k for k,s in states.items() if k in observed or frame-s['last']<=60)
  boxes=[]
  for k in keys:
   s=states[k]
   if k in observed:
    b=np.asarray(observed[k]['box'],float);gap=max(1,frame-s['last']);s['velocity']=(b-s['box'])/gap;s['last']=frame;s['box']=b
   b=s['box']+s['velocity']*(frame-s['last']);b[2:]=np.maximum(b[2:],b[:2]+1);boxes.append(b)
  feats=extractor(rgb,boxes)
  for j,k in enumerate(keys):
   s=states[k];b=boxes[j];present=k in observed
   if s['anchor'] is None:s['anchor']=feats[j,:128].copy()
   score=observed[k].get('score',1.) if present else 0.
   geom=[float(present),(frame-s['last'])/60,(b[2]-b[0])/width,(b[3]-b[1])/height,(b[0]+b[2])/(2*width),(b[1]+b[3])/(2*height),overlap(b,[v['box'] for kk,v in observed.items() if kk!=k]),score]
   rows.append({'frame':frame,'id':k,'present':present,'x':np.r_[feats[j],s['anchor'],geom].astype(np.float32)})
 return sorted(rows,key=lambda r:(r['id'],r['frame']))

def pack(rows,path,labels=None):
 x=np.stack([r.pop('x') for r in rows]);ids=np.asarray([r['id'] for r in rows]);frames=np.asarray([r['frame'] for r in rows]);present=np.asarray([r['present'] for r in rows])
 arrays={'x':x,'ids':ids,'frames':frames,'present':present}
 if labels is not None:
  gt=[labels[int(f),int(k)] for f,k in zip(frames,ids)];arrays.update(y=np.asarray([v['state'] for v in gt]),visibility=np.asarray([v['visibility'] for v in gt],np.float32))
 np.savez_compressed(path,**arrays)
 return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'rows':len(rows)}

def video_frames(p,expected):
 cap=cv2.VideoCapture(str(p));i=0
 try:
  while True:
   ok,bgr=cap.read()
   if not ok:break
   i+=1;yield i,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 finally:cap.release()
 assert i==expected,(p,i,expected)

def main():
 with Run('features',{'inputs':'RGB and past/current visible observations only','train_targets':'synthetic geometry only','backbone':'ResNet18 frozen through layer2','window':8,'real_GT':'not read','missing':'constant velocity; no hidden box','geometry':['present','gap/60','w/W','h/H','cx/W','cy/H','max intersection/own-area','score']}) as run:
  assert sha(ROOT/'real-videos.tar')==read(ROOT/'VIDEO_RECEIPT.json')['sha256']
  hashes=read(ROOT/'data/INPUT_FILES.json')
  for row in hashes:
   p=ROOT/'data'/row['path']
   if row['path']=='manifest.json':p=ROOT/'data/sequences.json'
   assert sha(p)==row['sha256'],row['path']
  extractor=Extractor();dest=run.path/'features';dest.mkdir();catalog=[]
  scenes=read(ROOT/'scenes.json')
  for cfg in scenes:
   folder=ROOT/'synthetic_v2'/('%03d'%cfg['id']);a=read(folder/'annotations.json');assert a['frames']==32 and len(a['records'])==128
   labels={(r['frame'],r['track_id']):r for r in a['records']};obs=collections.defaultdict(dict)
   for r in a['records']:
    if r['box'] is not None:obs[r['frame']][r['track_id']]={'box':[v*2 for v in r['box']],'score':1.}
   frames=((f,np.asarray(Image.open(folder/'rgb'/('%03d.png'%f)).convert('RGB'))) for f in range(32))
   rows=stream(extractor,frames,obs,384,256,cfg['seed'] if cfg['split']=='train' else None)
   entry=pack(rows,dest/('synthetic_%03d.npz'%cfg['id']),labels);entry.update(kind='synthetic',split=cfg['split'],scene=cfg['id'],annotation_sha256=sha(folder/'annotations.json'));catalog.append(entry)
   print('FEATURE_SYNTH',cfg['id'],len(rows),flush=True)
  for seq in read(ROOT/'data/sequences.json')['sequences']:
   name=seq['name'];base=load_tracks(name);assert len(base)==seq['frames'];obs=collections.defaultdict(dict)
   scale=np.asarray([640/seq['width'],360/seq['height']]*2)
   for f in base:
    obs[f['frame']]={t['base_id']:{'box':(np.asarray(t['box'])*scale).tolist(),'score':t['score']} for t in f['tracks']}
   rows=stream(extractor,video_frames(ROOT/'real_videos'/(name+'.mp4'),seq['frames']),obs,640,360)
   if name==read(ROOT/'data/sequences.json')['sequences'][0]['name']:
    def prefix():
     for f,rgb in video_frames(ROOT/'real_videos'/(name+'.mp4'),seq['frames']):
      if f>20:break
      yield f,rgb
    short=stream(extractor,prefix(),obs,640,360);expected=[r for r in rows if r['frame']<=20]
    assert len(short)==len(expected)
    for a,b in zip(short,expected):assert a['frame']==b['frame'] and a['id']==b['id'] and np.array_equal(a['x'],b['x']),'RGB feature prefix differs'
    save(run.path/'RGB_PREFIX_CAUSALITY.json',{'sequence':name,'prefix_frames':20,'rows':len(short),'exact_match':True})
   entry=pack(rows,dest/(name+'.npz'));entry.update(kind='real',split=seq['decision_split'],name=name);catalog.append(entry);print('FEATURE_REAL',name,len(rows),flush=True)
  save(run.path/'FEATURE_MANIFEST.json',catalog);save(run.path/'SUMMARY.json',{'entries':len(catalog),'synthetic_rows':sum(c['rows'] for c in catalog if c['kind']=='synthetic'),'real_rows':sum(c['rows'] for c in catalog if c['kind']=='real'),'GT_used_for_real_inputs':False})
