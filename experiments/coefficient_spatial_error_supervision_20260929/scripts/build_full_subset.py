import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F

def bounds(b,h=160,w=160):
 x1=max(0,min(w-1,int(torch.floor(torch.tensor(float(b[0])*w/640)).item()))); y1=max(0,min(h-1,int(torch.floor(torch.tensor(float(b[1])*h/640)).item()))); x2=max(x1+1,min(w,int(torch.ceil(torch.tensor(float(b[2])*w/640)).item()))); y2=max(y1+1,min(h,int(torch.ceil(torch.tensor(float(b[3])*h/640)).item()))); return x1,y1,x2,y2

def convert(rows,raw,limit):
 ids=sorted(int(p.stem) for p in Path(raw).rglob('*.pt'))
 if limit and limit > 0: ids=ids[:limit]
 ids=set(ids); files={int(p.stem):p for p in Path(raw).rglob('*.pt')}; chosen=[r for r in rows if int(r['image_id']) in ids]; cache={}; out=[]
 for n,r in enumerate(chosen,1):
  iid=int(r['image_id']); img=cache.get(iid)
  if img is None: img=torch.load(files[iid],map_location='cpu',weights_only=True); cache[iid]=img
  proto=img['proto'].float(); box=img['boxes'][int(r['raw_id'])].float(); x1,y1,x2,y2=bounds(box); p=F.adaptive_avg_pool2d(proto[:,y1:y2,x1:x2],(16,16)).contiguous()
  out.append({'image_id':iid,'annotation_id':int(r['annotation_id']),'raw_id':int(r['raw_id']),'h':r['h'].float(),'c0':r['c0'].float(),'delta':r['delta'].float(),'p':p,'box':box})
  if n%100==0: print('rows',n,flush=True)
 return out

def main(a):
 fit=torch.load(a.fit,map_location='cpu',weights_only=True); dev=torch.load(a.dev,map_location='cpu',weights_only=True); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); ff=convert(fit,a.raw,a.limit); dd=convert(dev,a.raw,a.limit); torch.save(ff,out/'FIT_FULL.pt'); torch.save(dd,out/'DEV_FULL.pt'); (out/'MANIFEST.json').write_text(json.dumps({'fit_rows':len(ff),'dev_rows':len(dd),'images_limit':a.limit,'p_shape':list(ff[0]['p'].shape) if ff else None},indent=2)); print(json.dumps({'fit_rows':len(ff),'dev_rows':len(dd)}))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--fit',required=True); p.add_argument('--dev',required=True); p.add_argument('--raw',required=True); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=200); main(p.parse_args())


