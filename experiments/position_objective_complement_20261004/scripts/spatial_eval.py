import argparse,json
from pathlib import Path
import numpy as np,torch
from ultralytics.utils import ops
from pycocotools.coco import COCO

def iou(a,b): return float(((a&b).sum())/((a|b).sum().clamp_min(1)))

def main(a):
 root=a.root; x=json.loads((root/'spatial_affine'/'SOLVE.json').read_text()); A=[torch.tensor(v,dtype=torch.float32) for v in x['A']]; st={int(k):(torch.tensor(v[0]),torch.tensor(v[1])) for k,v in x['stats'].items()}; coco=COCO(str(root/'data/annotations/instances_val2017.json')); split=json.loads((root/'assets'/'VAL_ONLY_SPLIT.json').read_text())['val']; rec=[]
 for iid in split:
  im=torch.load(root/'geometry_val'/'images'/f'{int(iid):012d}.pt',map_location='cpu',weights_only=False); rows=im['rows']; ids=[int(r['raw_id']) for r in rows]; proto=im['proto'].float(); boxes=im['boxes'].float(); c=im['coeff'][ids].float(); c2=c.clone(); sizes=[(80,80),(40,40),(20,20)]; offs=[0,6400,8000]
  for k,(raw,r) in enumerate(zip(ids,rows)):
   l=int(r['level']); w,h=sizes[l]; z=raw-offs[l]; xx=(z%w)/(w-1)*2-1; yy=(z//w)/(h-1)*2-1; q=torch.tensor([1.,xx,yy,xx*xx,yy*yy,xx*yy]); hh=torch.cat(((im['h'][raw].float()-st[l][0][:64])/st[l][1][:64],q,torch.ones(1))); c2[k]+=hh@A[l]
  ma=ops.process_mask(proto,c,boxes[ids],(640,640),upsample=True); mb=ops.process_mask(proto,c2,boxes[ids],(640,640),upsample=True); rp=((float(im['gain']),float(im['gain'])),(float(im['left']),float(im['top']))); sh=tuple(im['original_shape']); ma=ops.scale_masks(ma[:,None].float(),sh,ratio_pad=rp)[:,0]>.5; mb=ops.scale_masks(mb[:,None].float(),sh,ratio_pad=rp)[:,0]>.5; va=[];vb=[]
  for k,r in enumerate(rows):
   gt=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])])).bool(); va.append(iou(ma[k],gt)); vb.append(iou(mb[k],gt))
  rec.append({'image_id':int(iid),'n':len(va),'A':float(np.mean(va)),'B':float(np.mean(vb)),'delta':float(np.mean(vb)-np.mean(va)),'repairs':sum(x<.75<=y for x,y in zip(va,vb)),'damage':sum(x>=.75>y for x,y in zip(va,vb))})
 out=root/'spatial_affine'; (out/'PER_IMAGE.jsonl').write_text('\n'.join(json.dumps(r) for r in rec),encoding='utf-8'); d=np.array([r['delta'] for r in rec]); rng=np.random.default_rng(20261004); idx=rng.integers(0,len(rec),(5000,len(rec))); s={'images':len(rec),'candidates':sum(r['n'] for r in rec),'A_macro':float(np.mean([r['A'] for r in rec])),'B_macro':float(np.mean([r['B'] for r in rec])),'delta_pp':float(d.mean()*100),'ci_pp':(np.quantile(d[idx].mean(1),[.025,.975])*100).tolist(),'repairs':sum(r['repairs'] for r in rec),'damage':sum(r['damage'] for r in rec)}; (out/'SUMMARY.json').write_text(json.dumps(s,indent=2),encoding='utf-8'); print(json.dumps(s))

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); main(ap.parse_args())
