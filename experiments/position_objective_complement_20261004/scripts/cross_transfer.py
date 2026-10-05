import argparse,json
from pathlib import Path
import numpy as np, torch
from ultralytics.utils import ops
from pycocotools.coco import COCO
def iou(a,b): return float(((a&b).sum())/((a|b).sum().clamp_min(1)))
def main(a):
 root=a.root; coco=COCO(str(root/'data/annotations/instances_val2017.json')); split=json.loads((root/'assets/VAL_ONLY_SPLIT.json').read_text())['val']; arm=json.loads((root/'stage1'/'G-BIAS.json').read_text()); bias=[torch.tensor(x,dtype=torch.float32) for x in arm['A']]; rec=[]
 for iid in split:
  im=torch.load(root/'official_cache'/'images'/f'{int(iid):012d}.pt',map_location='cpu',weights_only=False); rows=im['rows']; ids=[int(r['raw_id']) for r in rows]; boxes=im['boxes'].float(); proto=im['proto'].float(); c=im['coeff'][ids].float(); c2=c.clone()
  for k,raw in enumerate(ids): c2[k]+=bias[int(im['levels'][raw])]
  ma=ops.process_mask(proto,c,boxes[ids],(640,640),upsample=True); mb=ops.process_mask(proto,c2,boxes[ids],(640,640),upsample=True); rp=im['ratio_pad']; sh=tuple(im['original_shape']); ma=ops.scale_masks(ma[:,None].float(),sh,ratio_pad=rp)[:,0]>.5; mb=ops.scale_masks(mb[:,None].float(),sh,ratio_pad=rp)[:,0]>.5; va=[];vb=[]
  for k,r in enumerate(rows):
   gt=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])])).bool(); va.append(iou(ma[k],gt)); vb.append(iou(mb[k],gt))
  rec.append({'image_id':int(iid),'n':len(va),'A':float(np.mean(va)),'OWN_AFF':float(np.mean(vb)),'delta':float(np.mean(vb)-np.mean(va)),'repairs':sum(x<.75<=y for x,y in zip(va,vb)),'damage':sum(x>=.75>y for x,y in zip(va,vb))})
 out=root/'cross_transfer'; out.mkdir(exist_ok=True); (out/'PER_IMAGE.jsonl').write_text('\n'.join(json.dumps(x) for x in rec),encoding='utf-8'); d=np.array([x['delta'] for x in rec]); rng=np.random.default_rng(20261004); idx=rng.integers(0,len(rec),(5000,len(rec))); s={'images':len(rec),'candidates':sum(x['n'] for x in rec),'A_macro':float(np.mean([x['A'] for x in rec])),'OWN_AFF_macro':float(np.mean([x['OWN_AFF'] for x in rec])),'delta_pp':float(d.mean()*100),'ci_pp':(np.quantile(d[idx].mean(1),[.025,.975])*100).tolist(),'repairs':sum(x['repairs'] for x in rec),'damage':sum(x['damage'] for x in rec),'method':'official TAL positions; own coefficient + frozen geometry-trained G-BIAS'}; (out/'SUMMARY.json').write_text(json.dumps(s,indent=2),encoding='utf-8'); print(json.dumps(s))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); main(ap.parse_args())
