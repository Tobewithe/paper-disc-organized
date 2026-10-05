"""Build native 160x160 objective rows at the frozen official TAL raw ids."""
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics.utils import ops
from pycocotools.coco import COCO

def build(group, cache, ann_json, out):
    coco=COCO(str(ann_json)); index=json.loads((cache/'INDEX.json').read_text())
    rows=[]
    for item in index[group]:
        iid=int(item['image_id']); im=torch.load(cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
        proto=im['proto'].float(); anns=coco.imgToAnns[iid]
        (gx,gy),(px,py)=im['ratio_pad']; gx=float(gx); gy=float(gy); px=float(px); py=float(py)
        oh,ow=map(int,im['original_shape']); gain=float(gx)
        nh,nw=round(oh*gain),round(ow*gain); top=round((640-nh)/2-.1); left=round((640-nw)/2-.1)
        for k,r in enumerate(im['rows']):
            ann=coco.anns[int(r['annotation_id'])]
            gt=torch.from_numpy(coco.annToMask(ann)).float()[None,None]
            padded=F.pad(F.interpolate(gt,(nh,nw),mode='nearest'),(left,640-nw-left,top,640-nh-top))
            native_gt=F.interpolate(padded,(160,160),mode='nearest')[0,0]
            box=im['target_boxes'][k].float(); nb=box/4.0
            xx=torch.arange(160)[None,:]; yy=torch.arange(160)[:,None]
            support=(xx>=nb[0])&(xx<nb[2])&(yy>=nb[1])&(yy<nb[3])
            if not bool(support.any()):
                # Preserve the official candidate identity.  A sub-pixel native
                # box has no integer support; use its clamped center pixel and
                # record the deterministic boundary fallback in the metadata.
                cx=int(torch.clamp(((nb[0]+nb[2])/2).floor(),0,159)); cy=int(torch.clamp(((nb[1]+nb[3])/2).floor(),0,159)); support=torch.zeros((160,160),dtype=torch.bool); support[cy,cx]=True
            y=native_gt[support].contiguous(); p=proto[:,support].T.contiguous()
            area=max(float(((nb[2:]-nb[:2]).clamp_min(0)).prod()),1.0)
            rows.append({'meta':{'image_id':iid,'annotation_id':int(r['annotation_id']),'raw_id':int(r['raw_id']),'level':int(r['level']),'box_iou':float(r.get('box_iou',float('nan'))),'native_support_fallback':bool(len(y)==1 and float(area)<1.0)},
                'h':im['h'][int(r['raw_id'])].float(),'c':im['coeff'][int(r['raw_id'])].float(),'p':p,'y':y,'factor':float(len(y)/max(area,1e-8)),'z_in':torch.empty(0)})
    torch.save(rows,out)
    print(json.dumps({'group':group,'rows':len(rows),'out':str(out)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--ann',type=Path,required=True); ap.add_argument('--group',required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); build(a.group,a.cache,a.ann,a.out)
