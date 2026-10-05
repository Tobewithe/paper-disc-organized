"""Preregistered GT-free same-image donor readout using frozen G-BIAS."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from ultralytics.utils import ops
from pycocotools.coco import COCO
def box_iou(a,b):
    lt=torch.maximum(a[:,:2],b[:2]); rb=torch.minimum(a[:,2:],b[2:]); wh=(rb-lt).clamp_min(0); inter=wh[:,0]*wh[:,1]; ar=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); br=(b[2]-b[0])*(b[3]-b[1]); return inter/(ar+br-inter).clamp_min(1e-9)
def iou(a,b):
    return float(((a&b).sum())/((a|b).sum().clamp_min(1)))
def main(a):
    root=a.root; coco=COCO(str(root/'data/annotations/instances_val2017.json')); split=json.loads((root/'assets/VAL_ONLY_SPLIT.json').read_text())['val']; arm=json.loads((root/'stage1'/'G-BIAS.json').read_text()); bias=[torch.tensor(x,dtype=torch.float32) for x in arm['A']]; recs=[]
    for iid in split:
        im=torch.load(root/'official_cache'/'images'/f'{int(iid):012d}.pt',map_location='cpu',weights_only=False); rows=im['rows']; boxes=im['boxes'].float(); coeff=im['coeff'].float(); proto=im['proto'].float(); n=len(rows); out_boxes=[]; anns=[]; own=[]
        for r in rows: own.append(int(r['raw_id'])); out_boxes.append(boxes[int(r['raw_id'])]); anns.append(int(r['annotation_id']))
        out_boxes=torch.stack(out_boxes); donor=[]
        for k,raw in enumerate(own):
            v=box_iou(boxes,out_boxes[k]); v[raw]=-1; donor.append(int(v.argmax()))
        donor=torch.tensor(donor,dtype=torch.long); c_d=coeff[donor]; c_da=c_d.clone()
        for k,did in enumerate(donor.tolist()): c_da[k]+=bias[int(im['levels'][did])]
        c_a=coeff[own]
        m_a=ops.process_mask(proto,c_a,out_boxes,(640,640),upsample=True); m_d=ops.process_mask(proto,c_d,out_boxes,(640,640),upsample=True); m_da=ops.process_mask(proto,c_da,out_boxes,(640,640),upsample=True)
        shape=tuple(im['original_shape']); rp=im['ratio_pad']; m_a=ops.scale_masks(m_a[:,None].float(),shape,ratio_pad=rp)[:,0]>.5; m_d=ops.scale_masks(m_d[:,None].float(),shape,ratio_pad=rp)[:,0]>.5; m_da=ops.scale_masks(m_da[:,None].float(),shape,ratio_pad=rp)[:,0]>.5
        va=[];vd=[];vda=[]
        for k,annid in enumerate(anns):
            gt=torch.from_numpy(coco.annToMask(coco.anns[annid])).bool(); va.append(iou(m_a[k],gt)); vd.append(iou(m_d[k],gt)); vda.append(iou(m_da[k],gt))
        recs.append({'image_id':int(iid),'n':n,'A':float(np.mean(va)),'DONOR':float(np.mean(vd)),'DONOR_AFF':float(np.mean(vda)),'delta_donor':float(np.mean(vd)-np.mean(va)),'delta_aff':float(np.mean(vda)-np.mean(va)),'repairs_donor':sum(x<.75<=y for x,y in zip(va,vd)),'damage_donor':sum(x>=.75>y for x,y in zip(va,vd)),'repairs_aff':sum(x<.75<=y for x,y in zip(va,vda)),'damage_aff':sum(x>=.75>y for x,y in zip(va,vda))})
    out=root/'stage2'; out.mkdir(exist_ok=True); (out/'PER_IMAGE.jsonl').write_text('\n'.join(json.dumps(x) for x in recs),encoding='utf-8'); d=np.array([r['delta_aff'] for r in recs]); dd=np.array([r['delta_donor'] for r in recs]); rng=np.random.default_rng(20261004); idx=rng.integers(0,len(recs),(5000,len(recs))); summary={'images':len(recs),'candidates':sum(r['n'] for r in recs),'A_macro':float(np.mean([r['A'] for r in recs])),'DONOR_macro':float(np.mean([r['DONOR'] for r in recs])),'DONOR_AFF_macro':float(np.mean([r['DONOR_AFF'] for r in recs])),'DONOR_delta_pp':float(dd.mean()*100),'DONOR_ci_pp':(np.quantile(dd[idx].mean(1),[.025,.975])*100).tolist(),'DONOR_AFF_delta_pp':float(d.mean()*100),'DONOR_AFF_ci_pp':(np.quantile(d[idx].mean(1),[.025,.975])*100).tolist(),'repairs_donor':sum(r['repairs_donor'] for r in recs),'damage_donor':sum(r['damage_donor'] for r in recs),'repairs_aff':sum(r['repairs_aff'] for r in recs),'damage_aff':sum(r['damage_aff'] for r in recs),'donor_excludes_self':True,'donor_selection':'same-image max IoU of predicted donor box with retained output box','affine':'G-BIAS frozen per-level constant correction'}; (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); main(ap.parse_args())
