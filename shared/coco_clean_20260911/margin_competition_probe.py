"""Risk-controlled same-class mask competition on frozen YOLO outputs.

Calibration uses only directed pair samples from the train cache. For a source
GT-exclusive pixel, the source candidate must remain active; the threshold is
the empirical 99th percentile of competitor-minus-source logit margins. At
inference, a candidate pixel is removed only when a same-class overlapping
candidate exceeds it by that calibrated margin. Ambiguous pixels are retained.
This is a deterministic, conservative control for the learned ownership
experiments, not a claim of a deployable method until held-out evaluation.
"""
from __future__ import annotations
import argparse, contextlib, gzip, io, json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import ops

ROOT=Path(__file__).resolve().parent

def sha(p):
    import hashlib
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def calibrate(cache, q=0.99):
    margins=[]; n=0
    for path in (cache/'train').glob('*.npz'):
        with np.load(path) as z:
            if not len(z['pair_source']): continue
            zi=np.einsum('psk,pk->ps',z['pair_p'],z['c'][z['pair_source']])
            zj=np.einsum('psk,pk->ps',z['pair_p'],z['c'][z['pair_neighbor']])
            margins.append((zj-zi).reshape(-1)); n+=len(zi)
    if not margins: raise RuntimeError('No pair margins in training cache')
    m=np.concatenate(margins); tau=float(np.quantile(m,q))
    return dict(q=q,tau=tau,pair_rows=n,pixels=int(len(m)),margin_p01=float(np.quantile(m,.01)),margin_median=float(np.median(m)),margin_p99=float(np.quantile(m,.99)))

def decode(item,tau):
    p=torch.tensor(item['proto'],device='cuda',dtype=torch.float32); c=torch.tensor(item['coeff'],device='cuda',dtype=torch.float32); b=torch.tensor(item['boxes'],device='cuda',dtype=torch.float32); d=torch.tensor(item['detections'],device='cuda',dtype=torch.float32)
    ishape=tuple(int(v) for v in item['input_shape']); h,w=ishape
    if not len(c): return torch.empty((0,h,w),device='cuda',dtype=torch.uint8),0
    z=(c@p.flatten(1)).reshape(1,-1,*p.shape[-2:]); z=torch.nn.functional.interpolate(z,(h,w),mode='bilinear',align_corners=False)[0]
    out=z>0; removed=0; pairs=0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if i==j or int(d[i,5])!=int(d[j,5]): continue
            wh=(torch.minimum(b[i,2:],b[j,2:])-torch.maximum(b[i,:2],b[j,:2])).clamp_min(0); inter=wh.prod(); ai=(b[i,2:]-b[i,:2]).clamp_min(0).prod(); aj=(b[j,2:]-b[j,:2]).clamp_min(0).prod(); iou=inter/(ai+aj-inter).clamp_min(1e-9)
            if float(iou)<=.05: continue
            # Candidate j can suppress i only inside their common box support.
            common=ops.crop_mask(torch.ones((1,h,w),device='cuda',dtype=torch.uint8),b[[i]])[0].bool() & ops.crop_mask(torch.ones((1,h,w),device='cuda',dtype=torch.uint8),b[[j]])[0].bool()
            drop_i=out[i]&common&(z[j]>0)&(z[j]-z[i]>tau)
            drop_j=out[j]&common&(z[i]>0)&(z[i]-z[j]>tau)
            removed+=int(drop_i.sum()+drop_j.sum()); out[i][drop_i]=False; out[j][drop_j]=False; pairs+=1
    # The binary mask is cropped once after competition, matching stock decode.
    out=ops.crop_mask(out.to(torch.uint8),b)
    return out,pairs,removed

def rle(mask):
    q=mu.encode(np.asfortranarray(mask.astype(np.uint8))); q['counts']=q['counts'].decode('ascii'); return q

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-cache',type=Path,required=True); ap.add_argument('--eval-cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--quantile',type=float,default=.99); a=ap.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False); (out/'predictions').mkdir()
    cal=calibrate(a.train_cache,a.quantile); sel=json.loads((a.eval_cache/'selection.json').read_text()); ids=list(sel['transfer']); ann=COCO(str(ROOT.parent.parent/'datasets/coco/annotations/instances_train2017.json')); cats=sorted(ann.cats); preds={'initial':[],'margin_gate':[]}; pairs=removed=0
    for iid in ids:
        with np.load(a.eval_cache/'images'/f'{iid}.npz') as q: item={k:q[k] for k in q.files}
        p=torch.tensor(item['proto'],device='cuda'); c=torch.tensor(item['coeff'],device='cuda'); b=torch.tensor(item['boxes'],device='cuda'); ishape=tuple(int(v) for v in item['input_shape']); stock=ops.process_mask(p,c,b,ishape,upsample=True); gated,n,r=decode(item,cal['tau']); pairs+=n; removed+=r; shape=tuple(int(v) for v in item['shape']); det=item['detections']
        for name,masks in [('initial',stock),('margin_gate',gated)]:
            restored=ops.scale_masks(masks[:,None],shape)[:,0].cpu().numpy()
            for j,m in enumerate(restored):
                if not bool(m.any()): continue
                preds[name].append(dict(image_id=iid,category_id=cats[int(det[j,5])],score=float(det[j,4]),segmentation=rle(m)))
    rows=[]
    for name,pp in preds.items():
        with gzip.open(out/'predictions'/f'{name}.json.gz','wt',encoding='utf8') as f: json.dump(pp,f,separators=(',',':'))
        with contextlib.redirect_stdout(io.StringIO()): dt=ann.loadRes(pp); ev=COCOeval(ann,dt,'segm'); ev.params.imgIds=ids; ev.evaluate(); ev.accumulate(); ev.summarize()
        rows.append(dict(arm=name,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pp)))
    (out/'SUMMARY.json').write_text(json.dumps({'status':'COMPLETE','images':len(ids),'calibration':cal,'rows':rows,'candidate_pairs_scanned':pairs,'removed_input_pixels':removed,'GT_used_in_inference':False},indent=2),encoding='utf8'); print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
