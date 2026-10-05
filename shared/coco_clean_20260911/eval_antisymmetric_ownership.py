"""Evaluate the antisymmetric pair gate on the held-out 300-image cache.

Inference uses only cached predictions (prototype, coefficient, boxes,
class/score). COCO annotations are loaded after masks are serialized solely for
official evaluation and reporting. Pair order is deterministic by candidate
index; each unordered same-class pair is corrected once on its joint support.
"""
from __future__ import annotations
import argparse, contextlib, gzip, io, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import ops
from antisymmetric_ownership import PairGate, pair_features

ROOT = Path(__file__).resolve().parent


def load_model(path):
    m=PairGate().cuda().eval(); state=torch.load(path,map_location="cuda",weights_only=True); m.load_state_dict(state["model"]); return m


@torch.inference_mode()
def decode(item, model):
    p=torch.tensor(item["proto"],device="cuda",dtype=torch.float32)
    c=torch.tensor(item["coeff"],device="cuda",dtype=torch.float32)
    boxes=torch.tensor(item["boxes"],device="cuda",dtype=torch.float32)
    det=torch.tensor(item["detections"],device="cuda",dtype=torch.float32)
    shape=tuple(int(v) for v in item["input_shape"]); h,w=shape
    if not len(c): return torch.empty((0,h,w),device="cuda",dtype=torch.uint8), 0
    full=F.interpolate((c@p.flatten(1)).reshape(1,-1,*p.shape[-2:]),shape,mode="bilinear",align_corners=False)[0]
    # Accumulate pair corrections against the same original logits. Applying
    # pairs sequentially would make the result depend on candidate ordering.
    correction=torch.zeros_like(full); pair_count=0; changed=0
    # Candidate masks have no pair input in the deployed path. Construct the
    # same deterministic overlap relation from predicted classes and boxes.
    for i in range(len(c)):
        for j in range(i+1,len(c)):
            if int(det[i,5]) != int(det[j,5]): continue
            wh=(torch.minimum(boxes[i,2:],boxes[j,2:])-torch.maximum(boxes[i,:2],boxes[j,:2])).clamp_min(0)
            inter=wh.prod(); ai=(boxes[i,2:]-boxes[i,:2]).clamp_min(0).prod(); aj=(boxes[j,2:]-boxes[j,:2]).clamp_min(0).prod()
            iou=inter/(ai+aj-inter).clamp_min(1e-9)
            if float(iou) <= .05: continue
            # Compute the pair gate on the complete joint box support. The
            # gate is applied only where both candidates have positive support.
            x0=max(0,int(torch.ceil(torch.maximum(boxes[i,0],boxes[j,0])).item())); y0=max(0,int(torch.ceil(torch.maximum(boxes[i,1],boxes[j,1])).item()))
            x1=min(w,int(torch.floor(torch.minimum(boxes[i,2],boxes[j,2])).item())); y1=min(h,int(torch.floor(torch.minimum(boxes[i,3],boxes[j,3])).item()))
            if x1<=x0 or y1<=y0: continue
            yy,xx=torch.meshgrid(torch.arange(y0,y1,device="cuda",dtype=torch.float32)+.5,torch.arange(x0,x1,device="cuda",dtype=torch.float32)+.5,indexing="ij")
            # Flattened input-grid index (integer y*w+x); yy/xx are centres.
            pos=(torch.floor(yy.reshape(-1)-.5)*w+torch.floor(xx.reshape(-1)-.5)).long()
            # Pair features use flattened pixel indices; retain only pixels
            # inside both predicted boxes, matching the training support gate.
            inside=(xx>=boxes[i,0])&(xx<boxes[i,2])&(yy>=boxes[i,1])&(yy<boxes[i,3])&(xx>=boxes[j,0])&(xx<boxes[j,2])&(yy>=boxes[j,1])&(yy<boxes[j,3])
            ids=pos[inside.reshape(-1)].long()
            if not len(ids): continue
            zi=full[i].flatten()[ids]; zj=full[j].flatten()[ids]
            # Candidate x consists of cached branch input; the gate only uses
            # geometry/score slices, so construct a compact 73-D placeholder
            # with those prediction fields in the documented positions.
            # Reconstruct the exact 73-D candidate input from the frozen cache
            # (h, level, boxes); this is the same input_features() layout used
            # during training and avoids silently replacing appearance with
            # zeros at evaluation time.
            def candidate_x(k):
                hfeat=torch.tensor(item["h"][k],device="cuda",dtype=torch.float32)
                lv=F.one_hot(torch.tensor(int(item["level"][k]),device="cuda"),3).float()
                bb=boxes[k]/boxes.new_tensor([w,h,w,h]); wh=(bb[2:]-bb[:2]).clamp_min(1e-5)
                return torch.cat([hfeat,lv,bb,wh])
            xi=candidate_x(i); xj=candidate_x(j)
            # Match ownership_ranking.input_features exactly: normalized
            # xyxy followed by normalized width/height (indices 67:73).
            bi=boxes[i]/boxes.new_tensor([w,h,w,h]); bj=boxes[j]/boxes.new_tensor([w,h,w,h])
            xi[67:73]=torch.cat([bi, (bi[2:]-bi[:2]).clamp_min(1e-5)])
            xj[67:73]=torch.cat([bj, (bj[2:]-bj[:2]).clamp_min(1e-5)])
            # pair_features expects one scalar vector per candidate and a
            # flattened location vector; use exact model-input dimensions.
            feats=pair_features(zi,zj,xi,xj,ids,(h,w))
            # The training cache contains both directed views when available.
            # Evaluate both directions and antisymmetrize; this removes an
            # arbitrary candidate-index orientation from the correction.
            feats_rev=pair_features(zj,zi,xj,xi,ids,(h,w))
            d=0.5*(model(feats)-model(feats_rev))
            # Selective gate: only ambiguous pixels where both candidates are
            # positive and neither response is decisively dominant. All other
            # pixels retain the stock logits exactly.
            active=(zi>0)&(zj>0)&((zi-zj).abs()<1.0)&(d.abs()>=0.5)
            d=torch.where(active,d,torch.zeros_like(d))
            correction[i].flatten()[ids]+=d; correction[j].flatten()[ids]-=d
            pair_count+=1; changed+=int((d.abs()>1e-6).sum())
    corrected=full+correction
    binary=(corrected>0).to(torch.uint8)
    # Apply the stock crop exactly once after all pair corrections.
    binary=ops.crop_mask(binary,boxes)
    return binary,pair_count,dict(correction_abs=float(correction.abs().sum()), correction_max=float(correction.abs().max()) if correction.numel() else 0., active_pixels=int((correction.abs()>1e-6).sum()))


def rle(mask):
    q=mu.encode(np.asfortranarray(mask.astype(np.uint8))); q["counts"]=q["counts"].decode("ascii"); return q


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--training",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False); (out/"predictions").mkdir()
    selection=json.loads((a.cache/"selection.json").read_text()); ids=list(selection["transfer"]); ids=ids[:a.limit] if a.limit else ids; ann=COCO(str(ROOT.parent.parent/"datasets/coco/annotations/instances_train2017.json")); cats=sorted(ann.cats); models={"initial":None}
    for seed in [0,1,2]: models[f"gate_s{seed}"]=load_model(a.training/f"pair_gate_s{seed}_epoch15.pt")
    preds={k:[] for k in models}; pair_counts={k:0 for k in models}
    for iid in ids:
        path=a.cache/"images"/f"{iid}.npz"; item={k:v for k,v in np.load(path).items()}; shape=tuple(int(v) for v in item["shape"])
        for name,model in models.items():
            if model is None:
                p=torch.tensor(item["proto"],device="cuda"); c=torch.tensor(item["coeff"],device="cuda"); b=torch.tensor(item["boxes"],device="cuda"); ishape=tuple(int(v) for v in item["input_shape"]); masks=ops.process_mask(p,c,b,ishape,upsample=True)
            else: masks,n,diag=decode(item,model); pair_counts[name]+=n; pair_counts[name+'_changed'] = pair_counts.get(name+'_changed',0)+diag['active_pixels']; pair_counts[name+'_abs'] = pair_counts.get(name+'_abs',0.)+diag['correction_abs']
            restored=ops.scale_masks(masks[:,None],shape)[:,0].cpu().numpy()
            det=item["detections"]
            for j,m in enumerate(restored):
                if not bool(m.any()): continue
                preds[name].append(dict(image_id=iid,category_id=cats[int(det[j,5])],score=float(det[j,4]),segmentation=rle(m)))
    rows=[]
    for name,pp in preds.items():
        with gzip.open(out/"predictions"/f"{name}.json.gz","wt",encoding="utf8") as f: json.dump(pp,f,separators=(",",":"))
        with contextlib.redirect_stdout(io.StringIO()):
            dt=ann.loadRes(pp); ev=COCOeval(ann,dt,"segm"); ev.params.imgIds=ids; ev.evaluate(); ev.accumulate(); ev.summarize()
        rows.append(dict(arm=name,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pp),pair_updates=pair_counts[name],changed_pixels=pair_counts.get(name+'_changed',0),correction_abs=pair_counts.get(name+'_abs',0.0)))
    (out/"SUMMARY.json").write_text(json.dumps({"status":"COMPLETE","images":len(ids),"ordinary_gt":sum(not x.get("iscrowd",0) for i in ids for x in ann.imgToAnns[i]),"rows":rows,"selection":"predeclared transfer split","GT_used_in_inference":False},indent=2),encoding="utf8")
    print(json.dumps(rows,ensure_ascii=False))


if __name__=="__main__": main()
