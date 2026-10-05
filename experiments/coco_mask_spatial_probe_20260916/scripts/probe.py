"""Per-instance GT-assisted optimization; not model training or AP evaluation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    for key in ("data","annotations","out"):ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--steps",type=int,default=120)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((a.data/"manifest.json").read_text())
    if a.limit:manifest=manifest[:a.limit]
    coco=COCO(str(a.annotations));start=time.monotonic();all_rows=[]
    modes=("scalar","coeff","coeff_bias","local4","coeff_local4")
    for number,row in enumerate(manifest):
        with np.load(a.data/f"{row['image_id']:012d}.npz") as z:
            proto=torch.as_tensor(z["proto"],device="cuda")
            coef=torch.as_tensor(z["coefficients"],device="cuda")
            shape=tuple(z["input_shape"].tolist());orig=tuple(z["original_shape"].tolist())
            box=torch.as_tensor(z["boxes_input"][row["raw_id"]],device="cuda")
        with torch.no_grad():base=(coef.T@proto.flatten(1)).reshape(-1,*proto.shape[-2:])[row["raw_id"]].detach()
        gt=torch.as_tensor(coco.annToMask(coco.anns[row["annotation_id"]]),device="cuda").float()
        ih,iw=shape;oh,ow=orig;gain=min(ih/oh,iw/ow)
        nh,nw=round(oh*gain),round(ow*gain);top,left=round((ih-nh)/2-.1),round((iw-nw)/2-.1)
        gt_input=F.pad(F.interpolate(gt[None,None],(nh,nw),mode="bilinear",align_corners=False)[0,0],(left,iw-nw-left,top,ih-nh-top))
        yy,xx=torch.meshgrid(torch.arange(ih,device="cuda"),torch.arange(iw,device="cuda"),indexing="ij")
        support=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])
        target=gt_input[support];assert len(target)>0
        # Grid is tied to the fixed instance's predicted box, never to GT shape.
        grid=torch.stack((2*(xx-box[0])/(box[2]-box[0]).clamp(min=1)-1,2*(yy-box[1])/(box[3]-box[1]).clamp(min=1)-1),-1).float()[None]
        scale=proto.square().mean((1,2)).sqrt().clamp(min=.1).detach()
        def render(logits):
            binary=((logits>0)&support).float()
            return F.interpolate(binary[None,None,top:top+nh,left:left+nw],orig,mode="bilinear",align_corners=False)[0,0].byte().bool()
        def metrics(logits):
            mask=render(logits);g=gt.bool();tp=int((mask&g).sum());fp=int((mask&~g).sum());fn=int((~mask&g).sum())
            return dict(iou=tp/max(tp+fp+fn,1),coverage=tp/max(tp+fn,1),purity=tp/max(tp+fp,1),tp=tp,fp=fp,fn=fn)
        base_up=F.interpolate(base[None,None],shape,mode="bilinear",align_corners=False)[0,0]
        baseline=metrics(base_up)
        assert abs(baseline["iou"]-row["baseline_iou"])<2e-6,(row["annotation_id"],baseline["iou"],row["baseline_iou"])
        result={**row,"baseline":baseline,"arms":{}}
        for mode in modes:
            dc=torch.zeros(proto.shape[0],device="cuda",requires_grad=True)
            bias=torch.zeros((),device="cuda",requires_grad=True)
            local=torch.zeros((1,1,4,4),device="cuda",requires_grad=True)
            use_c=mode in ("coeff","coeff_bias","coeff_local4");use_b=mode in ("scalar","coeff_bias");use_l=mode in ("local4","coeff_local4")
            params=([dc] if use_c else [])+([bias] if use_b else [])+([local] if use_l else [])
            opt=torch.optim.Adam(params,lr=.08)
            best=baseline.copy();best_step=0;history=[]
            for step in range(a.steps+1):
                low=base+(dc[:,None,None]*proto/scale[:,None,None]).sum(0) if use_c else base
                logits=F.interpolate(low[None,None],shape,mode="bilinear",align_corners=False)[0,0]
                if use_b:logits=logits+bias
                if use_l:logits=logits+F.grid_sample(local,grid,mode="bilinear",padding_mode="border",align_corners=True)[0,0]
                pred=logits[support];prob=pred.sigmoid()
                loss=F.binary_cross_entropy_with_logits(pred,target)+.5*(1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1))
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite optimization loss")
                if step%10==0 or step==a.steps:
                    with torch.no_grad():m=metrics(logits)
                    history.append(dict(step=step,loss=float(loss.detach()),**m))
                    if m["iou"]>best["iou"]:best=m.copy();best_step=step
                if step<a.steps:
                    opt.zero_grad();loss.backward();opt.step()
            result["arms"][mode]=dict(best=best,best_step=best_step,last=history[-1],trajectory=history,
                parameter_count=sum(x.numel() for x in params))
        all_rows.append(result)
        with (a.out/"instances.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
        progress=dict(instances=number+1,total=len(manifest),elapsed_s=round(time.monotonic()-start,2),last_gt=row["annotation_id"])
        dump(a.out/"progress.json",progress);print(json.dumps(progress),flush=True)
    def summarize(rows):
        return dict(n=len(rows),baseline_iou=float(np.mean([r["baseline"]["iou"] for r in rows])),
            arms={m:dict(mean_best_iou=float(np.mean([r["arms"][m]["best"]["iou"] for r in rows])),
                mean_last_iou=float(np.mean([r["arms"][m]["last"]["iou"] for r in rows])),
                best_repaired75=sum(r["baseline"]["iou"]<.75<=r["arms"][m]["best"]["iou"] for r in rows),
                last_repaired75=sum(r["baseline"]["iou"]<.75<=r["arms"][m]["last"]["iou"] for r in rows),
                last_damaged75=sum(r["arms"][m]["last"]["iou"]<.75<=r["baseline"]["iou"] for r in rows)) for m in modes})
    summary=dict(all=summarize(all_rows),by_outcome={g:summarize([r for r in all_rows if r["outcome"]==g]) for g in sorted({r["outcome"] for r in all_rows})},
        by_stratum={g:summarize([r for r in all_rows if r["stratum"]==g]) for g in sorted({r["stratum"] for r in all_rows})},
        environment=dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()),
        limitations=["Per-instance evaluation-GT optimization, not model training or deployable method.","Best trajectory uses GT IoU selection and includes baseline; final iterate also reported.","Failure to optimize is not proof of a representation upper bound.","Local bias adds spatial degrees of freedom without establishing their learnability."])
    dump(a.out/"SUMMARY.json",summary);dump(a.out/"COMPLETE.json",progress)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
