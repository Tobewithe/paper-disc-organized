"""Evaluate frozen predictions, full native projection, and original COCO masks."""
import argparse,json
from collections import defaultdict
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import rankdata
from ultralytics.utils import ops
from common import *

METRICS=("iou","auc","effect_cos","grid_cos","coverage","fpr_native","coeff_norm","projection_residual")
def cosine(x,y):
    return (x*y).sum(-1)/(x.square().sum(-1).sqrt()*y.square().sum(-1).sqrt()).clamp_min(1e-12)

def auc(y,z):
    n=int(y.sum());m=len(y)-n
    if not n or not m:return float("nan")
    return float((rankdata(z,method="average")[y].sum()-n*(n+1)/2)/(n*m))

def decode_logits(z,box,image):
    boxes=box[None].expand(len(z),-1).clone()/4
    z=ops.crop_mask(z,boxes)
    masks=F.interpolate(z[None],(640,640),mode="bilinear",align_corners=False)[0]>0
    return ops.scale_masks(masks.float()[None],image["original_shape"],ratio_pad=image["ratio_pad"])[0]>.5

@torch.no_grad()
def predict(test):
    queue=json.loads((ROOT/"QUEUE.json").read_text())
    predictions={}
    for seed in (0,1):
        for mode in MODES:
            for kind in KINDS:
                name=f"{mode}_{kind}_s{seed}"
                rid=queue["runs"]["train_"+name]
                checkpoint=load(ROOT/"runs"/rid/"BEST.pt")
                assert (checkpoint["mode"],checkpoint["kind"],checkpoint["seed"])==(mode,kind,seed)
                net=Probe(kind).cuda().eval();net.load_state_dict(checkpoint["state"])
                d=state_pack(test,checkpoint["norm"])
                outputs=[]
                for lo in range(0,len(d["h"]),512):
                    ix=torch.arange(lo,min(lo+512,len(d["h"])),device="cuda")
                    outputs.append(net(*inputs(d,ix,mode)).cpu())
                predictions[name]=torch.cat(outputs)
                del d,net
    return predictions

@torch.no_grad()
def main(a):
    setup()
    test=load(a.test/"TEST.pt")
    predictions=predict(test)
    a.out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(keys=test["keys"],predictions=predictions),a.out/"PREDICTIONS.pt")
    names=["original","oracle","oracle_grid","oracle_grid_projected"]
    for k in predictions:
        names.append(k)
        if "_spatial_" in k:names.append(k+"_projected")
    n=len(test["keys"]);values=np.full((n,len(names),len(METRICS)),np.nan,dtype=np.float32)
    boxes=np.full(n,np.nan);ranks=np.zeros(n,dtype=np.int32);condition=np.full(n,np.nan)
    coco=COCO(str(TEST_LABELS/"conversion_input/instances_val2017.json"))
    grouped=defaultdict(list)
    for i,key in enumerate(test["keys"]):grouped[key[0]].append(i)
    max_decode_error=0.
    for step,(iid,indices) in enumerate(grouped.items(),1):
        image=load(a.test/"images"/f"{iid:012d}.pt")
        proto=image["proto"].cuda().float()
        lookup={(r["annotation_id"],r["raw_id"]):j for j,r in enumerate(image["rows"])}
        for i in indices:
            _,aid,rid=test["keys"][i];j=lookup[(aid,rid)]
            c0=image["coeff"][j].cuda();box=image["boxes"][j].cuda()
            delta=image["delta"][j].cuda();boxes[i]=image["rows"][j]["box_iou"]
            z0=(c0@proto.flatten(1)).reshape(160,160)
            target=(delta@proto.flatten(1)).reshape(160,160)
            A=test["p"][i].cuda().flatten(1).T
            tgrid=A@delta
            support=ops.crop_mask(torch.ones(1,160,160,device="cuda"),box[None]/4)[0].bool()
            assert support.any()
            pixels=proto[:,support].T.double()
            gram=pixels.T@pixels
            eigen,Q=torch.linalg.eigh(gram)
            cutoff=eigen[-1].clamp_min(1e-20)*1e-10
            keep=eigen>cutoff
            inverse=(Q[:,keep]/eigen[keep][None])@Q[:,keep].T
            ranks[i]=int(keep.sum());condition[i]=float((eigen[-1]/eigen[keep][0]).sqrt()) if keep.any() else np.inf
            x1,y1,x2,y2=bounds(box)
            def expand(grid):
                result=torch.zeros_like(z0)
                result[y1:y2,x1:x2]=F.interpolate(grid.reshape(1,1,16,16),
                    (y2-y1,x2-x1),mode="bilinear",align_corners=False)[0,0]
                return result
            def project(spatial):
                dc=(inverse@(pixels.T@spatial[support].double())).float()
                result=(dc@proto.flatten(1)).reshape(160,160)
                resid=float((result[support]-spatial[support]).norm()/spatial[support].norm().clamp_min(1e-9))
                return result,float(dc.norm()),resid
            grid_delta=expand(tgrid)
            projected_grid,pnorm,pres=project(grid_delta)
            effects=[torch.zeros_like(z0),target,grid_delta,projected_grid]
            gn=[torch.zeros_like(tgrid),tgrid,tgrid,A@(inverse@(pixels.T@grid_delta[support].double())).float()]
            norms=[0.,float(delta.norm()),float("nan"),pnorm]
            residuals=[0.,0.,float("nan"),pres]
            for name,pred in predictions.items():
                out=pred[i].cuda()
                if "_coefficient_" in name:
                    effects.append((out@proto.flatten(1)).reshape(160,160))
                    gn.append(A@out);norms.append(float(out.norm()));residuals.append(0.)
                else:
                    dense=expand(out);projected,pnorm,pres=project(dense)
                    effects.extend([dense,projected])
                    dc=(inverse@(pixels.T@dense[support].double())).float()
                    gn.extend([out,A@dc]);norms.extend([float("nan"),pnorm])
                    residuals.extend([float("nan"),pres])
            effects=torch.stack(effects);gn=torch.stack(gn)
            assert len(effects)==len(names)
            logits=z0[None]+effects
            masks=[]
            for lo in range(0,len(names),16):
                masks.append(decode_logits(logits[lo:lo+16].clone(),box,image))
            masks=torch.cat(masks)
            if i<10:
                official=ops.process_mask(proto,c0[None],box[None],(640,640),upsample=True)
                official=ops.scale_masks(official.float()[None],image["original_shape"],ratio_pad=image["ratio_pad"])[0]>.5
                error=float((official[0]!=masks[0]).float().mean())
                max_decode_error=max(max_decode_error,error);assert error==0.
            gt=torch.from_numpy(coco.annToMask(coco.anns[aid]).astype(bool)).cuda()
            inter=(masks&gt).sum((1,2)).float()
            union=(masks|gt).sum((1,2)).clamp_min(1)
            ious=(inter/union).cpu().numpy()
            coverage=(inter/gt.sum().clamp_min(1)).cpu().numpy()
            ys=(image["owner160"].cuda()[support]==int(image["owners"][j])+1).cpu().numpy()
            s=logits[:,support].cpu().numpy()
            aucs=[auc(ys,row) for row in s]
            fprs=((s[:,~ys]>0).mean(1) if (~ys).any() else np.full(len(names),np.nan))
            fc=cosine(effects[:,support],target[support][None]).cpu().numpy()
            gc=cosine(gn,tgrid[None]).cpu().numpy()
            values[i]=np.stack([ious,aucs,fc,gc,coverage,fprs,np.array(norms),np.array(residuals)],1)
        if step%50==0 or step==len(grouped):
            progress=dict(images=step,total_images=len(grouped),last_image=iid,max_decode_error=max_decode_error)
            print(json.dumps(progress),flush=True);write(a.out/"PROGRESS.json",progress)
    np.savez_compressed(a.out/"METRICS.npz",values=values,keys=np.array(test["keys"]),
        names=np.array(names),metrics=np.array(METRICS),box_iou=boxes,rank=ranks,condition=condition)
    write(a.out/"COMPLETE.json",dict(instances=n,images=len(grouped),arms=names,metrics=METRICS,
        max_decode_error=max_decode_error,test_source=str(a.test),full_COCO_AP=False,
        auc_gt="official overlap owner map, nearest 160 grid, predicted support",
        iou_gt="original COCO annToMask",projection_rtol=1e-5))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--test",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);main(p.parse_args())

