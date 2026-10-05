"""Deterministic typical-case figures, not best-case selection."""
import os
os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from local_replay import Q,ROOT,DATA,load,preprocess,bounds
@torch.no_grad()
def main(a):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    d=np.load(a.source/"DECOMPOSITION.npz")
    t=load(Q/"runs/RUN_011491022c594a86aabaebd9d9d3a2a1/TEST.pt")
    pr=load(Q/"runs/RUN_e37f1bf4d7e6432187a451f23933c1e6/PREDICTIONS.pt")["predictions"]
    names=d["names"].tolist();inds=[names.index("p_spatial_s0"),names.index("p_spatial_s1")]
    gain=d["hybrid_iou"][:,1,:,:][:,:,inds].mean(2)-d["iou640"][:,np.array(inds)+2].mean(1)[:,None]
    good=(d["original_iou"]<.75)&(d["box_iou"]>=.75)
    chosen=[]
    for r in range(4):
        pool=np.where(good&(np.argmax(np.nan_to_num(gain,nan=-np.inf),1)==r)&(gain[:,r]>=.01))[0]
        if not len(pool):continue
        order=pool[np.argsort(gain[pool,r],kind="stable")]
        i=int(order[len(order)//2])
        chosen.append(dict(index=i,region=str(d["regions"][r]),image_id=int(d["keys"][i,0]),
            annotation_id=int(d["keys"][i,1]),gain=float(gain[i,r]),eligible=len(pool),
            selection="median gain among good-box failures with this region dominant and gain>=0.01"))
    coco=COCO(str(DATA/"annotations/instances_val2017.json"))
    fig,axs=plt.subplots(len(chosen),6,figsize=(17,3.3*len(chosen)),squeeze=False,layout="constrained")
    for row,info in enumerate(chosen):
        i=info["index"];iid=info["image_id"];aid=info["annotation_id"]
        im=load(a.source/"native_proto"/f"{iid:012d}.pt");p=im["proto"].cuda()
        x,shape,rp,hw=preprocess(iid);rgb=(x[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        box=t["box"][i].cuda();c=t["c0"][i].cuda();delta=t["delta"][i].cuda()
        z=(c@p.flatten(1)).reshape(160,160);o=(delta@p.flatten(1)).reshape(160,160)
        grid=pr["p_spatial_s0"][i].cuda().reshape(16,16);eff=torch.zeros_like(z)
        x1,y1,x2,y2=bounds(box);eff[y1:y2,x1:x2]=F.interpolate(grid[None,None],(y2-y1,x2-x1),mode="bilinear",align_corners=False)[0,0]
        z0,oracle,model=F.interpolate(torch.stack([z,o,eff])[None],(640,640),mode="bilinear",align_corners=False)[0].cpu().numpy()
        left,top=rp[1];hh,ww=hw
        gt=np.zeros((640,640),bool);gt[top:top+hh,left:left+ww]=cv2.resize(coco.annToMask(coco.anns[aid]).astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST).astype(bool)
        xx,yy=np.meshgrid(np.arange(640),np.arange(640));b=box.cpu().numpy()
        support=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])
        gx,gy,gw,gh=coco.anns[aid]["bbox"]
        # Union of prediction and GT boxes, with small context padding.
        qx1=max(0,int(min(b[0],gx*ww/shape[1]+left))-10);qx2=min(640,int(max(b[2],(gx+gw)*ww/shape[1]+left))+11)
        qy1=max(0,int(min(b[1],gy*hh/shape[0]+top))-10);qy2=min(640,int(max(b[3],(gy+gh)*hh/shape[0]+top))+11)
        sl=np.s_[qy1:qy2,qx1:qx2]
        axs[row,0].imshow(rgb[sl]);axs[row,0].contour(gt[sl],levels=[.5],colors=["#00ff80"],linewidths=.8)
        for col,logit in enumerate((z0,z0+model,z0+oracle),1):
            mask=(logit>0)&support
            err=np.ones((640,640,3),np.float32)*.96
            err[gt&mask]=(.25,.25,.25);err[gt&~mask]=(.2,.45,.95);err[~gt&mask]=(.95,.2,.15)
            axs[row,col].imshow(err[sl])
        limit=max(float(np.quantile(np.abs(oracle[sl]),.98)),.1)
        for col,effect in ((4,model),(5,oracle)):
            axs[row,col].imshow(effect[sl],cmap="coolwarm",vmin=-limit,vmax=limit)
            axs[row,col].contour(gt[sl],levels=[.5],colors=["black"],linewidths=.6)
        for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
        axs[row,0].set_ylabel(info["region"]+"\nimage "+str(iid),fontsize=9)
    for ax,title in zip(axs[0],("Image + GT outline","Original errors","Spatial predictor (seed 0)","Oracle errors","Predicted correction","Oracle correction")):ax.set_title(title,fontsize=10)
    fig.suptitle("Typical dominant-region examples (median, not largest gain)\n"
                 "Blue: missed GT; red: extra pixels; gray: correct target. Corrections share a scale within each row.",fontsize=12)
    a.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out/"TYPICAL_CASES.png",dpi=160);fig.savefig(a.out/"TYPICAL_CASES.pdf")
    (a.out/"EXAMPLES.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())


