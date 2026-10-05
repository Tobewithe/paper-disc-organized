"""7R regional 16x16 representation ceiling, using 7Q's oracle_grid operator."""
import argparse,json,math,time
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import numpy as np,cv2,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from local_replay import Q,ROOT,DATA,load,bounds
from decompose import morph,write,measure
@torch.no_grad()
def main(a):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(parents=True,exist_ok=True)
    write(a.out/"run.json",dict(run_id=a.out.name,status="running",started_at=datetime.now(timezone.utc).isoformat(),
        scope="7Q 16x16 oracle-grid representation ceiling by spatial region; no retraining"))
    t=load(Q/"runs/RUN_011491022c594a86aabaebd9d9d3a2a1/TEST.pt")
    group=defaultdict(list)
    for i,k in enumerate(t["keys"]):group[int(k[0])].append(i)
    n=len(t["keys"]);values=np.full((n,4,7),np.nan,np.float32)
    full=np.full((n,3),np.nan,np.float32)
    coco=COCO(str(DATA/"annotations/instances_val2017.json"));start=time.time()
    for step,(iid,ix) in enumerate(group.items(),1):
        file=a.source/"native_proto"/f"{iid:012d}.pt"
        while True:
            try:image=load(file);break
            except (FileNotFoundError,RuntimeError,EOFError):
                if (a.source/"FAILED.json").exists():raise RuntimeError("Source failed")
                time.sleep(.5)
        p=image["proto"].cuda();h,w=image["original_shape"]
        hh,ww=image["hw_resized"];left,top=image["ratio_pad"][1]
        masks={};union=np.zeros((640,640),bool)
        for ann in coco.imgToAnns[iid]:
            m=np.zeros((640,640),bool)
            m[top:top+hh,left:left+ww]=cv2.resize(coco.annToMask(ann).astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[ann["id"]]=m;union|=m
        for i in ix:
            box=t["box"][i].cuda();delta=t["delta"][i].cuda();c0=t["c0"][i].cuda()
            z=(c0@p.flatten(1)).reshape(160,160);target=(delta@p.flatten(1)).reshape(160,160)
            grid=(t["p"][i].cuda().flatten(1).T@delta).reshape(16,16)
            bx1,by1,bx2,by2=bounds(box);g=torch.zeros_like(target)
            g[by1:by2,bx1:bx2]=F.interpolate(grid[None,None],(by2-by1,bx2-bx1),mode="bilinear",align_corners=False)[0,0]
            maps=F.interpolate(torch.stack([z,target,g])[None],(640,640),mode="bilinear",align_corners=False)[0]
            x1=max(left,0,math.ceil(float(box[0])));y1=max(top,0,math.ceil(float(box[1])))
            x2=min(left+ww,640,math.ceil(float(box[2])));y2=min(top+hh,640,math.ceil(float(box[3])))
            if x2<=x1 or y2<=y1:full[i]=0;continue
            sl=np.s_[y1:y2,x1:x2];gtruth=masks[t["keys"][i][1]];total=gtruth.sum()
            er,di=morph(gtruth,3);other=union&~gtruth
            br=(di&~er)[sl];interior=er[sl];neighbor=(other&~di)[sl];bg=~(br|interior|neighbor)
            z0,tar,gg=maps[:,y1:y2,x1:x2].cpu().numpy().reshape(3,-1);y=gtruth[sl].flatten()
            binary=np.stack([z0>0,z0+tar>0,z0+gg>0])
            tp=(binary&y).sum(1);fp=(binary&~y).sum(1)
            full[i]=tp/np.maximum(total+fp,1)
            for j,reg in enumerate([br,interior,neighbor,bg]):
                q=reg.flatten();tt=tar[q];aa=gg[q];energy=np.sum(tt*tt)
                if not q.any():values[i,j]=[np.nan,np.nan,np.nan,0,0,0,0];continue
                dot=np.sum(aa*tt);pe=np.sum(aa*aa)
                co=dot/max(np.sqrt(energy*pe),1e-12) if energy>1e-10 and pe>1e-10 else np.nan
                err=np.sum((aa-tt)**2)/max(energy,1e-12) if energy>1e-10 else np.nan
                tpr=(binary[:,q]&y[q]).sum(1);fpr=(binary[:,q]&~y[q]).sum(1)
                # Grid everywhere, oracle only in region.
                hy=(tp[2]+tpr[1]-tpr[2])/max(total+fp[2]+fpr[1]-fpr[2],1)-full[i,2]
                # Original everywhere, oracle-grid only in region.
                only=(tp[0]+tpr[2]-tpr[0])/max(total+fp[0]+fpr[2]-fpr[0],1)-full[i,0]
                fullonly=(tp[0]+tpr[1]-tpr[0])/max(total+fp[0]+fpr[1]-fpr[0],1)-full[i,0]
                values[i,j]=[co,err,dot/max(energy,1e-12),hy,only,fullonly,energy]
        if step%50==0 or step==len(group):
            v=dict(images=step,total_images=len(group),elapsed_seconds=time.time()-start)
            write(a.out/"PROGRESS.json",v);print(json.dumps(v),flush=True)
    np.savez_compressed(a.out/"GRID_CEILING.npz",keys=np.asarray(t["keys"]),values=values,full_iou=full,
        metrics=["cos","nmse","amplitude","oracle_region_over_grid_gain","grid_only_gain","oracle_only_gain","energy"],
        regions=["boundary","interior","neighbor","background"])
    rec=json.loads((a.out/"run.json").read_text());write(a.out/"run.json",dict(rec,status="completed",finished_at=datetime.now(timezone.utc).isoformat()))
    write(a.out/"COMPLETE.json",dict(instances=n,images=len(group),elapsed_seconds=time.time()-start))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
