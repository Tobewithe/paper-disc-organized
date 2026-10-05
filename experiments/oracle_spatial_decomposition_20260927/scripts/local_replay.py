import json, math, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
Q=Path(r"C:\Dpan\codexproject\paper-disc-organized\experiments\coefficient_pixel_direction_20260927")
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:\Dpan\codexproject\paper-disc-organized\assets\datasets\coco")
MODEL=Path(r"C:\Dpan\codexproject\paper-disc-organized\assets\models\coco_clean_20260911\yolo26m-seg.pt")
sys.path.insert(0,str(Q/"scripts"))
from common import bounds, roi
def load(p): return torch.load(p,map_location="cpu",weights_only=True)
def preprocess(iid):
    im=cv2.imread(str(DATA/"images/val2017"/f"{iid:012d}.jpg"))
    if im is None: raise FileNotFoundError(iid)
    h,w=im.shape[:2];r=640/max(h,w)
    hh,ww=min(math.ceil(h*r),640),min(math.ceil(w*r),640)
    if r!=1: im=cv2.resize(im,(ww,hh),interpolation=cv2.INTER_LINEAR)
    im=LetterBox((640,640),auto=False,scaleup=False)(image=im)
    rp=((hh/h,ww/w),(round((640-ww)/2-.1),round((640-hh)/2-.1)))
    x=torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)))[None].cuda().float()/255
    return x,(h,w),rp,(hh,ww)
def main():
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    t=load(Q/"runs/RUN_011491022c594a86aabaebd9d9d3a2a1/TEST.pt")
    g=defaultdict(list)
    for i,k in enumerate(t["keys"]):g[k[0]].append(i)
    model=YOLO(str(MODEL)).model.cuda().float().eval()
    result=[]
    with torch.no_grad():
        for iid,ix in list(g.items())[:12]:
            x,shape,rp,hw=preprocess(iid)
            _,raw=model(x);pr=raw["one2one"];p=pr["proto"][0]
            c=pr["mask_coefficient"][0].T
            b=model.model[-1]._get_decode_boxes(pr)[0].T
            ids=[t["keys"][i][2] for i in ix]
            result.append(dict(image=iid,shape=shape,ratio_pad=rp,
                c_max=float((c[ids].cpu()-t["c0"][ix]).abs().max()),
                box_max=float((b[ids].cpu()-t["box"][ix]).abs().max()),
                p_max=float((torch.stack([roi(p,t["box"][i]) for i in ix]).cpu()-t["p"][ix]).abs().max())))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
