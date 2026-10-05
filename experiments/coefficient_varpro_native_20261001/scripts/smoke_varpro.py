from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

def load_input(path: Path, cache: dict) -> torch.Tensor:
    im=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    rh,rw=[float(x) for x in cache["ratio_pad"][0]]
    top,left=[int(round(float(x))) for x in (cache["ratio_pad"][1][1], cache["ratio_pad"][1][0])]
    nh,nw=round(im.shape[0]*rh),round(im.shape[1]*rw)
    im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((640,640,3),114,dtype=np.uint8); canvas[top:top+nh,left:left+nw]=im
    return torch.from_numpy(np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))).float().div_(255).unsqueeze(0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',type=Path,required=True); ap.add_argument('--c-final',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('refusing CPU execution')
    yolo=YOLO(str(a.weights)); model=yolo.model.float().to(device).eval(); head=model.model[-1]
    ck=torch.load(a.c_final,map_location='cpu',weights_only=False); head.one2one_cv4.load_state_dict(ck['cv4'],strict=True)
    idx=json.loads((a.cache/'INDEX.json').read_text(encoding='utf-8'))
    item=idx['fit'][0]; iid=int(item['image_id']); cache=torch.load(a.cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
    captured={}
    def hook(_m,inputs): captured['x']=[x.detach() for x in inputs[0]]
    handle=head.register_forward_pre_hook(hook)
    t0=time.time(); x=load_input(a.data/'images'/'train2017'/f'{iid:012d}.jpg',cache).to(device)
    with torch.inference_mode(): _=model(x)
    feats=captured['x']; hs=[]; cs=[]
    for branch,feat in zip(head.one2one_cv4,feats):
        h=branch[:-1](feat); c=branch(feat); hs.append(h.reshape(1,64,-1)); cs.append(c.reshape(1,32,-1))
    h=torch.cat(hs,2).squeeze(0).T; c=torch.cat(cs,2).squeeze(0).T
    handle.remove()
    result={'device':str(device),'torch':torch.__version__,'cuda':torch.cuda.get_device_name(0),'image_id':iid,'cache_rows':len(cache['rows']),'feature_shapes':[list(z.shape) for z in feats],'h_shape':list(h.shape),'c_shape':list(c.shape),'h_finite':bool(torch.isfinite(h).all()),'c_finite':bool(torch.isfinite(c).all()),'elapsed_s':time.time()-t0,'c_state_keys':len(ck['cv4'])}
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
