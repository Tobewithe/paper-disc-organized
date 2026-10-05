from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops

def load_input(path, cache):
    im=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    rh,rw=[float(x) for x in cache['ratio_pad'][0]]; left=int(round(float(cache['ratio_pad'][1][0]))); top=int(round(float(cache['ratio_pad'][1][1])))
    nh,nw=round(im.shape[0]*rh),round(im.shape[1]*rw); im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((640,640,3),114,dtype=np.uint8); canvas[top:top+nh,left:left+nw]=im
    return torch.from_numpy(np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))).float().div_(255).unsqueeze(0)

def capture(model,x):
    cap={}; head=model.model[-1]
    def hook(_m,ins): cap['x']=[v.detach().clone() for v in ins[0]]
    h=head.register_forward_pre_hook(hook)
    with torch.inference_mode(): model(x)
    h.remove(); return cap['x']

def area(box): return float((((box[2:]-box[:2])/640.) .prod()*640.*640.).item())
def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',type=Path,required=True); ap.add_argument('--c-final',type=Path,required=True); ap.add_argument('--readout',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--split',required=True,choices=['dev','val']); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--lambda-eval',type=float,default=.003)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type!='cuda': raise RuntimeError('refusing CPU execution')
    idx=json.loads((a.cache/'INDEX.json').read_text()); items=list(idx[a.split]); yolo=YOLO(str(a.weights)); model=yolo.model.float().to(dev).eval(); head=model.model[-1]; ck=torch.load(a.c_final,map_location='cpu',weights_only=False); head.one2one_cv4.load_state_dict(ck['cv4'],strict=True); ro=torch.load(a.readout,map_location='cpu',weights_only=False); readouts=[{'W':r['W'].float().to(dev),'b':r['b'].float().to(dev)} for r in ro['readouts']]
    gain_default=9.83241; sumb0=sumb1=sumr=0.; n=0; per=[]; t=time.monotonic()
    for pos,it in enumerate(items,1):
        iid=int(it['image_id']); x=torch.load(a.cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False); feats=capture(model,load_input(a.data/'images'/('val2017' if a.split=='val' else 'train2017')/f'{iid:012d}.jpg',x).to(dev)); proto=F.interpolate(x['proto'].float()[None].to(dev),(640,640),mode='bilinear',align_corners=False)[0]; masks=x['masks'].to(dev); gain=float(x.get('segmentation_gain',gain_default)); img0=img1=imgr=0.
        fields=[head.one2one_cv4[l][:-1](feats[l]).reshape(1,64,-1).squeeze(0).T for l in range(3)]
        offsets=[0]
        for f in fields: offsets.append(offsets[-1]+f.shape[0])
        for k,row in enumerate(x['rows']):
            rid=int(row['raw_id']); l=int(row['level']); h=fields[l][rid-offsets[l]]; cb=F.linear(h,readouts[l]['W'],readouts[l]['b']); ca=x['coeff'][rid].float().to(dev); box=x['target_boxes'][k].float(); support=ops.crop_mask(torch.ones((1,640,640),device=dev),box[None].to(dev))[0].bool(); p=proto[:,support].T; y=(masks[support]==int(x['owners'][k])+1).float(); ar=area(box); b0=float((gain*(F.softplus(p@ca)-y*(p@ca)).sum()/ar).detach().cpu()); b1=float((gain*(F.softplus(p@cb)-y*(p@cb)).sum()/ar).detach().cpu()); rg=float((a.lambda_eval*.5*(cb-ca).square().sum()).detach().cpu()); sumb0+=b0; sumb1+=b1; sumr+=rg; n+=1; img0+=b0; img1+=b1+rg; imgr+=rg
        per.append({'image_id':iid,'n_candidates':len(x['rows']),'A_J':img0/max(len(x['rows']),1),'B_J':img1/max(len(x['rows']),1),'delta':(img1-img0)/max(len(x['rows']),1)})
        if pos%50==0 or pos==len(items): print(json.dumps({'split':a.split,'images':pos,'total':len(items),'elapsed_s':time.monotonic()-t}),flush=True)
    result={'split':a.split,'images':len(items),'candidates':n,'A_BCE':sumb0/n,'B_BCE':sumb1/n,'B_regularizer':sumr/n,'B_J':(sumb1+sumr)/n,'delta_B_minus_A':(sumb1+sumr-sumb0)/n,'weights_sha256':sha(a.weights),'per_image':per}; (a.out/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
