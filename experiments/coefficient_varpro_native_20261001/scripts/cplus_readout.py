from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops

def load_input(path: Path, cache: dict) -> torch.Tensor:
    im=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    rh,rw=[float(x) for x in cache['ratio_pad'][0]]; top,left=[int(round(float(x))) for x in (cache['ratio_pad'][1][1],cache['ratio_pad'][1][0])]
    nh,nw=round(im.shape[0]*rh),round(im.shape[1]*rw); im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((640,640,3),114,dtype=np.uint8); canvas[top:top+nh,left:left+nw]=im
    x=torch.from_numpy(np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))).float().div_(255).unsqueeze(0); return x

def sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def compute_h(model, cache, image_id, data_root, device):
    head=model.model[-1]; cap={}
    def hook(_m,inputs): cap['x']=[v.detach() for v in inputs[0]]
    handle=head.register_forward_pre_hook(hook)
    with torch.inference_mode(): model(load_input(data_root/'images'/'train2017'/f'{image_id:012d}.jpg',cache).to(device))
    handle.remove(); fields=[]
    for branch,feat in zip(head.one2one_cv4,cap['x']): fields.append(branch[:-1](feat).reshape(1,64,-1).squeeze(0).T)
    return fields

def row_h(fields, row):
    return fields[int(row['level'])][int(row['raw_id'])-sum(x.shape[0] for x in fields[:int(row['level'])])]

def evaluate(readouts, cache_root, index_items, hcache_root, device, lambda_eval):
    total_bce=total_reg=0.0; n=0
    for item in index_items:
        iid=int(item['image_id']); cache=torch.load(cache_root/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False); proto=F.interpolate(cache['proto'].float()[None].to(device),(640,640),mode='bilinear',align_corners=False)[0]; masks=cache['masks'].to(device); hs=torch.load(hcache_root/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
        for k,row in enumerate(cache['rows']):
            h=hs[k].to(device); l=int(row['level']); c=F.linear(h,readouts[l]['W'],readouts[l]['b']); c0=cache['coeff'][int(row['raw_id'])].float().to(device); box=cache['target_boxes'][k].float(); support=ops.crop_mask(torch.ones((1,640,640),device=device),box[None].to(device))[0].bool(); p=proto[:,support].T; y=(masks[support]==int(cache['owners'][k])+1).float(); z=p@c; area=float((((box[2:]-box[:2])/640.0).prod()*640.0*640.0)); bce=float((float(cache['segmentation_gain'])*(F.softplus(z)-y*z).sum()/area).detach().cpu()); reg=float((float(lambda_eval)*0.5*(c-c0).square().sum()).detach().cpu()); total_bce+=bce; total_reg+=reg; n+=1
    return {'candidates':n,'bce':total_bce/n,'regularizer':total_reg/n,'J_eval':(total_bce+total_reg)/n}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',type=Path,required=True); ap.add_argument('--c-final',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--lambda-eval',type=float,default=0.003); ap.add_argument('--max-images',type=int,default=0); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    if device.type!='cuda': raise RuntimeError('refusing CPU execution')
    index=json.loads((args.cache/'INDEX.json').read_text(encoding='utf-8')); items=list(index['fit']);
    if args.max_images: items=items[:args.max_images]
    yolo=YOLO(str(args.weights)); model=yolo.model.float().to(device).eval(); head=model.model[-1]; ck=torch.load(args.c_final,map_location='cpu',weights_only=False); head.one2one_cv4.load_state_dict(ck['cv4'],strict=True)
    hdir=args.out/'h_cache'; hdir.mkdir(exist_ok=True); prep=[]; t0=time.time()
    for j,item in enumerate(items):
        iid=int(item['image_id']); cpath=args.cache/'images'/f'{iid:012d}.pt'; cache=torch.load(cpath,map_location='cpu',weights_only=False); fields=compute_h(model,cache,iid,args.data,device); hs=torch.stack([row_h(fields,r) for r in cache['rows']]).detach().cpu(); torch.save(hs,hdir/f'{iid:012d}.pt'); prep.append({'image_id':iid,'rows':len(cache['rows']),'h_shape':list(hs.shape)}); 
        if (j+1)%25==0 or j+1==len(items): print(json.dumps({'stage':'feature_extract','images':j+1,'total':len(items),'elapsed_s':time.time()-t0}),flush=True)
    # Initialize the shared readout from the C checkpoint's native 1x1 convolutions.
    readouts=[]
    for l in range(3):
        w=ck['cv4'][f'{l}.2.weight'].float().squeeze(-1).squeeze(-1).to(device).clone().requires_grad_(True); b=ck['cv4'][f'{l}.2.bias'].float().to(device).clone().requires_grad_(True); readouts.append({'W':w,'b':b})
    params=[q for r in readouts for q in (r['W'],r['b'])]; opt=torch.optim.Adam(params,lr=args.lr); losses=[]; n_total=sum(len(torch.load(args.cache/'images'/f'{int(i["image_id"]):012d}.pt',map_location='cpu',weights_only=False)['rows']) for i in items); 
    # Full objective gradient accumulation: one optimizer step per complete fit pass.
    for ep in range(args.epochs):
        opt.zero_grad(set_to_none=True); total=0.0; count=0; begin=time.time()
        for j,item in enumerate(items):
            iid=int(item['image_id']); cache=torch.load(args.cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False); proto=F.interpolate(cache['proto'].float()[None].to(device),(640,640),mode='bilinear',align_corners=False)[0]; masks=cache['masks'].to(device); hs=torch.load(hdir/f'{iid:012d}.pt',map_location='cpu')
            image_loss=torch.zeros((),device=device)
            for k,row in enumerate(cache['rows']):
                l=int(row['level']); c=F.linear(hs[k].to(device),readouts[l]['W'],readouts[l]['b']); c0=cache['coeff'][int(row['raw_id'])].float().to(device); box=cache['target_boxes'][k].float(); support=ops.crop_mask(torch.ones((1,640,640),device=device),box[None])[0].bool(); p=proto[:,support].T; y=(masks[support]==int(cache['owners'][k])+1).float(); z=p@c; area=float((((box[2:]-box[:2])/640.0).prod()*640.0*640.0)); image_loss=image_loss+(float(cache['segmentation_gain'])*(F.softplus(z)-y*z).sum()/area+float(args.lambda_eval)*0.5*(c-c0).square().sum())
            image_loss=(image_loss/max(len(cache['rows']),1))/max(n_total,1); image_loss.backward(); total+=float(image_loss.detach().cpu())*n_total; count+=len(cache['rows'])
            if (j+1)%50==0: print(json.dumps({'stage':'readout_gradient','epoch':ep+1,'images':j+1,'total':len(items)}),flush=True)
        grad=float(torch.nn.utils.clip_grad_norm_(params,float('inf'))); opt.step(); rec={'epoch':ep+1,'J_est':total,'candidates':count,'grad_norm':grad,'elapsed_s':time.time()-begin}; losses.append(rec); print(json.dumps({'stage':'readout_epoch',**rec}),flush=True)
    final={'device':str(device),'gpu':torch.cuda.get_device_name(0),'images':len(items),'candidates':n_total,'epochs':args.epochs,'lambda_eval':args.lambda_eval,'weights_sha256':sha256(args.weights),'c_checkpoint':str(args.c_final),'feature_cache':str(hdir),'train_log':losses,'readouts':[{'W':r['W'].detach().cpu(),'b':r['b'].detach().cpu()} for r in readouts]}
    torch.save(final,args.out/'C_READOUT.pt');
    # evaluate adapted readout and original c0 on same images
    adapted=[{'W':x['W'].to(device),'b':x['b'].to(device)} for x in final['readouts']]; ev=evaluate(adapted,args.cache,items,hdir,device,args.lambda_eval); a_bce=a_j=0.0; an=0
    for item in items:
        cache=torch.load(args.cache/'images'/f'{int(item["image_id"]):012d}.pt',map_location='cpu',weights_only=False)
        for k,row in enumerate(cache['rows']):
            c0=cache['coeff'][int(row['raw_id'])].float(); box=cache['target_boxes'][k].float(); proto=F.interpolate(cache['proto'].float()[None],(640,640),mode='bilinear',align_corners=False)[0]; support=ops.crop_mask(torch.ones((1,640,640)),box[None])[0].bool(); p=proto[:,support].T; y=(cache['masks'][support]==int(cache['owners'][k])+1).float(); z=p@c0; area=float((((box[2:]-box[:2])/640.0).prod()*640.0*640.0)); a_bce+=float((float(cache['segmentation_gain'])*(F.softplus(z)-y*z).sum()/area).cpu()); an+=1
    ev['A_bce']=a_bce/an; ev['A_J_eval']=ev['A_bce']; ev['delta_vs_A']=ev['J_eval']-ev['A_J_eval']; cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}; (args.out/'SUMMARY.json').write_text(json.dumps({'config':cfg,'prepare':prep,'evaluation':ev},indent=2,default=float),encoding='utf-8'); print(json.dumps({'stage':'complete','evaluation':ev}),flush=True)
if __name__=='__main__': main()

