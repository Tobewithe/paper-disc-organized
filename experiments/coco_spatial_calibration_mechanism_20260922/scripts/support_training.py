"""Loss-support intervention on the same analytic-template head and training bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from pycocotools.coco import COCO

import learn_refinement as original
from component_seed_probe import setup
from repair_refinement import per_loss,hard_iou
from shared_shape_head import SharedShapeHead,bank_indices,grid_tensor,roi_template,correct


def support_loss(z,y,mask):
    bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    prob=z.sigmoid()
    dice=1-(2*(prob*y*mask).sum((1,2))+1)/((prob*mask).sum((1,2))+(y*mask).sum((1,2))+1)
    return bce+.5*dice


@torch.no_grad()
def prepare(args):
    setup();start=time.monotonic();bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    split=json.loads(args.split.read_text());fit,dev=bank_indices(bank,split)
    coco=COCO(str(args.data/'annotations/instances_train2017.json'))
    model=YOLO(str(args.weights)).model.cuda().float().eval();head=model.model[-1]
    byimage=defaultdict(list)
    for idx in fit+dev:byimage[bank['records'][idx]['image_id']].append(idx)
    masks=torch.ones((len(bank['records']),32,32),dtype=torch.bool);matched=masks.clone();done=torch.zeros(len(masks),dtype=torch.bool)
    removed_foreground=0;removed_total=0
    transform=LetterBox((640,640),auto=True,stride=32)
    for ni,iid in enumerate(sorted(byimage)):
        image=cv2.imread(str(args.data/'images/train2017'/f'{iid:012d}.jpg'));assert image is not None
        resized=transform.apply_image({'img':image},transform.get_params({'img':image}))['img']
        inp=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
        shape=tuple(inp.shape[2:]);_,raw=model(inp);boxes=head._get_decode_boxes(raw['one2one'])[0].T
        gain=min(shape[0]/image.shape[0],shape[1]/image.shape[1]);nh,nw=round(image.shape[0]*gain),round(image.shape[1]*gain)
        top,left=round((shape[0]-nh)/2-.1),round((shape[1]-nw)/2-.1)
        for idx in byimage[iid]:
            row=bank['records'][idx];box=boxes[row['raw_id']:row['raw_id']+1]
            coords=original.roi_grid(box,shape)[0];x=(coords[...,0]+1)*shape[1]/2;y=(coords[...,1]+1)*shape[0]/2
            gx,gy,gw,gh=coco.anns[row['annotation_id']]['bbox']
            keep=((x>=gx*gain+left)&(x<(gx+gw)*gain+left)&(y>=gy*gain+top)&(y<(gy+gh)*gain+top)).cpu()
            assert keep.any();masks[idx]=keep;done[idx]=True
            target=bank['target'][idx]>=.5;removed_total+=int((~keep).sum());removed_foreground+=int((target&~keep).sum())
            rng=np.random.default_rng(20260924+idx);control=torch.ones(1024,dtype=torch.bool)
            for cls in [False,True]:
                candidates=torch.where(target.flatten()==cls)[0].numpy()
                n=int(((target==cls)&~keep).sum())
                if n:control[rng.choice(candidates,n,replace=False)]=False
            matched[idx]=control.reshape(32,32)
            assert int((~matched[idx]).sum())==int((~keep).sum())
            assert int((target&~matched[idx]).sum())==int((target&~keep).sum())
        if ni%100==0 or ni+1==len(byimage):
            p={'images':ni+1,'total':len(byimage),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',p);print(json.dumps(p),flush=True)
    assert bool(done[fit+dev].all())
    torch.save({'gt_support':masks,'matched_random':matched,'records':bank['records'],'fit':fit,'dev':dev},args.out/'supports.pt')
    original.save(args.out/'COMPLETE.json',{'images':len(byimage),'records':len(fit)+len(dev),'removed_pixels':removed_total,
        'removed_hard_foreground':removed_foreground,'random_matches_fg_bg_counts':True,'elapsed_s':time.monotonic()-start})


def train(args):
    setup();start=time.monotonic()
    bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    support=torch.load(args.supports,map_location='cpu',weights_only=False)
    geometry=torch.load(args.geometry,map_location='cpu',weights_only=False)
    split=json.loads(args.split.read_text());fit,dev=bank_indices(bank,split)
    assert support['records']==bank['records']==geometry['records']
    assert geometry['template']['field']==json.loads(args.template.read_text())['field']
    norm=torch.stack((geometry['mean'],geometry['std']),1)
    masks=support[args.mode];rt=roi_template(grid_tensor(args.template,'cpu'))
    fitset=TensorDataset(*(bank[k][fit] for k in ['x','base','target']),norm[fit],masks[fit])
    devset=TensorDataset(*(bank[k][dev] for k in ['x','base','target']),norm[dev])
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    net=SharedShapeHead().cuda();rt=rt.cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(fitset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(0),num_workers=0)
    original.save(args.out/'SPLIT.json',split)
    with torch.no_grad():
        z=bank['base'][:2].float();y=bank['target'][:2].float()
        error=float((support_loss(z,y,torch.ones_like(y))-per_loss(z,y)).abs().max());assert error<1e-6
    history=[]
    for epoch in range(1,9):
        net.train();losses=[]
        for x,b,y,nm,mask in loader:
            x,b,y,nm,mask=[v.cuda().float() for v in (x,b,y,nm,mask)]
            basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12)
            z=correct(b,basis,net(x));loss=support_loss(z,y,mask).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        torch.save({'state_dict':net.state_dict(),'seed':0,'epoch':epoch,'mode':'shared2','template':geometry['template'],
            'parameter_count':sum(p.numel() for p in net.parameters()),'loss_support':args.mode},args.out/f'epoch{epoch}.pt')
        p={'epoch':epoch,'loss':history[-1],'gradient_norm':float(grad),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',p);print(json.dumps(p),flush=True)
    net.eval();alphas=[0.,.25,.5,1.];stats={str(a):{'total':0.,'n':0,'repairs':0,'damages':0,'loss':0.} for a in alphas}
    with torch.no_grad():
        for x,b,y,nm in DataLoader(devset,batch_size=128):
            x,b,y,nm=[v.cuda().float() for v in (x,b,y,nm)];basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12)
            values=net(x);oi=hard_iou(b,y)
            for a in alphas:
                z=correct(b,basis,values*a);iou=hard_iou(z,y);item=stats[str(a)]
                item['total']+=float(iou.sum());item['n']+=len(x);item['repairs']+=int(((iou>=.75)&(oi<.75)).sum());item['damages']+=int(((iou<.75)&(oi>=.75)).sum());item['loss']+=float(per_loss(z,y).sum())
    for item in stats.values():item.update(mean_iou=item['total']/item['n'],mean_loss=item['loss']/item['n'])
    alpha=max(alphas,key=lambda a:(stats[str(a)]['mean_iou'],-a))
    result={'mode':'shared2','loss_support':args.mode,'seed':0,'alpha':alpha,'stats':stats,'history':history,
        'checkpoint':'epoch8.pt','parameters':sum(p.numel() for p in net.parameters()),'independent_head':True,
        'template_source':str(args.template),'full_mask_identity_loss_error':error,'elapsed_s':time.monotonic()-start}
    original.save(args.out/'SELECTION.json',result);original.save(args.out/'COMPLETE.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train'])
    for k in ['bank','split','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['data','weights','template','geometry','supports']:p.add_argument('--'+k,type=Path)
    p.add_argument('--mode',choices=['gt_support','matched_random']);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    (prepare if args.action=='prepare' else train)(args)
