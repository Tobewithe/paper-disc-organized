"""Seed-generalized copy of completed native output-layer control; no objective change."""
"""Same-data control: fine-tune only the existing coefficient readout's final layers."""
import argparse
from collections import defaultdict
import copy
import gzip
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

import learn_refinement as original
from component_seed_probe import setup
from shared_shape_head import bank_indices
from repair_refinement import per_loss,hard_iou


def load_model(args):
    m=YOLO(str(args.weights)).model.cuda().float().eval();assert m.model[-1].end2end;return m


def image_input(path,transform):
    im=cv2.imread(str(path));assert im is not None
    res=transform.apply_image({'img':im},transform.get_params({'img':im}))['img']
    return im.shape[:2],torch.from_numpy(np.ascontiguousarray(res[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255


@torch.no_grad()
def prepare(args):
    setup();start=time.monotonic();bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    split=json.loads(args.split.read_text());fit,dev=bank_indices(bank,split);model=load_model(args);head=model.model[-1]
    n=len(bank['records']);hidden=torch.zeros(n,64);levels=torch.zeros(n,dtype=torch.long);scale=torch.ones(n,32)
    done=torch.zeros(n,dtype=torch.bool);layer_states=[];captured={};hooks=[]
    for level,module in enumerate(head.one2one_cv4):
        layer=module[-1];assert layer.weight.shape==(32,64,1,1)
        layer_states.append({'weight':layer.weight.detach().cpu()[:,:,0,0],'bias':layer.bias.detach().cpu()})
        def hook(m,inputs,level=level):captured[level]=inputs[0].detach()
        hooks.append(layer.register_forward_pre_hook(hook))
    groups=defaultdict(list)
    for i in fit+dev:groups[bank['records'][i]['image_id']].append(i)
    transform=LetterBox((640,640),auto=True,stride=32);max_error=0.
    for ni,iid in enumerate(sorted(groups)):
        _,inp=image_input(args.data/'images/train2017'/f'{iid:012d}.jpg',transform)
        _,raw=model(inp);p=raw['one2one'];ph=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
        lv=torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],),l,device='cuda') for l in range(3)])
        ids=torch.tensor([bank['records'][i]['raw_id'] for i in groups[iid]],device='cuda')
        hidden[groups[iid]]=ph[ids].cpu();levels[groups[iid]]=lv[ids].cpu()
        ps=p['proto'][0].square().mean((1,2)).sqrt().clamp_min(.1)
        scale[groups[iid]]=ps.cpu();done[groups[iid]]=True
        rebuilt=torch.empty_like(p['mask_coefficient'][0].T[ids])
        for l in range(3):
            sel=lv[ids]==l
            if sel.any():rebuilt[sel]=F.linear(ph[ids][sel],head.one2one_cv4[l][-1].weight[:,:,0,0],head.one2one_cv4[l][-1].bias)
        error=float((rebuilt-p['mask_coefficient'][0].T[ids]).abs().max());max_error=max(max_error,error)
        if ni%100==0 or ni+1==len(groups):
            progress={'images':ni+1,'total':len(groups),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert bool(done[fit+dev].all()) and max_error<1e-4,max_error
    for hook in hooks:hook.remove()
    torch.save({'hidden':hidden,'level':levels,'scale':scale,'states':layer_states,'records':bank['records']},args.out/'coefficient_features.pt')
    original.save(args.out/'COMPLETE.json',{'images':len(groups),'records':len(fit)+len(dev),'max_reconstruction_error':max_error,'elapsed_s':time.monotonic()-start})


class Readout(nn.Module):
    def __init__(self,states):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(st['weight'].shape[1],st['weight'].shape[0]) for st in states])
        for layer,state in zip(self.layers,states):layer.load_state_dict(state)
    def forward(self,hidden,levels):
        out=torch.empty(len(hidden),32,device=hidden.device)
        for l in range(3):
            ix=levels==l
            if ix.any():out[ix]=self.layers[l](hidden[ix])
        return out


def train(args):
    setup();start=time.monotonic();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    features=torch.load(args.features,map_location='cpu',weights_only=False);assert features['records']==bank['records']
    fit,dev=bank_indices(bank,json.loads(args.split.read_text()))
    def dataset(indices):return TensorDataset(bank['x'][indices,:32],bank['base'][indices],bank['target'][indices],
        features['hidden'][indices],features['level'][indices],features['scale'][indices])
    fitset=dataset(fit);devset=dataset(dev);net=Readout(features['states']).cuda();frozen=copy.deepcopy(net).eval()
    for p in frozen.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(fitset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    history=[]
    for epoch in range(1,9):
        losses=[]
        for p,b,y,h,l,scale in loader:
            p,b,y,h,scale=[v.cuda().float() for v in (p,b,y,h,scale)];l=l.cuda()
            dc=(net(h,l)-frozen(h,l))*scale
            z=b+(p*dc[:,:,None,None]).sum(1);loss=per_loss(z,y).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)));torch.save({'states':[x.state_dict() for x in net.layers],'initial_states':features['states'],
            'epoch':epoch,'seed':args.seed,'parameters':sum(p.numel() for p in net.parameters())},args.out/f'epoch{epoch}.pt')
        progress={'epoch':epoch,'loss':history[-1],'gradient_norm':float(gn),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    alphas=[0.,.25,.5,1.];stats={str(a):{'sum_iou':0.,'n':0} for a in alphas}
    with torch.no_grad():
        for p,b,y,h,l,scale in DataLoader(devset,batch_size=128):
            p,b,y,h,scale=[v.cuda().float() for v in (p,b,y,h,scale)];l=l.cuda();dc=(net(h,l)-frozen(h,l))*scale
            delta=(p*dc[:,:,None,None]).sum(1)
            for a in alphas:stats[str(a)]['sum_iou']+=float(hard_iou(b+a*delta,y).sum());stats[str(a)]['n']+=len(h)
    for value in stats.values():value['mean_iou']=value['sum_iou']/value['n']
    alpha=max(alphas,key=lambda a:(stats[str(a)]['mean_iou'],-a))
    result={'alpha':alpha,'stats':stats,'history':history,'seed':args.seed,'epoch':8,'parameters':sum(p.numel() for p in net.parameters()),
            'scope':'Only native three final coefficient convolutions, dimensions inferred from checkpoint; same labeled candidates, not replayed TAL training.'}
    original.save(args.out/'SELECTION.json',result);original.save(args.out/'COMPLETE.json',result);print(json.dumps(result),flush=True)


@torch.no_grad()
def evaluate(args):
    setup();torch.backends.cudnn.benchmark=False;start=time.monotonic();model=load_model(args);head=model.model[-1]
    ck=torch.load(args.train/'epoch8.pt',weights_only=False);sel=json.loads((args.train/'SELECTION.json').read_text());alpha=sel['alpha']
    for module,state,initial in zip(head.one2one_cv4,ck['states'],ck['initial_states']):
        layer=module[-1]
        assert torch.equal(layer.weight[:,:,0,0].cpu(),initial['weight']) and torch.equal(layer.bias.cpu(),initial['bias'])
        layer.weight.copy_((initial['weight']+alpha*(state['weight'].cpu()-initial['weight']))[:,:,None,None])
        layer.bias.copy_(initial['bias']+alpha*(state['bias'].cpu()-initial['bias']))
    coco=COCO(str(args.data/'annotations/instances_val2017.json'));ids=sorted(coco.imgs);assert len(ids)==5000;cats=sorted(coco.cats)
    transform=LetterBox((640,640),auto=True,stride=32)
    with gzip.open(args.out/'predictions.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stream:
        for ni,iid in enumerate(ids):
            orig,inp=image_input(args.data/'images/val2017'/coco.imgs[iid]['file_name'],transform)
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];c=p['mask_coefficient'][0].T
            boxes=head._get_decode_boxes(p)[0].T;scores,classes,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            keep=scores[0,:,0]>.001;score=scores[0,keep,0];classes=classes[0,keep,0].long();rids=rids[0,keep,0]
            low=(c@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            for first in range(0,len(rids),16):
                ix=rids[first:first+16];z=F.interpolate(low[ix][None],shape,mode='bilinear',align_corners=False)[0]
                masks=ops.scale_masks(ops.crop_mask(z,boxes[ix]).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
                for j,mask in enumerate(masks):
                    if not mask.any():continue
                    k=first+j;rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                    stream.write(json.dumps({'image_id':iid,'category_id':cats[int(classes[k])],'score':float(score[k]),'raw_id':int(rids[k]),'segmentation':rle})+'\n')
            if ni%100==0 or ni+1==len(ids):
                stream.flush();pgr={'images':ni+1,'total':len(ids),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',pgr);print(json.dumps(pgr),flush=True)
    with gzip.open(args.out/'predictions.jsonl.gz','rt') as stream:pred=[json.loads(x) for x in stream]
    ev=COCOeval(coco,coco.loadRes(pred),'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
    for item in ev.evalImgs:
        if item is not None and item['aRng']==ev.params.areaRng[0]:good.update(int(g) for j,g in enumerate(item['gtIds']) if not item['gtIgnore'][j] and item['gtMatches'][ti,j]>0)
    reference=json.loads(args.reference.read_text());base=set(reference['baseline']);good&={a['id'] for a in coco.anns.values() if not a.get('iscrowd',0) and not a.get('ignore',0)}
    metrics=dict(zip(['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'],map(float,ev.stats)))
    result={'metrics':metrics,'matched75':len(good),'repaired75':len(good-base),'damaged75':len(base-good),'alpha':alpha,'parameters':sel['parameters']}
    original.save(args.out/'RESULTS.json',result);original.save(args.out/'MATCHED_GT75.json',{'baseline':sorted(base),'native_finetune':sorted(good)})
    original.save(args.out/'COMPLETE.json',{'images':5000,'elapsed_s':time.monotonic()-start,'result':result});print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate'])
    for k in ['out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['data','weights','bank','split','features','train','reference']:p.add_argument('--'+k,type=Path)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);globals()[args.action](args)
