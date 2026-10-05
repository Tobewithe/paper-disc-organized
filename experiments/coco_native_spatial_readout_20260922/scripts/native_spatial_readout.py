"""Reuse native mask-head features for low-dimensional box-relative correction."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

import learn_refinement as original
from component_seed_probe import setup
from repair_refinement import per_loss,hard_iou
from shared_shape_head import bank_indices,grid_tensor,roi_template,full_template,correct
from native_coefficient_control import load_model,image_input


class NativeSpatialHead(nn.Module):
    def __init__(self,dim,mean,std,scalar=False):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.layer=nn.Linear(dim,64);self.output=nn.Linear(64,2);self.scalar=scalar
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        values=self.output(F.silu(self.layer((x-self.mean)/self.std))).tanh()
        return values*values.new_tensor([1.,0.]) if self.scalar else values


def input_features(hidden,level,roi_logits=None):
    parts=[hidden,F.one_hot(level,3).float()]
    if roi_logits is not None:parts.append(F.adaptive_avg_pool2d(roi_logits[:,None],4).flatten(1).div(8).clamp(-4,4))
    return torch.cat(parts,1)


def train(args):
    setup();start=time.monotonic();bank=torch.load(args.bank,map_location='cpu',weights_only=False,mmap=True)
    native=torch.load(args.features,map_location='cpu',weights_only=False);geometry=torch.load(args.geometry,map_location='cpu',weights_only=False)
    assert native['records']==bank['records']==geometry['records'];split=json.loads(args.split.read_text());fit,dev=bank_indices(bank,split)
    use_response=args.mode=='response_spatial'
    features=input_features(native['hidden'].float(),native['level'],bank['base'].float() if use_response else None)
    mean=features[fit].mean(0);std=features[fit].std(0,unbiased=False).clamp_min(.05)
    norm=torch.stack((geometry['mean'],geometry['std']),1)
    rt=roi_template(torch.tensor(geometry['template']['field']).reshape(1,16)).cuda()
    def dataset(ix):return TensorDataset(features[ix],bank['base'][ix],bank['target'][ix],norm[ix])
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    net=NativeSpatialHead(features.shape[1],mean,std,args.mode=='native_scalar').cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(dataset(fit),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    history=[];original.save(args.out/'SPLIT.json',split)
    for epoch in range(1,9):
        losses=[]
        for x,b,y,nm in loader:
            x,b,y,nm=[v.cuda().float() for v in (x,b,y,nm)]
            basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12)
            z=correct(b,basis,net(x));loss=per_loss(z,y).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        ck={'state_dict':net.state_dict(),'dim':features.shape[1],'mode':args.mode,'seed':args.seed,'epoch':epoch,
            'template':geometry['template'],'parameters':sum(p.numel() for p in net.parameters())}
        torch.save(ck,args.out/f'epoch{epoch}.pt')
        progress={'epoch':epoch,'loss':history[-1],'gradient_norm':float(gn),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    alphas=[0.,.25,.5,1.];stats={str(a):{'sum_iou':0.,'n':0,'repairs':0,'harms':0} for a in alphas}
    net.eval()
    with torch.no_grad():
        for x,b,y,nm in DataLoader(dataset(dev),batch_size=128):
            x,b,y,nm=[v.cuda().float() for v in (x,b,y,nm)];basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12)
            values=net(x);base_iou=hard_iou(b,y)
            for a in alphas:
                iou=hard_iou(correct(b,basis,values*a),y);r=stats[str(a)];r['sum_iou']+=float(iou.sum());r['n']+=len(x)
                r['repairs']+=int(((iou>=.75)&(base_iou<.75)).sum());r['harms']+=int(((iou<.75)&(base_iou>=.75)).sum())
    for r in stats.values():r['mean_iou']=r['sum_iou']/r['n']
    alpha=max(alphas,key=lambda a:(stats[str(a)]['mean_iou'],-a))
    result={'mode':args.mode,'alpha':alpha,'stats':stats,'history':history,'parameters':ck['parameters'],'dim':features.shape[1],'seed':args.seed,'epoch':8}
    original.save(args.out/'SELECTION.json',result);original.save(args.out/'COMPLETE.json',result);print(json.dumps(result),flush=True)


def load_head(path):
    ck=torch.load(path/'epoch8.pt',map_location='cpu',weights_only=False);state=ck['state_dict']
    net=NativeSpatialHead(ck['dim'],state['mean'],state['std'],ck['mode']=='native_scalar').cuda().eval();net.load_state_dict(state)
    return net,json.loads((path/'SELECTION.json').read_text())['alpha'],ck


@torch.no_grad()
def evaluate(args):
    setup();torch.backends.cudnn.benchmark=False;start=time.monotonic();model=load_model(args);head=model.model[-1]
    config=json.loads(args.models.read_text());heads={name:load_head(Path(path)) for name,path in config['heads'].items()}
    template=list(heads.values())[0][2]['template'];assert all(h[2]['template']['field']==template['field'] for h in heads.values())
    grid=torch.tensor(template['field'],device='cuda').reshape(1,16)
    native_ck=torch.load(config['native_checkpoint'],map_location='cpu',weights_only=False)
    delta=[{k:(s[k]-i[k]).cuda() for k in ['weight','bias']} for s,i in zip(native_ck['states'],native_ck['initial_states'])]
    for module,state in zip(head.one2one_cv4,native_ck['initial_states']):
        assert torch.equal(module[-1].weight[:,:,0,0].cpu(),state['weight'])
        assert torch.equal(module[-1].bias.cpu(),state['bias'])
    captured={};hooks=[]
    for level,module in enumerate(head.one2one_cv4):
        def hook(m,inputs,level=level):captured[level]=inputs[0].detach()
        hooks.append(module[-1].register_forward_pre_hook(hook))
    coco=COCO(str(args.data/'annotations/instances_val2017.json'));ids=sorted(coco.imgs);cats=sorted(coco.cats)
    if args.smoke:ids=ids[:3]
    names=list(heads)+['native_alpha025','native_alpha050'];streams={n:gzip.open(args.out/f'predictions_{n}.jsonl.gz','wt',encoding='utf-8',compresslevel=1) for n in names}
    transform=LetterBox((640,640),auto=True,stride=32);identity_errors=0;feature_error=0.;counts={n:0 for n in names}
    try:
        for ni,iid in enumerate(ids):
            orig,inp=image_input(args.data/'images/val2017'/coco.imgs[iid]['file_name'],transform)
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];c=p['mask_coefficient'][0].T
            hidden=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            levels=torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],),l,device='cuda') for l in range(3)])
            boxes=head._get_decode_boxes(p)[0].T;scores,classes,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            keep=scores[0,:,0]>.001;score=scores[0,keep,0];classes=classes[0,keep,0].long();rids=rids[0,keep,0]
            # Same original all-raw mask multiplication as the existing reference evaluator.
            low=(c@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            dcoef=torch.empty(len(rids),32,device='cuda')
            for l in range(3):
                at=levels[rids]==l
                if at.any():dcoef[at]=F.linear(hidden[rids][at],delta[l]['weight'],delta[l]['bias'])
            for first in range(0,len(rids),16):
                ix=rids[first:first+16];bb=boxes[ix];roi_grid=original.roi_grid(bb,shape)
                roi_z=F.grid_sample(low[ix,None],roi_grid,align_corners=False)[:,0]
                if ni==0:
                    _,reference_roi,_=original.features(proto,c[ix],bb,shape)
                    feature_error=max(feature_error,float((F.adaptive_avg_pool2d(roi_z[:,None],4)-F.adaptive_avg_pool2d(reference_roi[:,None],4)).abs().max()))
                base=F.interpolate(low[ix][None],shape,mode='bilinear',align_corners=False)[0]
                basis,_,_,_=full_template(grid,bb,shape)
                if ni==0:
                    decoded=ops.crop_mask(base,bb).gt(0);official=ops.process_mask(proto,c[ix],bb,shape,upsample=True).bool()
                    identity_errors+=int((decoded!=official).sum())
                variants={}
                for name,(net,alpha,ck) in heads.items():
                    features=input_features(hidden[ix],levels[ix],roi_z if ck['mode']=='response_spatial' else None)
                    variants[name]=correct(base,basis,net(features)*alpha)
                dc=dcoef[first:first+len(ix)]
                for name,alpha in [('native_alpha025',.25),('native_alpha050',.5)]:
                    # Evaluate true interpolated native coefficients, not a separately thresholded correction.
                    newlow=((c[ix]+alpha*dc)@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
                    variants[name]=F.interpolate(newlow[None],shape,mode='bilinear',align_corners=False)[0]
                for name,z in variants.items():
                    masks=ops.scale_masks(ops.crop_mask(z,bb).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
                    for j,mask in enumerate(masks):
                        if not mask.any():continue
                        k=first+j;rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                        streams[name].write(json.dumps({'image_id':iid,'category_id':cats[int(classes[k])],'score':float(score[k]),'raw_id':int(rids[k]),'segmentation':rle})+'\n');counts[name]+=1
            if ni%100==0 or ni+1==len(ids):
                for stream in streams.values():stream.flush()
                progress={'images':ni+1,'total':len(ids),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    finally:
        for stream in streams.values():stream.close()
        for hook in hooks:hook.remove()
    assert identity_errors==0,identity_errors
    assert feature_error<1e-3,feature_error
    if args.smoke:
        original.save(args.out/'COMPLETE.json',{'smoke':True,'images':len(ids),'identity_pixel_errors':identity_errors,'roi_feature_error':feature_error});return
    assert len(ids)==5000
    reference=json.loads(args.reference.read_text());base=set(reference['baseline']);results={};matches={'baseline':sorted(base)}
    for name in names:
        with gzip.open(args.out/f'predictions_{name}.jsonl.gz','rt') as stream:pred=[json.loads(line) for line in stream]
        ev=COCOeval(coco,coco.loadRes(pred),'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
        for item in ev.evalImgs:
            if item is not None and item['aRng']==ev.params.areaRng[0]:
                good.update(int(g) for j,g in enumerate(item['gtIds']) if not item['gtIgnore'][j] and item['gtMatches'][ti,j]>0)
        metrics=dict(zip(['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'],map(float,ev.stats)))
        results[name]={'metrics':metrics,'matched75':len(good),'repaired75':len(good-base),'damaged75':len(base-good),'predictions':counts[name]}
        if name in heads:results[name].update(alpha=heads[name][1],parameters=heads[name][2]['parameters'])
        else:results[name].update(alpha=.25 if name.endswith('025') else .5,parameters=6240,sensitivity_only=True)
        matches[name]=sorted(good);original.save(args.out/'RESULTS.json',results);original.save(args.out/'MATCHED_GT75.json',matches)
        del ev,pred
    original.save(args.out/'COMPLETE.json',{'images':5000,'identity_pixel_errors':identity_errors,'roi_feature_error':feature_error,'elapsed_s':time.monotonic()-start,'models':config})
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--out',type=Path,required=True)
    for name in ['bank','features','geometry','split','data','weights','models','reference']:p.add_argument('--'+name,type=Path)
    p.add_argument('--mode',choices=['native_scalar','native_spatial','response_spatial']);p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);globals()[args.action](args)
