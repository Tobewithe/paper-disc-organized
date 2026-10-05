"""Fixed spatial priors and coefficient residual controls, not a full SipMask reproduction."""
import argparse
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
from shared_shape_head import bank_indices,roi_template,full_template
from native_coefficient_control import load_model,image_input
from native_spatial_readout import input_features

MODES=['global_scalar','global_spatial','native_coeff_mlp','native_quad_coeff_mlp']


class Control(nn.Module):
    def __init__(self,mode,mean,std):
        super().__init__();self.mode=mode
        self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        if mode.startswith('global'):
            self.values=nn.Parameter(torch.zeros(1 if mode=='global_scalar' else 2))
        else:
            self.hidden=nn.Linear(67,64);self.output=nn.Linear(64,32 if mode=='native_coeff_mlp' else 128)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        if self.mode.startswith('global'):return self.values.tanh()[None].expand(len(x),-1)
        return self.output(F.silu(self.hidden((x-self.mean)/self.std))).tanh()


def roi_quadrants(size,device):
    t=torch.arange(size,device=device)+.5
    return (t[None,:]>=size/2).long()+2*(t[:,None]>=size/2).long()


def full_quadrants(boxes,shape):
    y=torch.arange(shape[0],device=boxes.device)+.5;x=torch.arange(shape[1],device=boxes.device)+.5
    mid=(boxes[:,:2]+boxes[:,2:])/2
    return (x[None,None,:]>=mid[:,0,None,None]).long()+2*(y[None,:,None]>=mid[:,1,None,None]).long()


def apply_roi(base,proto,basis,values,mode):
    if mode=='global_scalar':return base+4*values[:,0,None,None]
    if mode=='global_spatial':return base+4*(values[:,0,None,None]+values[:,1,None,None]*basis)
    if mode=='native_coeff_mlp':return base+2*(values[:,:,None,None]*proto).sum(1)
    maps=2*torch.einsum('nqc,nchw->nqhw',values.reshape(-1,4,32),proto)
    q=roi_quadrants(base.shape[-1],base.device)[None,None].expand(len(base),1,-1,-1)
    return base+maps.gather(1,q)[:,0]


def full_coeff_delta(proto,scale,boxes,shape,values,mode):
    q=1 if mode=='native_coeff_mlp' else 4
    dc=2*values.reshape(-1,q,32)/scale[None,None]
    low=(dc.reshape(-1,32)@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
    maps=F.interpolate(low[None],shape,mode='bilinear',align_corners=False)[0].reshape(len(boxes),q,*shape)
    if q==1:return maps[:,0]
    return maps.gather(1,full_quadrants(boxes,shape)[:,None])[:,0]


def checks(args):
    # CPU algebra check: no image data or GPU interference with the running replication.
    torch.set_num_threads(2);torch.manual_seed(20260922)
    n=3;p=torch.randn(n,32,32,32);b=torch.randn(n,32,32);v=torch.randn(n,32)
    one=apply_roi(b,p,None,v,'native_coeff_mlp')
    four=apply_roi(b,p,None,v.repeat(1,4),'native_quad_coeff_mlp')
    error=float((one-four).abs().max());assert error<2e-5,error
    p=torch.zeros(1,32,32,32);p[:,0]=1;v=torch.zeros(1,4,32);v[0,:,0]=torch.tensor([1.,2.,3.,4.])
    got=apply_roi(torch.zeros(1,32,32),p,None,v.flatten(1),'native_quad_coeff_mlp')[0]
    expected=2*(roi_quadrants(32,'cpu')+1);assert torch.equal(got,expected)
    bb=torch.tensor([[0.,0.,32.,32.]]);assert torch.equal(full_quadrants(bb,(32,32))[0],roi_quadrants(32,'cpu'))
    # Full decode uses the same quadrant assembly and normalized coefficient convention.
    proto=torch.zeros(32,32,32);proto[0]=1
    full=full_coeff_delta(proto,torch.ones(32),bb,(32,32),v.flatten(1),'native_quad_coeff_mlp')
    assert torch.equal(full[0],expected)
    original.save(args.out/'COMPLETE.json',{'same_coefficients_max_error':error,'quadrant_order':'TL,TR,BL,BR','full_roi_partition_consistent':True})


def train(args):
    setup();start=time.monotonic()
    bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    native=torch.load(args.features,map_location='cpu',weights_only=False);geometry=torch.load(args.geometry,map_location='cpu',weights_only=False)
    assert bank['records']==native['records']==geometry['records']
    split=json.loads(args.split.read_text());fit,dev=bank_indices(bank,split)
    all_features=input_features(native['hidden'].float(),native['level'])
    mean=all_features[fit].mean(0);std=all_features[fit].std(0,unbiased=False).clamp_min(.05)
    norm=torch.stack([geometry['mean'],geometry['std']],1)
    rt=roi_template(torch.tensor(geometry['template']['field']).reshape(1,16)).cuda()
    def dataset(ix):return TensorDataset(all_features[ix],bank['x'][ix,:32],bank['base'][ix],bank['target'][ix],norm[ix])
    fit_dataset,dev_dataset=dataset(fit),dataset(dev)
    assert len(fit_dataset)==len(fit) and len(dev_dataset)==len(dev)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    net=Control(args.mode,mean,std).cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(fit_dataset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    history=[];original.save(args.out/'SPLIT.json',split)
    for epoch in range(1,9):
        losses=[]
        for x,p,b,y,nm in loader:
            x,p,b,y,nm=[v.cuda().float() for v in (x,p,b,y,nm)]
            basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12)
            z=apply_roi(b,p,basis,net(x),args.mode);loss=per_loss(z,y).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        torch.save({'mode':args.mode,'state_dict':net.state_dict(),'seed':args.seed,'epoch':epoch,
             'template':geometry['template'],'parameters':sum(v.numel() for v in net.parameters())},args.out/f'epoch{epoch}.pt')
        progress={'epoch':epoch,'loss':history[-1],'gradient_norm':float(gn),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    stats={str(a):{'n':0,'sum_iou':0.,'repairs':0,'harms':0} for a in [0.,.25,.5,1.]};net.eval()
    with torch.no_grad():
        for x,p,b,y,nm in DataLoader(dev_dataset,batch_size=128):
            x,p,b,y,nm=[v.cuda().float() for v in (x,p,b,y,nm)]
            basis=(rt[None]-nm[:,0,None,None])/nm[:,1,None,None].clamp_min(1e-12);values=net(x);ref=hard_iou(b,y)
            for key,st in stats.items():
                iou=hard_iou(apply_roi(b,p,basis,values*float(key),args.mode),y)
                st['n']+=len(x);st['sum_iou']+=float(iou.sum());st['repairs']+=int(((iou>=.75)&(ref<.75)).sum());st['harms']+=int(((iou<.75)&(ref>=.75)).sum())
    for st in stats.values():st['mean_iou']=st['sum_iou']/st['n']
    alpha=max([0.,.25,.5,1.],key=lambda a:(stats[str(a)]['mean_iou'],-a))
    result={'mode':args.mode,'seed':args.seed,'epoch':8,'alpha':alpha,'stats':stats,'history':history,'parameters':sum(v.numel() for v in net.parameters())}
    if args.mode.startswith('global'):result['effective_unscaled_outputs']=(4*net.values.tanh()).detach().cpu().tolist()
    original.save(args.out/'SELECTION.json',result);original.save(args.out/'COMPLETE.json',result);print(json.dumps(result),flush=True)


def load_control(folder):
    folder=Path(folder);ck=torch.load(folder/'epoch8.pt',map_location='cpu',weights_only=False);st=ck['state_dict']
    net=Control(ck['mode'],st['mean'],st['std']).cuda().eval();net.load_state_dict(st)
    return net,json.loads((folder/'SELECTION.json').read_text())['alpha'],ck


@torch.no_grad()
def evaluate(args):
    setup();torch.backends.cudnn.benchmark=False;start=time.monotonic();model=load_model(args);head=model.model[-1]
    config=json.loads(args.models.read_text());nets={k:load_control(v) for k,v in config.items()}
    grids={k:torch.tensor(v[2]['template']['field'],device='cuda').reshape(1,16) for k,v in nets.items()}
    captured={};hooks=[]
    for level,module in enumerate(head.one2one_cv4):
        def hook(m,inputs,level=level):captured[level]=inputs[0].detach()
        hooks.append(module[-1].register_forward_pre_hook(hook))
    coco=COCO(str(args.data/'annotations/instances_val2017.json'));ids=sorted(coco.imgs);cats=sorted(coco.cats)
    if args.smoke:ids=ids[:3]
    streams={k:gzip.open(args.out/f'predictions_{k}.jsonl.gz','wt',encoding='utf-8',compresslevel=1) for k in nets}
    counts={k:0 for k in nets};identity_errors=0;zero_errors=0
    transform=LetterBox((640,640),auto=True,stride=32)
    try:
        for ni,iid in enumerate(ids):
            orig,inp=image_input(args.data/'images/val2017'/coco.imgs[iid]['file_name'],transform);shape=tuple(inp.shape[2:])
            _,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];c=p['mask_coefficient'][0].T
            scale=proto.square().mean((1,2)).sqrt().clamp_min(.1)
            hidden=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            levels=torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],),l,device='cuda') for l in range(3)])
            boxes=head._get_decode_boxes(p)[0].T;scores,classes,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            keep=scores[0,:,0]>.001;scores=scores[0,keep,0];classes=classes[0,keep,0].long();rids=rids[0,keep,0]
            low=(c@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            for first in range(0,len(rids),16):
                ix=rids[first:first+16];bb=boxes[ix];features=input_features(hidden[ix],levels[ix])
                base=F.interpolate(low[ix][None],shape,mode='bilinear',align_corners=False)[0]
                if ni==0:
                    identity_errors+=int((ops.crop_mask(base,bb).gt(0)!=ops.process_mask(proto,c[ix],bb,shape,upsample=True).bool()).sum())
                for name,(net,alpha,ck) in nets.items():
                    v=net(features)*alpha
                    if name=='global_scalar':z=base+4*v[:,0,None,None]
                    elif name=='global_spatial':
                        basis,_,_,_=full_template(grids[name],bb,shape);z=base+4*(v[:,0,None,None]+v[:,1,None,None]*basis)
                    else:
                        delta=full_coeff_delta(proto,scale,bb,shape,v,name);z=base+delta
                        if ni==0:zero_errors+=int(torch.count_nonzero(full_coeff_delta(proto,scale,bb,shape,v*0,name)))
                    masks=ops.scale_masks(ops.crop_mask(z,bb).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
                    for j,mask in enumerate(masks):
                        if not mask.any():continue
                        k=first+j;rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                        streams[name].write(json.dumps({'image_id':iid,'category_id':cats[int(classes[k])],'score':float(scores[k]),'raw_id':int(rids[k]),'segmentation':rle})+'\n');counts[name]+=1
            if ni%100==0 or ni+1==len(ids):
                for stream in streams.values():stream.flush()
                pr={'images':ni+1,'total':len(ids),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',pr);print(json.dumps(pr),flush=True)
    finally:
        for stream in streams.values():stream.close()
        for hook in hooks:hook.remove()
    assert identity_errors==zero_errors==0,(identity_errors,zero_errors)
    if args.smoke:
        original.save(args.out/'COMPLETE.json',{'images':len(ids),'identity_errors':identity_errors,'zero_coefficient_errors':zero_errors});return
    assert len(ids)==5000;base=set(json.loads(args.reference.read_text())['baseline']);results={};matches={'baseline':sorted(base)}
    for name in nets:
        with gzip.open(args.out/f'predictions_{name}.jsonl.gz','rt') as f:pred=[json.loads(line) for line in f]
        ev=COCOeval(coco,coco.loadRes(pred),'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
        for item in ev.evalImgs:
            if item is not None and item['aRng']==ev.params.areaRng[0]:
                good.update(int(g) for j,g in enumerate(item['gtIds']) if not item['gtIgnore'][j] and item['gtMatches'][ti,j]>0)
        metrics=dict(zip(['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'],map(float,ev.stats)))
        results[name]={'metrics':metrics,'alpha':nets[name][1],'parameters':nets[name][2]['parameters'],'matched75':len(good),
            'repaired75':len(good-base),'damaged75':len(base-good),'predictions':counts[name]}
        matches[name]=sorted(good);original.save(args.out/'RESULTS.json',results);original.save(args.out/'MATCHED_GT75.json',matches)
        del ev,pred
    original.save(args.out/'COMPLETE.json',{'images':5000,'identity_errors':identity_errors,'zero_coefficient_errors':zero_errors,'elapsed_s':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['checks','train','evaluate']);p.add_argument('--out',type=Path,required=True)
    for name in ['bank','features','geometry','split','data','weights','models','reference']:p.add_argument('--'+name,type=Path)
    p.add_argument('--mode',choices=MODES);p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);globals()[args.action](args)
