"""Train-only strength selection and conservative objective, paired 800/200 split.

Reuse the original bank; each GT identity remains whole. No val2017 selection.
Full COCO predictions are retained as compressed RLE files after scoring.
"""
import argparse
from collections import defaultdict
import gc
import gzip
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
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from learn_refinement import Refiner, features, roi_correct, save


def per_loss(logits,target):
    p=logits.sigmoid()
    return F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean((1,2))+.5*(1-(2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1))


def soft_iou(z,y):
    # Sharper threshold surrogate than the ordinary sigmoid; original logits use threshold0.
    p=(z/.5).sigmoid();g=(y>=.5).float();inter=(p*g).sum((1,2))
    return (inter+1)/(p.sum((1,2))+g.sum((1,2))-inter+1)


def hard_iou(z,y):
    m=z>0;g=y>=.5
    return (m&g).sum((1,2))/(m|g).sum((1,2)).clamp(min=1)


def main():
    p=argparse.ArgumentParser()
    for k in ('bank','data','weights','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--eval-images',type=int,default=5000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic()
    def progress(stage,**extra):
        v=dict(stage=stage,elapsed_s=round(time.monotonic()-start,1),**extra);save(a.out/'progress.json',v);print(json.dumps(v),flush=True)
    bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True)
    owners=defaultdict(set)
    for r in bank['records']:owners[(r['image_id'],r['raw_id'])].add(r['annotation_id'])
    seen=set();keep=[]
    for i,r in enumerate(bank['records']):
        k=(r['image_id'],r['raw_id'])
        if len(owners[k])==1 and k not in seen:keep.append(i);seen.add(k)
    split=json.loads((a.bank.parent/'SPLIT.json').read_text())
    ids=list(split['train_ids']);random.Random(20260916).shuffle(ids);fit_ids=set(ids[:800]);dev_ids=set(ids[800:])
    assert len(fit_ids)==800 and len(dev_ids)==200 and not fit_ids&dev_ids
    fit=[i for i in keep if bank['records'][i]['image_id'] in fit_ids];dev=[i for i in keep if bank['records'][i]['image_id'] in dev_ids]
    save(a.out/'SPLIT.json',dict(fit_image_ids=sorted(fit_ids),selection_image_ids=sorted(dev_ids),
        fit_rows=len(fit),selection_rows=len(dev),val_ids=split['val_ids'][:a.eval_images],
        source=str(a.bank),selection='hash-independent seeded shuffle before training; no val label used for fitting or selection'))
    fitset=TensorDataset(*(bank[k][fit] for k in ('x','base','target')))
    devset=TensorDataset(*(bank[k][dev] for k in ('x','base','target')))
    fits={};selection={};alphas=(0.,.25,.5,1.)
    for mode,kind in [('scalar','plain'),('coeff_local4','plain'),('coeff_local4','safe')]:
        torch.manual_seed(0);net=Refiner(mode).cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(fitset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(0),num_workers=0)
        history=[]
        for epoch in range(a.epochs):
            losses=[]
            for x,b,y in loader:
                x=x.cuda().float();b=b.cuda().float();y=y.cuda().float();z=roi_correct(b,x,net(x),mode)
                loss=per_loss(z,y).mean()
                if kind=='safe':
                    current=soft_iou(z,y);reference=soft_iou(b,y).detach()
                    # Penalize per-instance soft quality regressions and unnecessary logit movement.
                    loss=loss+2*F.relu(reference-current).mean()+.01*(z-b).square().mean()
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step();losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            torch.save(dict(state_dict=net.state_dict(),mode=mode,kind=kind,epoch=epoch+1),a.out/f'{mode}_{kind}_epoch{epoch+1}.pt')
            progress('train',mode=mode,kind=kind,epoch=epoch+1,loss=history[-1])
        net.eval();stats={str(alpha):dict(total=0.,n=0,repairs=0,damages=0,loss=0.) for alpha in alphas}
        with torch.no_grad():
            for x,b,y in DataLoader(devset,batch_size=128):
                x=x.cuda().float();b=b.cuda().float();y=y.cuda().float();c=net(x);base_iou=hard_iou(b,y)
                for alpha in alphas:
                    z=roi_correct(b,x,c*alpha,mode);ious=hard_iou(z,y);v=stats[str(alpha)]
                    v['total']+=float(ious.sum());v['n']+=len(x);v['repairs']+=int(((ious>=.75)&(base_iou<.75)).sum())
                    v['damages']+=int(((ious<.75)&(base_iou>=.75)).sum());v['loss']+=float(per_loss(z,y).sum())
        for v in stats.values():v['mean_iou']=v['total']/v['n'];v['mean_loss']=v['loss']/v['n']
        # Select average hard ROI IoU. Tie preference: less intervention.
        chosen=max(alphas,key=lambda alpha:(stats[str(alpha)]['mean_iou'],-alpha))
        key=f'{mode}_{kind}';fits[key]=net;selection[key]=dict(alpha=chosen,stats=stats,history=history)
        save(a.out/'SELECTION.json',selection);progress('selected',key=key,alpha=chosen,details=stats)
    del bank,fitset,devset,loader;gc.collect();torch.cuda.empty_cache()
    specs=[('baseline',None,0.,None),('scalar_full','scalar_plain',1.,'scalar'),
           ('scalar_selected','scalar_plain',selection['scalar_plain']['alpha'],'scalar'),
           ('combo_full','coeff_local4_plain',1.,'coeff_local4'),
           ('combo_selected','coeff_local4_plain',selection['coeff_local4_plain']['alpha'],'coeff_local4'),
           ('safe_full','coeff_local4_safe',1.,'coeff_local4'),
           ('safe_selected','coeff_local4_safe',selection['coeff_local4_safe']['alpha'],'coeff_local4')]
    # Identical head+strength conditions share prediction artifacts and scores.
    aliases={};active=[];seen={}
    for name,key,alpha,mode in specs:
        ident=(key,alpha) if alpha else (None,0.)
        if ident in seen:aliases[name]=seen[ident]
        else:seen[ident]=name;active.append((name,key,alpha,mode))
    save(a.out/'VARIANTS.json',dict(active=active,aliases=aliases))
    coco=COCO(str(a.data/'annotations/instances_val2017.json'));val_ids=split['val_ids'][:a.eval_images]
    model=YOLO(str(a.weights)).model.cuda().float().eval();head=model.model[-1];assert head.end2end
    transform=LetterBox((640,640),auto=True,stride=32);cat_ids=sorted(coco.cats)
    streams={name:gzip.open(a.out/f'predictions_{name}.jsonl.gz','wt',encoding='utf-8',compresslevel=1) for name,_,_,_ in active}
    with torch.no_grad():
        for ni,image_id in enumerate(val_ids):
            im=cv2.imread(str(a.data/'images/val2017'/coco.imgs[image_id]['file_name']));assert im is not None
            orig=im.shape[:2];params=transform.get_params({'img':im});res=transform.apply_image({'img':im},params)['img']
            inp=torch.from_numpy(np.ascontiguousarray(res[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];allc=p['mask_coefficient'][0].T
            allboxes=head._get_decode_boxes(p)[0].T
            ts,tc,ti=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300);ok=ts[0,:,0]>.001
            scores=ts[0,ok,0];classes=tc[0,ok,0].long();ids=ti[0,ok,0];boxes=allboxes[ids];coeff=allc[ids]
            low=(allc@proto.flatten(1)).reshape(-1,*proto.shape[-2:]);scale=proto.square().mean((1,2)).sqrt().clamp(min=.1)
            ob=ops.scale_boxes(shape,boxes.clone(),orig)
            yy,xx=torch.meshgrid(torch.arange(shape[0],device='cuda'),torch.arange(shape[1],device='cuda'),indexing='ij')
            for first in range(0,len(boxes),16):
                sl=slice(first,first+16);bb=boxes[sl];x,_,_=features(proto,coeff[sl],bb,shape)
                base=F.interpolate(low[ids[sl]][None],shape,mode='bilinear',align_corners=False)[0]
                corrections={key:net(x) for key,net in fits.items()}
                for name,key,alpha,mode in active:
                    z=base
                    if key:
                        c=corrections[key]*alpha
                        if mode=='coeff_local4':
                            d=(2*c[:,:32]/scale[None])@proto.flatten(1)
                            z=z+F.interpolate(d.reshape(-1,*proto.shape[-2:])[None],shape,mode='bilinear',align_corners=False)[0]
                            gx=2*(xx[None]-bb[:,0,None,None])/(bb[:,2]-bb[:,0]).clamp(min=1)[:,None,None]-1
                            gy=2*(yy[None]-bb[:,1,None,None])/(bb[:,3]-bb[:,1]).clamp(min=1)[:,None,None]-1
                            z=z+F.grid_sample(4*c[:,-16:].reshape(-1,1,4,4),torch.stack((gx,gy),-1),padding_mode='border',align_corners=True)[:,0]
                        else:z=z+4*c[:,-1,None,None]
                    masks=ops.scale_masks(ops.crop_mask(z.clone(),bb).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
                    for j,mask in enumerate(masks):
                        if not mask.any():continue
                        k=first+j;rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                        streams[name].write(json.dumps(dict(image_id=image_id,category_id=cat_ids[int(classes[k])],score=float(scores[k]),segmentation=rle,raw_id=int(ids[k])))+'\n')
            if ni%25==0 or ni+1==len(val_ids):
                for stream in streams.values():stream.flush()
                progress('predict',images=ni+1,total=len(val_ids))
    for stream in streams.values():stream.close()
    names=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL');results={};matched={};base_good=None
    ordinary={a['id']:a for image_id in val_ids for a in coco.imgToAnns[image_id] if not a.get('iscrowd',0) and not a.get('ignore',0)}
    def size(a):return 'small' if a['area']<32**2 else 'medium' if a['area']<96**2 else 'large'
    for name,_,_,_ in active:
        with gzip.open(a.out/f'predictions_{name}.jsonl.gz','rt') as f:pred=[json.loads(line) for line in f]
        dt=coco.loadRes(pred);ev=COCOeval(coco,dt,'segm');ev.params.imgIds=val_ids;ev.evaluate();ev.accumulate();ev.summarize()
        ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
        for entry in ev.evalImgs:
            if entry is not None and entry['aRng']==ev.params.areaRng[0]:
                good.update(int(g) for j,g in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti,j]>0)
        good &= ordinary.keys()
        if name=='baseline':base_good=good
        results[name]=dict(metrics=dict(zip(names,map(float,ev.stats))),predictions=len(pred),matched75=len(good),repaired75=len(good-base_good),damaged75=len(base_good-good),
            size_recall75={g:dict(gt=sum(size(a)==g for a in ordinary.values()),matched=sum(size(ordinary[i])==g for i in good)) for g in ('small','medium','large')})
        matched[name]=sorted(good);save(a.out/'RESULTS.json',results);save(a.out/'MATCHED_GT75.json',matched);progress('score',mode=name,mask_ap=100*ev.stats[0])
        del pred,dt,ev;gc.collect()
    for name,source in aliases.items():results[name]={**results[source],'identical_to':source};matched[name]=matched[source]
    save(a.out/'RESULTS.json',results);save(a.out/'MATCHED_GT75.json',matched);save(a.out/'COMPLETE.json',dict(images=len(val_ids),fit_images=800,selection_images=200,protocol='Frozen epoch8, train-only strength selection; full COCO evaluation.'))


if __name__=='__main__':main()
