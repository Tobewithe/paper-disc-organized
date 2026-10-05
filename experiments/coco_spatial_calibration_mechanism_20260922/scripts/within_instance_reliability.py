"""Within-instance logit-bin reliability: fixed phase2 representatives, no new selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO

import learn_refinement as original
from component_seed_probe import setup
from native_coefficient_control import image_input, load_model


def summarize(rows):
    result={}
    for split in ['explore','confirm']:
        selected=[r for r in rows if r['split']==split]
        images=sorted({r['image_id'] for r in selected});lookup={v:i for i,v in enumerate(images)}
        # Weight by the harmonic effective sample count, within the SAME object and SAME logit bin.
        # This removes differences in category, size and global morphology between objects.
        rng=np.random.default_rng(20260925)
        draws=rng.multinomial(len(images),np.ones(len(images))/len(images),size=2000)
        bins=[]
        for bi in range(10):
            sums=np.zeros((len(images),3));pairs=0
            for r in selected:
                c,o=np.asarray(r['bins'],dtype=float)[:,bi]
                if min(c[0],o[0])<3:continue
                w=c[0]*o[0]/(c[0]+o[0]);pairs+=1
                sums[lookup[r['image_id']]] += [w*(c[1]/c[0]-o[1]/o[0]),w*(c[2]/c[0]-o[2]/o[0]),w]
            total=sums.sum(0);boot=draws@sums;valid=boot[:,2]>0
            if total[2]==0:bins.append({'pairs':0});continue
            bins.append({'pairs':pairs,'weight':float(total[2]),
                'foreground_difference_pp':float(100*total[0]/total[2]),
                'model_probability_difference_pp':float(100*total[1]/total[2]),
                'residual_difference_pp':float(100*(total[0]-total[1])/total[2]),
                'foreground_ci95_pp':np.quantile(100*boot[valid,0]/boot[valid,2],[.025,.975]).tolist(),
                'residual_ci95_pp':np.quantile(100*(boot[valid,0]-boot[valid,1])/boot[valid,2],[.025,.975]).tolist()})
        result[split]={'instances':len(selected),'images':len(images),'bins':bins}
    return result


@torch.no_grad()
def main(args):
    setup();start=time.monotonic();args.out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(x) for x in args.instances.read_text().splitlines()]
    groups=defaultdict(list)
    for r in records:groups[r['image_id']].append(r)
    coco=COCO(str(args.data/'annotations/instances_train2017.json'));model=load_model(args);head=model.model[-1]
    transform=LetterBox((640,640),auto=True,stride=32)
    edges=torch.tensor([-4,-2,-1,-.5,0,.5,1,2,4],device='cuda')
    t=(torch.arange(32,device='cuda')+.5)/32;yy,xx=torch.meshgrid(t,t,indexing='ij')
    center=(xx>=.25)&(xx<.75)&(yy>=.25)&(yy<.75);rows=[];max_box_error=0.
    with (args.out/'instances.jsonl').open('w') as stream:
        for ni,iid in enumerate(sorted(groups)):
            shape0,inp=image_input(args.data/'images/train2017'/coco.imgs[iid]['file_name'],transform)
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];boxes_all=head._get_decode_boxes(p)[0].T
            gain=min(shape[0]/shape0[0],shape[1]/shape0[1]);nh,nw=round(shape0[0]*gain),round(shape0[1]*gain)
            top,left=round((shape[0]-nh)/2-.1),round((shape[1]-nw)/2-.1)
            for first in range(0,len(groups[iid]),8):
                rr=groups[iid][first:first+8];ix=torch.tensor([r['raw_id'] for r in rr],device='cuda')
                boxes=boxes_all[ix];_,z,grid=original.features(p['proto'][0],p['mask_coefficient'][0].T[ix],boxes,shape)
                expected=torch.tensor([r['input_box_wh'] for r in rr],device='cuda')
                max_box_error=max(max_box_error,float(((boxes[:,2:]-boxes[:,:2])-expected).abs().max()))
                gt=torch.from_numpy(np.stack([coco.annToMask(coco.anns[r['annotation_id']]) for r in rr])).cuda().float()
                rx=(grid[...,0]+1)*shape[1]/2;ry=(grid[...,1]+1)*shape[0]/2
                native_grid=torch.stack((2*(rx-left)/gain/shape0[1]-1,2*(ry-top)/gain/shape0[0]-1),-1)
                target=F.grid_sample(gt[:,None],native_grid,mode='nearest',align_corners=False)[:,0]
                gb=torch.tensor([coco.anns[r['annotation_id']]['bbox'] for r in rr],device='cuda');gb[:,2:]+=gb[:,:2]
                gb[:,[0,2]]=gb[:,[0,2]]*gain+left;gb[:,[1,3]]=gb[:,[1,3]]*gain+top
                support=ops.crop_mask(torch.ones(len(rr),*shape,device='cuda'),gb)
                support=F.grid_sample(support[:,None],grid,mode='nearest',align_corners=False)[:,0]>.5
                bi=torch.bucketize(z.contiguous(),edges);out=torch.zeros(len(rr),2,10,3,device='cuda')
                for region,region_mask in enumerate([center,~center]):
                    for b in range(10):
                        mask=(bi==b)&region_mask[None]&support
                        out[:,region,b,0]=mask.sum((1,2));out[:,region,b,1]=(target*mask).sum((1,2));out[:,region,b,2]=(z.sigmoid()*mask).sum((1,2))
                for r,bins in zip(rr,out.cpu().tolist()):
                    item={k:r[k] for k in ['image_id','annotation_id','raw_id','split','category_id','area']};item['bins']=bins
                    rows.append(item);stream.write(json.dumps(item)+'\n')
            if ni%100==0 or ni+1==len(groups):
                progress={'images':ni+1,'total':len(groups),'elapsed_s':time.monotonic()-start};original.save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert max_box_error<1e-4,max_box_error
    result=summarize(rows);original.save(args.out/'RESULTS.json',result)
    original.save(args.out/'COMPLETE.json',{'images':len(groups),'instances':len(rows),'max_box_error':max_box_error,
        'scope':'Exploratory within-instance follow-up on the existing phase2 panel; not another independently held-out replication.',
        'condition':'Within original GT box, nearest original hard mask labels, same object + logit bin, >=3 ROI samples per region.',
        'edges':['-inf',-4,-2,-1,-.5,0,.5,1,2,4,'inf'],'elapsed_s':time.monotonic()-start})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','weights','instances','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
