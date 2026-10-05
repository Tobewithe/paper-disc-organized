"""Serial, paired inference timing on64 fixed val images. Excludes loading and RLE encoding."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

import learn_refinement as original
from component_seed_probe import setup
from shared_shape_head import SharedShapeHead,full_template,correct
from native_coefficient_control import image_input,load_model
from native_spatial_readout import load_head,input_features


@torch.no_grad()
def main(args):
    setup();torch.backends.cudnn.benchmark=False;args.out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.models.read_text());heads={name:load_head(Path(path)) for name,path in config['heads'].items()}
    ck=torch.load(args.roi_checkpoint,map_location='cpu',weights_only=False);roi=SharedShapeHead().cuda().eval();roi.load_state_dict(ck['state_dict'])
    grid=torch.tensor(ck['template']['field'],device='cuda').reshape(1,16)
    assert all(v[2]['template']['field']==ck['template']['field'] for v in heads.values())
    initial=torch.load(config['native_checkpoint'],map_location='cpu',weights_only=False)
    coco=json.loads((args.data/'annotations/instances_val2017.json').read_text())
    images=sorted(coco['images'],key=lambda r:r['id']);rng=np.random.default_rng(20260922)
    images=[images[i] for i in sorted(rng.choice(len(images),64,replace=False))]
    transform=LetterBox((640,640),auto=True,stride=32)
    inputs=[(row['id'],*image_input(args.data/'images/val2017'/row['file_name'],transform)) for row in images]
    model=load_model(args);head=model.model[-1];captured={};results={}
    for mode in ['baseline','native_finetune','roi_shared2','native_scalar','native_spatial','response_spatial']:
        for layer,st,init in zip(head.one2one_cv4,initial['states'],initial['initial_states']):
            values=st if mode=='native_finetune' else init
            layer[-1].weight.copy_(values['weight'][:,:,None,None]);layer[-1].bias.copy_(values['bias'])
        hooks=[]
        if mode in heads:
            for level,module in enumerate(head.one2one_cv4):
                def hook(m,args,level=level):captured[level]=args[0].detach()
                hooks.append(module[-1].register_forward_pre_hook(hook))
        def infer(orig,inp):
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];c=p['mask_coefficient'][0].T
            boxes=head._get_decode_boxes(p)[0].T;scores,_,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            ix=rids[0,scores[0,:,0]>.001,0];low=(c@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            if mode in heads:
                hidden=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
                levels=torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],),l,device='cuda') for l in range(3)])
            output=[]
            for first in range(0,len(ix),16):
                ids=ix[first:first+16];bb=boxes[ids];z=F.interpolate(low[ids][None],shape,mode='bilinear',align_corners=False)[0]
                if mode=='roi_shared2':
                    x,_,_=original.features(proto,c[ids],bb,shape);basis,_,_,_=full_template(grid,bb,shape);z=correct(z,basis,roi(x)*.5)
                elif mode in heads:
                    roi_z=F.grid_sample(low[ids,None],original.roi_grid(bb,shape),align_corners=False)[:,0] if mode=='response_spatial' else None
                    net,alpha,_=heads[mode];v=net(input_features(hidden[ids],levels[ids],roi_z))*alpha
                    if mode=='native_scalar':z=z+4*v[:,0,None,None]
                    else:
                        basis,_,_,_=full_template(grid,bb,shape);z=correct(z,basis,v)
                output.append(ops.scale_masks(ops.crop_mask(z,bb).gt(0).byte()[None],orig)[0].byte())
            return output,len(ix)
        for i in range(12):infer(*inputs[i%len(inputs)][1:])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[];counts=[]
        for repeat in range(3):
            order=np.random.default_rng(20260923+repeat).permutation(len(inputs))
            for j in order:
                iid,orig,inp=inputs[j];torch.cuda.synchronize();start=time.perf_counter();out,n=infer(orig,inp);torch.cuda.synchronize()
                timings.append({'image_id':iid,'repeat':repeat,'ms':(time.perf_counter()-start)*1000,'predictions':n});counts.append(n);del out
        vals=np.array([r['ms'] for r in timings]);results[mode]={'mean_ms':float(vals.mean()),'median_ms':float(np.median(vals)),
            'p25_ms':float(np.quantile(vals,.25)),'p75_ms':float(np.quantile(vals,.75)),'peak_allocated_mb':torch.cuda.max_memory_allocated()/1e6,
            'mean_retained_predictions':float(np.mean(counts)),'timings':timings}
        for hook in hooks:hook.remove()
        print(json.dumps({'mode':mode,**{k:v for k,v in results[mode].items() if k!='timings'}}),flush=True)
        original.save(args.out/'RESULTS.json',results)
    original.save(args.out/'PROTOCOL.json',{'images':[r['id'] for r in images],'batch':1,'repeats':3,'warmup_per_mode':12,
        'confidence':.001,'topk':300,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'includes':'model forward,raw decode,correction,crop,binary resize to original dimensions',
        'excludes':'disk loading,CPU RLE encoding,COCOeval','scope':'COCO evaluation operating point, not a confidence0.25 deployment benchmark; same all-raw prototype multiplication in all modes.'})
    original.save(args.out/'COMPLETE.json',{'status':'completed','modes':list(results),'images':64,'repeats':3})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['out','data','weights','models','roi-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
