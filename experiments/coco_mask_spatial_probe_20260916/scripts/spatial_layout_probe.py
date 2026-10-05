"""Intervene on learned local correction layout at fixed original identities."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools import mask as mu
import learn_refinement as original
from component_seed_probe import image_correct,setup


def measures(mask,gt,base,near,same,other,background):
    tp=int((mask&gt).sum());fp=int((mask&~gt).sum());fn=int((~mask&gt).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(tp+fp+fn,1),'coverage':tp/max(tp+fn,1),'purity':tp/max(tp+fp,1),
            'fn_fixed':int((mask&~base&gt).sum()),'tp_lost':int((~mask&base&gt).sum()),
            'fp_fixed':int((~mask&base&~gt).sum()),'tn_lost':int((mask&~base&~gt).sum()),
            'fp_near':int((mask&~gt&near).sum()),'fp_far':int((mask&~gt&~near).sum()),
            'fn_near':int((~mask&gt&near).sum()),'fn_interior':int((~mask&gt&~near).sum()),
            'fp_same_neighbor':int((mask&same).sum()),'fp_other_neighbor':int((mask&other).sum()),
            'fp_background':int((mask&background).sum())}


def histogram(mask,box):
    yy,xx=np.nonzero(mask)
    if len(xx)==0:return np.zeros(16)
    gx=np.clip(((xx+.5-box[0])/max(box[2]-box[0],1)*4).astype(int),0,3)
    gy=np.clip(((yy+.5-box[1])/max(box[3]-box[1],1)*4).astype(int),0,3)
    return np.bincount(gy*4+gx,minlength=16).astype(float)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    for k in ['cohort','data','weights','local-weights','scalar-weights','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--limit',type=int,default=100000)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);setup();torch.backends.cudnn.benchmark=False
    start=time.monotonic();coco=COCO(str(a.data/'annotations/instances_val2017.json'))
    with gzip.open(a.cohort,'rt') as f:rows=[json.loads(s) for s in f]
    by_image=defaultdict(list)
    for r in rows:
        if 'raw_id' in r:by_image[r['image_id']].append(r)
    ids=sorted(by_image)[:a.limit]
    local=original.Refiner('local4').cuda().eval();local.load_state_dict(torch.load(a.local_weights,weights_only=False)['state_dict'])
    scalar=original.Refiner('scalar').cuda().eval();scalar.load_state_dict(torch.load(a.scalar_weights,weights_only=False)['state_dict'])
    model=YOLO(str(a.weights)).model.cuda().float().eval();head=model.model[-1];assert head.end2end
    transform=LetterBox((640,640),auto=True,stride=32)
    records=(a.out/'instances.jsonl').open('w');count=0;checks=defaultdict(int)
    original.save(a.out/'SETUP.json',{'images':len(ids),'gt':sum(len(by_image[i]) for i in ids),'alpha':.5,
        'coordinate':'pixel_center','candidate':'best original same-class mask, fixed raw_id','no_training':True})
    for number,image_id in enumerate(ids):
        im=cv2.imread(str(a.data/'images/val2017'/coco.imgs[image_id]['file_name']));orig=im.shape[:2]
        resized=transform.apply_image({'img':im},transform.get_params({'img':im}))['img']
        inp=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
        shape=tuple(inp.shape[-2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];allc=p['mask_coefficient'][0].T
        allboxes=head._get_decode_boxes(p)[0].T
        anns=coco.imgToAnns[image_id];gt_masks={ann['id']:coco.annToMask(ann).astype(bool) for ann in anns}
        unions={};crowd=np.zeros(orig,dtype=bool)
        for ann in anns:
            if ann.get('iscrowd',0) or ann.get('ignore',0):crowd|=gt_masks[ann['id']]
            else:unions.setdefault(ann['category_id'],np.zeros(orig,dtype=bool))[:] |=gt_masks[ann['id']]
        union=np.logical_or.reduce(list(unions.values()))
        for first in range(0,len(by_image[image_id]),16):
            batch_rows=by_image[image_id][first:first+16];raw_ids=torch.tensor([r['raw_id'] for r in batch_rows],device='cuda')
            boxes=allboxes[raw_ids];coeff=allc[raw_ids]
            x,_,_=original.features(proto,coeff,boxes,shape);lc=local(x)*.5;sc=scalar(x)*.5
            low=(coeff@proto.flatten(1)).reshape(-1,*proto.shape[-2:]);base=F.interpolate(low[None],shape,mode='bilinear',align_corners=False)[0]
            zeros=torch.zeros_like(base)
            field=image_correct(zeros,proto,None,boxes,shape,lc,'local4','pixel_center')
            rotated=image_correct(zeros,proto,None,boxes,shape,lc.reshape(-1,4,4).flip((1,2)).reshape(-1,16),'local4','pixel_center')
            support=ops.crop_mask(torch.ones_like(base),boxes).bool();n=support.sum((1,2)).clamp_min(1)
            mean=(field*support).sum((1,2))/n
            std=(((field-mean[:,None,None]).square()*support).sum((1,2))/n).sqrt()
            rmean=(rotated*support).sum((1,2))/n
            rstd=(((rotated-rmean[:,None,None]).square()*support).sum((1,2))/n).sqrt()
            rotated=(rotated-rmean[:,None,None])* (std/rstd.clamp_min(1e-12))[:,None,None]+mean[:,None,None]
            rms=(std.square()+mean.square()).sqrt();uniform_rms=torch.where(mean<0,-rms,rms)
            variants={'baseline':base,'scalar':base+4*sc[:,-1,None,None], 'local':base+field,
                      'field_mean':base+mean[:,None,None],'field_rms_scalar':base+uniform_rms[:,None,None],
                      'field_rotated':base+rotated}
            decoded={name:ops.scale_masks(ops.crop_mask(z.clone(),boxes).gt(0).byte()[None],orig)[0].byte().cpu().numpy().astype(bool)
                     for name,z in variants.items()}
            field_orig=ops.scale_masks(field[None],orig)[0].cpu().numpy()
            original_boxes=ops.scale_boxes(shape,boxes.clone(),orig).cpu().numpy()
            for j,r in enumerate(batch_rows):
                aid=r['annotation_id'];ann=coco.anns[aid];gt=gt_masks[aid];m0=decoded['baseline'][j]
                if not support[j].any():raise RuntimeError('Empty support on selected reference')
                for name in ['baseline','scalar','local']:
                    ref=mu.decode(r['reference'][name]).astype(bool) if name in r['reference'] else np.zeros(orig,dtype=bool)
                    mismatch=int((ref!=decoded[name][j]).sum());checks[name+'_pixels_mismatch']+=mismatch
                    # A repeated execution must reproduce the frozen cohort response.
                    assert mismatch==0,(image_id,aid,name,mismatch)
                    checks[name+'_instances_verified']+=1
                gtbox=np.array([ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]])
                box=original_boxes[j];inter=np.maximum(np.minimum(box[2:],gtbox[2:])-np.maximum(box[:2],gtbox[:2]),0).prod()
                biou=inter/max(np.maximum(box[2:]-box[:2],0).prod()+ann['bbox'][2]*ann['bbox'][3]-inter,1e-9)
                width=max(1,int(np.ceil(.02*np.sqrt(gt.sum()))));dist=np.where(gt,distance_transform_edt(gt),distance_transform_edt(~gt));near=dist<=width
                same=unions[ann['category_id']]&~gt;other=union&~gt&~unions[ann['category_id']];background=~union&~crowd
                fp=m0&~gt;fn=~m0&gt;tp=m0&gt;tn=~m0&~gt
                hfp=histogram(fp,box);hfn=histogram(fn,box)
                pfp=hfp/max(hfp.sum(),1);pfn=hfn/max(hfn.sum(),1)
                both=bool(fp.any() and fn.any())
                layout={'fp_cells':int((hfp>0).sum()),'fn_cells':int((hfn>0).sum()),
                    'fp_fraction_largest_cell':float(pfp.max()),'fn_fraction_largest_cell':float(pfn.max()),
                    'fn_fp_spatial_tv':float(.5*np.abs(pfp-pfn).sum()) if both else None,
                    'both_error_types':both,'fp_histogram':hfp.tolist(),'fn_histogram':hfn.tolist()}
                fo=field_orig[j]
                layout.update({'field_mean':float(mean[j]),'field_std':float(std[j]),'field_rms':float(rms[j]),
                    'field_positive_fraction':float((field[j][support[j]]>0).float().mean()),
                    'grid':lc[j].reshape(4,4).cpu().tolist(),
                    'field_region_means':{k:float(fo[v].mean()) if v.any() else None for k,v in [('fp',fp),('fn',fn),('tp',tp)]}})
                metrics={name:measures(mm[j],gt,m0,near,same,other,background) for name,mm in decoded.items()}
                result={k:v for k,v in r.items() if k!='reference'}
                result.update(box_iou=float(biou),fill=float(gt.sum()/max(ann['bbox'][2]*ann['bbox'][3],1)),
                              layout=layout,metrics=metrics)
                records.write(json.dumps(result,allow_nan=False)+'\n');count+=1
        records.flush();progress={'images':number+1,'total':len(ids),'instances':count,'elapsed_s':round(time.monotonic()-start,2)}
        original.save(a.out/'progress.json',progress)
        if number==0 or (number+1)%50==0:print(json.dumps(progress),flush=True)
    records.close();original.save(a.out/'REPRODUCTION_CHECK.json',dict(checks));original.save(a.out/'COMPLETE.json',progress);print(json.dumps(progress),flush=True)


if __name__=='__main__':main()
