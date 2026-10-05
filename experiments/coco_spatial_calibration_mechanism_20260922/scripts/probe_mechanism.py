"""Fixed-candidate spatial/decode interventions; no parameter updates or val tuning."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.models.yolo.segment.val import SegmentationValidator
from pycocotools.coco import COCO

import learn_refinement as original
from component_seed_probe import setup
from shared_shape_head import SharedShapeHead, grid_tensor, full_template


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows, image_ids):
    outputs = {}
    contrasts = {
        'decoder_on_baseline': ('base_half', 'base_byte'),
        'shared_gain_byte': ('shared_byte', 'base_byte'),
        'shared_gain_half': ('shared_half', 'base_half'),
        'shared_minus_scalar_byte': ('shared_byte', 'scalar_byte'),
        'shared_minus_scalar_half': ('shared_half', 'scalar_half'),
        'space_given_bias_byte': ('shared_byte', 'bias_byte'),
        'space_given_bias_half': ('shared_half', 'bias_half'),
        'space_orientation_half': ('shared_half', 'reverse_half'),
        'gt_inside_gain_half': ('inside_half', 'base_half'),
        'gt_outside_gain_half': ('outside_half', 'base_half'),
    }
    if rows and 'projected_half' in rows[0]['metrics']:
        contrasts.update({
            'projection_retains_space_half':('projected_half','bias_half'),
            'outside_span_space_half':('orthogonal_half','bias_half'),
            'full_minus_projected_half':('shared_half','projected_half'),
            'space_inside_given_bias_half':('space_inside_half','bias_half'),
            'space_outside_given_bias_half':('space_outside_half','bias_half'),
            'projection_retains_space_byte':('projected_byte','bias_byte'),
            'full_minus_projected_byte':('shared_byte','projected_byte'),
        })
    for split in ['explore', 'confirm']:
        selected = [r for r in rows if r['split'] == split]
        iids = image_ids[split]
        lookup = {v: i for i, v in enumerate(iids)}
        draws = np.random.default_rng(20260922).integers(0, len(iids), (1000, len(iids)))
        groups = {}
        for group in ['all', 'small', 'medium_large', 'box75']:
            chosen = [r for r in selected if group == 'all' or
                      (group == 'small' and r['area'] < 1024) or
                      (group == 'medium_large' and r['area'] >= 1024) or
                      (group == 'box75' and r['box_iou'] >= .75)]
            if not chosen:
                continue
            counts = np.zeros(len(iids))
            for r in chosen:
                counts[lookup[r['image_id']]] += 1
            results = {}
            for name, (a, b) in contrasts.items():
                metrics = {}
                for metric in ['iou', 'coverage', 'purity', 'mask75']:
                    values = np.zeros(len(iids))
                    for r in chosen:
                        values[lookup[r['image_id']]] += r['metrics'][a][metric]-r['metrics'][b][metric]
                    den = counts[draws].sum(1)
                    sample = 100*values[draws].sum(1)/den.clip(min=1)
                    metrics[metric] = {'delta_pp': 100*values.sum()/counts.sum(),
                                       'image_ci95_pp': np.quantile(sample, [.025, .975]).tolist()}
                results[name] = metrics
            vals = np.zeros(len(iids))
            for r in chosen:
                m = r['metrics']
                vals[lookup[r['image_id']]] += (m['shared_half']['iou']-m['base_half']['iou'])-(m['shared_byte']['iou']-m['base_byte']['iou'])
            samples = 100*vals[draws].sum(1)/counts[draws].sum(1).clip(min=1)
            interaction = {'delta_pp':100*vals.sum()/counts.sum(), 'image_ci95_pp':np.quantile(samples,[.025,.975]).tolist()}
            groups[group] = {'instances': len(chosen), 'contrasts': results, 'decoder_x_shared_interaction_iou':interaction,
                            'mean_metrics': {mode:{metric:float(np.mean([r['metrics'][mode][metric] for r in chosen]))
                                                  for metric in ['iou','coverage','purity','mask75']}
                                             for mode in chosen[0]['metrics']},
                            'a_positive_fraction':float(np.mean([r['a'] > 0 for r in chosen]))}
        outputs[split] = groups
    return outputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ['data','weights','split','scalar','shared','template','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--decompose', action='store_true')
    parser.add_argument('--sample-seed',type=int,default=20260922)
    parser.add_argument('--exclude-panel',type=Path)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    setup(); torch.backends.cudnn.benchmark=False; start=time.monotonic()
    coco=COCO(str(args.data/'annotations/instances_train2017.json'))
    old=json.loads(args.split.read_text()); excluded=set(old['fit_image_ids']+old['selection_image_ids'])
    if args.exclude_panel:
        panel=json.loads(args.exclude_panel.read_text());excluded.update(panel['explore']+panel['confirm'])
    all_ids=sorted(set(coco.imgs)-excluded); random.Random(args.sample_seed).shuffle(all_ids)
    ids=all_ids[:1000] if not args.smoke else [old['fit_image_ids'][0],old['fit_image_ids'][1]]
    split_ids={'explore':ids[:len(ids)//2], 'confirm':ids[len(ids)//2:]}
    save(args.out/'SPLIT.json', {**split_ids, 'excluded_images':sorted(excluded), 'smoke':args.smoke,
         'sample_seed':args.sample_seed,'decompose':args.decompose})
    (args.out/'official_decode_source.py.txt').write_text('\n\n'.join(inspect.getsource(o) for o in
        [ops.scale_masks, ops.process_mask, ops.process_mask_native, SegmentationValidator.scale_preds]),encoding='utf-8')
    model=YOLO(str(args.weights)).model.cuda().float().eval();head=model.model[-1];assert head.end2end
    scalar=original.Refiner('scalar').cuda().eval();scalar.load_state_dict(torch.load(args.scalar,weights_only=False)['state_dict'])
    shared=SharedShapeHead().cuda().eval();ck=torch.load(args.shared,weights_only=False);shared.load_state_dict(ck['state_dict'])
    assert ck['template']['field']==json.loads(args.template.read_text())['field']
    grid=grid_tensor(args.template);transform=LetterBox((640,640),auto=True,stride=32)
    cats=sorted(coco.cats); cat_to_cls={c:i for i,c in enumerate(cats)}
    logit_edges=torch.tensor([-float('inf'),-4,-2,-1,-.5,0,.5,1,2,4,float('inf')],device='cuda')
    t=(torch.arange(32,device='cuda')+.5)/32; yy32,xx32=torch.meshgrid(t,t,indexing='ij')
    center=((xx32>=.25)&(xx32<.75)&(yy32>=.25)&(yy32<.75))
    rows=[];calibration=[];identity_differences=0
    with (args.out/'instances.jsonl').open('w',encoding='utf-8') as stream:
        for ni,iid in enumerate(ids):
            split='explore' if iid in split_ids['explore'] else 'confirm'
            anns=[a for a in coco.imgToAnns[iid] if not a.get('iscrowd',0) and not a.get('ignore',0)]
            if not anns:continue
            image=cv2.imread(str(args.data/'images/train2017'/coco.imgs[iid]['file_name']));assert image is not None
            orig=image.shape[:2];resized=transform.apply_image({'img':image},transform.get_params({'img':image}))['img']
            inp=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0]
            allboxes=head._get_decode_boxes(p)[0].T;allc=p['mask_coefficient'][0].T
            scores,classes,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            keep=scores[0,:,0]>.001;classes=classes[0,keep,0].long();rids=rids[0,keep,0]
            if not len(rids):continue
            predorig=ops.scale_boxes(shape,allboxes[rids].clone(),orig)
            gtboxes=torch.tensor([a['bbox'] for a in anns],device='cuda',dtype=torch.float32);gtboxes[:,2:]+=gtboxes[:,:2]
            q=box_iou(predorig,gtboxes);gtcls=torch.tensor([cat_to_cls[a['category_id']] for a in anns],device='cuda')
            q[classes[:,None]!=gtcls[None]]=-1;best,owner=q.max(1)
            rawowners=defaultdict(set);pool=[]
            for j in torch.where(best>=.5)[0].tolist():
                rid=int(rids[j]);own=int(owner[j]);rawowners[rid].add(own);pool.append((float(best[j]),rid,own,j))
            chosen={}
            for quality,rid,own,j in sorted(pool,key=lambda v:(-v[0],v[1])):
                if len(rawowners[rid])==1 and own not in chosen:chosen[own]=(quality,rid,j)
            selected=list(chosen);image_bins=torch.zeros((2,10,3),device='cuda')
            inside_bins=torch.zeros_like(image_bins);hard_inside_bins=torch.zeros_like(image_bins)
            low=(allc@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            if args.decompose:
                proto_scale=proto.square().mean((1,2)).sqrt().clamp_min(.1)
                normalized_proto=F.interpolate((proto/proto_scale[:,None,None])[None],shape,mode='bilinear',align_corners=False)[0]
            gain=min(shape[0]/orig[0],shape[1]/orig[1]);nh,nw=round(orig[0]*gain),round(orig[1]*gain)
            top,left=round((shape[0]-nh)/2-.1),round((shape[1]-nw)/2-.1)
            for first in range(0,len(selected),8):
                owns=selected[first:first+8];indices=torch.tensor([chosen[o][1] for o in owns],device='cuda')
                boxes=allboxes[indices];ob=ops.scale_boxes(shape,boxes.clone(),orig)
                targets=torch.from_numpy(np.stack([coco.annToMask(anns[o]) for o in owns])).cuda().bool()
                input_gt=F.pad(F.interpolate(targets[:,None].float(),(nh,nw),mode='bilinear',align_corners=False),
                    (left,shape[1]-nw-left,top,shape[0]-nh-top))
                feats,roi_base,roi_grid=original.features(proto,allc[indices],boxes,shape)
                roi_target=F.grid_sample(input_gt,roi_grid,align_corners=False)[:,0]
                bins=torch.bucketize(roi_base.contiguous(),logit_edges[1:-1])
                for region,mask in enumerate([center,~center]):
                    for bi in range(10):
                        selected_pixels=(bins==bi)&mask[None]
                        image_bins[region,bi,0]+=selected_pixels.sum()
                        image_bins[region,bi,1]+=(roi_target*selected_pixels).sum()
                        image_bins[region,bi,2]+=(roi_base.sigmoid()*selected_pixels).sum()
                base=F.interpolate(low[indices][None],shape,mode='bilinear',align_corners=False)[0]
                basis,_,std,count=full_template(grid,boxes,shape);assert bool(torch.isfinite(basis).all())
                values=shared(feats)*.5;b=4*values[:,0,None,None];a=4*values[:,1,None,None]
                gtinput=gtboxes[owns].clone();gtinput[:,[0,2]]=gtinput[:,[0,2]]*gain+left;gtinput[:,[1,3]]=gtinput[:,[1,3]]*gain+top
                gt_support=ops.crop_mask(torch.ones_like(base),gtinput).bool()
                roi_support=F.grid_sample(gt_support[:,None].float(),roi_grid,mode='nearest',align_corners=False)[:,0]>.5
                # Original-pixel nearest sampling supplies a hard-label check without soft resize targets.
                rx=(roi_grid[...,0]+1)*shape[1]/2;ry=(roi_grid[...,1]+1)*shape[0]/2
                ox=(rx-left)/gain;oy=(ry-top)/gain
                native_grid=torch.stack((2*ox/orig[1]-1,2*oy/orig[0]-1),-1)
                hard_roi=F.grid_sample(targets[:,None].float(),native_grid,mode='nearest',align_corners=False)[:,0]
                for region,mask in enumerate([center,~center]):
                    for bi in range(10):
                        selected_pixels=(bins==bi)&mask[None]&roi_support
                        for dst,label in [(inside_bins,roi_target),(hard_inside_bins,hard_roi)]:
                            dst[region,bi,0]+=selected_pixels.sum()
                            dst[region,bi,1]+=(label*selected_pixels).sum()
                            dst[region,bi,2]+=(roi_base.sigmoid()*selected_pixels).sum()
                delta=b+a*basis
                variants={'base':base,'scalar':base+2*scalar(feats)[:,0,None,None],
                    'shared':base+delta,'bias':base+b,'space':base+a*basis,
                    'reverse':base+b-a*basis,'inside':base+delta*gt_support,'outside':base+delta*(~gt_support)}
                projection_info=[]
                if args.decompose:
                    crop=ops.crop_mask(torch.ones_like(base),boxes).bool()
                    projected=[]
                    for j in range(len(boxes)):
                        support=crop[j]
                        mat=normalized_proto[:,support].double()
                        target_space=(a[j]*basis[j])[support].double()
                        if mat.shape[1]==0:
                            coef=torch.zeros(32,device='cuda',dtype=torch.float64);rank=0
                        else:
                            gram=mat@mat.T/max(1,mat.shape[1]);rhs=mat@target_space/max(1,mat.shape[1])
                            eigen,vec=torch.linalg.eigh(gram)
                            kept=eigen>eigen.max().clamp_min(1e-12)*1e-8
                            coef=vec[:,kept]@((vec[:,kept].T@rhs)/eigen[kept]);rank=int(kept.sum())
                        proj=(coef.float()@normalized_proto.flatten(1)).reshape(shape)
                        residual=(a[j]*basis[j]-proj)[support].double()
                        energy=float(target_space.square().sum())
                        projection_info.append({'rank':rank,'retained_energy':1-float(residual.square().sum())/max(energy,1e-12),
                            'target_rms':float(target_space.square().mean().sqrt()) if len(target_space) else 0.,
                            'coef_norm':float(coef.norm()),'projection_finite':bool(torch.isfinite(proj).all())})
                        projected.append(proj)
                    projected=torch.stack(projected);assert torch.isfinite(projected).all()
                    variants.update(projected=base+b+projected,orthogonal=base+b+a*basis-projected,
                        space_inside=base+b+a*basis*gt_support,space_outside=base+b+a*basis*(~gt_support))
                masks={}
                for name,z in variants.items():
                    binary=ops.crop_mask(z.clone(),boxes).gt(0).byte()
                    if name=='base' and ni==0:
                        official=ops.process_mask(proto,allc[indices],boxes,shape,upsample=True)
                        identity_differences+=int((official!=binary).sum())
                    scaled=ops.scale_masks(binary[None],orig)[0]
                    masks[name+'_byte']=scaled.byte().bool()
                    masks[name+'_half']=scaled.gt(.5)
                measured={};changes={};target_area=targets.sum((1,2)).clamp_min(1)
                for name,mask in masks.items():
                    tp=(mask&targets).sum((1,2));predarea=mask.sum((1,2));union=(mask|targets).sum((1,2)).clamp_min(1)
                    iou=tp/union
                    measured[name]=torch.stack((iou,tp/target_area,tp/predarea.clamp_min(1),(iou>=.75).float()),1).cpu().tolist()
                    if name.startswith('shared_'):
                        original_mask=masks['base_'+name.split('_')[-1]]
                        changes[name]=torch.stack([((mask&~original_mask)&targets).sum((1,2))/target_area,
                            ((~mask&original_mask)&targets).sum((1,2))/target_area,
                            ((~mask&original_mask)&~targets).sum((1,2))/target_area,
                            ((mask&~original_mask)&~targets).sum((1,2))/target_area],1).cpu().tolist()
                for j,own in enumerate(owns):
                    ann=anns[own];qual,rid,_=chosen[own]
                    row={'image_id':iid,'annotation_id':ann['id'],'raw_id':rid,'split':split,'area':ann['area'],
                         'category_id':ann['category_id'],'box_iou':qual,'input_box_wh':(boxes[j,2:]-boxes[j,:2]).tolist(),
                         'resize_gain':gain,'fill_fraction':ann['area']/max(1,ann['bbox'][2]*ann['bbox'][3]),
                         'b':float(b[j,0,0]),'a':float(a[j,0,0]),'crop_pixels':int(count[j]),'template_std':float(std[j]),
                         'metrics':{name:dict(zip(['iou','coverage','purity','mask75'],v[j])) for name,v in measured.items()},
                         'pixel_changes':{name:dict(zip(['added_TP','lost_TP','removed_FP','added_FP'],v[j])) for name,v in changes.items()}}
                    if args.decompose:row['projection']=projection_info[j]
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            calibration.append({'image_id':iid,'split':split,'bins':image_bins.cpu().tolist(),
                'inside_gt_bins':inside_bins.cpu().tolist(),'hard_inside_gt_bins':hard_inside_bins.cpu().tolist()})
            if ni%50==0 or ni+1==len(ids):
                progress={'images':ni+1,'total':len(ids),'instances':len(rows),'elapsed_s':time.monotonic()-start}
                save(args.out/'progress.json',progress);print(json.dumps(progress),flush=True);stream.flush()
    assert identity_differences==0,identity_differences
    save(args.out/'CALIBRATION.json',{'columns':['count','target_sum','probability_sum'],'regions':['center_quarter','outer_ring'],
        'logit_edges':['-inf',-4,-2,-1,-.5,0,.5,1,2,4,'inf'],'images':calibration})
    summary=summarize(rows,split_ids);save(args.out/'SUMMARY.json',summary)
    save(args.out/'COMPLETE.json',{'images':len(ids),'instances':len(rows),'identity_pixel_differences':identity_differences,
         'elapsed_s':time.monotonic()-start,'smoke':args.smoke,'scope':'Fixed selected representatives, not COCO AP.'})
    print(json.dumps({split:summary.get(split,{}).get('all',{}) for split in split_ids}),flush=True)


if __name__=='__main__':main()
