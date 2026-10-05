"""Localize pixel competition and compare error-focused one-step diagnostics.

GT-dependent masks define training losses; all are same-image diagnostics, not
deployable refinements or held-out generalization measurements.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from pycocotools.coco import COCO
from prepare import key
from supervision_probe import IdentityFormat, overlap_stats, dump
from parameter_probe import flatten, set_parameters


def pixel_transitions(mask, base, gt):
    return {'fn_fixed':int((~base & gt & mask).sum()),
            'tp_lost':int((base & gt & ~mask).sum()),
            'fp_fixed':int((base & ~gt & ~mask).sum()),
            'tn_lost':int((~base & ~gt & mask).sum())}


def objective_vectors(c, pf, target, support, area, gain, generator):
    """Analytical coefficient gradients, same official bbox-area normalization."""
    z=c@pf;prob=z.sigmoid();residual=prob-target.float()
    positive=z>0
    regions={'tp':support & target & positive, 'fn':support & target & ~positive,
             'fp':support & ~target & positive, 'tn':support & ~target & ~positive}
    components={k:(pf[:,m]@residual[m])*gain/area for k,m in regions.items()}
    fg=regions['tp']|regions['fn'];bg=regions['fp']|regions['tn']
    ordinary=sum(components.values());errors=components['fn']+components['fp']
    balanced=.5*int(support.sum())*((components['tp']+components['fn'])/max(int(fg.sum()),1)
                                  +(components['fp']+components['tn'])/max(int(bg.sum()),1))
    selected=torch.zeros_like(support)
    for pool,n in [(fg,int(regions['fn'].sum())),(bg,int(regions['fp'].sum()))]:
        ix=pool.nonzero().flatten()
        if n:
            selected[ix[torch.randperm(len(ix),device='cuda',generator=generator)[:n]]]=True
    random=(pf[:,selected]@residual[selected])*gain/area
    pt=torch.where(target,prob,1-prob)
    bce=F.binary_cross_entropy_with_logits(z,target.float(),reduction='none')
    focal_factor=(1-pt).square()+2*pt*(1-pt)*bce
    focal=(pf[:,support]@(residual*focal_factor)[support])*gain/area
    intersection=(prob[support]*target[support]).sum()
    union=(prob[support]+target[support]-prob[support]*target[support]).sum()
    derivative=((intersection+1)*(~target)-(union+1)*target)/(union+1).square()
    soft_iou=(pf[:,support]@(derivative*prob*(1-prob))[support])*gain
    vectors={'ordinary':ordinary,'balanced':balanced,'errors_only':errors,
             'correct_only':components['tp']+components['tn'],'random_equal_pixels':random,
             'focal2':focal,'soft_iou':soft_iou}
    counts={k:int(m.sum()) for k,m in regions.items()}
    return vectors,components,counts,regions


def main():
    ap=argparse.ArgumentParser()
    for k in ['root','raw','source','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--limit',type=int,default=100000)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    source=[json.loads(s) for s in a.source.read_text().splitlines()]
    primary=[r for r in source if r['geometry_state'] in ('joint_good','box_good_mask_unavailable')
             and r.get('readout_probe',{}).get('actual_assignment_owner',{}).get('one2one')==r['annotation_id']]
    image_ids=sorted({r['image_id'] for r in primary})[:a.limit]
    primary_ids={r['annotation_id'] for r in primary};by_image=defaultdict(list)
    for r in source:
        if 'readout_probe' in r:by_image[r['image_id']].append(r)
    coco=COCO(str(a.root/'assets/datasets/coco/annotations/instances_train2017.json'))
    ledger=json.loads((a.root/'LABEL_LEDGER.json').read_text())
    yolo=YOLO(str(a.root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'))
    model=yolo.model.cuda().float().eval();model.args=SimpleNamespace(**yolo.ckpt['train_args'])
    for p in model.parameters():p.requires_grad_(False)
    head=model.model[-1];module=head.one2one_cv4;params=list(module.parameters())
    for p in params:p.requires_grad_(True)
    initial=flatten([p.detach() for p in params]).clone()
    optimizer=BaseTrainer.build_optimizer(SimpleNamespace(args=model.args,data={'nc':80}),module,
        name='MuSGD',lr=model.args.lr0,momentum=model.args.momentum,decay=model.args.weight_decay)
    criterion=v8SegmentationLoss(model,tal_topk=7,tal_topk2=1)
    data=YOLODataset(img_path=str(a.root/'explore_images.txt'),data={'names':model.names,'nc':80},
        task='segment',imgsz=640,batch_size=1,augment=False,hyp=model.args,cache=False,rect=False,stride=32,prefix='regions: ')
    by_id={int(Path(v['im_file']).stem):i for i,v in enumerate(data.labels)}
    letterbox=LetterBox((640,640),auto=True,stride=32)
    formatter=IdentityFormat(return_mask=True,mask_ratio=1,mask_overlap=True,bgr=0.)
    head.training=True;head.proto.training=True
    dump(a.out/'SETUP.json',{'images':image_ids,'primary_instances':sum(r['image_id'] in image_ids for r in primary),
        'parameter_count':initial.numel(),'head_scales':[1,3],'coefficient_rms_scales':[.1,.3],
        'scope':'same-image fixed candidate diagnostic; head resets per image, coefficient resets per instance; not AP'})
    records=(a.out/'instances.jsonl').open('w');image_records=(a.out/'images.jsonl').open('w')
    start=time.monotonic();count=0
    for number,image_id in enumerate(image_ids):
        set_parameters(params,initial)
        label=deepcopy(data.labels[by_id[image_id]]);binding={v['key']:v['annotation_id'] for v in ledger[str(image_id)]}
        ann_ids=[binding[key(cls[0],seg)] for cls,seg in zip(label['cls'],label['segments'])]
        im=cv2.imread(str(a.root/'assets/datasets/coco/images/train2017'/coco.imgs[image_id]['file_name']));orig=im.shape[:2]
        label['img']=im;label.pop('shape',None);batch=formatter(letterbox(data.update_labels_info(label)))
        ordered=[ann_ids[k] for k in formatter.order];mapping={ann:i for i,ann in enumerate(ordered)}
        for k in ['cls','bboxes','batch_idx','masks','sem_masks']:batch[k]=batch[k].cuda()
        batch['img']=batch['img'].cuda().float()[None]/255;batch['sem_masks']=batch['sem_masks'][None]
        shape=tuple(batch['img'].shape[-2:]);hh,ww=shape
        with torch.no_grad():
            pred=model(batch['img'])['one2one']
            fg_mask,owner,assigned_boxes,_,_=criterion.get_assigned_targets_and_loss(pred,batch)[0]
            positives=fg_mask[0].nonzero().flatten();proto=pred['proto'][0][0]
            pf=F.interpolate(proto[None],shape,mode='bilinear',align_corners=False)[0].flatten(1)
            feats=[x.detach() for x in pred['feats']];canvas=batch['masks'][0].long()
        def forward_coeff():return torch.cat([module[i](feats[i]).view(1,head.nm,-1) for i in range(head.nl)],2)[0].T
        coeff=forward_coeff()
        assert torch.allclose(coeff.detach(),pred['mask_coefficient'][0].T,atol=2e-5,rtol=2e-5)
        def vjp(indices,vectors):
            return flatten([g.detach() for g in torch.autograd.grad(coeff[indices],params,grad_outputs=vectors,retain_graph=True)])
        vectors_by_arm=defaultdict(list);parts_by_region=defaultdict(list);metadata={};own_error_grads={};own_correct_grads={}
        with np.load(a.raw/'images'/f'{image_id:012d}.npz') as cache:boxes=torch.from_numpy(cache['boxes_input']).cuda()
        for ri_tensor in positives:
            ri=int(ri_tensor);gi=int(owner[0,ri]);ann_id=ordered[gi]
            with torch.no_grad():
                box=assigned_boxes[0,ri:ri+1];support=ops.crop_mask(torch.ones((1,hh,ww),device='cuda'),box)[0].bool().flatten()
                target=(canvas==gi+1).flatten();area=(box[0,2:]-box[0,:2]).prod().clamp_min(1e-9)
                gen=torch.Generator(device='cuda').manual_seed(ann_id+20260920)
                vec,components,counts,regions=objective_vectors(coeff.detach()[ri],pf,target,support,area,model.args.box*.2/len(positives),gen)
                metadata[ri]={'counts':counts,'vectors':vec,'components':components,'support':support,'regions':regions,'annotation_id':ann_id}
                for k,v in vec.items():vectors_by_arm[k].append(v)
                for k,v in components.items():parts_by_region[k].append(v)
            if ann_id in primary_ids:
                own_error_grads[ri]=vjp([ri],vec['errors_only'][None])
                own_correct_grads[ri]=vjp([ri],vec['correct_only'][None])
        gradients={k:vjp(positives,torch.stack(v)) for k,v in vectors_by_arm.items()}
        part_gradients={k:vjp(positives,torch.stack(v)) for k,v in parts_by_region.items()}
        if number==0:
            pred_check=dict(pred);pred_check['mask_coefficient']=coeff.T[None]
            official_loss=criterion.loss(pred_check,batch)[0][1]*.2
            exact=flatten(torch.autograd.grad(official_loss,params,retain_graph=True))
            rel=float((gradients['ordinary']-exact).norm()/exact.norm().clamp_min(1e-12))
            assert rel<2e-4
            # Validate non-BCE derivatives against autograd on a real positive.
            first=int(positives[0]);meta=metadata[first];idx=(positives==first).nonzero().item();gi=int(owner[0,first])
            cc=coeff.detach()[first].clone().requires_grad_();z=cc@pf;t=(canvas==gi+1).flatten();s=meta['support'];p=z.sigmoid()
            area=(assigned_boxes[0,first,2:]-assigned_boxes[0,first,:2]).prod();gain=model.args.box*.2/len(positives)
            pt=torch.where(t,p,1-p);focal=((1-pt).square()*F.binary_cross_entropy_with_logits(z,t.float(),reduction='none'))[s].sum()*gain/area
            g1=torch.autograd.grad(focal,cc,retain_graph=True)[0]
            inter=(p[s]*t[s]).sum();union=(p[s]+t[s]-p[s]*t[s]).sum();iou_loss=(1-(inter+1)/(union+1))*gain
            g2=torch.autograd.grad(iou_loss,cc)[0]
            errors={'official_gradient_relative':rel,'focal_relative':float((g1-meta['vectors']['focal2']).norm()/g1.norm().clamp_min(1e-12)),
                    'soft_iou_relative':float((g2-meta['vectors']['soft_iou']).norm()/g2.norm().clamp_min(1e-12))}
            assert max(errors.values())<2e-4;dump(a.out/'GRADIENT_CHECK.json',errors)
        def proposal(gradient):
            set_parameters(params,initial);optimizer.state.clear();optimizer.zero_grad(set_to_none=True);offset=0
            for p in params:p.grad=gradient[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
            torch.nn.utils.clip_grad_norm_(params,10.);optimizer.step()
            update=initial-flatten([p.detach() for p in params]);set_parameters(params,initial);return update
        ordinary=proposal(gradients['ordinary']);size=ordinary.norm();directions={'ordinary':ordinary}
        for name,g in gradients.items():
            if name=='ordinary':continue
            u=proposal(g);directions[name]=u/u.norm().clamp_min(1e-20)*size
        image_records.write(json.dumps({'image_id':image_id,'positives':len(positives),'step_l2':float(size),
            'gradient_norms':{k:float(g.norm()) for k,g in gradients.items()},
            'region_gradient_norms':{k:float(g.norm()) for k,g in part_gradients.items()}})+'\n');image_records.flush()
        base_coeff=coeff.detach().clone();del coeff
        with torch.no_grad():
            outputs={}
            for name,direction in directions.items():
                for scale in (1.,3.):
                    set_parameters(params,initial-scale*direction);outputs[f'head/{name}:{scale:g}']=forward_coeff().detach()
            set_parameters(params,initial)
            eval_rows=by_image[image_id]
            # Ordinary COCO instances define neighbors; crowd regions retained separately.
            anns=coco.loadAnns(coco.getAnnIds(imgIds=[image_id]));unions={};crowd=np.zeros(orig,dtype=bool)
            for ann in anns:
                m=coco.annToMask(ann).astype(bool)
                if ann.get('iscrowd',0):crowd|=m
                else:unions.setdefault(ann['category_id'],np.zeros(orig,dtype=bool))[:] |= m
            union_all=np.logical_or.reduce(list(unions.values())) if unions else np.zeros(orig,dtype=bool)
            for r in eval_rows:
                ann_id=r['annotation_id'];ri=r['readout_probe']['raw_id'];gi=mapping[ann_id]
                gt_np=coco.annToMask(coco.anns[ann_id]).astype(bool);gt=torch.from_numpy(gt_np).cuda();ownbox=boxes[ri:ri+1]
                gtbox=ops.xywh2xyxy(batch['bboxes'][gi:gi+1])*torch.tensor([ww,hh,ww,hh],device='cuda')
                support=ops.crop_mask(torch.ones((1,hh,ww),device='cuda'),gtbox)[0].bool();target=canvas==gi+1
                def logit(c):return F.interpolate((c@proto.flatten(1)).reshape(1,1,*proto.shape[-2:]),shape,mode='bilinear',align_corners=False)[0,0]
                def decode(z):return ops.scale_masks((ops.crop_mask(z[None].clone(),ownbox)>0).byte()[None],orig)[0,0].byte().bool()
                base_z=logit(base_coeff[ri]);base=decode(base_z)
                width=max(1,int(np.ceil(.02*np.sqrt(gt_np.sum()))))
                distance=np.where(gt_np,distance_transform_edt(gt_np),distance_transform_edt(~gt_np))
                near=torch.from_numpy(distance<=width).cuda()
                same=torch.from_numpy(unions[coco.anns[ann_id]['category_id']] & ~gt_np).cuda()
                other=torch.from_numpy(union_all & ~gt_np & ~unions[coco.anns[ann_id]['category_id']]).cuda()
                background=torch.from_numpy(~union_all & ~crowd).cuda()
                result={'image_id':image_id,'annotation_id':ann_id,'geometry_state':r['geometry_state'],'primary':ann_id in primary_ids,
                        'raw_id':ri,'boundary_width_original_pixels':width,'metrics':{}}
                zs={'baseline':base_z}
                for name,c in outputs.items():zs[name]=logit(c[ri])
                if ann_id in primary_ids:
                    meta=metadata[ri];assert meta['annotation_id']==ann_id
                    err=own_error_grads[ri];correct=own_correct_grads[ri]
                    result['competition']={'counts':meta['counts'],
                        'own_correct_error_cosine':float(err@correct/(err.norm()*correct.norm()).clamp_min(1e-20)),
                        'own_correct_gradient_opposes_errors':bool(err@correct<0),
                        'ordinary_optimizer_harms_error_bce':bool(ordinary@err<0),
                        'own_correct_error_norm_ratio':float(correct.norm()/err.norm().clamp_min(1e-12)),
                        'image_region_dot_own_errors':{k:float(g@err) for k,g in part_gradients.items()},
                        'coefficient_region_dot_own_errors':{k:float(g@meta['vectors']['errors_only']) for k,g in meta['components'].items()}}
                    for name,g in meta['vectors'].items():
                        dz=(g@pf).reshape(shape);rms=dz[support].square().mean().sqrt()
                        if rms>1e-12:
                            for scale in (.1,.3):zs[f'coefficient/{name}:{scale:g}']=base_z-scale*dz/rms
                training_regions={'tp':support & target & (base_z>0),'fn':support & target & (base_z<=0),
                                  'fp':support & ~target & (base_z>0),'tn':support & ~target & (base_z<=0)}
                for name,z in zs.items():
                    mask=decode(z);m=overlap_stats(mask,gt);m.update(pixel_transitions(mask,base,gt))
                    m['fp_same_neighbor']=int((mask & same).sum());m['fp_other_neighbor']=int((mask & other).sum())
                    m['fp_background']=int((mask & background).sum())
                    m['fp_near_boundary']=int((mask & ~gt & near).sum());m['fp_far_boundary']=int((mask & ~gt & ~near).sum())
                    m['fn_near_boundary']=int((~mask & gt & near).sum());m['fn_interior']=int((~mask & gt & ~near).sum())
                    bce=F.binary_cross_entropy_with_logits(z,target.float(),reduction='none')
                    m['crop_bce']=float(bce[support].mean())
                    m['region_bce']={k:float(bce[v].mean()) if v.any() else None for k,v in training_regions.items()}
                    result['metrics'][name]=m
                assert abs(result['metrics']['baseline']['iou']-r['readout_probe']['baseline']['iou'])<2e-6
                records.write(json.dumps(result,allow_nan=False)+'\n');count+=1
        records.flush();progress={'images':number+1,'total':len(image_ids),'instances':count,'elapsed_s':round(time.monotonic()-start,2)}
        dump(a.out/'progress.json',progress)
        if number==0 or (number+1)%10==0:print(json.dumps(progress),flush=True)
        del outputs,pred,gradients,part_gradients,own_error_grads,own_correct_grads,pf,metadata,zs
    records.close();image_records.close();dump(a.out/'COMPLETE.json',progress);print(json.dumps(progress),flush=True)


if __name__=='__main__':main()
