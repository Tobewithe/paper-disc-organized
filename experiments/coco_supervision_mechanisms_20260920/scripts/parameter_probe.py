"""Controlled one-step updates of the actual one2one coefficient head.

Each image starts from the pretrained checkpoint. This is an in-sample mechanism
probe, not training or a claim of held-out AP. Boxes, prototypes and BN stay fixed.
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
from scipy.optimize import minimize
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


def flatten(xs):
    return torch.cat([x.reshape(-1) for x in xs])


@torch.no_grad()
def set_parameters(params, flat):
    offset = 0
    for p in params:
        p.copy_(flat[offset:offset+p.numel()].view_as(p))
        offset += p.numel()


def project_direction(direction, fg_grads):
    """Euclidean projection onto g_foreground_i dot update >= 0 for all positives."""
    size = direction.norm()
    unit = direction / size.clamp_min(1e-20)
    lengths = fg_grads.norm(dim=1)
    normals = fg_grads[lengths > 1e-15] / lengths[lengths > 1e-15, None]
    if not len(normals):
        return direction, {'constraints': 0, 'success': True}
    gram = (normals @ normals.T).double().cpu().numpy()
    rhs = (normals @ unit).double().cpu().numpy()
    fit = minimize(lambda x: (.5*x@gram@x + rhs@x, gram@x+rhs),
                   np.zeros(len(rhs)), jac=True, bounds=[(0, None)]*len(rhs),
                   method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 3000})
    projected = unit + torch.from_numpy(fit.x).to(unit) @ normals
    # Resolve small numerical infeasibility without changing the constraint set.
    for _ in range(1000):
        dots = normals @ projected
        worst = int(dots.argmin())
        if float(dots[worst]) >= -1e-7:
            break
        projected = projected - dots[worst]*normals[worst]
    norm = projected.norm()
    result = projected/norm.clamp_min(1e-20)*size if norm > 1e-8 else torch.zeros_like(direction)
    return result, {'constraints': len(normals), 'success': bool(fit.success),
                    'pre_normalization_norm': float(norm),
                    'min_constraint': float((normals @ (result/size.clamp_min(1e-20))).min()),
                    'before_violations': int((rhs < -1e-7).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=100000)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = [json.loads(s) for s in a.source.read_text().splitlines()]
    primary = [r for r in source if r['geometry_state'] in ('joint_good', 'box_good_mask_unavailable')
               and r.get('readout_probe', {}).get('actual_assignment_owner', {}).get('one2one') == r['annotation_id']]
    image_ids = sorted({r['image_id'] for r in primary})[:a.limit]
    primary_ids = {r['annotation_id'] for r in primary}
    by_image = defaultdict(list)
    for r in source:
        if 'readout_probe' in r:
            by_image[r['image_id']].append(r)
    coco = COCO(str(a.root/'assets/datasets/coco/annotations/instances_train2017.json'))
    ledger = json.loads((a.root/'LABEL_LEDGER.json').read_text())
    yolo = YOLO(str(a.root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'))
    model = yolo.model.cuda().float().eval()
    model.args = SimpleNamespace(**yolo.ckpt['train_args'])
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]; module = head.one2one_cv4
    params = list(module.parameters())
    for p in params:
        p.requires_grad_(True)
    initial = flatten([p.detach() for p in params]).clone()
    optimizer = BaseTrainer.build_optimizer(SimpleNamespace(args=model.args, data={'nc':80}), module,
                    name='MuSGD', lr=model.args.lr0, momentum=model.args.momentum, decay=model.args.weight_decay)
    criterion = v8SegmentationLoss(model, tal_topk=7, tal_topk2=1)
    data = YOLODataset(img_path=str(a.root/'explore_images.txt'), data={'names':model.names, 'nc':80},
                       task='segment', imgsz=640, batch_size=1, augment=False, hyp=model.args,
                       cache=False, rect=False, stride=32, prefix='parameter: ')
    by_id = {int(Path(v['im_file']).stem):i for i,v in enumerate(data.labels)}
    letterbox = LetterBox((640,640), auto=True, stride=32)
    formatter = IdentityFormat(return_mask=True, mask_ratio=1, mask_overlap=True, bgr=0.0)
    head.training = True; head.proto.training = True
    dump(a.out/'SETUP.json', {'images':len(image_ids), 'image_ids':image_ids, 'primary_instances':sum(r['image_id'] in image_ids for r in primary),
         'parameter_count':initial.numel(), 'optimizer':'MuSGD cold start, no historical state',
         'lr0':model.args.lr0, 'momentum':model.args.momentum, 'weight_decay':model.args.weight_decay,
         'box_gain':model.args.box, 'one2one_weight':.2, 'bn':'eval', 'scales':[1.0,3.0],
         'scope':'fixed candidate/prototype/box; same-image one-step head-only diagnostic, not AP'})
    records = (a.out/'instances.jsonl').open('w', encoding='utf-8')
    image_records = (a.out/'images.jsonl').open('w', encoding='utf-8')
    start = time.monotonic(); count = 0
    for number, image_id in enumerate(image_ids):
        set_parameters(params, initial)
        label = deepcopy(data.labels[by_id[image_id]])
        binding = {v['key']:v['annotation_id'] for v in ledger[str(image_id)]}
        ann_ids = [binding[key(cls[0],seg)] for cls,seg in zip(label['cls'], label['segments'])]
        im = cv2.imread(str(a.root/'assets/datasets/coco/images/train2017'/coco.imgs[image_id]['file_name']))
        orig = im.shape[:2]
        label['img'] = im; label.pop('shape', None)
        batch = formatter(letterbox(data.update_labels_info(label)))
        ordered = [ann_ids[k] for k in formatter.order]
        for k in ['cls','bboxes','batch_idx','masks','sem_masks']:
            batch[k] = batch[k].cuda()
        batch['img'] = batch['img'].cuda().float()[None]/255
        batch['sem_masks'] = batch['sem_masks'][None]
        shape = tuple(batch['img'].shape[-2:]); hh,ww = shape
        with torch.no_grad():
            pred = model(batch['img'])['one2one']
            fg_mask, owner, assigned_boxes, _, _ = criterion.get_assigned_targets_and_loss(pred,batch)[0]
            positives = fg_mask[0].nonzero().flatten()
            proto = pred['proto'][0][0]
            pf = F.interpolate(proto[None], shape, mode='bilinear', align_corners=False)[0].flatten(1)
            feats = [x.detach() for x in pred['feats']]
            canvas = batch['masks'][0].long()
        def forward_coeff():
            return torch.cat([module[i](feats[i]).view(1,head.nm,-1) for i in range(head.nl)],2)[0].T
        coeff = forward_coeff()
        assert torch.allclose(coeff.detach(),pred['mask_coefficient'][0].T,atol=2e-5,rtol=2e-5)
        def vjp(indices, vectors):
            grads = torch.autograd.grad(coeff[indices], params, grad_outputs=vectors, retain_graph=True)
            return flatten([g.detach() for g in grads])
        gfc=[]; gbc=[]; balanced=[]; fgrads=[]; bgrads={}; positive_lookup={}
        with np.load(a.raw/'images'/f'{image_id:012d}.npz') as cache:
            boxes = torch.from_numpy(cache['boxes_input']).cuda()
            assert torch.allclose(proto,torch.from_numpy(cache['proto']).cuda(),atol=2e-5,rtol=2e-5)
        for j,ri_tensor in enumerate(positives):
            ri=int(ri_tensor); gi=int(owner[0,ri]); ann_id=ordered[gi]
            positive_lookup[ri]=(j,ann_id)
            with torch.no_grad():
                box=assigned_boxes[0,ri:ri+1]
                support=ops.crop_mask(torch.ones((1,hh,ww),device='cuda'),box)[0].bool().flatten()
                target=(canvas==gi+1).flatten()
                pos=support & target; neg=support & ~target
                bboxarea=(box[0,2:]-box[0,:2]).prod().clamp_min(1e-9)
                weight=model.args.box*.2/len(positives)/bboxarea
                prob=(coeff.detach()[ri]@pf).sigmoid()
                gf=(pf[:,pos]@(prob[pos]-1))*weight
                gb=(pf[:,neg]@prob[neg])*weight
                gfc.append(gf);gbc.append(gb)
                n=int(support.sum())
                balanced.append(.5*n*(gf/max(int(pos.sum()),1)+gb/max(int(neg.sum()),1)))
            fgrads.append(vjp([ri],gf[None]))
            if ann_id in primary_ids:
                bgrads[ri]=vjp([ri],gb[None])
        gf_coeff=torch.stack(gfc);gb_coeff=torch.stack(gbc)
        joint=vjp(positives,gf_coeff+gb_coeff)
        balanced_gradient=vjp(positives,torch.stack(balanced))
        fgrads=torch.stack(fgrads)
        # Direct official loss autograd verifies the analytical VJP and normalization once.
        if number==0:
            pred_check=dict(pred);pred_check['mask_coefficient']=coeff.T[None]
            official_loss=criterion.loss(pred_check,batch)[0][1]*.2
            exact=flatten(torch.autograd.grad(official_loss,params,retain_graph=True))
            rel=float((joint-exact).norm()/exact.norm().clamp_min(1e-12))
            assert rel<2e-4, f'Official gradient mismatch {rel}'
            dump(a.out/'GRADIENT_CHECK.json',{'relative_error':rel,'official_mask_loss':float(official_loss.detach())})
        def proposal(gradient):
            set_parameters(params,initial);optimizer.state.clear();optimizer.zero_grad(set_to_none=True)
            offset=0
            for p in params:
                p.grad=gradient[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
            norm=torch.nn.utils.clip_grad_norm_(params,10.)
            optimizer.step()
            update=initial-flatten([p.detach() for p in params])
            set_parameters(params,initial)
            return update,float(norm)
        ordinary,gradnorm=proposal(joint)
        balance,_=proposal(balanced_gradient)
        size=ordinary.norm()
        assert size>0
        balance=balance/balance.norm().clamp_min(1e-20)*size
        protect,projection=project_direction(ordinary,fgrads)
        generator=torch.Generator(device='cuda').manual_seed(image_id+20260920)
        random=torch.randn(ordinary.shape,device='cuda',generator=generator)
        unit=ordinary/size
        random=random-(random@unit)*unit;random=random/random.norm().clamp_min(1e-20)
        cosine=float((unit@(protect/size)).clamp(-1,1))
        angle_control=size*(cosine*unit+np.sqrt(max(0.,1-cosine*cosine))*random)
        directions={'ordinary':ordinary,'balanced':balance,'foreground_protected':protect,'angle_control':angle_control}
        image_record={'image_id':image_id,'positives':len(positives),'joint_gradient_norm':gradnorm,
                      'step_l2':float(size),'projection':projection,'protected_cosine':cosine}
        image_records.write(json.dumps(image_record)+'\n');image_records.flush()
        eval_rows=by_image[image_id]
        base_coeff=coeff.detach().clone()
        del coeff
        outputs={}
        with torch.no_grad():
            outputs['baseline']=base_coeff
            for name,direction in directions.items():
                for scale in (1.,3.):
                    set_parameters(params,initial-scale*direction)
                    outputs[f'{name}:{scale:g}']=forward_coeff().detach()
            set_parameters(params,initial)
            for r in eval_rows:
                ann_id=r['annotation_id'];ri=r['readout_probe']['raw_id']
                gi=ordered.index(ann_id)
                gt=torch.from_numpy(coco.annToMask(coco.anns[ann_id]).astype(bool)).cuda()
                ownbox=boxes[ri:ri+1]
                gtbox=ops.xywh2xyxy(batch['bboxes'][gi:gi+1])*torch.tensor([ww,hh,ww,hh],device='cuda')
                support=ops.crop_mask(torch.ones((1,hh,ww),device='cuda'),gtbox)[0].bool()
                target=canvas==gi+1
                pos=support & target;neg=support & ~target
                result={'image_id':image_id,'annotation_id':ann_id,'geometry_state':r['geometry_state'],
                        'primary':ann_id in primary_ids,'raw_id':ri,'metrics':{}}
                if ri in positive_lookup and positive_lookup[ri][1]==ann_id and ann_id in primary_ids:
                    idx=positive_lookup[ri][0];gf=fgrads[idx];gb=bgrads[ri]
                    result['parameter_gradient']={'fg_bg_cosine':float(gf@gb/(gf.norm()*gb.norm()).clamp_min(1e-20)),
                      'own_joint_harms_fg':bool((gf+gb)@gf<0), 'image_joint_harms_fg':bool(joint@gf<0),
                      'optimizer_harms_fg':bool(ordinary@gf<0),
                      'first_order_fg_change':{name:-float(direction@gf) for name,direction in directions.items()}}
                for name,c in outputs.items():
                    # Match normal decoder order: low-resolution combination then interpolation.
                    low=(c[ri]@proto.flatten(1)).reshape(proto.shape[-2:])
                    z=F.interpolate(low[None,None],shape,mode='bilinear',align_corners=False)[0,0]
                    binary=ops.crop_mask(z[None].clone(),ownbox)>0
                    mask=ops.scale_masks(binary.byte()[None],orig)[0,0].byte().bool()
                    m=overlap_stats(mask,gt)
                    m['fg_bce']=float(F.softplus(-z[pos]).mean()) if pos.any() else None
                    m['bg_bce']=float(F.softplus(z[neg]).mean()) if neg.any() else None
                    m['crop_bce']=float(F.binary_cross_entropy_with_logits(z[support],target[support].float()))
                    result['metrics'][name]=m
                assert abs(result['metrics']['baseline']['iou']-r['readout_probe']['baseline']['iou'])<2e-6
                records.write(json.dumps(result,allow_nan=False)+'\n');count+=1
        records.flush()
        progress={'images':number+1,'total':len(image_ids),'evaluated_instances':count,'elapsed_s':round(time.monotonic()-start,2)}
        dump(a.out/'progress.json',progress)
        if number==0 or (number+1)%10==0:print(json.dumps(progress),flush=True)
        del outputs,fgrads,bgrads,directions,pred,base_coeff,pf
    records.close();image_records.close()
    dump(a.out/'COMPLETE.json',progress)
    print(json.dumps(progress),flush=True)


if __name__=='__main__':
    main()
