"""Real-image label support and fixed-readout gradient probes. No network training."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import polygons2masks_overlap
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from pycocotools.coco import COCO
from prepare import key


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class IdentityFormat(Format):
    def _format_segments(self, instances, cls, w, h):
        _, self.order = polygons2masks_overlap((h, w), instances.segments, self.mask_ratio)
        return super()._format_segments(instances, cls, w, h)


def overlap_stats(mask, gt):
    tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'iou': tp/max(tp+fp+fn, 1),
            'coverage': tp/max(tp+fn, 1), 'purity': tp/max(tp+fp, 1)}


def summary(rows):
    results = {}
    for state in ['all', 'joint_good', 'box_good_mask_unavailable', 'separate_good_no_joint',
                  'mask_good_box_unavailable', 'neither_good']:
        rs = [r for r in rows if state=='all' or r['geometry_state']==state]
        labelled = [r for r in rs if 'training_label' in r]
        witnessed = [r for r in rs if 'readout_probe' in r]
        failed = [r for r in witnessed if r['readout_probe']['baseline']['iou'] < .75]
        result = {'gt': len(rs), 'labelled_gt': len(labelled), 'box75_witnesses': len(witnessed),
                  'failed_box75_witnesses': len(failed)}
        if labelled:
            result['nearest_instance_vanished'] = sum(r['training_label']['nearest_cells']==0 for r in labelled)
            result['mean_semantic_coverage'] = float(np.mean([r['training_label']['semantic_self_coverage'] for r in labelled]))
        if witnessed:
            result['oracle_repaired'] = {name:sum(r['readout_probe']['oracles'][name]>=.75 for r in failed)
                for name in ['remove_fp_inside_training_crop', 'remove_fp_outside_training_crop',
                             'remove_all_fp', 'fill_fn_within_predicted_support']}
            result['coefficient_gradient_negative_dot'] = sum(r['readout_probe']['fg_bg_cosine'] < 0 for r in witnessed)
            result['coefficient_gradient_negativity_denominator'] = len(witnessed)
            arms = sorted(set(k for r in witnessed for k in r['readout_probe']['arms']))
            result['arms'] = {}
            for arm in arms:
                ss = [r['readout_probe'] for r in witnessed if arm in r['readout_probe']['arms']]
                result['arms'][arm] = {
                    'n': len(ss), 'mean_delta_iou_pp': float(np.mean([v['arms'][arm]['iou']-v['baseline']['iou'] for v in ss])*100),
                    'repaired75': sum(v['baseline']['iou']<.75<=v['arms'][arm]['iou'] for v in ss),
                    'damaged75': sum(v['arms'][arm]['iou']<.75<=v['baseline']['iou'] for v in ss)}
        results[state] = result
    return {'groups': results, 'limitations': [
        'Exploratory train2017 diagnostics; confirmation cohort has not been inspected.',
        'GT-assisted candidate selection and per-instance coefficient steps are oracle probes, not deployable predictions or AP.',
        'Nearest label support and coefficient gradients alone do not establish causal shared-feature training conflict.',
        'No augmentation or optimizer update; current assignment replay is not historical pretrained assignment.',
        'Summary is descriptive; matched controls and image-cluster inference are required before mechanism claims.']}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    for name in ['root', 'raw', 'out']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--limit', type=int, default=500)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    coco = COCO(str(a.root/'assets/datasets/coco/annotations/instances_train2017.json'))
    ids = json.loads((a.root/'explore_ids.json').read_text())[:a.limit]
    ledger = json.loads((a.root/'LABEL_LEDGER.json').read_text())
    yolo = YOLO(str(a.root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'))
    model = yolo.model.cuda().float().eval()
    model.args = SimpleNamespace(**yolo.ckpt['train_args'])
    assert model.args.mask_ratio==1 and model.args.overlap_mask
    criteria = {'one2many': v8SegmentationLoss(model, tal_topk=10),
                'one2one': v8SegmentationLoss(model, tal_topk=7, tal_topk2=1)}
    data = YOLODataset(img_path=str(a.root/'explore_images.txt'), data={'names': model.names, 'nc':80},
                       task='segment', imgsz=640, batch_size=1, augment=False, hyp=model.args,
                       cache=False, rect=False, stride=32, prefix='mechanism: ')
    by_id = {int(Path(v['im_file']).stem): i for i,v in enumerate(data.labels)}
    letterbox = LetterBox((640,640), auto=True, stride=32)
    formatter = IdentityFormat(return_mask=True, mask_ratio=1, mask_overlap=True, bgr=0.0)
    model.model[-1].training = True
    model.model[-1].proto.training = True
    rows=[]; start=time.monotonic()
    record = (a.out/'instances.jsonl').open('w', encoding='utf-8')
    for number, image_id in enumerate(ids):
        cached = json.loads((a.raw/'images'/f'{image_id:012d}.json').read_text())
        source_rows = {v['annotation_id']:v for v in cached['gt_rows']}
        label = deepcopy(data.labels[by_id[image_id]])
        binding = {v['key']:v['annotation_id'] for v in ledger[str(image_id)]}
        ann_ids = [binding[key(cls[0], segment)] for cls,segment in zip(label['cls'], label['segments'])]
        im = cv2.imread(str(a.root/'assets/datasets/coco/images/train2017'/coco.imgs[image_id]['file_name']))
        orig = im.shape[:2]
        label['img'] = im; label.pop('shape', None)
        label = data.update_labels_info(label)
        label = letterbox(label)
        batch = formatter(label)
        ordered = [ann_ids[k] for k in formatter.order] if ann_ids else []
        for k in ['cls','bboxes','batch_idx','masks','sem_masks']:
            batch[k] = batch[k].cuda()
        batch['img'] = batch['img'].cuda().float()[None]/255
        batch['sem_masks'] = batch['sem_masks'][None]
        shape=tuple(batch['img'].shape[-2:]); hh,ww=shape
        p=model(batch['img'])
        assignments={}; assignment_maps={}
        for branch in criteria:
            assigned=criteria[branch].get_assigned_targets_and_loss(p[branch],batch)[0]
            fg,owner=assigned[:2]
            assignments[branch]=[(owner[0][fg[0]]==i).sum().item() for i in range(len(ordered))]
            assignment_maps[branch]=(fg[0],owner[0])
        proto=p['one2one']['proto'][0][0]
        # Proto26 auxiliary semantic output is at P3 resolution.
        semantic_shape=p['one2one']['proto'][1].shape[-2:]
        canvas=batch['masks'][0].long()
        nearest=F.interpolate(canvas[None,None].float(),semantic_shape,mode='nearest')[0,0].long()
        semantic=batch['sem_masks'][0].long()
        present=canvas!=0
        nearest_sem=F.interpolate(semantic[None,None].float(),semantic_shape,mode='nearest')
        nearest_pres=F.interpolate(present[None,None].float(),semantic_shape,mode='nearest')
        sem_back=F.interpolate(nearest_sem,shape,mode='nearest')[0,0].long()
        pres_back=F.interpolate(nearest_pres,shape,mode='nearest')[0,0].bool()
        mapping={ann:i for i,ann in enumerate(ordered)}
        with np.load(a.raw/'images'/f'{image_id:012d}.npz') as cache:
            assert tuple(cache['input_shape'])==shape
            coefficients=torch.from_numpy(cache['coefficients']).cuda().T
            raw_proto=torch.from_numpy(cache['proto']).cuda()
            assert torch.allclose(proto,raw_proto,atol=2e-5,rtol=2e-5), 'Training-view preprocessing changed raw input/proto'
            boxes=torch.from_numpy(cache['boxes_input']).cuda()
            all_low=(coefficients@raw_proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            for ann_id,row in source_rows.items():
                ann=coco.anns[ann_id]
                res={'image_id':image_id,'annotation_id':ann_id,'category_id':ann['category_id'],
                     'area':ann['area'],'geometry_state':row['geometry_state'],
                     'box_max':row['box_max'],'mask_max':row['mask_max']}
                if ann_id not in mapping:
                    res['training_label_missing']=True
                    rows.append(res);record.write(json.dumps(res)+'\n');continue
                gi=mapping[ann_id]
                gt_train=canvas==gi+1
                area=int(gt_train.sum())
                box=ops.xywh2xyxy(batch['bboxes'][gi:gi+1])*torch.tensor([ww,hh,ww,hh],device='cuda')
                train_support=ops.crop_mask(torch.ones((1,hh,ww),device='cuda'),box)[0].bool()
                cls=int(batch['cls'][gi,0])
                sem_covered=gt_train & pres_back & (sem_back==cls)
                nearest_cells=int((nearest==gi+1).sum())
                res['training_label']={'area':area,'bbox':box[0].tolist(),
                    'fill_fraction':int((gt_train & train_support).sum())/max(int(train_support.sum()),1),
                    'nearest_cells':nearest_cells,'semantic_self_coverage':int(sem_covered.sum())/max(area,1),
                    'one2many_positives':assignments['one2many'][gi],
                    'one2one_positives':assignments['one2one'][gi]}
                witness=row['best_mask_with_box75']
                if witness is not None and area>0 and train_support.any():
                    ri=witness['raw_id']; c=coefficients[ri]
                    z=F.interpolate(all_low[ri][None,None],shape,mode='bilinear',align_corners=False)[0,0]
                    bb=boxes[ri:ri+1]
                    def decode(logit):
                        binary=ops.crop_mask(logit[None].clone(),bb)>0
                        return ops.scale_masks(binary.byte()[None],orig)[0,0].byte().bool()
                    gt=torch.from_numpy(coco.annToMask(ann).astype(bool)).cuda()
                    mask=decode(z)
                    base=overlap_stats(mask,gt)
                    assert abs(base['iou']-witness['mask_iou'])<2e-6
                    ts=ops.scale_masks(train_support[None,None].byte(),orig)[0,0].byte().bool()
                    pred_support=ops.crop_mask(torch.ones((1,*shape),device='cuda'),bb)
                    ps=ops.scale_masks(pred_support[None],orig)[0,0].byte().bool()
                    fp=mask & ~gt
                    oracles={
                        'remove_fp_inside_training_crop':overlap_stats(mask & ~(fp & ts),gt)['iou'],
                        'remove_fp_outside_training_crop':overlap_stats(mask & ~(fp & ~ts),gt)['iou'],
                        'remove_all_fp':overlap_stats(mask & gt,gt)['iou'],
                        'fill_fn_within_predicted_support':overlap_stats(mask | (gt & ps),gt)['iou']}
                    # Exact coefficient gradients for BCE on the actual training crop.
                    # Upsampling a prototype is linear; no model parameter changes here.
                    pf=F.interpolate(raw_proto[None],shape,mode='bilinear',align_corners=False)[0].flatten(1)
                    fg=(gt_train & train_support).flatten(); bg=(~gt_train & train_support).flatten()
                    probability=z.sigmoid().flatten()
                    denom=max(int(train_support.sum()),1)
                    gf=(pf[:,fg]@(probability[fg]-1))/denom
                    gb=(pf[:,bg]@probability[bg])/denom
                    dot=float(gf@gb); cosine=dot/max(float(gf.norm()*gb.norm()),1e-20)
                    generator=torch.Generator(device='cuda').manual_seed(ann_id)
                    directions={'joint':gf+gb,'foreground_only':gf,'background_only':gb,
                        'balanced':gf*denom/max(int(fg.sum()),1)+gb*denom/max(int(bg.sum()),1),
                        'random':torch.randn(c.shape,generator=generator,device='cuda')}
                    arms={}
                    for name,direction in directions.items():
                        dz=(direction@pf).reshape(shape)
                        norm=dz[train_support].square().mean().sqrt() if train_support.any() else torch.tensor(0.,device='cuda')
                        if norm<1e-10:continue
                        for step in (.1,.3):
                            changed=z-step*dz/norm
                            metrics=overlap_stats(decode(changed),gt)
                            metrics['training_bce']=float(F.binary_cross_entropy_with_logits(changed[train_support],gt_train[train_support].float()))
                            arms[name+':'+str(step)]=metrics
                    res['readout_probe']={'raw_id':ri,'box_iou':witness['box_iou'],'baseline':base,
                        'actual_assignment_owner':{branch:(ordered[int(owner[ri])] if bool(fg[ri]) else None)
                            for branch,(fg,owner) in assignment_maps.items()},
                        'training_bce':float(F.binary_cross_entropy_with_logits(z[train_support],gt_train[train_support].float())),
                        'fp_inside_training_crop':int((fp&ts).sum()),'fp_outside_training_crop':int((fp&~ts).sum()),
                        'oracles':oracles,'fg_bg_cosine':cosine,'fg_norm':float(gf.norm()),'bg_norm':float(gb.norm()),
                        'joint_dot_fg':float((gf+gb)@gf),'joint_dot_bg':float((gf+gb)@gb),'arms':arms}
                    del pf
                rows.append(res);record.write(json.dumps(res,allow_nan=False)+'\n')
        record.flush()
        progress={'images':number+1,'total':len(ids),'gt':len(rows),'elapsed_s':round(time.monotonic()-start,2)}
        dump(a.out/'progress.json',progress)
        if (number+1)%10==0 or number==0:print(json.dumps(progress),flush=True)
    record.close()
    dump(a.out/'SUMMARY.json',summary(rows))
    dump(a.out/'COMPLETE.json',progress)
    print(json.dumps(progress),flush=True)


if __name__ == '__main__':
    main()
