"""Frozen-checkpoint audit of the executed gate, teacher edit and official ownership.

No training. Official augmentation recipe, but Mosaic source sampling is uniform
over the full training pool (not a short diagnostic loader's rolling buffer).
Gradients refer to detection-head outputs, not model-parameter gradients.
"""
from __future__ import annotations
import argparse, csv, json, random, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.augment import Mosaic, MixUp
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors
import method_source as ms

EVENTS = {}
for transform, name in [(Mosaic, 'mosaic'), (MixUp, 'mixup')]:
    original = transform.apply_image
    def tracked(self, labels, params=None, _original=original, _name=name):
        EVENTS[_name] = EVENTS.get(_name, 0) + 1
        return _original(self, labels, params)
    transform.apply_image = tracked
# Controlled replay panel: avoid repeated copies from a nearly empty buffer.
Mosaic.get_indexes = lambda self: [random.randrange(len(self.dataset)) for _ in range(self.n-1)]

def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def unpack(model, raw):
    raw=ms.branch(raw)
    criterion=model.criterion.one2many
    points,strides=make_anchors(raw['feats'],criterion.stride,.5)
    boxes=criterion.bbox_decode(points,raw['boxes'].permute(0,2,1))*strides
    return raw,boxes,points,strides

def cosine(a,b):
    denom=float(a.norm()*b.norm())
    return float(torch.dot(a.flatten(),b.flatten())/denom) if denom>1e-16 else None

def target_attributes(batch,gi):
    bi=int(batch['batch_idx'][gi]);ids=(batch['batch_idx'].view(-1).long()==bi).nonzero().flatten().tolist()
    own=batch['masks'][bi].long()==ids.index(gi)+1
    area=int(own.sum());radius=max(4,min(16,int(round(area**.5*.20))))
    region=F.max_pool2d(own[None,None].float(),2*radius+1,1,radius)[0,0].bool()
    neighbor=region&(batch['masks'][bi]!=0)&~own;bg=region&(batch['masks'][bi]==0)
    fgmean=batch['img'][bi,:,own].mean(1);bgmean=batch['img'][bi,:,bg].mean(1)
    contrast=float((fgmean-bgmean).abs().mean())
    return dict(target_area=area,neighbor_pixels=int(neighbor.sum()),neighbor_fraction=float(neighbor.sum())/max(area,1),contrast=contrast,vulnerability=float(neighbor.sum())/max(area,1)+1/(contrast+.05),n_augmented_instances=len(ids))

def audit_student(arm,model,batch,teacher_original,teacher_edited,attributes,draws):
    # Training mode produces the exact raw heads and uses per-batch BN statistics.
    # Restore all buffers afterwards; no checkpoint or model parameters are updated.
    buffers={n:b.clone() for n,b in model.named_buffers()}
    with torch.no_grad(),torch.amp.autocast('cuda',enabled=True):out=model(batch['img'])
    with torch.no_grad():
        for n,b in model.named_buffers():b.copy_(buffers[n])
    raw=ms.branch(out)
    raw={**raw,'boxes':raw['boxes'].detach().float().requires_grad_(True),'scores':raw['scores'].detach().float().requires_grad_(True)}
    criterion=model.criterion.one2many
    (fg,assigned,_,points,strides),loss,_=criterion.get_assigned_targets_and_loss(raw,batch)
    boxes=criterion.bbox_decode(points,raw['boxes'].permute(0,2,1))*strides
    bsz=boxes.shape[0];h,w=batch['img'].shape[-2:];p3=raw['feats'][0].shape[-2]*raw['feats'][0].shape[-1]
    gtboxes=ops.xywh2xyxy(batch['bboxes'])*torch.tensor([w,h,w,h],device=boxes.device)
    leaves=[raw['boxes'],raw['scores']]
    native_grads=torch.autograd.grad(loss.sum()*bsz*.8,leaves,retain_graph=True,allow_unused=True)
    records=[];terms=[];gated_records=[]
    for gi,pos,tiou in batch['_cf_teacher_guidance']:
        bi=int(batch['batch_idx'][gi]);cls=int(batch['cls'][gi]);gt=gtboxes[gi]
        ids=(batch['batch_idx'].view(-1).long()==bi).nonzero().flatten().tolist();local=ids.index(gi)
        ious=ms.pairwise_iou(boxes[bi].detach(),gt);siou=float(ious[pos]);gate=siou<.5 and tiou>=.5 and tiou>=siou+.1
        oiou=ms.pairwise_iou(teacher_original[bi],gt);eiou=ms.pairwise_iou(teacher_edited[bi],gt)
        assert abs(float(eiou[pos])-tiou)<.005
        if not bool(fg[bi,pos]):owner='background';owner_global=-1
        else:
            owner_global=ids[int(assigned[bi,pos])]
            owner='self' if owner_global==gi else ('other_same_class' if int(batch['cls'][owner_global])==cls else 'other_different_class')
        selected_class=raw['scores'][bi].detach().argmax(0)==cls
        same_best=float(ious[selected_class].max()) if selected_class.any() else 0.
        center=points[pos]*strides[pos]
        row={**draws[bi],**attributes[gi],'arm':arm,'global_gt_index':gi,'local_gt_index':local,'class_index':cls,'position':pos,'gate':int(gate),'owner':owner,'owner_global_gt_index':owner_global,'anchor_inside_gt':int(bool(((center>=gt[:2])&(center<=gt[2:])).all())),
             'student_same_position_iou':siou,'student_best_p3_iou':float(ious[:p3].max()),'student_best_all_iou':float(ious.max()),'student_best_same_class_iou':same_best,
             'teacher_edited_position_iou':tiou,'teacher_original_position_iou':float(oiou[pos]),'teacher_edit_delta':tiou-float(oiou[pos]),'teacher_original_best_p3_iou':float(oiou[:p3].max()),'teacher_edited_best_p3_iou':float(eiou[:p3].max()),
             'edit_needed_to_cross50':int(float(oiou[pos])<.5<=tiou),'native_box_grad_norm':None,'aux_box_grad_norm':None,'box_grad_cosine':None,'native_class_grad':None,'aux_class_grad':None,'class_grad_conflict':None}
        records.append(row)
        if gate:
            wh=(gt[2:]-gt[:2]).clamp_min(8);error=(boxes[bi,pos]-gt)/torch.cat([wh,wh])
            term=F.smooth_l1_loss(error,torch.zeros_like(error),beta=.1,reduction='mean')+.05*F.binary_cross_entropy_with_logits(raw['scores'][bi,cls,pos],torch.ones_like(raw['scores'][bi,cls,pos]))
            terms.append(term);gated_records.append((row,bi,cls,pos))
    if terms:
        aux=torch.stack(terms).mean()*bsz*.5
        aux_grads=torch.autograd.grad(aux,leaves)
        for row,bi,cls,pos in gated_records:
            a=native_grads[0][bi,:,pos];b=aux_grads[0][bi,:,pos]
            nc=float(native_grads[1][bi,cls,pos]);ac=float(aux_grads[1][bi,cls,pos])
            row.update(native_box_grad_norm=float(a.norm()),aux_box_grad_norm=float(b.norm()),box_grad_cosine=cosine(a,b),native_class_grad=nc,aux_class_grad=ac,class_grad_conflict=int(nc*ac<0))
        # Same outputs and guidance must reproduce the actual executed auxiliary.
        replay=ms.counterfactual_aux(model,{'one2many':raw},batch)*bsz*.5
        assert torch.allclose(aux.detach(),replay.detach(),atol=1e-6,rtol=1e-5)
    return records

def main():
    p=argparse.ArgumentParser()
    for key in ('data','source','baseline','method','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--full-count',type=int,default=256);p.add_argument('--pilot-count',type=int,default=128)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    seed=20260915;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    cfg=get_cfg(overrides={**ms.official_weight_config(args.source),'task':'segment','data':str(args.data),'batch':2,'workers':0,'device':0,'cache':False,'plots':False,'seed':seed})
    assert cfg.mask_ratio==1 and cfg.overlap_mask
    data=check_det_dataset(str(args.data));dataset=build_yolo_dataset(cfg,data['train'],batch=2,data=data,mode='train',stride=32)
    with args.pilot.open() as f:pilot_ids={int(r['image_id']) for r in csv.DictReader(f)}
    pilot_indices=[i for i,x in enumerate(dataset.im_files) if int(Path(x).stem) in pilot_ids]
    full_indices=[i for i,x in enumerate(dataset.im_files) if int(Path(x).stem) not in pilot_ids]
    rng=np.random.default_rng(seed)
    panel=[(int(i),'random_nonpilot') for i in rng.choice(full_indices,args.full_count,replace=False)]+[(int(i),'pilot_anchor') for i in rng.choice(pilot_indices,args.pilot_count,replace=False)]
    rng.shuffle(panel)
    anchors=[dict(draw=i,anchor_image_id=int(Path(dataset.im_files[idx]).stem),stratum=stratum) for i,(idx,stratum) in enumerate(panel)]
    write_csv(args.out/'anchor_panel.csv',anchors)
    models={}
    for arm,path in [('initial',args.source),('baseline',args.baseline),('method',args.method)]:
        model=YOLO(str(path)).model.to('cuda').float().train().requires_grad_(False);model.args=cfg;model.criterion=model.init_criterion();models[arm]=model
    teacher=copy.deepcopy(models['initial']);teacher.criterion=None
    for module in teacher.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    ms.METHOD_ENABLED=True;ms.CF_TEACHER=teacher
    trainer=object.__new__(ms.CounterfactualTrainer);trainer.args=cfg;trainer.device=torch.device('cuda:0');trainer.amp=True;trainer.model=models['initial'];trainer.stride=32
    captured={}
    def capture(module,inputs,output):captured['raw']=output
    handle=teacher.register_forward_hook(capture)
    rows=[];image_rows=[]
    for start in range(0,len(panel),2):
        items=[];draws=[]
        for k in range(start,min(start+2,len(panel))):
            EVENTS.clear();item=dataset[panel[k][0]];items.append(item);draws.append({**anchors[k],**{x+'_calls':EVENTS.get(x,0) for x in ('mosaic','mixup')}})
        batch=dataset.collate_fn(items);batch=trainer.preprocess_batch(batch)
        with torch.no_grad():_,edited_boxes,_,_=unpack(models['initial'],captured['raw'])
        edited_boxes=edited_boxes.detach().clone()
        with torch.inference_mode(),torch.amp.autocast('cuda',enabled=True):original_raw=teacher(batch['img']);_,original_boxes,_,_=unpack(models['initial'],original_raw)
        original_boxes=original_boxes.detach().clone()
        attrs={gi:target_attributes(batch,gi) for gi in batch['_cf_selected'] if gi>=0}
        for bi,draw in enumerate(draws):
            gi=batch['_cf_selected'][bi];image_rows.append({**draw,'selected':int(gi>=0),'n_augmented_instances':int((batch['batch_idx']==bi).sum())})
        for arm,model in models.items():rows.extend(audit_student(arm,model,batch,original_boxes,edited_boxes,attrs,draws))
        if (start+2)%16==0:
            write_csv(args.out/'gate_per_target.csv',rows);write_csv(args.out/'image_draws.csv',image_rows)
            print('AUDIT',start+2,'/',len(panel),'records',len(rows),flush=True)
    handle.remove();write_csv(args.out/'gate_per_target.csv',rows);write_csv(args.out/'image_draws.csv',image_rows)
    summaries=[]
    for arm in models:
        for stratum in ['all','random_nonpilot','pilot_anchor']:
            subset=[r for r in rows if r['arm']==arm and (stratum=='all' or r['stratum']==stratum)];gated=[r for r in subset if r['gate']]
            mean=lambda key:float(np.mean([r[key] for r in gated if r[key] is not None])) if any(r[key] is not None for r in gated) else None
            s=dict(arm=arm,stratum=stratum,n_selected=len(subset),n_gated=len(gated),owner_counts={k:sum(r['owner']==k for r in gated) for k in ['self','background','other_same_class','other_different_class']},already_has_any_box50=sum(r['student_best_all_iou']>=.5 for r in gated),already_has_same_class_box50=sum(r['student_best_same_class_iou']>=.5 for r in gated),already_has_p3_box50=sum(r['student_best_p3_iou']>=.5 for r in gated),edit_needed_to_cross50=sum(r['edit_needed_to_cross50'] for r in gated),no_neighbor=sum(r['neighbor_pixels']==0 for r in gated),edit_mean_delta=mean('teacher_edit_delta'),box_grad_cosine=mean('box_grad_cosine'),class_grad_conflict=sum(r['class_grad_conflict']==1 for r in gated))
            summaries.append(s)
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',anchor_images=len(panel),seed=seed,training=False,augmentation='official weight recipe; Mosaic sources uniformly sampled from all train2017, not rolling loader buffer',targets='augmented instances, not anchor original GT identities',scope='frozen checkpoints and matched batches; not historical gate frequency',gradient_scope='native one2many detection outputs versus auxiliary; excludes native mask loss',native_o2m_weight=.8,aux_weight=.5),indent=2))

if __name__=='__main__':main()
