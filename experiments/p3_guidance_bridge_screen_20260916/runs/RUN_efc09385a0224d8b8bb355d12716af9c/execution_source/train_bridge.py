"""Four-arm short screen: teacher view controls selection, GT remains regression target."""
import argparse, copy, csv, json, math, os, random, shutil
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel, torch_safe_load
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model
from recording import atomic_json, now

ARM='baseline'; TEACHER=None; WEIGHT=.5; OUT=None
ORIGINAL_LOSS=SegmentationModel.loss
STATS=dict(batches=0,selected=0,eligible_targets=0,selected_positions=0,
           eligible_positions=0,student_iou_sum=0.,teacher_iou_sum=0.,aux_sum=0.,
           selected_teacher_better=0,selected_teacher_cross50=0,
           paired_view_targets=0,paired_view_position_changed=0,paired_view_selected_iou_delta_sum=0.)
CHECKED=False

def official_config(weight):
    ckpt,_=torch_safe_load(str(weight)); source=ckpt['train_args']
    keys='imgsz optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr nbs amp deterministic rect cos_lr close_mosaic multi_scale semseg_loss overlap_mask mask_ratio dropout box cls dfl o2m muon_w sgd_w cls_w stride_ratio topk hungarian hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr mosaic mixup cutmix copy_paste copy_paste_mode auto_augment erasing'.split()
    return {k:source[k] for k in keys if k in source and k in DEFAULT_CFG_DICT}

def branch(preds):
    if isinstance(preds,tuple):preds=preds[1]
    return preds['one2many'] if 'one2many' in preds else preds

def iou(boxes,gt):
    inter=(torch.minimum(boxes[:,2:],gt[2:])-torch.maximum(boxes[:,:2],gt[:2])).clamp_min(0).prod(1)
    return inter/((boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)+(gt[2:]-gt[:2]).clamp_min(0).prod()-inter).clamp_min(1e-9)

def select_and_edit(batch,edit):
    """The same GT-only selection in B/C/D; edit changes teacher pixels only."""
    image=batch['img']; masks=batch['masks'].to(image.device)
    if masks.ndim==4:masks=masks[:,0]
    if tuple(masks.shape[-2:])!=tuple(image.shape[-2:]):
        masks=F.interpolate(masks[:,None].float(),image.shape[-2:],mode='nearest')[:,0]
    masks=masks.long(); edited=image.detach().clone() if edit else image
    selected=[]
    for bi in range(len(image)):
        gids=(batch['batch_idx'].view(-1).long()==bi).nonzero().view(-1)
        candidates=[]
        for lid,gi in enumerate(gids.tolist(),1):
            box=batch['bboxes'][gi]
            if float(box[2]*box[3])>.01:continue
            own=masks[bi]==lid; area=int(own.sum())
            if area<4:continue
            radius=max(4,min(16,int(round(area**.5*.20))))
            dilated=F.max_pool2d(own[None,None].float(),2*radius+1,1,radius)[0,0].bool()
            neighbor=dilated&(masks[bi]!=0)&~own; ring=dilated&(masks[bi]==0)
            if not ring.any():continue
            fg=image[bi,:,own].mean(1); bg=image[bi,:,ring].mean(1)
            score=float(neighbor.sum())/area+1/(float((fg-bg).abs().mean())+.05)
            candidates.append((score,gi,own,neighbor,fg,bg))
        if not candidates:continue
        _,gi,own,neighbor,fg,bg=max(candidates,key=lambda x:x[0]); selected.append(gi)
        if edit:
            if neighbor.any():edited[bi,:,neighbor]=bg[:,None]
            sign=1. if float(fg.mean())>=float(bg.mean()) else -1.
            if abs(float(fg.mean()-bg.mean()))<.01:sign=1. if float(fg.mean())<.5 else -1.
            edited[bi,:,own]=(edited[bi,:,own]+sign*.11).clamp(0,1)
    return selected,edited

def auxiliary(self,preds,batch,assigned):
    raw=branch(preds); crit=self.criterion.one2many
    fg,owners,_,points,strides=assigned
    boxes=crit.bbox_decode(points,raw['boxes'].permute(0,2,1).contiguous())*strides
    h,w=batch['img'].shape[-2:]
    gt=ops.xywh2xyxy(batch['bboxes'].to(boxes.device))*boxes.new_tensor([w,h,w,h])
    p3=raw['feats'][0].shape[-2]*raw['feats'][0].shape[-1]
    teacher=batch.get('_teacher_boxes'); terms=[]; traces=[]
    for gi in batch['_selected']:
        bi=int(batch['batch_idx'][gi]); ids=(batch['batch_idx'].view(-1).long()==bi).nonzero().view(-1)
        local=int((ids==gi).nonzero().item())
        candidates=(fg[bi,:p3]&(owners[bi,:p3]==local)).nonzero().view(-1)
        if not len(candidates):continue
        siou=iou(boxes[bi,candidates].detach(),gt[gi]); keep=siou<.5
        candidates=candidates[keep]; siou=siou[keep]
        if not len(candidates):continue
        if ARM=='hard_gt':index=int(siou.argmin()); tq=None
        else:
            tq=iou(teacher[bi,candidates],gt[gi])
            cls=int(batch['cls'][gi]); score=batch['_teacher_scores'][bi,candidates,cls]
            index=int((tq+.02*score).argmax())
        pos=int(candidates[index]); assert bool(fg[bi,pos]) and int(owners[bi,pos])==local
        if '_original_teacher_boxes' in batch:
            oq=iou(batch['_original_teacher_boxes'][bi,candidates],gt[gi])
            oscore=batch['_original_teacher_scores'][bi,candidates,int(batch['cls'][gi])]
            opos=int((oq+.02*oscore).argmax())
            STATS['paired_view_targets']+=1
            STATS['paired_view_position_changed']+=int(opos!=index)
            STATS['paired_view_selected_iou_delta_sum']+=float(tq[index]-oq[opos])
        wh=(gt[gi,2:]-gt[gi,:2]).clamp_min(8.)
        error=(boxes[bi,pos]-gt[gi])/torch.cat((wh,wh))
        terms.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=.1,reduction='mean'))
        STATS['eligible_targets']+=1; STATS['selected_positions']+=1;STATS['eligible_positions']+=len(candidates)
        STATS['student_iou_sum']+=float(siou[index])
        if tq is not None:
            STATS['teacher_iou_sum']+=float(tq[index])
            STATS['selected_teacher_better']+=int(tq[index]>siou[index])
            STATS['selected_teacher_cross50']+=int(tq[index]>=.5)
        if len(traces)<4:traces.append(dict(image=bi,global_gt=gi,local_gt=local,position=pos,
            candidate_count=len(candidates),student_iou=float(siou[index]),teacher_iou=None if tq is None else float(tq[index])))
    aux=torch.stack(terms).mean() if terms else boxes.sum()*0
    if not torch.isfinite(aux):raise FloatingPointError('Nonfinite auxiliary loss')
    STATS['batches']+=1;STATS['selected']+=len(batch['_selected']);STATS['aux_sum']+=float(aux.detach())
    if traces and not (OUT/'first_active_positions.json').exists():atomic_json(OUT/'first_active_positions.json',traces)
    return aux

def bridge_loss(self,batch,preds=None):
    global CHECKED
    if ARM=='baseline' or '_selected' not in batch:return ORIGINAL_LOSS(self,batch,preds)
    if getattr(self,'criterion',None) is None:self.criterion=self.init_criterion()
    if preds is None:preds=self.forward(batch['img'])
    crit=self.criterion.one2many; original=crit.get_assigned_targets_and_loss; captured=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);captured.append(result[0]);return result
    crit.get_assigned_targets_and_loss=capture
    try:base,items=self.criterion(preds,batch)
    finally:crit.get_assigned_targets_and_loss=original
    assert len(captured)==1,'Native O2M assignment must execute exactly once'
    # Same forward tensors, official full loss, no custom matching and no BN second forward.
    if not CHECKED:
        ref,refitems=self.criterion(preds,batch)
        torch.testing.assert_close(base,ref,rtol=0,atol=0)
        torch.testing.assert_close(items,refitems,rtol=0,atol=0)
        atomic_json(OUT/'native_loss_check.json',dict(native_loss_equal=True,assignment_calls=1,
                    supervision='official own-GT P3 positives only',checked_at=now()))
        CHECKED=True
    aux=auxiliary(self,preds,batch,captured[0]); augmented=base.clone()
    weight=WEIGHT*float(self.criterion.o2m)/float(self.criterion.o2m_copy)
    augmented[0]=augmented[0]+weight*aux*batch['img'].shape[0]
    return augmented,items

class BridgeTrainer(SegmentationTrainer):
    def _setup_train(self):
        global TEACHER
        super()._setup_train()
        if ARM.startswith('teacher_'):
            TEACHER=copy.deepcopy(unwrap_model(self.model)).to(self.device).requires_grad_(False).train()
            for mod in TEACHER.modules():
                if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        if ARM=='baseline':return batch
        selected,view=select_and_edit(batch,ARM=='teacher_edited');batch['_selected']=selected
        if ARM.startswith('teacher_'):
            model=unwrap_model(self.model)
            if getattr(model,'criterion',None) is None:model.criterion=model.init_criterion()
            crit=model.criterion.one2many
            with torch.no_grad(),torch.amp.autocast(device_type=self.device.type,enabled=bool(self.amp)):
                raw=branch(TEACHER(view));points,strides=make_anchors(raw['feats'],crit.stride,.5)
                batch['_teacher_boxes']=(crit.bbox_decode(points,raw['boxes'].permute(0,2,1).contiguous())*strides).detach()
                batch['_teacher_scores']=raw['scores'].permute(0,2,1).sigmoid().detach()
                if ARM=='teacher_edited' and STATS['batches']<32:
                    original=branch(TEACHER(batch['img']))
                    batch['_original_teacher_boxes']=(crit.bbox_decode(points,original['boxes'].permute(0,2,1).contiguous())*strides).detach()
                    batch['_original_teacher_scores']=original['scores'].permute(0,2,1).sigmoid().detach()
        return batch

def main():
    global ARM,WEIGHT,OUT
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=['baseline','hard_gt','teacher_original','teacher_edited'],required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());ARM=a.arm;WEIGHT=cfg['aux_weight'];OUT=a.out
    assert ultralytics.__version__==cfg['ultralytics'];assert not (a.out/'args.yaml').exists()
    shutil.copyfile(cfg['amp_cache'],Path.cwd()/'yolo26n.pt')
    settings.update({k:False for k in ['wandb','comet','mlflow','clearml','neptune'] if k in settings})
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    inherited=official_config(cfg['weight']);assert inherited['mask_ratio']==1 and inherited['overlap_mask']
    args=dict(inherited);args.update(data=str(a.study/'data/train.yaml'),epochs=cfg['epochs'],batch=cfg['batch'],
        workers=cfg['workers'],device=0,project=str(a.out.parent),name=a.out.name,exist_ok=True,seed=cfg['seed'],
        save=True,save_period=1,val=True,plots=False,cache=False,patience=0,verbose=True,resume=False)
    atomic_json(a.out/'design.json',dict(arm=ARM,training_config=args,source_checkpoint_training_config=inherited,
        teacher='frozen initial weights' if ARM.startswith('teacher') else None,
        target='GT box; no extra classification loss',pool='official own-GT O2M P3 positives with current IoU < .5',
        lambda0=WEIGHT,quality_gate='none; teacher ranks same eligible pool',checkpoint='last primary'))
    SegmentationModel.loss=bridge_loss
    model=YOLO(cfg['weight']);epochs=[];previous=dict(STATS)
    def epoch_end(trainer):
        nonlocal previous
        total=dict(STATS);delta={k:total[k]-previous[k] for k in total};previous=total
        row=dict(epoch=trainer.epoch+1,at=now(),auxiliary=delta);epochs.append(row)
        atomic_json(a.out/'auxiliary_epochs.json',epochs);print('AUX_EPOCH',json.dumps(row),flush=True)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.train(trainer=BridgeTrainer,**args)
    with (a.out/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==cfg['epochs']
    bad=[dict(epoch=row.get('epoch'),field=k,value=v) for row in rows for k,v in row.items() if k and v and not math.isfinite(float(v))]
    assert not [x for x in bad if not x['field'].startswith('val/')],bad
    saved=YOLO(str(a.out/'weights/last.pt')).model
    assert all(torch.isfinite(v).all() for v in saved.state_dict().values()),'Nonfinite saved model'
    if ARM!='baseline':assert STATS['selected_positions']>0,'No eligible auxiliary training exposures'
    atomic_json(a.out/'TRAINING_COMPLETE.json',dict(status='complete',arm=ARM,epochs=len(rows),training_losses_finite=True,
        weights_finite=True,native_validation_nonfinite=bad,
        method_stats=STATS,finished_at=now(),checkpoint='last',source_weight=cfg['weight'],version=ultralytics.__version__))
if __name__=='__main__':main()
