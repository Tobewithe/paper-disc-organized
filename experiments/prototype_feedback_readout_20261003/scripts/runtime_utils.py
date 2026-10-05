"""Train native cv4 + optional box attention; fixed frozen-prefix supervision."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import ultralytics
from ultralytics.utils.loss import v8SegmentationLoss

from box_guided_head import CoefficientReadout


def dump(path, data):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load(path): return torch.load(path, map_location='cpu', weights_only=False)


def setup(seed):
    assert ultralytics.__version__ == '8.4.100'
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def read_image(cfg, iid):
    return load(Path(cfg['cache'])/'images'/f'{iid:012d}.pt')


def gpu_image(x):
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in x.items() if k != 'F'}


def feed(images):
    features = [torch.stack([x['F'][l] for x in images]).cuda() for l in range(3)]
    selected = [dict(raw_ids=x['raw_ids'].cuda(), boxes=x['boxes'].cuda()) for x in images]
    return features, selected


def loss_and_grad(x, coefficients, normalizer):
    """Exact chain rule, chunk only pixel workspace, never change loss weights.

    Compute dL/dc with independent coefficient leaves, then backpropagate that
    gradient through the native branch. Avoid retaining all640^2 pixel graphs.
    """
    proto = F.interpolate(x['proto'][None].float(), (640,640), mode='bilinear', align_corners=False)[0]
    gradient = torch.zeros_like(coefficients)
    total = 0.
    for lo in range(0, len(coefficients), 8):
        hi = min(lo+8, len(coefficients))
        leaf = coefficients[lo:hi].detach().requires_grad_(True)
        gt = (x['masks'][None] == (x['owners'][lo:hi]+1)[:,None,None]).float()
        box = x['target_boxes'][lo:hi]
        area = ((box[:,2:] - box[:,:2])/640).prod(1)
        loss = v8SegmentationLoss.single_mask_loss(gt,leaf,proto,box,area)*float(x['segmentation_gain'])
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite mask objective')
        gradient[lo:hi] = torch.autograd.grad(loss/normalizer, leaf)[0]
        total += float(loss.detach())
    return total, gradient


def optimizer_for(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        is_fusion = name.startswith('fusion.')
        decay = p.ndim > 1
        k=(is_fusion,decay)
        groups.setdefault(k, []).append(p)
    return torch.optim.AdamW([dict(params=params,
        lr=cfg['fusion_lr'] if fusion else cfg['branch_lr'],
        initial_lr=cfg['fusion_lr'] if fusion else cfg['branch_lr'],
        weight_decay=cfg['weight_decay'] if decay else 0.)
        for (fusion,decay),params in groups.items()], betas=(.9,.999), eps=1e-8)


def lr_factor(progress, cfg):
    warm=cfg['warmup_epochs']
    if progress < warm: return max(.01, progress/warm)
    phase=min(1., (progress-warm)/max(1,cfg['epochs']-warm))
    return cfg['eta_ratio']+(1-cfg['eta_ratio'])*.5*(1+math.cos(math.pi*phase))


def bn_state(model):
    return {n:b.detach().cpu().clone() for n,b in model.named_buffers()
            if n.endswith(('running_mean','running_var','num_batches_tracked'))}


def verify_bn(model, original):
    for n,b in model.named_buffers():
        if n in original: torch.testing.assert_close(b.cpu(),original[n],atol=0,rtol=0)
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def save_checkpoint(path, model, optimizer, epoch, mode, cfg, history):
    temp=Path(str(path)+'.tmp')
    torch.save(dict(mode=mode,epoch=epoch,state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),config=cfg,history=history,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
    temp.replace(path)


def smoke(cfg, out, index):
    original=YOLO(cfg['weights']).model.eval().requires_grad_(False)
    chosen=[r['image_id'] for r in index['fit'] if r['n']][:8]
    images=[read_image(cfg,i) for i in chosen]
    assert images, 'No positive images for smoke'
    report=dict(ultralytics_import=ultralytics.__version__, source=str(ultralytics.__file__),
        image_ids=chosen, arms={}, scope='code verification only; reset all model states before formal training')
    for mode in cfg['arms']:
        setup(cfg['seed']); model=CoefficientReadout(original.model[-1].one2one_cv4,mode).cuda().train()
        optimizer=optimizer_for(model,cfg); buffers=bn_state(model)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        first_max=0.; losses=[]; grads={}
        # Every cached smoke image participates; two passes let zero-initialized
        # final residual move before checking gradients in upstream attention.
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            total=sum(len(x['rows']) for x in images); total_loss=0.
            for lo in range(0,len(images),cfg['microbatch_images']):
                batch=images[lo:lo+cfg['microbatch_images']]
                fs,selected=feed(batch); cs=model(fs,selected); gs=[]
                for x,c in zip(batch,cs):
                    if step==0:
                        first_max=max(first_max,float((c.detach().cpu()-x['c0']).abs().max()))
                        torch.testing.assert_close(c.detach().cpu(),x['c0'],atol=2e-5,rtol=2e-5)
                    value,g=loss_and_grad(gpu_image(x),c,total);gs.append(g);total_loss+=value
                    # Independent ordinary autograd check once per arm.
                    if step==0 and lo==0:
                        gx=gpu_image(x); leaf=c.detach().requires_grad_(True)
                        proto=F.interpolate(gx['proto'][None],(640,640),mode='bilinear',align_corners=False)[0]
                        gt=(gx['masks'][None]==(gx['owners']+1)[:,None,None]).float()
                        area=((gx['target_boxes'][:,2:]-gx['target_boxes'][:,:2])/640).prod(1)
                        direct=v8SegmentationLoss.single_mask_loss(gt,leaf,proto,gx['target_boxes'],area)*gx['segmentation_gain']/total
                        torch.testing.assert_close(torch.autograd.grad(direct,leaf)[0],g,atol=3e-5,rtol=3e-5)
                torch.autograd.backward(cs,gs)
            for n,p in model.named_parameters():
                if p.grad is not None: grads[n]=max(grads.get(n,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step(); losses.append(total_loss/total);verify_bn(model,buffers)
        changed={n:float((p.detach().cpu()-before[n]).norm()) for n,p in model.named_parameters()}
        for level in range(3):
            assert any(v>0 for n,v in changed.items() if n.startswith(f'native_cv4.{level}.0.')), f'{mode}: first convolution {level} did not update'
        if mode!='N':
            assert any(v>0 for n,v in grads.items() if '.attention.in_proj_weight' in n), 'attention did not receive gradient after residual initialization'
        report['arms'][mode]=dict(initial_coefficient_max_error=first_max,
            parameter_counts=model.parameter_counts(),losses=losses,gradient_norms=grads,parameter_changes=changed,bn_unchanged=True)
        del optimizer,model;torch.cuda.empty_cache()
    assert report['arms']['P']['parameter_counts']==report['arms']['R']['parameter_counts']
    dump(out/'SMOKE.json',report);dump(out/'COMPLETE.json',dict(passed=True,kind='smoke',formal_training_steps=0))


def train(cfg,out,index,mode,deadline):
    setup(cfg['seed']); original=YOLO(cfg['weights']).model.eval().requires_grad_(False)
    model=CoefficientReadout(original.model[-1].one2one_cv4,mode).cuda().train();del original
    optimizer=optimizer_for(model,cfg);frozen_bn=bn_state(model)
    dump(out/'MODEL.json',dict(mode=mode,counts=model.parameter_counts(),
        trainable=model.trainable_parameter_names(),bn_statistics='frozen; affine trainable',
        actual_ultralytics=ultralytics.__version__,source=str(ultralytics.__file__)))
    items=[r for r in index['fit'] if r['n']>0]
    dump(out/'COHORT.json',dict(planned_images=len(index['fit']),effective_images=len(items),
        candidates=sum(r['n'] for r in items),no_positive_images=[r['image_id'] for r in index['fit'] if not r['n']]))
    start_epoch=0;history=[];last=out/'last.pt'
    if last.exists():
        ck=load(last); assert ck['mode']==mode and ck['config']==cfg
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.cuda()
        start_epoch=ck['epoch'];history=ck['history']
    started=time.monotonic()
    for epoch in range(start_epoch,cfg['epochs']):
        order=list(items);random.Random(cfg['seed']+epoch).shuffle(order)
        epoch_start=time.monotonic();total_loss=0.;seen=0
        groups=math.ceil(len(order)/cfg['effective_batch_images'])
        for step,lo in enumerate(range(0,len(order),cfg['effective_batch_images'])):
            if deadline and time.time()>deadline:
                dump(out/'BUDGET_STOP.json',dict(epoch_completed=epoch,mid_epoch_step=step,
                    note='No method conclusion; rerun resumes last completed epoch. Do not evaluate partial parameters as epoch15.'))
                raise TimeoutError('Shared training resource budget reached')
            rows=order[lo:lo+cfg['effective_batch_images']];denominator=sum(r['n'] for r in rows)
            factor=lr_factor(epoch+(step+1)/groups,cfg)
            for g in optimizer.param_groups:g['lr']=g['initial_lr']*factor
            optimizer.zero_grad(set_to_none=True)
            for j in range(0,len(rows),cfg['microbatch_images']):
                batch=[read_image(cfg,r['image_id']) for r in rows[j:j+cfg['microbatch_images']]]
                fs,selection=feed(batch);cs=model(fs,selection);gs=[]
                for x,c in zip(batch,cs):
                    value,g=loss_and_grad(gpu_image(x),c,denominator);gs.append(g)
                    total_loss+=value;seen+=len(x['rows'])
                torch.autograd.backward(cs,gs)
                del batch,fs,cs,gs
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True))
            optimizer.step()
            if step%25==0 or step+1==groups:
                state=dict(mode=mode,epoch=epoch+1,epochs=cfg['epochs'],step=step+1,steps=groups,
                    trajectory_bce=total_loss/max(1,seen),grad_norm_before_clip=grad,
                    elapsed_s=time.monotonic()-started,epoch_elapsed_s=time.monotonic()-epoch_start)
                dump(out/'PROGRESS.json',state);print(json.dumps(state),flush=True)
        verify_bn(model,frozen_bn)
        history.append(dict(epoch=epoch+1,trajectory_bce=total_loss/seen,candidates=seen,
            seconds=time.monotonic()-epoch_start))
        save_checkpoint(last,model,optimizer,epoch+1,mode,cfg,history)
        dump(out/'HISTORY.json',history)
    # Deliberately select only the fixed final epoch, never a favorable dev epoch.
    save_checkpoint(out/'final.pt',model,optimizer,cfg['epochs'],mode,cfg,history)
    dump(out/'COMPLETE.json',dict(completed=True,mode=mode,epochs=cfg['epochs'],elapsed_s=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--index');p.add_argument('--mode',choices=['N','P','R']);p.add_argument('--smoke',action='store_true')
    p.add_argument('--deadline',type=float,default=0.)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    setup(cfg['seed']);index=json.loads(Path(a.index or str(Path(cfg['cache'])/'INDEX.json')).read_text())
    if a.smoke:smoke(cfg,out,index)
    else:
        assert a.mode
        assert (Path(cfg['cache'])/'COMPLETE.json').exists(),'Full frozen cohort cache must complete first'
        train(cfg,out,index,a.mode,a.deadline)


if __name__=='__main__':main()
