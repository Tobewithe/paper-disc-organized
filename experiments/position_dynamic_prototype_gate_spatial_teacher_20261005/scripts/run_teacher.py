from __future__ import annotations
import argparse, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np

FAST = Path(os.environ.get('FAST_SCREEN_SCRIPTS', '/root/prototype_readout_fast_screen_20261003/scripts'))
sys.path.insert(0, str(FAST))
sys.path.insert(0, '/root/autodl-tmp/prototype_guided_evidence_selection_20261003/py')
import torch
import torch.nn.functional as F
from ultralytics.utils import ops
from pycocotools.coco import COCO
from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, tensor_sha
from runtime_utils import setup, bn_state, verify_bn, lr_factor
from dynamic_gate import DynamicPrototypeReadout
from evaluation_metrics import _padded_gt, _pixel_auc_fpr, _box_iou, _scale_binary

ARMS = ('G','P')
def cfg_load(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    cfg['server_root'] = str(Path(cfg['server_root']))
    cfg['cache'] = str(Path(cfg['cache']))
    cfg['assets'] = cfg['cache']
    return cfg

def load_oracle(path):
    oracle = torch.load(path, map_location='cpu', weights_only=False)
    delta = oracle['delta'].float()
    identities = oracle['identities']
    if len(delta) != len(identities):
        raise RuntimeError(f'oracle length mismatch: {len(delta)} vs {len(identities)}')
    result = {}
    for d, ident in zip(delta, identities):
        key = (int(ident['image_id']), int(ident['annotation_id']), int(ident['raw_id']), int(ident['pyramid_level']), int(ident['target_gt_idx']))
        if key in result:
            raise RuntimeError(f'duplicate oracle identity: {key}')
        result[key] = d
    return result

def oracle_delta_for_image(image, oracle, device):
    values = []
    for row in image['rows']:
        key = (int(row['image_id']), int(row['annotation_id']), int(row['raw_id']), int(row.get('pyramid_level', row.get('level'))), int(row.get('target_gt_idx', row.get('gt_index'))))
        if key not in oracle:
            raise RuntimeError(f'missing oracle identity: {key}')
        values.append(oracle[key])
    return torch.stack(values).to(device).float()

def teacher_target(image, oracle, device):
    delta = oracle_delta_for_image(image, oracle, device)
    proto = image['proto'].to(device).float()
    return (proto[None] * delta[:, :, None, None]).sum(1)

def teacher_loss(image, residual, target, tau):
    pred = F.interpolate(residual.float()[:, None], (640, 640), mode='bilinear', align_corners=False)[:, 0]
    tgt = F.interpolate(target.float()[:, None], (640, 640), mode='bilinear', align_corners=False)[:, 0]
    boxes = image['target_boxes'].to(residual.device).float()
    support = ops.crop_mask(torch.ones_like(pred), boxes).bool()
    pixel = F.huber_loss(pred, tgt, reduction='none', delta=float(tau))
    value = (pixel * support).sum((1, 2)) / support.sum((1, 2)).clamp_min(1)
    return value.sum()

def loss_terms(image, logits, residual, teacher, normalizer, cfg):
    device = logits.device
    z = F.interpolate(logits.float()[:, None], (640, 640), mode='bilinear', align_corners=False)[:, 0]
    target = (image['masks'][None] == (image['owners']+1)[:, None, None]).float()
    box = image['target_boxes'].float()
    area = ((box[:, 2:] - box[:, :2]) / 640).prod(1)
    pix = F.binary_cross_entropy_with_logits(z, target, reduction='none')
    bce = ops.crop_mask(pix, box).mean((1, 2)) / area * float(image['segmentation_gain'])
    teacher_value = teacher_loss(image, residual, teacher, cfg['teacher_huber_tau'])
    return (bce.sum() + float(cfg['teacher_beta']) * teacher_value) / normalizer, bce.sum(), teacher_value

def payload(image, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in image.items() if k in ('proto','masks','owners','target_boxes','segmentation_gain')}

def logits_loss(image, logits, normalizer):
    """Official full640 crop/area BCE, with a 160x160 logit leaf."""
    device = logits.device
    z = F.interpolate(logits.float()[:,None], (640,640), mode='bilinear', align_corners=False)[:,0]
    target = (image['masks'][None] == (image['owners']+1)[:,None,None]).float()
    box = image['target_boxes'].float()
    area = ((box[:,2:] - box[:,:2])/640).prod(1)
    pix = F.binary_cross_entropy_with_logits(z, target, reduction='none')
    value = ops.crop_mask(pix, box).mean((1,2))/area*float(image['segmentation_gain'])
    total = value.sum()
    grad = torch.autograd.grad(total/normalizer, logits, create_graph=False, retain_graph=True)[0]
    return float(total.detach()), grad

def make_selected(image, device):
    return dict(raw_ids=image['raw_ids'].to(device).long(), boxes=image['boxes'].to(device).float(), proto=image['proto'].to(device).float())

def native_z(image, device):
    p = image['proto'].to(device).float()
    c = image['c0'].to(device).float()
    return (p[None] * c[:, :, None, None]).sum(1)

def save_ck(path, model, opt, mode, cfg, history):
    tmp = Path(str(path)+'.tmp')
    torch.save(dict(mode=mode, state_dict=model.state_dict(), optimizer=opt.state_dict(), epoch=cfg['epochs'], config=cfg, history=history), tmp)
    tmp.replace(path)

def build_model(replay, mode, cfg):
    return DynamicPrototypeReadout(replay.native_cv4, replay.feature_channels, mode, float(cfg.get('rho_init', 0.0))).cuda().float().train()

def optimizer(model, cfg):
    native, dyn = [], []
    for n,p in model.named_parameters():
        (native if n.startswith('native_cv4.') else dyn).append(p)
    groups=[]
    for ps,lr,name in ((native,cfg['branch_lr'],'native'),(dyn,cfg['new_lr'],'dynamic')):
        if ps: groups.append(dict(params=ps,lr=lr,initial_lr=lr,weight_decay=cfg['weight_decay'],group_name=name))
    return torch.optim.AdamW(groups, betas=(.9,.999), eps=1e-8)

def smoke(cfg, out):
    idx=load_index(cfg); ids=[r['image_id'] for r in idx['fit'] if r['n']][:2]
    replay=FrozenReplay(cfg); oracle=load_oracle(cfg['oracle_path']); images=[load_asset(cfg,i,verify=True) for i in ids]
    report={'images':ids,'arms':{},'source':replay.import_info,'oracle_records':len(oracle)}
    for mode in ARMS:
        setup(0); model=build_model(replay,mode,cfg); feats=replay.replay(images)
        selected=[make_selected(x,replay.device) for x in images]
        outz,bases=model(feats,selected,return_base=True)
        for x,z in zip(images,outz):
            z0=native_z(x,replay.device)
            torch.testing.assert_close(z,z0,atol=2e-5,rtol=2e-5)
            assert torch.isfinite(z).all()
        loss,grad=logits_loss(payload(images[0],replay.device),outz[0],len(images[0]['raw_ids']))
        teacher=teacher_target(images[0],oracle,replay.device)
        residual=outz[0]-bases[0]
        tv=teacher_loss(images[0],residual,teacher,cfg['teacher_huber_tau'])
        tg=torch.autograd.grad(tv, [p for p in model.parameters() if p.requires_grad], allow_unused=True, retain_graph=True)
        nonzero=sum(g is not None and float(g.norm()) > 1e-8 for g in tg)
        gnames=[n for n,p in model.named_parameters() if p.requires_grad]
        report['arms'][mode]={'counts':model.parameter_counts(),'initial_identity':True,'dynamic_parameters':gnames,'loss':loss,'mask_grad_norm':float(grad.norm()),'teacher_value':float(tv.detach()),'teacher_nonzero_parameter_grads':nonzero}
        del model
    replay.assert_unchanged(); dump(Path(out)/'SMOKE.json',report); dump(Path(out)/'COMPLETE.json',{'passed':True,'kind':'smoke','oracle_records':len(oracle)})

def train(cfg, out, mode, deadline):
    idx=load_index(cfg); planned=idx['fit']; assert len(planned)==1024 and len(idx['dev'])==256 and not idx.get('val')
    setup(0); replay=FrozenReplay(cfg); oracle=load_oracle(cfg['oracle_path']); model=build_model(replay,mode,cfg); opt=optimizer(model,cfg); frozen=bn_state(model)
    dump(Path(out)/'MODEL.json',{'mode':mode,'counts':model.parameter_counts(),'ultralytics':replay.import_info,'oracle_records':len(oracle),'teacher_beta':cfg['teacher_beta'],'teacher_huber_tau':cfg['teacher_huber_tau']})
    items=[x for x in planned if x['n']>0]; history=[]; started=time.time()
    for epoch in range(cfg['epochs']):
        if time.time()>=deadline: raise TimeoutError('budget exhausted')
        order=list(items); random.Random(epoch).shuffle(order); total=seen=0; groups=math.ceil(len(order)/cfg['effective_batch_images'])
        teacher_total=0.0
        for step in range(0,len(order),cfg['effective_batch_images']):
            rows=order[step:step+cfg['effective_batch_images']]; den=sum(x['n'] for x in rows)
            fac=lr_factor(epoch+(step//cfg['effective_batch_images']+1)/groups,cfg)
            for g in opt.param_groups:g['lr']=g['initial_lr']*fac
            opt.zero_grad(set_to_none=True)
            for j in range(0,len(rows),cfg['microbatch_images']):
                batch=[load_asset(cfg,x['image_id'],verify=True) for x in rows[j:j+cfg['microbatch_images']]]
                feats=replay.replay(batch); selected=[make_selected(x,replay.device) for x in batch]
                zs,bases=model(feats,selected,return_base=True); loss_batch=0.0
                for bi,(x,z) in enumerate(zip(batch,zs)):
                    target_teacher=teacher_target(x,oracle,replay.device)
                    residual=z-bases[bi]
                    # The immutable cache is CPU-resident; all tensors used by
                    # the differentiable loss must be moved to the replay GPU.
                    xd=payload(x,replay.device)
                    val,bv,tv=loss_terms(xd,z,residual,target_teacher,den,cfg)
                    loss_batch=loss_batch+val; total+=float(bv.detach()); teacher_total+=float(tv.detach()); seen+=len(x['raw_ids'])
                loss_batch.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)); opt.step()
            if time.time()>=deadline: raise TimeoutError('budget exhausted')
        verify_bn(model,frozen); history.append({'epoch':epoch+1,'bce':total/seen,'teacher_huber':teacher_total/max(seen,1),'total_loss':total/max(seen,1)+float(cfg['teacher_beta'])*teacher_total/max(seen,1),'seconds':time.time()-started,'grad_norm':gn}); dump(Path(out)/'HISTORY.json',history)
        save_ck(Path(out)/'last.pt',model,opt,mode,cfg,history)
    save_ck(Path(out)/'final.pt',model,opt,mode,cfg,history); dump(Path(out)/'COMPLETE.json',{'completed':True,'mode':mode,'epochs':cfg['epochs'],'elapsed_s':time.time()-started})

def padded_gt(gt, ratio_pad, shape):
    oh,ow=gt.shape; ih,iw=shape
    ratios,pads=ratio_pad; r=float(ratios) if isinstance(ratios,(float,int)) else float(ratios[0]); px,py=(float(pads),float(pads)) if isinstance(pads,(float,int)) else (float(pads[0]),float(pads[1]))
    nh,nw=round(oh*r),round(ow*r); left,top=round(px-.1),round(py-.1); right,bottom=iw-nw-left,ih-nh-top
    return F.pad(F.interpolate(gt.float()[None,None],(nh,nw),mode='nearest'),(left,right,top,bottom))[0,0].bool()

def evaluate(cfg,out,checkpoints):
    idx=load_index(cfg); replay=FrozenReplay(cfg); coco=COCO(cfg['annotations_train']); models={}
    for label,path in checkpoints.items():
        # Teacher checkpoints share the underlying G/P architecture; the
        # suffix is an evaluation label, not the architecture mode.
        mode = label[:-2] if label.endswith('-T') else label
        ck=torch.load(path,map_location='cpu',weights_only=False); assert ck['mode']==mode and ck['epoch']==cfg['epochs']
        model=build_model(replay,mode,cfg); model.load_state_dict(ck['state_dict']); model.eval().requires_grad_(False); models[label]=model
    rows=[]; activation=[]; start=time.time(); baseline_max_pixels=0
    for pos,e in enumerate(idx['dev']):
        if not e['n']: continue
        x=load_asset(cfg,e['image_id'],verify=True); feats=replay.replay([x]); selected=[make_selected(x,replay.device)]
        zs={'A':native_z(x,replay.device)}
        with torch.no_grad():
            for mode,m in models.items(): zs[mode]=m(feats,selected)[0]
            for mode in ('G','P','G-T','P-T'):
                m=models[mode]; native_mode=mode[-1] if mode.endswith('-T') else mode; m.mode='N'; native=m(feats,selected)[0]; m.mode=native_mode
                residual=zs[mode]-native
                activation.append(dict(image_id=int(x['image_id']),mode=mode,
                    residual_rms=float(residual.square().mean().sqrt()),
                    residual_max=float(residual.abs().max()),
                    changed_proto_pixels=int(((zs[mode]>0)!=(native>0)).sum())))
            for label in ('P','P-T'):
                m=models[label]; native_mode='P'; m.mode='G'; without_local=m(feats,selected)[0]; m.mode=native_mode
                activation.append(dict(image_id=int(x['image_id']),mode=label+'-local-response',
                    residual_rms=float((zs[label]-without_local).square().mean().sqrt()),
                    residual_max=float((zs[label]-without_local).abs().max()),
                    changed_proto_pixels=int(((zs[label]>0)!=(without_local>0)).sum())))
        proto=x['proto'].to(replay.device).float(); shape=(640,640); boxes=x['boxes'].to(replay.device).float(); support=ops.crop_mask(torch.ones((len(x['rows']),*shape),device=replay.device),boxes).bool()
        gt_list=[]; pad_list=[]
        for row in x['rows']:
            gt=torch.as_tensor(coco.annToMask(coco.anns[int(row['annotation_id'])]).astype(bool),device=replay.device); gt_list.append(gt); pad_list.append(_padded_gt(gt,x['ratio_pad'],shape))
        vals={a:[] for a in ('A','N','G','P','G-T','P-T')};
        for arm,z in zs.items():
            z640=F.interpolate(z[:,None],shape,mode='bilinear',align_corners=False)[:,0]; binary=ops.crop_mask(z640.clone(),boxes)>0; scaled=ops.scale_masks(binary[:,None].float(),x['original_shape'],ratio_pad=x['ratio_pad'])[:,0]>.5
            if arm=='A':
                official=ops.process_mask(proto,x['c0'].to(replay.device).float(),boxes,shape,upsample=True).bool()
                changed=int((official!=binary).sum()); baseline_max_pixels=max(baseline_max_pixels,changed)
                if changed: raise AssertionError(f'Native baseline decode differs at {changed} pixels')
            for k,row in enumerate(x['rows']):
                inter=int((scaled[k]&gt_list[k]).sum()); union=int((scaled[k]|gt_list[k]).sum()); posn=int((support[k]&pad_list[k]).sum()); truth=pad_list[k][support[k]]; scores=z640[k][support[k]]; neg=(~truth).sum(); auc=float('nan')
                auc,fpr=_pixel_auc_fpr(z640[k],pad_list[k],support[k])
                vals.setdefault(arm,[]).append({'split':'dev','image_id':int(row['image_id']),'annotation_id':int(row['annotation_id']),'raw_id':int(row['raw_id']),'pyramid_level':int(row['pyramid_level']),'target_gt_idx':int(row['target_gt_idx']),'box_iou':_box_iou(boxes[k].cpu().numpy(),x['target_boxes'][k].cpu().numpy()),'iou':inter/max(union,1),'mask75':int(inter/max(union,1)>=.75),'coverage':inter/max(int(gt_list[k].sum()),1),'auc':auc,'fpr':fpr})
        for k in range(len(x['rows'])):
            row={'split':'dev','image_id':int(x['image_id']),'annotation_id':int(x['rows'][k]['annotation_id']),'raw_id':int(x['rows'][k]['raw_id']),'pyramid_level':int(x['rows'][k]['pyramid_level']),'target_gt_idx':int(x['rows'][k]['target_gt_idx']),'box_iou':vals['A'][k]['box_iou']}
            for arm in ('A','N','G','P','G-T','P-T'):
                if k < len(vals.get(arm,[])): row.update({f'{m}_{arm}':vals[arm][k][m] for m in ('iou','mask75','coverage','auc','fpr')})
            row['box_good_mask_bad']=bool(row['box_iou']>=.75 and row['iou_A']<.75); rows.append(row)
        if pos%20==0: dump(Path(out)/'PROGRESS.json',{'images':pos+1,'candidates':len(rows),'elapsed_s':time.time()-start})
    with (Path(out)/'PER_CANDIDATE.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps({k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()},allow_nan=False)+'\n')
    dump(Path(out)/'ACTIVATION.json',{'baseline_changed_pixels':baseline_max_pixels,'rows':activation})
    groups={}
    for r in rows: groups.setdefault(r['image_id'],[]).append(r)
    def stat(sub,arm,metric): return float(np.mean([np.mean([r[f'{metric}_{arm}'] for r in rs if math.isfinite(float(r[f'{metric}_{arm}']))]) for rs in sub.values() if any(math.isfinite(float(r[f'{metric}_{arm}'])) for r in rs)]))
    def pair(sub,arm,ref,metric):
        arr=[];rep=dam=0
        for rs in sub.values():
            a=[r for r in rs if math.isfinite(float(r.get(f'{metric}_{arm}',float('nan')))) and math.isfinite(float(r.get(f'{metric}_{ref}',float('nan'))))]
            if a: arr.append(np.mean([r[f'{metric}_{arm}']-r[f'{metric}_{ref}'] for r in a]))
            if metric=='iou':
                rep+=sum(r['mask75_'+arm]==1 and r['mask75_'+ref]==0 for r in a); dam+=sum(r['mask75_'+arm]==0 and r['mask75_'+ref]==1 for r in a)
        rng=np.random.default_rng(20261004); draws=[]
        if arr:
            aa=np.asarray(arr); draws=np.mean(aa[rng.integers(0,len(aa),(1000,len(aa)))],axis=1).tolist()
        return {'delta':float(np.mean(arr)) if arr else None,'ci95':np.quantile(draws,[.025,.975]).tolist() if draws else [None,None],'repair':rep,'damage':dam,'net':rep-dam}
    allg={k:v for k,v in groups.items()}; target={k:[r for r in v if r['box_good_mask_bad']] for k,v in groups.items()}; target={k:v for k,v in target.items() if v}
    summary={'population':{'images':len(groups),'candidates':len(rows)},'comparisons':{},'means':{}}
    for name,sub in (('all',allg),('target',target)):
        summary['means'][name]={a:{m:stat(sub,a,m) for m in ('iou','coverage','auc','fpr')} for a in ('A','N','G','P','G-T','P-T')}
        for arm in ('N','G','P','G-T','P-T'):
            for ref in ('A','G') if arm=='P' else ('A',):
                summary['comparisons'][f'{name}:{arm}-{ref}']={m:pair(sub,arm,ref,m) for m in ('iou','coverage','auc','fpr')}
        for arm,ref in (('G-T','G'),('P-T','P')):
            summary['comparisons'][f'{name}:{arm}-{ref}']={m:pair(sub,arm,ref,m) for m in ('iou','coverage','auc','fpr')}
    dump(Path(out)/'SUMMARY.json',summary); Path(out/'REPORT.md').write_text('# Position-conditioned prototype gate\n\n'+json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); dump(Path(out)/'COMPLETE.json',{'completed':True,'kind':'evaluation','candidates':len(rows),'images':len(groups)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=('smoke','train','eval'),required=True);p.add_argument('--mode');p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,default=0.);p.add_argument('--checkpoints');a=p.parse_args(); cfg=cfg_load(a.config); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    try:
        if a.stage=='smoke': smoke(cfg,out)
        elif a.stage=='train': train(cfg,out,a.mode,a.deadline)
        else: evaluate(cfg,out,json.loads(Path(a.checkpoints).read_text(encoding='utf-8')))
    except BaseException as e:
        dump(out/'FAILURE.json',{'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}); raise
if __name__=='__main__': main()
