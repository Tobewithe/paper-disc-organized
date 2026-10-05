"""S030: same S027 labels/targets, bounded relative-norm oracle sensitivity.

Independent GT-assisted coefficient fits only. No shared model training or
test-image selection of regularization, thresholds, iteration or final state.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,contextlib,io,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from rich_pixel_readout import loss_value
from readout_fit_transfer_diagnostic import read_np,numbers,csv_save
from eval_readout_input_pilot import ici


def solve_regularized(p,y,c0,factor,lam):
    scale=p.square().mean(0).sqrt().clamp_min(.01)
    x=(p/scale).double();yy=y.double();s=scale.double();base=c0.double()
    w=torch.nn.Parameter((c0*scale).double());den=base.square().sum().clamp_min(1e-12)
    def parts():
        loss=loss_value((x@w)[None],yy[None],yy.new_tensor([factor]))
        penalty=lam*(w/s-base).square().sum()/den
        return loss,penalty
    init=parts();initial=float((init[0]+init[1]).detach())
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=120,max_eval=240,tolerance_grad=1e-7,
        tolerance_change=1e-10,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss,penalty=parts();total=loss+penalty
        if not torch.isfinite(total):raise RuntimeError('Nonfinite regularized objective')
        total.backward();return total
    opt.step(closure);loss,penalty=parts();total=loss+penalty
    grad=torch.autograd.grad(total,w)[0];c=(w/s).detach();state=opt.state[w]
    delta_ratio=float((c-base).norm()/base.norm().clamp_min(1e-12))
    bound=np.sqrt((initial+1e-7)/lam)
    if float(total.detach())>initial+1e-7:raise RuntimeError('Regularized total increased')
    if delta_ratio>bound+1e-5:raise RuntimeError('Nonnegative-objective norm bound violated')
    if not torch.isfinite(c).all():raise RuntimeError('Nonfinite coefficient')
    return c,dict(lambda_reg=lam,initial_total=initial,final_total=float(total.detach()),
        data_objective=float(loss.detach()),penalty=float(penalty.detach()),relative_delta_norm=delta_ratio,
        relative_delta_bound=bound,coeff_norm_ratio=float(c.norm()/base.norm().clamp_min(1e-12)),
        iterations=int(state['n_iter']),hit_cap=int(state['n_iter'])>=120,gradient_max=float(grad.abs().max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'coefficients').mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous=json.loads((a.previous/'protocol.json').read_text());targets=previous['oracle_targets']
    hashes={k.replace('\\','/'):v for k,v in json.loads((a.cache/'COMPLETE.json').read_text())['hashes'].items()}
    oldhash={k.replace('\\','/'):v for k,v in json.loads((a.previous/'COMPLETE.json').read_text())['hashes'].items()}
    ids=sorted({int(iid) for rr in targets.values() for iid,aid in rr})
    config=dict(experiment='S030 regularized finite GT oracle',network_training=False,gt_assisted=True,
        targets=targets,arms=['original','unconstrained_saved','unconstrained_norm_restored','ridge_0.01','ridge_0.1'],
        primary_lambda=.01,sensitivity_lambda=.1,no_arm_selection=True,
        regularizer='lambda*||c-c0||_2^2/||c0||_2^2. Data objective identical512nativeGTboxBCE+Dice toS027. '
                    'Prototype RMS preconditioning sameS027, FP64 LBFGS120 fromoriginal, finalstate, noIoUselection.',
        bound='Since data loss nonnegative and total final<=total original, ||c-c0||/||c0||<=sqrt(initialTotal/lambda). '
              'This is parameterization-dependent norm regularization, not invariant prototype geometry or a novel method.',
        norm_restore='Positive rescale savedunconstrained c to originalnorm, same zero-threshold mask expected; no refit. '
                     'Separates rawscale probability overconfidence from learned direction errors. Exact binary parity measured.',
        evaluation='Same first512fitpixels; unique unusedpositions exclude every trainingposition; originalCOCO fullmask fixedpredbox. '
                   'IndependentGT fits on both fit andtransfer images are NOT crossimagegeneralization. Dense11/12targets limited.',
        stop='Report allfixedarms, all160targets. No studenttraining justified solely by oracle. '
             'Need unusedpixel probability stability, coverage and rawCOCO shape retention, not only trainloss.',
        sources=dict(previous=sha(a.previous/'COMPLETE.json'),cache=sha(a.cache/'COMPLETE.json'),script=sha(Path(__file__))))
    write_json(out/'protocol.json',config);shutil.copy2(__file__,out/Path(__file__).name)
    subset=a.cache/'conversion_input/instances_probe.json'
    if sha(subset)!=hashes['conversion_input/instances_probe.json']:raise RuntimeError('RawJSON subset changed')
    obj=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(images=[r for r in obj['images'] if r['id'] in ids],
         categories=obj['categories'],annotations=[r for r in obj['annotations'] if r['image_id'] in ids])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del obj
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    pixrows=[];spatial=[];solver=[];witness=[];targetrows=[];done=0
    for split in ['fit','transfer']:
        image_ids=sorted({int(q[0]) for q in targets[split]})
        for iid in image_ids:
            path=a.cache/'images'/f'{iid}.npz'
            if sha(path)!=hashes[f'images/{iid}.npz']:raise RuntimeError('Cache changed')
            item=read_np(path);proto=torch.tensor(item['proto'],device='cuda').double();shape=tuple(item['shape']);ishape=tuple(item['input_shape'])
            ordinary=[r for r in gt.imgToAnns[iid] if not r.get('iscrowd',0)]
            masks={r['id']:gt.annToMask(r).astype(bool) for r in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
            for r in gt.imgToAnns[iid]:
                if r.get('iscrowd',0):crowd|=masks[r['id']]
                else:union|=masks[r['id']]
            for _,aid in [q for q in targets[split] if int(q[0])==iid]:
                aid=int(aid);k=int(np.flatnonzero(item['annotation_ids']==aid)[0]);j=int(item['prediction_indices'][k]);aa=gt.anns[aid]
                pp=torch.tensor(item['sample_p'][k],device='cuda');yy=torch.tensor(item['sample_y'][k],device='cuda').float()
                c0=torch.tensor(item['coeff'][j],device='cuda');factor=float(item['loss_factor'][k]);box=torch.tensor(item['boxes'][j:j+1],device='cuda')
                oldpath=a.previous/'oracle_coefficients'/f'{iid}_{aid}.npz'
                if sha(oldpath)!=oldhash[f'oracle_coefficients/{iid}_{aid}.npz']:raise RuntimeError('Previous oracle changed')
                cu=torch.tensor(read_np(oldpath)['coefficient'],device='cuda').double()
                restored=cu*(c0.double().norm()/cu.norm().clamp_min(1e-12))
                cs={'original':c0.double(),'unconstrained_saved':cu,'unconstrained_norm_restored':restored}
                base=dict(split=split,image_id=iid,annotation_id=aid,ici=ici(aa,ordinary),high=ici(aa,ordinary)>.5+1e-10)
                for lam in [.01,.1]:
                    name=f'ridge_{lam:g}';c,info=solve_regularized(pp[:512],yy[:512],c0,factor,lam)
                    cs[name]=c;solver.append(dict(**base,mode=name,**info))
                np.savez_compressed(out/'coefficients'/f'{iid}_{aid}.npz',**{m:c.cpu().numpy() for m,c in cs.items()})
                positions=item['sample_positions'][k];seen=set(map(int,positions[:512]));held=[]
                for t in range(512,len(positions)):
                    v=int(positions[t])
                    if v not in seen:held.append(t);seen.add(v)
                targetrows.append(dict(**base,unused_n=len(held),positive512=int(yy[:512].sum())))
                binaries={}
                for mode,c in cs.items():
                    z=pp.double()@c
                    for domain,idx in [('train512',slice(0,512)),('unused',held)]:
                        pixrows.append(dict(**base,mode=mode,domain=domain,**numbers(z[idx],yy[idx].double(),factor)))
                    with torch.inference_mode():
                        lo=(c@proto.flatten(1)).reshape(1,1,*proto.shape[-2:])
                        dense=F.interpolate(lo,ishape,mode='bilinear',align_corners=False)[0,0]
                        binaries[mode]=ops.crop_mask((dense>0).byte()[None],box)[0]
                diff=int(torch.count_nonzero(binaries['unconstrained_saved']!=binaries['unconstrained_norm_restored']))
                witness.append(dict(**base,norm_restore_pixel_xor=diff,
                    unconstrained_norm_ratio=float(cu.norm()/c0.double().norm()),restored_norm_ratio=float(restored.norm()/c0.double().norm())))
                names=list(binaries);origmask=ops.scale_masks(torch.stack(list(binaries.values()))[:,None],shape)[:,0]>.5
                own=masks[aid]&~crowd;area=int(own.sum());same=np.zeros(shape,bool)
                for r in ordinary:
                    if r['id']!=aid and r['category_id']==aa['category_id']:same|=masks[r['id']]
                for t,mode in enumerate(names):
                    pred=origmask[t].cpu().numpy();vp=pred&~crowd;inter=int((pred&masks[aid]).sum());un=int((pred|masks[aid]).sum())
                    spatial.append(dict(**base,mode=mode,coco_iou=inter/un if un else 1.,
                       coverage=float((vp&own).sum()/area) if area else None,
                       neighbor=float((vp&same&~own).sum()/area) if area else None,
                       background=float((vp&~union).sum()/area) if area else None))
                done+=1
                if done%20==0:progress('solve',targets=done,total=160)
    for name,rows in [('pixels',pixrows),('spatial',spatial),('solver',solver),('witness',witness),('targets',targetrows)]:csv_save(out/f'{name}.csv',rows)
    progress('COMPLETE',targets=done)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=done,regularized_solves=len(solver),
       norm_restored_pixel_xor=sum(r['norm_restore_pixel_xor'] for r in witness),
       hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
