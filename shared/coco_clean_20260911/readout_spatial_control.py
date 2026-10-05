"""S040: fixed-prototype spatial control versus global readouts, GT diagnostic only.

Fits finite, regularized logistic readouts at the same 512 coordinates. No shared
network updates, no AP, no validation-set parameter choice. Three saved states.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import argparse, contextlib, csv, io, json, shutil, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from rich_pixel_readout import GlobalHead, instance_features
from readout_input_probe import sha, write_json
from run_rich_pixel_readout import read_np, csv_save, cuda, norm
from readout_selective_direction import full_metrics, pixel_metrics
from eval_readout_input_pilot import ici

ARMS = ['global32', 'spatial_bias4', 'global_spatial_bias36', 'spatial_coeff128', 'global_nonlinear128']


def phi(p, xy, arm, q):
    """p already divided by the direction-pixel channel RMS, xy is box relative."""
    x, y = xy.unbind(-1)
    space = torch.stack([torch.ones_like(x), x, y, x*y], -1)
    if arm == 'global32': return p
    if arm == 'spatial_bias4': return space
    if arm == 'global_spatial_bias36': return torch.cat([p, space], -1)
    if arm == 'spatial_coeff128': return (space[..., :, None]*p[..., None, :]).flatten(-2)
    if arm == 'global_nonlinear128': return torch.cat([p, torch.sin(p@q)], -1)
    raise ValueError(arm)


def xy_at(pos, box, width):
    xx = (pos % width).float(); yy = (pos // width).float()
    return torch.stack([2*(xx-box[0])/(box[2]-box[0]).clamp_min(1)-1,
                        2*(yy-box[1])/(box[3]-box[1]).clamp_min(1)-1], -1).clamp(-1, 1)


def objective(a, x, z, y):
    delta = x@a
    return F.binary_cross_entropy_with_logits(z+delta, y) + .01*delta.square().mean() + .0001*a.square().sum()


def solve(x, z, y):
    """Strictly convex fit, common objective and maximum iteration budget."""
    x=x.double(); z=z.double(); y=y.double(); n,d=x.shape
    a=torch.zeros(d, device=x.device, dtype=x.dtype); eye=torch.eye(d, device=x.device, dtype=x.dtype)
    trace=[]; converged=False
    for iteration in range(30):
        delta=x@a; prob=(z+delta).sigmoid()
        grad=x.T@((prob-y+.02*delta)/n)+.0002*a
        grad_inf=float(grad.abs().max()); value=objective(a,x,z,y)
        trace.append([iteration,float(value),grad_inf])
        if grad_inf<1e-7: converged=True; break
        curvature=(prob*(1-prob)+.02)/n
        hessian=x.T@(curvature[:,None]*x)+.0002*eye
        step=torch.linalg.solve(hessian, grad); descent=grad@step
        accepted=False
        for backtrack in range(20):
            scale=.5**backtrack; trial=a-scale*step
            if float(objective(trial,x,z,y)) <= float(value-1e-4*scale*descent):
                a=trial; accepted=True; break
        if not accepted: break
    delta=x@a; prob=(z+delta).sigmoid()
    final_grad=x.T@((prob-y+.02*delta)/n)+.0002*a
    final_inf=float(final_grad.abs().max()); converged=converged or final_inf<1e-7
    return a.float(),dict(iterations=len(trace),converged=converged,gradient_inf=final_inf,
        objective_before=float(objective(torch.zeros_like(a),x,z,y)),objective_after=float(objective(a,x,z,y)),
        parameter_norm=float(a.norm()),fit_delta_rms=float(delta.square().mean().sqrt()),trace=trace)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'fits').mkdir();start=time.monotonic()
    old=json.loads((a.prior/'protocol.json').read_text());src=Path(old['source']);cache=Path(old['cache']);s038=Path(old['prior'])
    previous=json.loads((s038/'protocol.json').read_text());targets=[tuple(t) for t in old['targets']]
    grouped=defaultdict(list)
    for iid,aid in targets:grouped[iid].append(aid)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json']:raise RuntimeError('ChangedGT')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    normalizer=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('ChangedNormalizer')
    heads={};weights={}
    for seed in range(3):
        path=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('ChangedHead')
        h=GlobalHead().cuda();h.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);heads[seed]=h.eval().requires_grad_(False);weights[str(path)]=sha(path)
    with (a.prior/'full_masks.csv').open() as f:
        replay={(int(r['image_id']),int(r['annotation_id']),int(r['seed'])):float(r['iou_raw_before']) for r in csv.DictReader(f)}
    protocol=dict(experiment='S040_OUTPUT_SPATIAL_CONTROL',training=False,source=str(src.resolve()),cache=str(cache.resolve()),
        prior=str(a.prior.resolve()),targets=targets,seeds=[0,1,2],arms=ARMS,
        objective='MeanBCE(rawCOCO) + .01*mean(delta_logit^2) + .0001*sum(standardized_basis_weights^2). '
            'Strictlyconvex; identical Newton budget30, tolerance1e-7; backtrack<=20. No selected best settings.',
        inputs='Frozen P, predicted box and savedS032 coefficients; same128 S039 targets, not failure-selected. '
            'All three saved heads. No shared training. Normalize P RMS and basis RMS from first512 only, floor1e-3. '
            'Spatial basis [1,x,y,xy], boxrelative integer input-grid positions clipped[-1,1]. '
            'Spatialcoeff has128weights; global nonlinear128 is P32 plus96fixedsin(randomlinearP) features, seed20260912. '
            'Same parameter count does not establish identical effective capacity. Spatialbias has4weights; jointcontrol36.',
        pixels='First512positions including their original repetitions for optimization. Remaining1536deduplicated '
            'and strictly excluding first512coords for unused evaluation, spatiallycorrelated. No unusedGT in fit or normalization.',
        fullmask='Official baseline input640logits plus learned residual on upsampledP; same originalcrop and scale_masks. '
            'Zeroresidual XOR witness required. No boxes/classes/scores/candidatechanges. OriginalCOCO fixedIoU75 only, notAP.',
        claim='Conditional regularized readout experiment. Not capacityupperbound, not causal identification, notdeployablemethod. '
            'Proceed only if spatialcoeff exceeds global nonlinear128 and global+spatialbias36 in unusedANDfullIoU '
            'with own/neighbor/background accounting. Otherwise stop this finite recipe. '
            'BlendMask(arXiv2001.00309)/CondInst(arXiv2003.05664) precedent: spatial/dynamic decoding not a novelty claim.',
        hashes=dict(script=sha(__file__),prior_receipt=sha(a.prior/'COMPLETE.json'),normalizer=sha(src/'normalizer.pt'),heads=weights))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    rng=np.random.default_rng(20260912);q=cuda(rng.standard_normal((32,96)).astype(np.float32)/np.sqrt(32)).float()
    pixelrows=[];fullrows=[];fitrows=[];witness=[];max_replay=0;max_zero_xor=0
    for number,(iid,aids) in enumerate(sorted(grouped.items()),1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('ChangedCache')
        item=read_np(path);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        targetmap={int(aid):(k,int(j)) for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices']))}
        ordinary=[r for r in gt.imgToAnns[iid] if not r.get('iscrowd',0)];masks={};raster={}
        crowd=np.zeros(shape,bool);union=np.zeros(shape,bool);crowd640=np.zeros(ishape,bool);union640=np.zeros(ishape,bool)
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        valid640=np.zeros(ishape,bool);valid640[top:top+rh,left:left+rw]=True
        for ann in gt.imgToAnns[iid]:
            aid=ann['id'];m=gt.annToMask(ann).astype(bool);masks[aid]=m;r=np.zeros(ishape,bool)
            r[top:top+rh,left:left+rw]=F.interpolate(cuda(m).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool().cpu().numpy();raster[aid]=r
            if ann.get('iscrowd',0):crowd|=m;crowd640|=r
            else:union|=m;union640|=r
        with torch.no_grad():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalizer)
            baseline={s:c+head(x) for s,head in heads.items()}
            p640=F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0].flatten(1).T.contiguous()
        for aid in aids:
            k,j=targetmap[aid];ann=gt.anns[aid];split='fit' if iid in previous['fit_images'] else 'transfer';density=ici(ann,ordinary)
            base=dict(split=split,image_id=iid,annotation_id=aid,high=density>.5+1e-10,ici=density)
            same=np.zeros(shape,bool);same640=np.zeros(ishape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']];same640|=raster[other['id']]
            zones=np.full(ishape,3,np.int64);zones[union640]=2;zones[same640]=1;zones[raster[aid]]=0;zones[crowd640|~valid640]=4
            positions=item['sample_positions'][k];unused,first=np.unique(positions[512:],return_index=True)
            keep=~np.isin(unused,np.unique(positions[:512]));unused=unused[keep];heldidx=first[keep]+512
            if np.intersect1d(positions[:512],unused).size:raise RuntimeError('PixelLeak')
            pos=cuda(positions).long();samplep=cuda(item['sample_p'][k]).float();sampley=cuda(raster[aid].flatten()[positions]).float();samplezone=cuda(zones.flatten()[positions]).long()
            prms=samplep[:512].square().mean(0).sqrt().clamp_min(1e-3);xy=xy_at(pos,boxes[j],iw)
            witness.append(dict(**base,unused_unique=len(unused),direction_unique=len(np.unique(positions[:512])),intersection=0))
            if split=='fit':
                labels=src/'labels'/f'{iid}.npz'
                if sha(labels)!=sr[f'labels/{iid}.npz'] or not np.array_equal(read_np(labels)['raw_coco'][k],sampley[:512].cpu().numpy()):raise RuntimeError('FitLabelReplay')
            saved=dict(q=q.cpu().numpy(),prms=prms.cpu().numpy(),positions=positions,direction_positions=positions[:512],unused_positions=unused)
            # Compute every full basis once per target, chunked to limit working memory.
            for seed in range(3):
                coeff=baseline[seed][j].detach();initial=(samplep*coeff[None]).sum(-1)
                raw640=F.interpolate((coeff@p.flatten(1)).reshape(1,1,*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0]
                original=ops.process_mask(p,coeff[None],boxes[j:j+1],ishape,upsample=True)
                zero=ops.crop_mask((raw640>0).byte(),boxes[j:j+1]);zero_xor=int((zero!=original).sum());max_zero_xor=max(max_zero_xor,zero_xor)
                if zero_xor:raise RuntimeError('ZeroDecoderMismatch')
                basepred=(ops.scale_masks(original[:,None],shape)[0,0]>.5).cpu().numpy();before=full_metrics(basepred,masks[aid],same,union,crowd)
                err=abs(before['iou_raw']-replay[iid,aid,seed]);max_replay=max(max_replay,err)
                if err>1e-12:raise RuntimeError('PreviousIoUMismatch')
                for arm in ARMS:
                    rawphi=phi(samplep/prms,xy,arm,q);scale=rawphi[:512].square().mean(0).sqrt().clamp_min(1e-3)
                    design=rawphi/scale;weight,fit=solve(design[:512],initial[:512],sampley[:512])
                    shifted=initial+design@weight
                    residual=[]
                    for offset in range(0,ih*iw,16384):
                        fpos=torch.arange(offset,min(offset+16384,ih*iw),device='cuda')
                        residual.append((phi(p640[fpos]/prms,xy_at(fpos,boxes[j],iw),arm,q)/scale)@weight)
                    delta=torch.cat(residual).reshape(1,ih,iw);binary=ops.crop_mask((raw640+delta>0).byte(),boxes[j:j+1])
                    output=(ops.scale_masks(binary[:,None],shape)[0,0]>.5).cpu().numpy();after=full_metrics(output,masks[aid],same,union,crowd)
                    identity=dict(**base,seed=seed,arm=arm)
                    fullrows.append(dict(**identity,**{name+'_before':value for name,value in before.items()},
                        **{name+'_after':value for name,value in after.items()},mask_xor=int(np.count_nonzero(output!=basepred))))
                    for domain,idx in [('direction',np.arange(512)),('unused',heldidx)]:
                        take=cuda(idx).long();pixelrows.append(dict(**identity,domain=domain,**pixel_metrics(initial[take],shifted[take],sampley[take],samplezone[take]),
                            iou_before=float(((initial[take]>0)&sampley[take].bool()).sum()/((initial[take]>0)|sampley[take].bool()).sum().clamp_min(1)) if len(idx) else None,
                            iou_after=float(((shifted[take]>0)&sampley[take].bool()).sum()/((shifted[take]>0)|sampley[take].bool()).sum().clamp_min(1)) if len(idx) else None))
                    trace=fit.pop('trace');fitrows.append(dict(**identity,parameters=len(weight),**fit))
                    saved[f'{arm}_s{seed}_weight']=weight.cpu().numpy();saved[f'{arm}_s{seed}_scale']=scale.cpu().numpy();saved[f'{arm}_s{seed}_trace']=np.asarray(trace)
            np.savez_compressed(out/'fits'/f'{iid}_{aid}.npz',**saved)
        if number%10==0 or number==len(grouped):
            progress=dict(images=number,total=len(grouped),targets=len(witness),seconds=time.monotonic()-start)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for name,rows in [('pixels',pixelrows),('full_masks',fullrows),('fits',fitrows),('witness',witness)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',targets=len(targets),images=len(grouped),seeds=3,fullmask_rows=len(fullrows),
        pixel_rows=len(pixelrows),seconds=time.monotonic()-start,max_previous_iou_error=max_replay,max_zero_decoder_xor=max_zero_xor,
        nonconverged=sum(not r['converged'] for r in fitrows),max_gradient_inf=max(r['gradient_inf'] for r in fitrows),
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.is_dir():write_json(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
