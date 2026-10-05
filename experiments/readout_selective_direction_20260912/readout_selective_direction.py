"""S039: GT-assisted own-BCE-constrained local directions on held-out pixels.

No training or parameter search. 128 S038 targets, all three saved heads.
Direction pixels are the original512; unused coords strictly exclude them.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,contextlib,csv,io,json,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from readout_input_probe import sha,write_json
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici

REGIONS=['own','same','other','background','ignored']
ARMS=['total','same','same_projected','background','background_projected','constant_down']


def pixel_metrics(z0,z1,y,zone):
    if not len(y):return dict(n=0)
    before=F.binary_cross_entropy_with_logits(z0,y,reduction='none')
    after=F.binary_cross_entropy_with_logits(z1,y,reduction='none');delta=z1-z0
    energy=float(delta.square().mean());mean=float(delta.mean())
    out=dict(n=len(y),logit_rms=energy**.5,mean_logit_change=mean,constant_energy_fraction=mean*mean/energy if energy else 0.)
    for k,name in enumerate(REGIONS):
        take=zone==k;count=int(take.sum());out[name+'_n']=count
        out[name+'_bce_change']=float((after-before)[take].mean()) if count else None
        out[name+'_logit_change']=float(delta[take].mean()) if count else None
        out[name+'_positive_change']=float(((z1[take]>0).float()-(z0[take]>0).float()).mean()) if count else None
    return out


def full_metrics(pred,own,same,union,crowd):
    valid=~crowd;ownvalid=own&valid;area=int(ownvalid.sum());vp=pred&valid
    intersection=int((vp&ownvalid).sum());den=int((vp|ownvalid).sum());rawden=int((pred|own).sum())
    return dict(iou_valid=intersection/den if den else None,iou_raw=int((pred&own).sum())/rawden if rawden else None,
        coverage=intersection/area if area else None,neighbor=int((vp&same&~own).sum())/area if area else None,
        background=int((vp&~union).sum())/area if area else None,own_area=area,predicted_area=int(pred.sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'directions').mkdir();start=time.monotonic()
    priorcfg=json.loads((a.prior/'protocol.json').read_text());src=Path(priorcfg['source']);cache=Path(priorcfg['cache'])
    targets=[tuple(t) for t in priorcfg['gradient_targets']];grouped=defaultdict(list)
    for iid,aid in targets:grouped[iid].append(aid)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json'] or sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Changedinputs')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    with (a.prior/'gradients.csv').open() as f:previous={(int(r['image_id']),int(r['annotation_id']),int(r['seed'])):r for r in csv.DictReader(f)}
    with (src/'spatial.csv').open() as f:spatial={(int(r['annotation_id']),r['arm']):float(r['iou']) for r in csv.DictReader(f)}
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True);heads={};checkpoints={}
    for seed in range(3):
        path=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changedcheckpoint')
        head=GlobalHead().cuda();head.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);heads[seed]=head.eval().requires_grad_(False);checkpoints[str(path)]=sha(path)
    protocol=dict(experiment='S039_SELECTIVE_LOCAL_DIRECTION',training=False,targets=targets,prior=str(a.prior.resolve()),source=str(src),cache=str(cache),
        seeds=[0,1,2],arms=ARMS,logit_rms=[.001,.01],
        basis='ExactS038raw512pixelBCE+Dice derivative partitions for same/background; own constraint uses actual own-region MEAN BCE gradient.',
        projection='d=-g_negative; if ownBCEgradient dot d>0, d_projected=d-own_grad*(own_grad dot d)/||own_grad||^2. '
            'FP64directionarithmetic at current saved coefficient. This is a standardfirstorderhalfspaceprojection, notnewmethod.',
        norm='Scale each nonzero direction to RMS .001 or .01 on original512samplelogits. No chosenbeststep. '
            'Zero/no-own/no-negative directions remainunchanged and recorded, includedineveryarm. '
            'Constant_down adds scalar -RMS to fullinputlogits, same RMS ordinarycalibrationcontrol.',
        holdout='Original2048positions[512:] deduplicated bycoordinate, then removeeverycoordinate occurringinfirst512. '
            'NoheldoutGTusedindirection/projection/norm; labelatunusedpositionusedonlymeasurement. Empty/smallperregion explicit. '
            'Unusedpixels in sameimage/support are spatiallycorrelated, not newimagegeneralization.',
        decode='Fulloriginalpredictedcrop+official process_mask/scale_masks, rawCOCOoriginalresolution spatialmetrics. '
            'OriginalS032fullbatchcoefficients forbaseline; compare exactpriorfixedIoU ontransfer. '
            'Allprototypes/heads/boxes remainfrozen, noGTcandidatechoice.',
        study='Same128hashS038targets,32perfit-transfer/high-nonhigh; no residualfailureselection. '
            'Threeheadseeds paired; temporaryGTdc not sharedparametertraining, notAP or deployment.',
        gate='Onlya conditional directionopportunity if unusedpixels AND normalfullmasks preserveown whileloweringneighbor. '
            'Tiny ortraining-onlyeffects do notjustify formaltraining or claiminginnovation.',
        hashes=dict(prior_receipt=sha(a.prior/'COMPLETE.json'),prior_gradients=sha(a.prior/'gradients.csv'),cache_receipt=sha(cache/'COMPLETE.json'),
            source_receipt=sha(src/'COMPLETE.json'),script=sha(__file__),checkpoints=checkpoints))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    pixelrows=[];fullrows=[];directionrows=[];witness=[];maxloss=0.;maxiou=0.
    for number,(iid,aids) in enumerate(sorted(grouped.items()),1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changedcache')
        item=read_np(path);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        targetmap={int(aid):(k,int(j)) for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices']))}
        ordinary=[r for r in gt.imgToAnns[iid] if not r.get('iscrowd',0)];masks={};raster={};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        valid640=np.zeros(ishape,bool);valid640[top:top+rh,left:left+rw]=True;crowd640=np.zeros(ishape,bool);union640=np.zeros(ishape,bool)
        for ann in gt.imgToAnns[iid]:
            aid=ann['id'];m=gt.annToMask(ann).astype(bool);masks[aid]=m;r=np.zeros(ishape,bool)
            r[top:top+rh,left:left+rw]=F.interpolate(cuda(m).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool().cpu().numpy();raster[aid]=r
            if ann.get('iscrowd',0):crowd|=m;crowd640|=r
            else:union|=m;union640|=r
        with torch.no_grad():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalization)
            baseline={s:c+head(x) for s,head in heads.items()}
        for aid in aids:
            k,j=targetmap[aid];ann=gt.anns[aid];split='fit' if iid in priorcfg['fit_images'] else 'transfer';density=ici(ann,ordinary)
            base=dict(split=split,image_id=iid,annotation_id=aid,high=density>.5+1e-10,ici=density)
            same=np.zeros(shape,bool);same640=np.zeros(ishape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']];same640|=raster[other['id']]
            zones=np.full(ishape,3,np.int64);zones[union640]=2;zones[same640]=1;zones[raster[aid]]=0;zones[crowd640|~valid640]=4
            allpos=item['sample_positions'][k];trainpos=allpos[:512];unused,first=np.unique(allpos[512:],return_index=True)
            keep=~np.isin(unused,np.unique(trainpos));unused=unused[keep];heldidx=first[keep]+512
            if np.intersect1d(trainpos,unused).size or len(np.unique(unused))!=len(unused):raise RuntimeError('Pixelholdoutleak')
            selections={'direction':np.arange(512),'unused':heldidx};samplep=cuda(item['sample_p'][k]).float()
            ally=cuda(raster[aid].flatten()[allpos]).float();allzone=cuda(zones.flatten()[allpos]).long()
            if split=='fit':
                labels=src/'labels'/f'{iid}.npz'
                if sha(labels)!=sr[f'labels/{iid}.npz']:raise RuntimeError('ChangedY')
                lab=read_np(labels)
                if not np.array_equal(lab['raw_coco'][k],ally[:512].cpu().numpy()):raise RuntimeError('FitYreplay')
            witness.append(dict(**base,direction_samples=512,direction_unique=len(np.unique(trainpos)),unused_unique=len(unused),
                unused_own=int((allzone[cuda(heldidx).long()]==0).sum()),unused_same=int((allzone[cuda(heldidx).long()]==1).sum()),
                unused_background=int((allzone[cuda(heldidx).long()]==3).sum()),zero_unused=len(unused)==0,coordinate_intersection=0))
            for seed in range(3):
                coeff=baseline[seed][j].detach();pp=samplep[:512];y=ally[:512];zone=allzone[:512]
                z=(pp*coeff[None]).sum(-1).detach().requires_grad_(True);factor=c.new_tensor(float(item['loss_factor'][k]))
                loss=loss_value(z[None],y[None],factor[None]);gz=torch.autograd.grad(loss,z)[0].detach()
                lossdiff=abs(float(loss.detach())-float(previous[iid,aid,seed]['loss']));maxloss=max(maxloss,lossdiff)
                if lossdiff>2e-6:raise RuntimeError('Priorlossreplaychanged')
                own=zone==0
                g_own=((pp[own].double()*(z.detach()[own].sigmoid().double()-1)[:,None]).mean(0)
                    if own.any() else torch.zeros(32,device='cuda',dtype=torch.float64))
                gs={r:(pp.double()*(gz.double()*(zone==idx))[:,None]).sum(0) for idx,r in enumerate(REGIONS)}
                rawdirs={'total':-sum(gs.values()),'same':-gs['same'],'background':-gs['background']};directions={};meta={}
                for arm in ARMS:
                    parent=arm.replace('_projected','');projected=arm.endswith('_projected');d=rawdirs.get(parent,torch.zeros_like(g_own)).clone()
                    active=arm=='total' or arm=='constant_down' or bool((zone==(1 if parent=='same' else 3)).any())
                    dot=float(torch.dot(g_own,d));removed=0.
                    if projected:
                        if not own.any():active=False
                        elif dot>0:
                            remove=g_own*(torch.dot(g_own,d)/g_own.square().sum().clamp_min(1e-30));d-=remove
                            removed=float(remove.norm()/rawdirs[parent].norm().clamp_min(1e-30))
                    response=(pp.double()*d[None]).sum(-1);rms=float(response.square().mean().sqrt())
                    if arm!='constant_down' and (not active or rms<1e-12):active=False;d.zero_()
                    directions[arm]=d;meta[arm]=dict(active=active,projected=projected,own_firstorder_before=dot,
                        own_firstorder_after=float(torch.dot(g_own,d)),projection_removed_norm_fraction=removed,
                        unnormalized_logit_rms=rms,negative_firstorder=float(torch.dot(gs.get(parent,torch.zeros_like(g_own)),d)))
                    if active and projected and float(torch.dot(g_own,d))>1e-10:raise RuntimeError('Ownfirstorderconstraintfailed')
                # Same single-target decoder call for every arm, baseline officialIoU on transfer.
                basebinary=ops.process_mask(p,coeff[None],boxes[j:j+1],ishape,upsample=True)
                basepred=(ops.scale_masks(basebinary[:,None],shape)[0,0]>.5).cpu().numpy()
                baseline_metrics=full_metrics(basepred,masks[aid],same,union,crowd)
                if split=='transfer':
                    err=abs(baseline_metrics['iou_raw']-spatial[aid,f'raw_coco_s{seed}_d0']);maxiou=max(maxiou,err)
                    if err>1e-12:raise RuntimeError('PriorfullmaskIoUreplaychanged')
                raw640=F.interpolate((coeff@p.flatten(1)).reshape(1,1,*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0]
                initial=(samplep*coeff[None]).sum(-1)
                saved={};
                for magnitude in [.001,.01]:
                    for arm in ARMS:
                        mm=meta[arm];d=directions[arm];delta=(d*(magnitude/mm['unnormalized_logit_rms'])).float() if mm['active'] and arm!='constant_down' else torch.zeros_like(coeff)
                        if arm=='constant_down':
                            shifted=initial-magnitude
                            binary=ops.crop_mask((raw640-magnitude>0).byte(),boxes[j:j+1])
                        else:
                            shifted=(samplep*(coeff+delta)[None]).sum(-1)
                            binary=ops.process_mask(p,(coeff+delta)[None],boxes[j:j+1],ishape,upsample=True)
                        output=(ops.scale_masks(binary[:,None],shape)[0,0]>.5).cpu().numpy();current=full_metrics(output,masks[aid],same,union,crowd)
                        identity=dict(**base,seed=seed,arm=arm,magnitude=magnitude,active=mm['active'])
                        fullrows.append(dict(**identity,**{name+'_before':value for name,value in baseline_metrics.items()},
                            **{name+'_after':value for name,value in current.items()},
                            mask_xor=int(np.count_nonzero(output!=basepred))))
                        directionrows.append(dict(**identity,**{key:value for key,value in mm.items() if key!='active'},
                            actual_direction_rms=float((shifted[:512]-initial[:512]).square().mean().sqrt()),
                            relative_coefficient_step=float(delta.norm()/coeff.norm().clamp_min(1e-12))))
                        for domain,idx in selections.items():
                            tt=cuda(idx).long();pixelrows.append(dict(**identity,domain=domain,
                                **pixel_metrics(initial[tt],shifted[tt],ally[tt],allzone[tt])))
                        saved[f'{arm}_{magnitude}']=delta.cpu().numpy()
                np.savez_compressed(out/'directions'/f'{iid}_{aid}_s{seed}.npz',coefficient=coeff.cpu().numpy(),
                    direction_positions=trainpos,unused_positions=unused,**saved)
        if number%20==0 or number==len(grouped):
            progress=dict(images=number,total=len(grouped),targets=len(witness),seconds=time.monotonic()-start)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for name,rows in [('pixels',pixelrows),('full_masks',fullrows),('directions',directionrows),('witness',witness)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',targets=len(targets),seeds=3,fullmask_rows=len(fullrows),pixel_rows=len(pixelrows),
        max_prior_loss_error=maxloss,max_prior_fixedIoU_error=maxiou,seconds=time.monotonic()-start,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.is_dir():write_json(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
