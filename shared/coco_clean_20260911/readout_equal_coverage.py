"""S037: frozen learned rankings at equal true-positive coverage on train images.

GT-assisted diagnostic only. No fitted coefficients/heads or deployable AP.
Adds shared-head comparisons, error exposure, and fixed coverage to S021/S034.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,csv,io,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici
from readout_composition_control import bbox_iou


def ranking(values,regions,area):
    """Whole-tie score thresholds. Regions own=0, same=1, other=2, bg=3."""
    n=len(values)
    if not n:return dict(status='empty_support'),[]
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite logit')
    order=np.argsort(-values,kind='stable');v=values[order];lab=regions[order]
    ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),n-1]
    counts=np.stack([np.cumsum(lab==j,dtype=np.int64) for j in range(4)])[:,ends]
    tp=counts[0];fp=counts[1:].sum(0);iou=tp/(area+fp)
    best=int(np.argmax(iou));exposure=np.bincount(regions,minlength=4)
    no_neighbor=tp/(area+counts[2]+counts[3]);no_background=tp/(area+counts[1]+counts[2])
    row=dict(status='complete',crop_pixels=n,own_area=area,own_in_crop=int(exposure[0]),
        crop_coverage=float(exposure[0]/area),neighbor_available=int(exposure[1]),background_available=int(exposure[3]),
        neighbor_exposure=float(exposure[1]/area),background_exposure=float(exposure[3]/area),
        threshold_best_iou=float(iou[best]),threshold_best_coverage=float(tp[best]/area),
        threshold_best=float(v[ends[best]]),threshold_best_neighbor=float(counts[1,best]/area),
        threshold_best_background=float(counts[3,best]/area),
        best_without_neighbor=float(no_neighbor.max()),best_without_background=float(no_background.max()),
        scalar_recovers75=bool(iou[best]>=.75),crop_precludes75=bool(exposure[0]/area<.75))
    quantiles=[]
    for coverage in [.8,.9,.95]:
        target=int(math.ceil(coverage*area))
        if exposure[0]<target:
            quantiles.append(dict(requested_coverage=coverage,status='crop_infeasible',target_tp=target));continue
        k=int(np.searchsorted(tp,target,side='left'))
        quantiles.append(dict(requested_coverage=coverage,status='complete',target_tp=target,actual_tp=int(tp[k]),
            tie_overshoot=int(tp[k]-target),coverage=float(tp[k]/area),threshold=float(v[ends[k]]),
            iou=float(iou[k]),neighbor=float(counts[1,k]/area),background=float(counts[3,k]/area),other=float(counts[2,k]/area),
            neighbor_fpr=float(counts[1,k]/exposure[1]) if exposure[1] else None,
            background_fpr=float(counts[3,k]/exposure[3]) if exposure[3] else None))
    return row,quantiles


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);start=time.monotonic();src=a.source
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=Path(json.loads((src/'protocol.json').read_text())['cache'])
    images=json.loads((cache/'selection.json').read_text())['transfer']
    receipt={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=receipt['conversion_input/instances_probe.json']:raise RuntimeError('Changed COCO')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Changed normalizer')
    normalizer=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    heads={};hashes={}
    for s in range(3):
        path=src/f'raw_coco_s{s}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changed checkpoint')
        head=GlobalHead().cuda();head.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model'])
        heads[f'coefficient_s{s}']=head.eval().requires_grad_(False);hashes[str(path)]=sha(path)
    with (src/'gt_recovery.csv').open() as f:task={(r['arm'],int(r['annotation_id'])):r['hit75']=='True' for r in csv.DictReader(f)}
    with (src/'spatial.csv').open() as f:prior={(r['arm'],int(r['annotation_id'])):r for r in csv.DictReader(f)}
    protocol=dict(experiment='S037_EQUAL_OWN_COVERAGE',training=False,images=images,source=str(src.resolve()),cache=str(cache.resolve()),
        arms=['original',*heads],seeds=[0,1,2],
        increment='Existing shared heads versus original ranking at equalTPcoverage, post-calibration residuals and negativeexposure controls. '
            'No freecoeff oracle or repeated learned-area transfer; differs fromS021native-label/freecoeff andS034predictedarea.',
        domain='Native640 continuous c@P then bilinear640; same original predicted crop. RawCOCOannToMask nearest-exact roundedletterbox. '
            'Exclude padding/crowd. Ownmask retains shared GT pixels; negative regions mutuallyexclusive sameclass,otherclass,bg.',
        coverage='Primary90% of full valid ownGTarea;80/95sensitivity. If originalcrop lacks this TPcount, report infeasible. '
            'Use highest whole-tie logit threshold attaining ceil(coverage*area), include all equal scores. Recordactualcoverage/tieovershoot. '
            'Same TPcount allarms except tieovershoot; predictions uniformlyshifted byinstancebias have exactly same rankingcurve.',
        scalar='Sweep ALL thresholds grouped byties, maximize same640validGTIoU in fixedcrop; no originalresolution optimality claim. '
            'Also removeGTneighbor orGTbackground and resweep only to classify interference, not a method.',
        cohort='All existing1815fixedbbox50matches across300exploredtrain2017images, not all2002GT. '
            'Posthoc subset all3S032heads failofficialMask75 is descriptive, not densityeffect or fullRecall.',
        control='Compare exposed-neighborFPR separately from neighborFP/own; category+size+boxIoU common strata >=3/group. '
            'Report allmatches and residualsubset; imagecluster2000pointwiseCI, no causal claim.',
        freeze='No refitting,tuning,valquery,newendtoendtraining. GT thresholds diagnostic only, not AP or deployable recovery.',
        hashes=dict(cache_receipt=sha(cache/'COMPLETE.json'),source_receipt=sha(src/'COMPLETE.json'),script=sha(__file__),checkpoints=hashes))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    rows=[];qrows=[];skips=[];maxerr=0.;replay_count=0;allgt=[]
    for num,iid in enumerate(images,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Changed cache')
        item=read_np(path);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        ordinary=[aa for aa in gt.imgToAnns[iid] if not aa.get('iscrowd',0)]
        mapping={int(aid):int(j) for aid,j in zip(item['annotation_ids'],item['prediction_indices'])}
        for ann in ordinary:
            aid=ann['id'];val=ici(ann,ordinary)
            allgt.append(dict(image_id=iid,annotation_id=aid,ici=val,matched=aid in mapping,
                original_hit=task['original_s-1_d0',aid],coefficient_all_fail=all(not task[f'raw_coco_s{s}_d0',aid] for s in range(3))))
        if not mapping:continue
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain)
        top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        valid=np.zeros(ishape,bool);valid[top:top+rh,left:left+rw]=True
        rasters={};masks={};crowd=np.zeros(ishape,bool);union=np.zeros(ishape,bool)
        for ann in gt.imgToAnns[iid]:
            aid=ann['id'];m=gt.annToMask(ann).astype(bool);masks[aid]=m
            r=np.zeros(ishape,bool)
            r[top:top+rh,left:left+rw]=F.interpolate(torch.tensor(m,device='cuda').float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool().cpu().numpy()
            if ann.get('iscrowd',0):crowd|=r
            else:rasters[aid]=r;union|=r
        valid&=~crowd
        bases={};domains={};regions={}
        for aid,j in mapping.items():
            ann=gt.anns[aid];own=rasters[aid]&valid;area=int(own.sum())
            if not area:
                skips.append(dict(image_id=iid,annotation_id=aid,reason='zero_valid640own'));continue
            box=item['boxes'][j];xx=np.arange(iw);yy=np.arange(ih)
            crop=(xx[None,:]>=box[0])&(xx[None,:]<box[2])&(yy[:,None]>=box[1])&(yy[:,None]<box[3])&valid
            same=np.zeros(ishape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==ann['category_id']:same|=rasters[other['id']]
            zone=np.full(ishape,3,dtype=np.uint8);zone[union]=2;zone[same]=1;zone[own]=0
            density=ici(ann,ordinary);gtbox=np.asarray(ann['bbox'],dtype=float);gtbox[2:]+=gtbox[:2]
            biou=bbox_iou(gtbox,item['detections'][j,:4])
            if biou<.5-1e-6:raise RuntimeError('Frozen bbox attribution changed')
            bases[aid]=dict(image_id=iid,annotation_id=aid,prediction_index=j,source_index=int(item['source_index'][j]),
                category_id=ann['category_id'],size='small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large',
                ici=density,high=density>.5+1e-10,group='low' if density<=1e-10 else 'middle' if density<=.5+1e-10 else 'high',
                box_iou=biou,box_bin='50_75' if biou<.75 else '75_90' if biou<.9 else '90_100',own_area=area,
                original_official_hit=task['original_s-1_d0',aid],
                coefficient_all_fail=all(not task[f'raw_coco_s{s}_d0',aid] for s in range(3)))
            domains[aid]=crop;regions[aid]=zone[crop]
        with torch.inference_mode():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalizer)
            for arm in ['original',*heads]:
                cc=c if arm=='original' else c+heads[arm](x)
                z=F.interpolate((cc@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0]
                binary=ops.crop_mask((z>0).byte(),boxes)
                restored=(ops.scale_masks(binary[:,None],shape)[:,0]>.5).cpu().numpy()
                zn=z.cpu().numpy();bn=binary.cpu().numpy()
                for aid,base in bases.items():
                    j=base['prediction_index'];oldarm='original_s-1_d0' if arm=='original' else arm.replace('coefficient_','raw_coco_')+'_d0'
                    ownoriginal=masks[aid];pred=restored[j];un=int((ownoriginal|pred).sum());oiou=float((ownoriginal&pred).sum()/un)
                    err=abs(oiou-float(prior[oldarm,aid]['iou']));maxerr=max(maxerr,err);replay_count+=1
                    if err>1e-12:raise RuntimeError('OriginalCOCO fixedIoU replay mismatch')
                    area=base['own_area'];domain=domains[aid];reg=regions[aid];values=zn[j][domain]
                    result,quantiles=ranking(values,reg,area)
                    if result['status']!='complete':
                        skips.append(dict(image_id=iid,annotation_id=aid,reason='empty_crop'));continue
                    pred640=bn[j][domain].astype(bool);tp=int((pred640&(reg==0)).sum());fp=int((pred640&(reg!=0)).sum())
                    row=dict(**base,arm=arm,**{k:v for k,v in result.items() if k!='own_area'},
                        fixed_original_iou=oiou,fixed640_iou=tp/(area+fp),fixed640_coverage=tp/area,
                        official_hit=task[oldarm,aid])
                    rows.append(row)
                    for q in quantiles:qrows.append(dict(**base,arm=arm,**q))
        if num%25==0 or num==len(images):
            progress=dict(images=num,total=len(images),seconds=time.monotonic()-start,rows=len(rows));write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    csv_save(out/'ranking.csv',rows);csv_save(out/'coverage.csv',qrows);csv_save(out/'all_gt.csv',allgt)
    write_json(out/'skips.json',skips)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(images),all_gt=len(allgt),ranking_rows=len(rows),coverage_rows=len(qrows),
        fixed_iou_replays=replay_count,max_replay_error=maxerr,seconds=time.monotonic()-start,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.is_dir():write_json(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
