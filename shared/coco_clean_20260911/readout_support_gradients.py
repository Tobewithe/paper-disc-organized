"""S038: actual saved sampling support and within-target regional gradients.

No network training. Fit and transfer explicitly separated; the latter was
never supervised by these heads. Small coefficient steps are local probes.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,contextlib,csv,io,json,shutil,time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from readout_input_probe import sha,stable_seed,write_json
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici

REGIONS=['own','same','other','background','ignored']


def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/den) if den>1e-12 else None


def gradient_probe(head,x,c,p,y,factor,zone,base):
    """Current-state exact logit loss derivative partitions; shared gradient VJPs."""
    coeff=c+head(x[None])[0];z=(p*coeff[None]).sum(-1)
    loss=loss_value(z[None],y[None],factor[None]);gz=torch.autograd.grad(loss,z,retain_graph=True)[0]
    gcs={r:(p*(gz*(zone==k))[:,None]).sum(0) for k,r in enumerate(REGIONS)}
    gcs['total']=sum(gcs.values());direct=torch.autograd.grad(loss,coeff,retain_graph=True)[0]
    if not torch.allclose(gcs['total'],direct,rtol=2e-5,atol=2e-7):raise RuntimeError('Regional coefficient gradient sum failed')
    params=list(head.parameters());gps={}
    for region,gc in gcs.items():
        gp=torch.autograd.grad(coeff,params,grad_outputs=gc.detach(),retain_graph=True)
        gps[region]=torch.cat([v.flatten() for v in gp]).detach()
    error=float((sum(gps[r] for r in REGIONS)-gps['total']).abs().max())
    if error>2e-5:raise RuntimeError('Shared regional VJP sum failed')
    row=dict(**base,loss=float(loss.detach()),coefficient_sum_error=float((gcs['total']-direct).abs().max()),shared_sum_error=error)
    for region in REGIONS:
        row[f'{region}_samples']=int((zone==REGIONS.index(region)).sum())
        row[f'{region}_coefficient_norm']=float(gcs[region].norm())
        row[f'{region}_shared_norm']=float(gps[region].norm())
        row[f'{region}_total_coefficient_cos']=cosine(gcs[region],gcs['total'])
        row[f'{region}_total_shared_cos']=cosine(gps[region],gps['total'])
        if region!='own':
            row[f'own_{region}_coefficient_cos']=cosine(gcs['own'],gcs[region])
            row[f'own_{region}_shared_cos']=cosine(gps['own'],gps[region])
    steps=[]
    with torch.no_grad():
        initial=z.detach();before_bce=F.binary_cross_entropy_with_logits(initial,y,reduction='none')
        for direction in ['total','same','background']:
            dc=-gcs[direction].detach();effect=(p*dc[None]).sum(-1);rms=effect.square().mean().sqrt()
            if float(rms)<1e-12:continue
            for magnitude in [.001,.01]:
                delta=dc*(magnitude/rms);zz=(p*(coeff.detach()+delta)[None]).sum(-1)
                bce=F.binary_cross_entropy_with_logits(zz,y,reduction='none')
                item=dict(**base,direction=direction,requested_logit_rms=magnitude,
                    actual_logit_rms=float((zz-initial).square().mean().sqrt()),loss_before=float(loss),
                    loss_after=float(loss_value(zz[None],y[None],factor[None])),
                    coefficient_relative_step=float(delta.norm()/coeff.norm().clamp_min(1e-12)))
                for k,region in enumerate(REGIONS):
                    take=zone==k
                    item[f'{region}_bce_change']=float((bce-before_bce)[take].mean()) if take.any() else None
                    item[f'{region}_logit_change']=float((zz-initial)[take].mean()) if take.any() else None
                    item[f'{region}_positive_change']=int(((zz[take]>0).long()-(initial[take]>0).long()).sum())
                steps.append(item)
    return row,steps


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;out=a.out;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=Path(json.loads((src/'protocol.json').read_text())['cache']);selection=json.loads((cache/'selection.json').read_text())
    fit=sorted(selection['fit'],key=lambda i:stable_seed('s038-fit',i))[:160];transfer=selection['transfer'];images=fit+transfer
    subset=cache/'conversion_input/instances_probe.json';cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    if sha(subset)!=cr['conversion_input/instances_probe.json']:raise RuntimeError('Changed COCO')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    cfg=get_cfg(overrides=dict(task='segment',imgsz=640,mask_ratio=1,overlap_mask=True,rect=False,cache=False,workers=0,fraction=1.0))
    data=dict(names={k:gt.cats[c]['name'] for k,c in enumerate(sorted(gt.cats))},nc=80,channels=3)
    with contextlib.redirect_stdout(io.StringIO()):ds=build_yolo_dataset(cfg,str(cache/'converted/images/probe'),1,data,mode='val',rect=False)
    dsindex={int(Path(l['im_file']).stem):k for k,l in enumerate(ds.labels)}
    matched=[]
    with (cache/'gt_status.csv').open() as f:
        states={(int(r['image_id']),int(r['annotation_id'])):r for r in csv.DictReader(f) if r['status']=='matched'}
    for (iid,aid),state in states.items():
        if iid not in images:continue
        ann=gt.anns[aid];ordinary=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]
        matched.append((iid,aid,'fit' if iid in fit else 'transfer',ici(ann,ordinary)>.5+1e-10))
    probe=set()
    for split in ['fit','transfer']:
        for high in [False,True]:
            pool=sorted([r for r in matched if r[2]==split and r[3]==high],key=lambda r:stable_seed('s038-gradient',r[0],r[1]))[:32]
            probe.update((r[0],r[1]) for r in pool)
    protocol=dict(experiment='S038_SAMPLING_AND_REGION_GRADIENTS',training=False,source=str(src.resolve()),cache=str(cache.resolve()),
        fit_images=fit,transfer_images=transfer,gradient_targets=sorted(probe),seeds=[0,1,2],pixels=512,
        sampling='Exactly first512fixedreplacement samples reused EACHepoch byS032. NativeFormat-derivedGTbox support rebuilt and count/positions verified. '
            'Not historicalofficialYOLOtraining. Transferpoints never used for fitting; only analogousgeometry diagnostics.',
        zones='OwnrawCOCO positive first, then exclusive sameclassordinaryGT, otherclassGT, background; crowd/padding taggedignored fordiagnostics. '
            'ActualS032loss includes ALL512samples, so ignored contribution explicitly retained in exactgradientpartition.',
        gradients='At saved epoch15 heads; dL/dz of EXACT sampleBCE+Dice partitioned byregion, summed to dc and pertargetsharedMLPparameterVJP. '
            'No independentregionDice redefinition. Within-target gradients, notS022cross-target1x1 or historicalAdamtrajectory.',
        step='Only temporarydc in same32D output, normalize predictedsamplelogitRMS .001/.01; total/same/backgrounddirections. '
            'Measure exactsameobjective, regionalBCE andsignchanges. Not a sharedparametertrainingstep, not heldoutorAP gain.',
        selection='160hashfitimages + all300existingtransfer, allmatchedtargets forsupport;32targets eachsplit/density forgradient hash before outputs.',
        scope='Exploratorysupport and localfirstordertradeoffs; cannot provepastoptimizercause,newmethod,or blameusertraining.',
        hashes=dict(source_receipt=sha(src/'COMPLETE.json'),cache_receipt=sha(cache/'COMPLETE.json'),script=sha(__file__)))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    normalizer=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True);heads={}
    for seed in range(3):
        path=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changedhead')
        head=GlobalHead().cuda();head.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);heads[seed]=head.eval()
    supportrows=[];errorrows=[];gradrows=[];steprows=[];witness=[]
    for number,iid in enumerate(images,1):
        split='fit' if iid in fit else 'transfer';path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changedcache')
        item=read_np(path)
        if not len(item['annotation_ids']):continue
        imageinfo=json.loads((cache/'images'/f'{iid}.json').read_text());lab=ds.labels[dsindex[iid]]
        labelpath=cache/'converted/labels/probe'/f'{iid:012d}.txt'
        if sha(labelpath)!=imageinfo['label_sha256']:raise RuntimeError('ChangednativeTXT')
        raw=deepcopy(lab);raw.pop('shape',None);raw['img']=cv2.imread(lab['im_file']);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        raw=ds.update_labels_info(raw);raw['ori_shape']=shape;raw['ratio_pad']=(1.,1.);raw=LetterBox(new_shape=ishape,auto=False,scaleup=True)(raw)
        native=format_native(raw,imageinfo['native_ids'],1,True)
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        valid=np.zeros(ishape,bool);valid[top:top+rh,left:left+rw]=True
        rasters={};crowd=np.zeros(ishape,bool);union=np.zeros(ishape,bool);ordinary=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]
        for ann in gt.imgToAnns[iid]:
            r=np.zeros(ishape,bool);m=gt.annToMask(ann)
            r[top:top+rh,left:left+rw]=F.interpolate(cuda(m).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool().cpu().numpy()
            if ann.get('iscrowd',0):crowd|=r
            else:rasters[ann['id']]=r;union|=r
        if split=='fit':
            label_file=src/'labels'/f'{iid}.npz'
            if sha(label_file)!=sr[f'labels/{iid}.npz']:raise RuntimeError('Changed training labels')
            saved=read_np(label_file)
            if not np.array_equal(saved['annotation_ids'],item['annotation_ids']):raise RuntimeError('Label identities changed')
        with torch.no_grad():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalizer)
            coeff={s:c+head(x) for s,head in heads.items()}
            binaries={s:ops.process_mask(p,cc,boxes,ishape,upsample=True).cpu().numpy().astype(bool) for s,cc in coeff.items()}
        for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices'])):
            aid,j=int(aid),int(j);ann=gt.anns[aid];own=rasters[aid];same=np.zeros(ishape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==ann['category_id']:same|=rasters[other['id']]
            zone=np.full(ishape,3,np.int8);zone[union]=2;zone[same]=1;zone[own]=0;zone[crowd|~valid]=4
            nbox=cuda(native['boxes'][native['position'][aid]:native['position'][aid]+1]).float()
            bbox=ops.xywh2xyxy(nbox)*640;support=ops.crop_mask(torch.ones((1,ih,iw),device='cuda'),bbox)[0].bool().cpu().numpy()
            pos=item['sample_positions'][k,:512];codes=zone.flatten()[pos];y=own.flatten()[pos].astype(np.float32)
            if int(support.sum())!=int(states[iid,aid]['support_pixels']) or not support.flatten()[pos].all():raise RuntimeError('Actualsupport replay mismatch')
            if split=='fit' and not np.array_equal(y,saved['raw_coco'][k]):raise RuntimeError('ActualYreplay mismatch')
            factor=float(item['loss_factor'][k]);expected=float(support.sum())/(640*640*float(nbox[0,2:].prod()))
            if abs(factor-expected)>2e-6:raise RuntimeError('Actualarea factor changed')
            density=ici(ann,ordinary);base=dict(split=split,image_id=iid,annotation_id=aid,category_id=ann['category_id'],
                size='small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large',ici=density,high=density>.5+1e-10)
            record=dict(**base,support_pixels=int(support.sum()),unique_sample_pixels=len(np.unique(pos)),sample_points=512,
                positive_samples=int(y.sum()),negative_samples=int((1-y).sum()))
            for region,name in enumerate(REGIONS):
                record[f'sample_{name}']=int((codes==region).sum());record[f'support_{name}']=int(((zone==region)&support).sum())
            supportrows.append(record)
            for seed in range(3):
                pred=binaries[seed][j];wrong=pred&(zone==1);bgwrong=pred&(zone==3);fp=int(wrong.sum())
                errorrows.append(dict(**base,seed=seed,neighbor_fp=fp,neighbor_fp_inside_support=int((wrong&support).sum()),
                    neighbor_fp_outside_support=int((wrong&~support).sum()),neighbor_fp_sample_occurrences=int(wrong.flatten()[pos].sum()),
                    neighbor_fp_sample_unique=int(wrong.flatten()[np.unique(pos)].sum()),background_fp=int(bgwrong.sum()),
                    background_fp_inside_support=int((bgwrong&support).sum()),sample_same=int((codes==1).sum()),
                    pred_own_tp=int((pred&(zone==0)).sum()),own_valid_area=int((zone==0).sum())))
                if (iid,aid) in probe:
                    pp=cuda(item['sample_p'][k,:512]).float()
                    gr,ss=gradient_probe(heads[seed],x[j].detach(),c[j].detach(),pp,cuda(y).float(),
                        c.new_tensor(factor),cuda(codes).long(),dict(**base,seed=seed))
                    gradrows.append(gr);steprows.extend(ss)
            witness.append(dict(split=split,image_id=iid,annotation_id=aid,support_count_match=True,sample_inside=True,
                exact_trainingY=split=='fit'))
        if number%40==0 or number==len(images):
            row=dict(images=number,total=len(images),targets=len(supportrows),gradient_rows=len(gradrows),seconds=time.monotonic()-start)
            write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    for name,rr in [('support',supportrows),('errors',errorrows),('gradients',gradrows),('steps',steprows),('witness',witness)]:csv_save(out/f'{name}.csv',rr)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(images),targets=len(supportrows),gradient_targets=len(probe),gradient_rows=len(gradrows),
        seconds=time.monotonic()-start,hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.is_dir():write_json(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
