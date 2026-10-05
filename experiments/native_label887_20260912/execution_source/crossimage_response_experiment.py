"""Fit frozen response readouts on train, train-calibrate, lock, evaluate."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,Capture,ownership,sha,write_json
from relative_ownership_experiment import read,spatial_and_predictions,cohort_summary
from three_region_probe import write_csv
from crossimage_response_decoder import prepare,features,score,decode


def rank(value):return hashlib.sha256(f'crossimage-response:20260911:{value}'.encode()).hexdigest()


def choose(config):
    old=set(json.loads((ROOT/'diagnostics/coefficient_pilot_cache_v4_20260911/selection.json').read_text())['train'])
    old|=set(json.loads((ROOT/'diagnostics/structure_train_witness32_v2_20260911/protocol.json').read_text())['images'])
    relative=json.loads((ROOT/'diagnostics/relative_ownership_20260911/selection.json').read_text());old|=set(relative['development'])
    pool={int(p.stem) for p in (ROOT/'data/images/train2017').glob('*.jpg')}-old
    candidates=sorted(pool,key=lambda iid:rank(f'train:{iid}'));n=config['fit_images'];m=config['calibration_images']
    assert len(candidates)>=n+m
    usedval=set(relative['evaluation'])|set(json.loads((ROOT/'diagnostics/structure_main300_20260911/protocol.json').read_text())['images'])
    vp={int(p.stem) for p in (ROOT/'diagnostics/full_val_cache_20260911/val').glob('*.npz')}-usedval
    val=sorted(sorted(vp,key=lambda iid:rank(f'val:{iid}'))[:config['evaluation_images']])
    result=dict(fit=sorted(candidates[:n]),calibration=sorted(candidates[n:n+m]),evaluation=val,available_train=len(pool),excluded_train=len(old),excluded_val=len(usedval))
    assert not set(result['fit'])&set(result['calibration']) and not set(candidates[:n+m])&set(val)
    return result


def load(path):
    with np.load(path) as q:return {k:q[k] for k in ['proto','coeff','boxes','detections','shape','input_shape','mapping_gt','mapping_pred']}


def gt_input_regions(gt,iid,item):
    h,w=map(int,item['shape']);ih,iw=map(int,item['input_shape']);gain=min(ih/h,iw/w)
    rh,rw=round(h*gain),round(w*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
    rasters={};union=torch.zeros((ih,iw),device='cuda',dtype=torch.bool);crowd=torch.zeros_like(union)
    for a in gt.imgToAnns[iid]:
        original=torch.tensor(gt.annToMask(a),device='cuda',dtype=torch.float32)
        resized=F.interpolate(original[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
        mask=torch.zeros_like(union);mask[top:top+rh,left:left+rw]=resized
        if a.get('iscrowd',0):crowd|=mask
        else:rasters[a['id']]=mask;union|=mask
    valid=torch.zeros_like(union);valid[top:top+rh,left:left+rw]=True;valid&=~crowd
    return rasters,union,valid


def fit_models(out,gt,chosen):
    pieces={seed:[] for seed in range(3)};status=[];started=time.monotonic()
    for num,iid in enumerate(chosen['fit'],1):
        item=load(out/'fit_cache'/f'{iid}.npz');prepared=prepare(item)
        raster,union,valid=gt_input_regions(gt,iid,item);mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
        for aid,j in mapping.items():
            if not len(prepared['neighbors'][j]):status.append(dict(image_id=iid,annotation_id=aid,status='no_predicted_neighbor'));continue
            support=prepared['support'][j]&valid;own=raster[aid]&support;near=union&~raster[aid]&support;bg=~union&support
            pools=[r.flatten().nonzero().flatten().cpu().numpy() for r in [own,near,bg]]
            if not len(pools[0]) or not (len(pools[1]) or len(pools[2])):
                status.append(dict(image_id=iid,annotation_id=aid,status='insufficient_positive_or_negative'));continue
            counts=[64,32 if len(pools[1]) and len(pools[2]) else 64 if len(pools[1]) else 0,32 if len(pools[1]) and len(pools[2]) else 64 if len(pools[2]) else 0]
            x=features(prepared,j,iid).reshape(-1,3)
            for seed in range(3):
                rng=np.random.default_rng(20260911+seed*7907+iid*31+aid*7)
                pos=np.concatenate([rng.choice(pool,n,replace=len(pool)<n) for pool,n in zip(pools,counts) if n])
                xx=x[torch.tensor(pos,device='cuda')].double().cpu().numpy()
                sham=features(prepared,j,iid,seed,True).reshape(-1,3)[torch.tensor(pos,device='cuda')].double().cpu().numpy()
                y=np.concatenate([np.ones(64),-np.ones(64)])
                pieces[seed].append((xx,sham,y,np.full(128,iid),np.full(128,aid)))
            status.append(dict(image_id=iid,annotation_id=aid,status='ok',own_pool=len(pools[0]),neighbor_pool=len(pools[1]),background_pool=len(pools[2]),
                               own_samples=counts[0],neighbor_samples=counts[1],background_samples=counts[2]))
        if num%40==0 or num==len(chosen['fit']):print(json.dumps(dict(phase='fit_samples',images=num,total=len(chosen['fit']),seconds=round(time.monotonic()-started,1))),flush=True)
    models={};ownmodels={}
    for seed,entries in pieces.items():
        assert entries,'No training samples'
        x,sham,y,iids,aids=[np.concatenate([entry[j] for entry in entries]) for j in range(5)]
        np.savez_compressed(out/f'fit_samples_s{seed}.npz',joint=x.astype(np.float32),sham=sham.astype(np.float32),y=y.astype(np.int8),image_id=iids,annotation_id=aids)
        for kind,data in [('own',x[:,:1]),('joint',x),('sham',sham)]:
            mu=data.mean(0);std=data.std(0).clip(.01);tr=(data-mu)/std
            w=np.linalg.solve(tr.T@tr/len(tr)+.1*np.eye(data.shape[1]),tr.T@(y-y.mean())/len(tr))
            raw=w/std;intercept=float(y.mean()-mu@raw);slope=float(raw[0]/5)
            assert np.isfinite(raw).all() and np.isfinite(intercept) and slope>1e-8,(seed,kind,raw,intercept)
            if kind=='own':raw=np.r_[raw,0.,0.]
            model=dict(kind=kind,seed=seed,raw_weights=raw.tolist(),raw_intercept=intercept,own_logit_slope=slope,
                       samples=len(y),targets=len(entries),fit_images=len(np.unique(iids)),feature_mean=mu.tolist(),feature_std=std.tolist())
            if kind=='own':ownmodels[f'own_s{seed}']=model
            else:models[f'{kind}_s{seed}']=model
    write_json(out/'MODELS.json',dict(models=models,own_models=ownmodels,selection='No validation selection, fixed ridge0.1; all fitted arms retained'))
    write_csv(out/'fit_target_status.csv',status)
    return models,ownmodels


@torch.inference_mode()
def calibration_pass(out,gt,meta,ids,models,grid,label,ownmodels=None):
    totals={arm:np.zeros(len(values)) for arm,values in grid.items()};base=0.;targets=0;eligible=0;ownxor=0
    pertarget=[]
    for iid in ids:
        item=load(out/'calibration_cache'/f'{iid}.npz');pr=prepare(item);shape=tuple(map(int,item['shape']));mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
        crowd=torch.zeros(shape,device='cuda',dtype=torch.bool)
        for a in gt.imgToAnns[iid]:
            if a.get('iscrowd',0):crowd|=torch.tensor(gt.annToMask(a).astype(bool),device='cuda')
        for aid,j in mapping.items():
            if float(meta[aid]['ici_same'])<=.5+1e-10:continue
            own=torch.tensor(gt.annToMask(gt.anns[aid]).astype(bool),device='cuda')&~crowd;area=int(own.sum())
            if not area:continue
            initial=ops.scale_masks(pr['initial'][j:j+1,None],shape)[0,0]>.5
            coverage=float((initial&own).sum()/area);base+=coverage;targets+=1
            active=bool(len(pr['neighbors'][j]));eligible+=active
            for arm,thresholds in grid.items():
                if not active:
                    values=np.full(len(thresholds),coverage)
                else:
                    model=models[arm];xx=features(pr,j,iid,model['seed'],model['kind']=='sham');s=score(xx,model)
                    values=[]
                    for offset in range(0,len(thresholds),4):
                        tau=torch.tensor(thresholds[offset:offset+4],device='cuda',dtype=torch.float32)
                        binary=((s[None]>tau[:,None,None])&pr['support'][j][None]).to(torch.uint8)
                        mask=ops.scale_masks(binary[:,None],shape)[:,0]>.5
                        values.extend(((mask&own[None]).flatten(1).sum(1).double()/area).cpu().tolist())
                    values=np.array(values)
                totals[arm]+=values
                pertarget.extend(dict(phase=label,image_id=iid,target_annotation=aid,arm=arm,threshold=float(tau),coverage=float(v),initial_coverage=coverage,eligible=active) for tau,v in zip(thresholds,values))
            if ownmodels and active:
                xx=features(pr,j,iid)
                for ownmodel in ownmodels.values():
                    # In a positive affine own-only readout, the intercept is the exact original-sign threshold.
                    original_sign=(score(xx,ownmodel)>ownmodel['raw_intercept']/ownmodel['own_logit_slope'])&pr['support'][j]
                    ownxor+=int((original_sign!=pr['initial'][j].bool()).sum())
    assert targets>0
    # Floating point cancellation can create ties close to zero; require exact binary parity here.
    if ownmodels:assert ownxor==0,('own-only affine parity',ownxor)
    rows=[dict(phase=label,arm=arm,threshold=float(tau),coverage=float(value/targets),initial_coverage=base/targets,high_targets=targets,eligible_high=eligible) for arm,values in totals.items() for tau,value in zip(grid[arm],values)]
    write_csv(out/f'{label}_curve.csv',rows);write_csv(out/f'{label}_targets.csv',pertarget)
    print(json.dumps(dict(phase=label,high_targets=targets,eligible_high=eligible,initial_coverage=base/targets,own_only_xor=ownxor)),flush=True)
    return {arm:totals[arm]/targets for arm in totals},base/targets,targets,ownxor


def calibrate(out,gt,meta,chosen,models,ownmodels):
    initial=[-20.,-10.,-5.,-2.,-1.,0.,1.,2.,5.,10.,20.];grid={arm:initial for arm in models}
    curves,target,n,xor=calibration_pass(out,gt,meta,chosen['calibration'],models,grid,'calibration_grid',ownmodels)
    bracket={};selected={}
    for arm,values in curves.items():
        assert np.all(np.diff(values)<=1e-12) and values[0]>=target>=values[-1],('unbracketed',arm,target,values)
        candidates=np.flatnonzero(values>=target);j=min(int(candidates[-1]),len(initial)-2)
        lo,hi=initial[j],initial[j+1];vlo,vhi=values[j],values[j+1]
        bracket[arm]=[lo,hi,float(vlo),float(vhi)]
        selected[arm]=float(lo+(hi-lo)*(vlo-target)/max(vlo-vhi,1e-12))
    for attempt in range(4):
        grid={arm:[tau] for arm,tau in selected.items()};result,_,_,_=calibration_pass(out,gt,meta,chosen['calibration'],models,grid,f'calibration_confirm{attempt}')
        gaps={arm:float(values[0]-target) for arm,values in result.items()}
        if all(abs(gap)<=.0025 for gap in gaps.values()):break
        if attempt==3:
            write_json(out/'CALIBRATION_FAILED.json',dict(gaps=gaps,thresholds=selected));raise RuntimeError('Train coverage confirmation failed, no val rescue')
        for arm,gap in gaps.items():
            if abs(gap)<=.0025:continue
            lo,hi,vlo,vhi=bracket[arm]
            if gap>0:lo=selected[arm];vlo=float(result[arm][0])
            else:hi=selected[arm];vhi=float(result[arm][0])
            bracket[arm]=[lo,hi,vlo,vhi];selected[arm]=(lo+hi)/2
    lock=dict(status='LOCKED_BEFORE_EVALUATION',models=models,thresholds=selected,calibration_high_targets=n,
        calibration_initial_coverage=target,confirmed_gap_pp={k:v*100 for k,v in gaps.items()},own_affine_pixel_xor=xor,
        sources={p.name:sha(p) for p in [out/'selection.json',out/'MODELS.json',out/'protocol.json',Path(__file__),Path(__file__).with_name('crossimage_response_decoder.py')]})
    write_json(out/'LOCKED_SETTINGS.json',lock);return lock


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['fit','evaluate'],required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=a.out
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();configpath=Path(__file__).with_name('crossimage_response_protocol.json')
    if a.phase=='fit':
        out.mkdir(parents=True,exist_ok=False);config=json.loads(configpath.read_text())
        if a.smoke:config.update(fit_images=8,calibration_images=8,evaluation_images=8)
        chosen=choose(config);write_json(out/'selection.json',chosen)
        write_json(out/'protocol.json',dict(config=config,smoke=a.smoke,script_sha256=sha(__file__),decoder_sha256=sha(Path(__file__).with_name('crossimage_response_decoder.py')),
            selection_sha256=sha(out/'selection.json'),weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),ops_sha256=sha(ops.__file__),
            annotations={s:sha(ROOT/f'data/annotations/instances_{s}2017.json') for s in ['train','val']}))
        with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_train2017.json'))
        meta={int(r['annotation_id']):r for r in read(ROOT/'census/train2017_instances.csv')}
        model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);hashes={}
        for split in ['fit','calibration']:
            directory=out/f'{split}_cache';directory.mkdir()
            for num,iid in enumerate(chosen[split],1):
                with torch.inference_mode():model.predict(str(ROOT/'data/images/train2017'/gt.imgs[iid]['file_name']),predictor=Capture,imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,half=False,retina_masks=False,device=0,verbose=False)
                cap=model.predictor.capture;mapping=ownership(gt,iid,cap['detections']);item={k:v.cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in cap.items()}
                item.update(mapping_gt=np.array(list(mapping),dtype=np.int64),mapping_pred=np.array(list(mapping.values()),dtype=np.int64))
                path=directory/f'{iid}.npz';np.savez_compressed(path,**item);hashes[str(path.relative_to(out))]=sha(path)
                if num%40==0 or num==len(chosen[split]):print(json.dumps(dict(phase=split+'_cache',images=num,total=len(chosen[split]),seconds=round(time.monotonic()-started,1))),flush=True)
        write_json(out/'TRAIN_CACHE_HASHES.json',hashes);del model;torch.cuda.empty_cache()
        models,ownmodels=fit_models(out,gt,chosen);lock=calibrate(out,gt,meta,chosen,models,ownmodels)
        write_json(out/'FIT_COMPLETE.json',dict(status='COMPLETE',network_training=False,seconds=time.monotonic()-started,
            hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(phase='fit_complete',thresholds=lock['thresholds'],confirmed_gap_pp=lock['confirmed_gap_pp'])),flush=True)
    else:
        receipt=json.loads((out/'FIT_COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
        for name,digest in receipt['hashes'].items():assert sha(out/name)==digest,name
        protocol=json.loads((out/'protocol.json').read_text());assert protocol['script_sha256']==sha(__file__);assert protocol['decoder_sha256']==sha(Path(__file__).with_name('crossimage_response_decoder.py'))
        chosen=json.loads((out/'selection.json').read_text());lock=json.loads((out/'LOCKED_SETTINGS.json').read_text())
        assert not (out/'EVALUATION_COMPLETE.json').exists()
        with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
        meta={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')}
        allrows=[];changes=[];skips=[];counts=[];hashes={};predhash={}
        for num,iid in enumerate(chosen['evaluation'],1):
            path=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';item=load(path);hashes[str(iid)]=sha(path)
            inputs={k:item[k] for k in ['proto','coeff','boxes','detections','input_shape']}
            for domain,expansion in [('normal',0.),('expand20',.2)]:
                prepared=prepare(inputs,expansion);masks,cc=decode(prepared,iid,lock['models'],lock['thresholds'])
                rows,preds,ss=spatial_and_predictions(gt,iid,item,masks,meta,True)
                allrows.extend(dict(domain=domain,**r) for r in rows);changes.extend(dict(domain=domain,**r) for r in cc);skips.extend(dict(domain=domain,**r) for r in ss)
                for arm,pp in preds.items():
                    directory=out/'predictions_by_image'/domain/arm;directory.mkdir(parents=True,exist_ok=True)
                    path=directory/f'{iid}.json.gz'
                    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
                    predhash[str(path.relative_to(out))]=sha(path)
                del masks,prepared
            counts.append(dict(image_id=iid,gt=sum(not x.get('iscrowd',0) for x in gt.imgToAnns[iid]),matched=len(item['mapping_gt']),predictions=len(item['coeff'])))
            if num%20==0 or num==len(chosen['evaluation']):
                progress=dict(phase='evaluation',images=num,total=len(chosen['evaluation']),seconds=round(time.monotonic()-started,1));print(json.dumps(progress),flush=True);write_json(out/'progress.json',progress)
        write_csv(out/'evaluation_spatial.csv',allrows);write_csv(out/'evaluation_changes.csv',changes);write_csv(out/'evaluation_counts.csv',counts);write_csv(out/'evaluation_skips.csv',skips)
        summary=[]
        for domain in ['normal','expand20']:summary.extend(dict(domain=domain,**r) for r in cohort_summary([r for r in allrows if r['domain']==domain]))
        write_csv(out/'spatial_summary.csv',summary);write_json(out/'EVALUATION_CACHE_HASHES.json',hashes);write_json(out/'PREDICTION_HASHES.json',predhash)
        write_json(out/'EVALUATION_COMPLETE.json',dict(status='COMPLETE',network_training=False,images=len(chosen['evaluation']),seconds=time.monotonic()-started,lock_sha256=sha(out/'LOCKED_SETTINGS.json'),
            hashes={p.name:sha(p) for p in [out/'evaluation_spatial.csv',out/'evaluation_changes.csv',out/'evaluation_counts.csv',out/'spatial_summary.csv',out/'EVALUATION_CACHE_HASHES.json',out/'PREDICTION_HASHES.json']}))


if __name__=='__main__':main()
