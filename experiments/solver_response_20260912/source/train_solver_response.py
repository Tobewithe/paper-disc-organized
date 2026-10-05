"""S041: train-only regularized teacher responses versus equal-budget controls.

Three frozen S032 starting states, same h->coefficient head. Ground truth is
used to solve teachers on fit images only; inference takes predictions only.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,contextlib,csv,gzip,io,json,shutil,time
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from readout_spatial_control import solve
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate

MODES=['direct','self_response','solver_response']


def soft_kl(z,target,factor):
    # Constant target entropy removed so reported loss is KL rather than CE.
    prob=target.sigmoid()
    entropy=F.binary_cross_entropy_with_logits(target,prob,reduction='none')
    return ((F.binary_cross_entropy_with_logits(z,prob,reduction='none')-entropy).mean(1)*factor).mean()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;out=a.out;out.mkdir(exist_ok=False);(out/'source').mkdir();(out/'teachers').mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=Path(json.loads((src/'protocol.json').read_text())['cache']);selection=json.loads((cache/'selection.json').read_text())
    fit=selection['fit'];transfer=selection['transfer']
    if len(fit)!=1200 or len(transfer)!=300 or set(fit)&set(transfer):raise RuntimeError('Cohort changed')
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    protocol=dict(experiment='S041_SOLVER_RESPONSE_LEARNABILITY',source=str(src.resolve()),cache=str(cache.resolve()),
        fit_images=fit,transfer_images=transfer,seeds=[0,1,2],modes=MODES,epochs=15,batch_targets=32,pixels=512,
        lr=1e-4,optimizer='Adam with S032 saved moment states',additional_updates_per_arm=3675,
        init='Eachseed continues its S032 rawCOCO epoch15 GlobalHead, same normalizer and optimizer state. '
             'Reuse saved NumPy orderstate and torch RNG for all three arms. Final additionalepoch15 only; 3arms*3seeds*15=135 checkpoints.',
        teacher='Fitimagesonly; same original512coordinates/rawCOCOY. S040 global32 regularized convex solve: '
            'meanBCE + .01 mean(delta_logits^2) + .0001 sum(normalizedweights^2). '
            'RMSnormalizePchannels and basis using fit512,floor.001. Max30Newton/tol1e-7/backtrack20, no new parameters. '
            'Teacher logits=savedseedmodel logits plus optimum response. Keep all finite fits, record convergence and extremes, no outcome filtering.',
        loss='direct: original factor-weighted BCE+Dice; self_response: same plus factor-weighted Bernoulli KL from savedinitial logits; '
            'solver_response: same plus identical KL from GTfit teacher logits. KL weight1,T1,512samepixels. '
            'Sigmoid converts teacher responses to bounded probabilities; no direct coefficient regression or clipping chosen after results.',
        freeze='Backbone/prototypes/h/boxes/scores/classes/candidate identity/fixedbbox50 fit attribution stay frozen. '
            'Only existing GlobalHead parameters updated, output original32 coefficient and official normal crop. NoGT input at inference.',
        evaluation='All300existingtransfertrain2017 images and allordinaryGT, officialCOCOeval before instanceICI grouping. '
            'Explored screening set not new confirmation set. No new val evaluation. Compare all3seeds against current direct and self-response controls, not only old original.',
        stopping='Fixed one recipe, no epoch/lambda/seed selection. Need positive highR75 vs both equalbudgetcontrols with AP maintained '
            'and gap nonhigh-high not worsened before new confirmation images. If no effect stop this recipe. '
            'Positive on screening is not method acceptance, novelty or causality.',
        source_hashes=dict(script=sha(__file__),solver=sha(Path(__file__).with_name('readout_spatial_control.py')),cache_receipt=sha(cache/'COMPLETE.json'),source_receipt=sha(src/'COMPLETE.json')))
    write_json(out/'protocol.json',protocol)
    for name in ['train_solver_response.py','readout_spatial_control.py','rich_pixel_readout.py','summarize_relative_ownership.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Changed normalizer')
    torch.save({k:v.cpu() for k,v in normalization.items()},out/'normalizer.pt')
    checkpoints={};starts={};weights={}
    for seed in [0,1,2]:
        path=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changed startingstate')
        ck=torch.load(path,map_location='cuda',weights_only=False);checkpoints[seed]=ck;weights[str(seed)]=sha(path)
        model=GlobalHead().cuda();model.load_state_dict(ck['model']);starts[seed]=model.eval().requires_grad_(False)
    records=[];identities=[];progress('load_fit')
    for number,iid in enumerate(fit,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changed fitcache')
        item=read_np(path);idx=item['prediction_indices']
        if not len(idx):continue
        label=src/'labels'/f'{iid}.npz'
        if sha(label)!=sr[f'labels/{iid}.npz']:raise RuntimeError('Changed fitlabel')
        lab=read_np(label)
        if not np.array_equal(lab['annotation_ids'],item['annotation_ids']):raise RuntimeError('Label identity')
        with torch.no_grad():
            c=cuda(item['coeff']).float();boxes=cuda(item['boxes']).float();h=cuda(item['h']).float();lv=cuda(item['level']).long()
            xx=norm(instance_features(h,lv,boxes,tuple(item['input_shape'])),normalization)
            cc=torch.stack([(c+starts[s](xx))[idx] for s in range(3)],1)
        records.append(dict(x=xx[idx].cpu(),c=c[idx].cpu(),startc=cc.cpu(),p=torch.tensor(item['sample_p'][:,:512]).float(),
            y=torch.tensor(lab['raw_coco']).float(),factor=torch.tensor(item['loss_factor']).float()))
        identities.extend((iid,int(aid),int(item['source_index'][j])) for aid,j in zip(item['annotation_ids'],idx))
        if number%200==0:progress('load_fit',images=number,total=1200)
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records
    count=len(identities)
    if count!=7811:raise RuntimeError('Fit target denominator')
    write_json(out/'fit_metadata.json',dict(targets=count,identities=identities,starting_checkpoint_sha256=weights))
    teacherrows=[];targetlogits={};startlogits={}
    progress('teacher',targets=count,seeds=3)
    for seed in range(3):
        basez=(data['p']*data['startc'][:,seed,None]).sum(-1);teach=torch.empty_like(basez);deltas=[]
        for k in range(count):
            pp=data['p'][k];rms=pp.square().mean(0).sqrt().clamp_min(.001);phi=pp/rms
            scale=phi.square().mean(0).sqrt().clamp_min(.001);design=phi/scale
            w,info=solve(design,basez[k],data['y'][k]);dc=w/(rms*scale);tt=basez[k]+design@w
            if not torch.isfinite(tt).all():raise RuntimeError('Nonfinite teacher')
            teach[k]=tt;deltas.append(dc);info.pop('trace')
            teacherrows.append(dict(seed=seed,image_id=identities[k][0],annotation_id=identities[k][1],
                coefficient_norm_ratio=float((data['startc'][k,seed]+dc).norm()/data['startc'][k,seed].norm().clamp_min(1e-12)),
                bce_before=float(F.binary_cross_entropy_with_logits(basez[k],data['y'][k])),
                bce_after=float(F.binary_cross_entropy_with_logits(tt,data['y'][k])),**info))
            if (k+1)%2000==0:progress('teacher',seed=seed,targets=k+1,total=count)
        torch.save(dict(logits=teach.cpu(),start_logits=basez.cpu(),delta_coefficients=torch.stack(deltas).cpu(),identities=identities),out/'teachers'/f'seed{seed}.pt')
        targetlogits[seed]=teach;startlogits[seed]=basez
        csv_save(out/'teacher_fit.csv',teacherrows);progress('teacher_seed_complete',seed=seed)
    write_json(out/'TEACHERS_COMPLETE.json',dict(status='COMPLETE',fit_images=1200,fit_targets=count,seeds=3,
        nonconverged=sum(not r['converged'] for r in teacherrows),max_gradient_inf=max(r['gradient_inf'] for r in teacherrows),
        hashes={p.name:sha(p) for p in (out/'teachers').glob('*.pt')}))
    trained=[];history=[];fitmetrics=[]
    def measure(model,seed):
        sums=dict(hard=0.,teacher_kl=0.,start_kl=0.,teacher_mse=0.,sample_iou=0.)
        with torch.no_grad():
            for first in range(0,count,128):
                sl=slice(first,first+128);pp=data['p'][sl];z=(pp*(data['c'][sl]+model(data['x'][sl]))[:,None]).sum(-1)
                y=data['y'][sl];factor=data['factor'][sl];n=len(y)
                sums['hard']+=float(loss_value(z,y,factor))*n
                sums['teacher_kl']+=float(soft_kl(z,targetlogits[seed][sl],factor))*n
                sums['start_kl']+=float(soft_kl(z,startlogits[seed][sl],factor))*n
                sums['teacher_mse']+=float((z-targetlogits[seed][sl]).square().mean())*n
                binary=z>0;yy=y.bool();sums['sample_iou']+=float(((binary&yy).sum(1)/(binary|yy).sum(1).clamp_min(1)).sum())
        return {k:v/count for k,v in sums.items()}
    for seed in range(3):
        initial=measure(starts[seed],seed);fitmetrics.append(dict(mode='saved',seed=seed,**initial))
        ck=checkpoints[seed]
        for mode in MODES:
            model=GlobalHead().cuda();model.load_state_dict(ck['model']);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            opt.load_state_dict(deepcopy(ck['optimizer']))
            torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
            rng=np.random.default_rng();rng.bit_generator.state=deepcopy(ck['numpy_rng'])
            dest=out/f'{mode}_s{seed}';(dest/'checkpoints').mkdir(parents=True);updates=0
            for epoch in range(15):
                model.train();order=rng.permutation(count);hardtotal=0.;softtotal=0.
                for first in range(0,count,32):
                    ids=cuda(order[first:first+32]).long();z=(data['p'][ids]*(data['c'][ids]+model(data['x'][ids]))[:,None]).sum(-1)
                    hard=loss_value(z,data['y'][ids],data['factor'][ids]);soft=z.new_zeros(())
                    if mode!='direct':soft=soft_kl(z,(startlogits if mode=='self_response' else targetlogits)[seed][ids],data['factor'][ids])
                    loss=hard+soft
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    hardtotal+=float(hard.detach())*len(ids);softtotal+=float(soft.detach())*len(ids);updates+=1
                row=dict(mode=mode,seed=seed,epoch=epoch+1,total_epoch=epoch+16,updates=updates,hard_loss=hardtotal/count,soft_loss=softtotal/count)
                history.append(row)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,updates=updates,total_epoch=epoch+16,
                    mode=mode,seed=seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state),dest/'checkpoints'/f'epoch{epoch+1:03d}.pt')
                write_json(out/'history.json',history)
                if (epoch+1)%5==0:progress('train',**row)
            model.eval();fitmetrics.append(dict(mode=mode,seed=seed,**measure(model,seed)));trained.append((mode,seed,model))
            write_json(dest/'COMPLETE.json',dict(status='COMPLETE',updates=updates,checkpoints=15,parameters=sum(v.numel() for v in model.parameters()),
                initial_sha256=weights[str(seed)],final_sha256=sha(dest/'checkpoints/epoch015.pt')))
    write_json(out/'fit_metrics.json',fitmetrics);del data,targetlogits,startlogits;torch.cuda.empty_cache();progress('training_complete',runs=9,checkpoints=135)
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json']:raise RuntimeError('Changed rawGT')
    obj=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(info=obj.get('info',{}),categories=obj['categories'],
        images=[im for im in obj['images'] if im['id'] in transfer],annotations=[an for an in obj['annotations'] if an['image_id'] in transfer])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    meta={ann['id']:{'ici_same':ici(ann,[b for b in gt.imgToAnns[ann['image_id']] if not b.get('iscrowd',0)])} for ann in gt.anns.values() if not ann.get('iscrowd',0)}
    categories=sorted(gt.cats);arms=[('original',-1,None)]+[('saved',s,h) for s,h in starts.items()]+trained
    predictions={f'{m}_s{s}_d0':[] for m,s,_ in arms};spatial=[]
    for num,iid in enumerate(transfer,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changed evalcache')
        item=read_np(path);c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
        det=cuda(item['detections']).float();h=cuda(item['h']).float();level=cuda(item['level']).long();shape=tuple(item['shape']);ishape=tuple(item['input_shape'])
        mapping={int(j):int(t) for t,j in zip(item['annotation_ids'],item['prediction_indices'])}
        masks={ann['id']:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        with torch.inference_mode():
            x=norm(instance_features(h,level,boxes,ishape),normalization)
            for mode,seed,model in arms:
                name=f'{mode}_s{seed}_d0';coeff=c if model is None else c+model(x)
                bb=ops.process_mask(p,coeff,boxes,ishape,upsample=True);pm=ops.scale_masks(bb[:,None],shape)[:,0]>.5 if len(c) else bb
                for j in range(len(c)):
                    pred=pm[j].cpu().numpy()
                    if bool(bb[j].any()):
                        rle=mu.encode(np.asfortranarray(pred.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        predictions[name].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in mapping:
                        aid=mapping[j];ann=gt.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());vp=pred&~crowd;same=np.zeros(shape,bool)
                        for other in gt.imgToAnns[iid]:
                            if not other.get('iscrowd',0) and other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']]
                        un=int((pred|masks[aid]).sum())
                        spatial.append(dict(arm=name,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],iou=int((pred&masks[aid]).sum())/un if un else 1.,
                            coverage=float((vp&own).sum()/area) if area else None,neighbor=float((vp&same&~own).sum()/area) if area else None,background=float((vp&~union).sum()/area) if area else None))
        if num%30==0:progress('decode',images=num,total=300)
    (out/'predictions').mkdir();parity=[];task=[];gtrows=[];pairs=[]
    for name,pp in predictions.items():
        with gzip.open(out/'predictions'/f'{name}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        if name.startswith('original') or name.startswith('saved'):
            oldname=name.replace('saved','raw_coco')
            with gzip.open(src/'predictions'/f'{oldname}.json.gz','rt',encoding='utf-8') as f:old=json.load(f)
            exact=old==pp;parity.append(dict(arm=name,exact=exact,predictions=len(pp)))
            if not exact:raise RuntimeError('Saved prediction replay')
        if not pp:raise RuntimeError('All-empty arm needs explicit evaluation')
        row,rr,pr=evaluate(gt,meta,transfer,pp,name)
        if len(rr)!=len(meta):raise RuntimeError('GT denominator')
        row['gap']=row['r75_low']-row['r75_high'];task.append(row);gtrows.extend(rr);pairs.extend(pr);progress('task',**row)
    for name,rows in [('task_summary',task),('gt_recovery',gtrows),('pair_recovery',pairs),('spatial',spatial)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'prediction_parity.json',parity)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,fit_targets=count,teacher_fits=len(teacherrows),
        transfer_images=len(transfer),ordinary_gt=len(meta),checkpoints=135,runs=9,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'checkpoints' not in str(p)}))
    progress('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        dest=Path(sys.argv[sys.argv.index('--out')+1])
        if dest.is_dir():write_json(dest/'FAILED.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise
