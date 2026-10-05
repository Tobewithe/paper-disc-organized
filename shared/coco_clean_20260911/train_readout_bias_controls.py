"""S035: train-only scalar logit offsets vs the saved S032 coefficient head.

All inputs, native640 positions, rawCOCO labels and prediction slots inherited.
No test-GT threshold search; no coefficient refit or new end-to-end training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,gzip,io,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate


class ConstantBias(nn.Module):
    def __init__(self):
        super().__init__();self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x):return self.bias.expand(len(x))


class InstanceBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(73,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)[:,0]


def fit_constant_fullbatch(z,y,factor):
    """One deterministic train-only numerical reference; not equal-budget SGD."""
    zz=z.double();yy=y.double();ff=factor.double();bias=nn.Parameter(torch.zeros((),dtype=torch.float64,device=z.device))
    opt=torch.optim.LBFGS([bias],lr=1.,max_iter=100,max_eval=200,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    initial=float(loss_value(zz,yy,ff))
    def closure():
        opt.zero_grad();loss=loss_value(zz+bias,yy,ff)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite fullbatch loss')
        loss.backward();return loss
    opt.step(closure);loss=loss_value(zz+bias,yy,ff);gradient=torch.autograd.grad(loss,bias)[0]
    info=dict(bias=float(bias.detach()),initial_loss=initial,final_loss=float(loss.detach()),gradient=float(gradient.detach()),
        iterations=int(opt.state[bias]['n_iter']),evaluations=int(opt.state[bias]['func_evals']),
        interpretation='One train-only stationary search fromzero,FP64fullbatch100LBFGS;not equalAdam budget or guaranteedglobaloptimum.')
    if info['final_loss']>initial+1e-8:raise RuntimeError('Fullbatch scalar loss increased')
    return info


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;out=a.out;out.mkdir(exist_ok=False);(out/'source').mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=Path(json.loads((src/'protocol.json').read_text())['cache'])
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    selection=json.loads((cache/'selection.json').read_text());fit=selection['fit'];transfer=selection['transfer']
    protocol=dict(experiment='S035_LOGIT_BIAS_CONTROLS',source=str(src.resolve()),cache=str(cache.resolve()),
        fit_images=fit,transfer_images=transfer,network_training='Onlysmallbiasheads;originalYOLOandS032coefficientsfrozen',
        seeds=[0,1,2],epochs=15,batch_targets=32,lr=1e-4,optimizer='Adam',pixels=512,
        new_modes=['constant','instance_bias'],saved_control='S032raw_coco3seedglobal32Dcoefficientresidual,final15',
        constant='One global scalar b, z_new(x)=z_original(x)+b; zero initial.',
        instance_bias='73Dsameh+level+box normalized fromS032 ->128SiLU->128SiLU->1; zero finalprojection. '
            'One scalar per prediction, constant overpixels, rankingpreserving. Samehiddenwidth but differentparametercount from32Dcontrol.',
        objective='ExactsameS032rawCOCOY,first512nativeGTboxsamplepositions,factors,BCE+Dice,initial0residual,seedshuffledorder,Adam,gradclip10,15epochs.',
        scalar_fullbatch_reference='Additional single deterministic train-only scalar stationary solve fromzero,FP64LBFGS100/maxeval200. '
            'Reports differing solverbudget explicitly, no testlabels, no bestof testchoices; not countedas a3seednewmodel.',
        decode='Official originalc@P then bilinearupsample640, add scalar at640, >0binary then SAMEoriginalpredictedcrop andscale_masks. '
            'NoGTinfeatures,bias,ranking,selectionorfinalcrop. Existingoriginal andS032coeffexactJSONreplay required.',
        evaluation='All300exploredtrain2017images/all2002ordinaryGT officialCOCOmatchingbeforeICI. '
            'Reportallseeds/arms/finalepochs, no thresholdsearch ontransfer. Newempty masks followstockfilter.',
        limits='Existingcalibration controls,notnovelmethod; sametrainbudget notsameparametercapacity. '
            'Remainingneighborrankingfailure and densegaptarget stillunresolved. NoformalCCLorendtoendtraining.',
        source_hashes=dict(source_receipt=sha(src/'COMPLETE.json'),cache_receipt=sha(cache/'COMPLETE.json'),script=sha(__file__)))
    write_json(out/'protocol.json',protocol)
    for name in ['train_readout_bias_controls.py','rich_pixel_readout.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Changednormalizer')
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    models={};checkpoints=[]
    for seed in [0,1,2]:
        cp=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(cp)!=json.loads((cp.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changedcoeffcheckpoint')
        m=GlobalHead().cuda();m.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['model'])
        models['coefficient',seed]=m.eval();checkpoints.append(dict(seed=seed,path=str(cp),sha256=sha(cp)))
    write_json(out/'saved_coefficient_control.json',checkpoints)
    records=[];identities=[];progress('build_fit')
    for num,iid in enumerate(fit,1):
        npz=cache/'images'/f'{iid}.npz'
        if sha(npz)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changedcache')
        item=read_np(npz);idx=item['prediction_indices']
        if not len(idx):continue
        labels=src/'labels'/f'{iid}.npz'
        if sha(labels)!=sr[f'labels/{iid}.npz']:raise RuntimeError('Changedrawlabels')
        target=read_np(labels)
        if not np.array_equal(target['annotation_ids'],item['annotation_ids']):raise RuntimeError('Labelidentitymismatch')
        h=cuda(item['h']).float();level=cuda(item['level']).long();boxes=cuda(item['boxes']).float()
        xx=instance_features(h,level,boxes,tuple(item['input_shape']))[idx]
        pp=cuda(item['sample_p'][:,:512]).float();cc=cuda(item['coeff'][idx]).float()
        z=(pp*cc[:,None]).sum(-1)
        records.append(dict(x=xx.cpu(),z=z.cpu(),y=torch.tensor(target['raw_coco']).float(),factor=torch.tensor(item['loss_factor']).float()))
        identities.extend((iid,int(item['source_index'][j])) for j in idx)
        if num%200==0:progress('build_fit',images=num,total=len(fit))
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records
    data['xn']=norm(data['x'],normalization);count=len(identities)
    if count!=7811 or [list(q) for q in identities]!=json.loads((src/'fit_metadata.json').read_text())['identities']:raise RuntimeError('Fittargetschanged')
    write_json(out/'fit_metadata.json',dict(targets=count,identities=identities,normalizer_sha256=sha(src/'normalizer.pt')))
    history=[];fit_metrics=[]
    for seed in [0,1,2]:
        for mode in ['constant','instance_bias']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=(ConstantBias() if mode=='constant' else InstanceBias()).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            rng=np.random.default_rng(seed);dest=out/f'{mode}_s{seed}';(dest/'checkpoints').mkdir(parents=True);updates=0
            for epoch in range(15):
                order=rng.permutation(count);total=0.
                for first in range(0,count,32):
                    ix=cuda(order[first:first+32]).long();z=data['z'][ix]+model(data['xn'][ix])[:,None]
                    loss=loss_value(z,data['y'][ix],data['factor'][ix])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfiniteloss')
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    total+=float(loss.detach())*len(ix);updates+=1
                row=dict(mode=mode,seed=seed,epoch=epoch+1,updates=updates,loss=total/count);history.append(row)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),mode=mode,seed=seed,epoch=epoch+1,updates=updates,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state),dest/'checkpoints'/f'epoch{epoch+1:03d}.pt')
                if (epoch+1)%5==0:progress('train',**row)
            models[mode,seed]=model.eval();write_json(dest/'COMPLETE.json',dict(status='COMPLETE',parameters=sum(p.numel() for p in model.parameters()),
                updates=updates,checkpoints=15,final_sha256=sha(dest/'checkpoints/epoch015.pt')))
            write_json(out/'history.json',history)
            with torch.no_grad():
                b=model(data['xn']);fit_metrics.append(dict(mode=mode,seed=seed,final_fullfit_loss=float(loss_value(data['z']+b[:,None],data['y'],data['factor'])),
                    bias_mean=float(b.mean()),bias_min=float(b.min()),bias_max=float(b.max()),positive_fraction=float((b>0).float().mean())))
    reference=fit_constant_fullbatch(data['z'],data['y'],data['factor']);write_json(out/'constant_fullbatch.json',reference)
    write_json(out/'fit_metrics.json',fit_metrics);progress('training_complete',new_checkpoints=90,fullbatch_bias=reference['bias'])
    del data;torch.cuda.empty_cache()
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json']:raise RuntimeError('Changedannotations')
    obj=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(info=obj.get('info',{}),categories=obj['categories'],
        images=[r for r in obj['images'] if r['id'] in transfer],annotations=[r for r in obj['annotations'] if r['image_id'] in transfer])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del obj
    categories=sorted(gt.cats);meta={ann['id']:{'ici_same':ici(ann,[q for q in gt.imgToAnns[ann['image_id']] if not q.get('iscrowd',0)])}
        for ann in gt.anns.values() if not ann.get('iscrowd',0)}
    keys=[('original',-1),('constant_fullbatch',-1)]+[(m,s) for s in [0,1,2] for m in ['constant','instance_bias','coefficient']]
    predictions={f'{m}_s{s}_d0':[] for m,s in keys};spatial=[];bias_rows=[];decoder_xor=0
    for num,iid in enumerate(transfer,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changedpredictioncache')
        item=read_np(path);c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
        det=cuda(item['detections']).float();shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']))
        masks={ann['id']:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        mapping={int(j):int(t) for t,j in zip(item['annotation_ids'],item['prediction_indices'])}
        with torch.inference_mode():
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalization)
            raw=F.interpolate((c@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0] if len(c) else c.new_empty((0,*ishape))
            original=ops.process_mask(p,c,boxes,ishape,upsample=True);diff=int(torch.count_nonzero(ops.crop_mask((raw>0).byte(),boxes)!=original));decoder_xor+=diff
            if diff:raise RuntimeError('Originaldecoderchanged')
            for mode,seed in keys:
                name=f'{mode}_s{seed}_d0';b=None
                if mode=='original':binary=original
                elif mode=='coefficient':binary=ops.process_mask(p,c+models[mode,seed](x),boxes,ishape,upsample=True)
                else:
                    b=c.new_full((len(c),),reference['bias']) if mode=='constant_fullbatch' else models[mode,seed](x)
                    binary=ops.crop_mask((raw+b[:,None,None]>0).byte(),boxes)
                restored=ops.scale_masks(binary[:,None],shape)[:,0]>.5 if len(c) else binary
                for j in range(len(c)):
                    pred=restored[j].cpu().numpy()
                    if b is not None:bias_rows.append(dict(mode=mode,seed=seed,image_id=iid,source_index=int(item['source_index'][j]),
                        annotation_id=mapping.get(j,''),bias=float(b[j]),original_input_area=int(original[j].sum()),predicted_input_area=int(binary[j].sum())))
                    if bool(binary[j].any()):
                        rle=mu.encode(np.asfortranarray(pred.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        predictions[name].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in mapping:
                        aid=mapping[j];ann=gt.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());vp=pred&~crowd;same=np.zeros(shape,bool)
                        for other in gt.imgToAnns[iid]:
                            if not other.get('iscrowd',0) and other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']]
                        un=int((pred|masks[aid]).sum())
                        spatial.append(dict(arm=name,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],
                            iou=int((pred&masks[aid]).sum())/un if un else 1.,coverage=int((vp&own).sum())/area if area else None,
                            neighbor=int((vp&same&~own).sum())/area if area else None,background=int((vp&~union).sum())/area if area else None))
        if num%30==0:progress('decode',images=num,total=300)
    (out/'predictions').mkdir();task=[];gtrows=[];pairs=[];parity=[]
    for arm,pp in predictions.items():
        with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        if arm.startswith('original_') or arm.startswith('coefficient_'):
            priorname=arm.replace('coefficient_','raw_coco_');priorpath=src/'predictions'/f'{priorname}.json.gz'
            if sha(priorpath)!=sr[f'predictions/{priorname}.json.gz']:raise RuntimeError('Changedpriorpredictions')
            with gzip.open(priorpath,'rt',encoding='utf-8') as f:old=json.load(f)
            if old!=pp:raise RuntimeError('Exactpredictionreplayfailed '+arm)
            parity.append(dict(arm=arm,exact=True,predictions=len(pp)))
        if not pp:raise RuntimeError('Allemptyarmrequiresmanualevaluationhandling')
        row,rr,pr=evaluate(gt,meta,transfer,pp,arm)
        if len(rr)!=len(meta):raise RuntimeError('AllGTdenominatorchanged')
        row['gap']=row['r75_low']-row['r75_high'];task.append(row);gtrows.extend(rr);pairs.extend(pr);progress('task',**row)
    for name,rows in [('task_summary',task),('gt_recovery',gtrows),('pair_recovery',pairs),('spatial',spatial),('bias_predictions',bias_rows)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'prediction_parity.json',parity)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,fit_targets=count,ordinary_gt=len(meta),
        new_checkpoints=90,saved_coefficient_checkpoints=45,original_decoder_xor=decoder_xor,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'checkpoints' not in str(p)}))
    progress('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
