"""S029: fixed fitted affine versus remaining spatial residual, no retraining.

The fit-only prediction projector from S028 supplies three immutable (a,b) pairs.
All original detections and all task GT retained; no GT enters any inference arm.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,contextlib,gzip,io,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from rich_pixel_readout import PixelHead
from pixel_ownership_refiner import build_features
from readout_input_probe import sha,write_json
from readout_fit_transfer_diagnostic import norm,read_np,csv_save
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--trained',type=Path,required=True);ap.add_argument('--projection',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids=json.loads((a.cache/'selection.json').read_text())['transfer']
    projector=json.loads((a.projection/'ANALYSIS.json').read_text())['projection']
    ab={r['seed']:(r['delta_slope'],r['intercept']) for r in projector if r['mode']=='rich_neighbor'}
    if set(ab)!={0,1,2}:raise RuntimeError('Missing fixed projector')
    write_json(out/'protocol.json',dict(experiment='S029 affine-component removal from saved rich_neighbor',
       images=ids,seeds=[0,1,2],projector=ab,
       arms={'original':'z','full':'z+r(x)','affine':'(1+a)z+b','remainder':'z+r(x)-az-b'},
       coefficients='Frozen S028 prediction-only leastsquares on 540targets in80fitimages; never refitted or selected ontransferGT.',
       scope='Exploratory300transfertrain2017, same asS026. No shared training, inferenceGT, box/score change or threshold search. '
             'Subtraction intervention tests whether learnednonaffine residual is useful after removing this estimated global component. '
             'Not originalmodel rootcause or densityspecific causality. Fit support was GTboxdiagnostic inheritedfromS028.',
       source_hashes={'cache':sha(a.cache/'COMPLETE.json'),'trained':sha(a.trained/'COMPLETE.json'),
                      'projection':sha(a.projection/'ANALYSIS.json'),'script':sha(Path(__file__))}))
    def progress(stage,**kw):
        r=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',r);print(json.dumps(r),flush=True)
    receipt={k.replace('\\','/'):v for k,v in json.loads((a.cache/'COMPLETE.json').read_text())['hashes'].items()}
    subset=a.cache/'conversion_input/instances_probe.json'
    if sha(subset)!=receipt['conversion_input/instances_probe.json']:raise RuntimeError('Annotation subset changed')
    obj=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(info=obj.get('info',{}),categories=obj['categories'],
       images=[x for x in obj['images'] if x['id'] in ids],annotations=[x for x in obj['annotations'] if x['image_id'] in ids])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del obj
    meta={aid:dict(ici_same=ici(x,[q for q in gt.imgToAnns[x['image_id']] if not q.get('iscrowd',0)]))
            for aid,x in gt.anns.items() if not x.get('iscrowd',0)}
    norm0=torch.load(a.trained/'normalizer.pt',weights_only=True)
    normal={k:{kk:vv.cuda() for kk,vv in v.items()} for k,v in norm0.items()};models={}
    for seed in ab:
        path=a.trained/f'rich_neighbor_s{seed}/checkpoints/epoch015.pt'
        if sha(path)!=json.loads((a.trained/f'rich_neighbor_s{seed}/COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Weights changed')
        model=PixelHead().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);models[seed]=model.eval()
    arms=[('original',-1)]+[(mode,s) for s in ab for mode in ['full','affine','remainder']]
    preds={f'{m}_s{s}':[] for m,s in arms};spatial=[];categories=sorted(gt.cats)
    with torch.inference_mode():
        for number,iid in enumerate(ids,1):
            path=a.cache/'images'/f'{iid}.npz'
            if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Cache changed')
            item=read_np(path);c=torch.tensor(item['coeff'],device='cuda');p=torch.tensor(item['proto'],device='cuda')
            boxes=torch.tensor(item['boxes'],device='cuda');det=torch.tensor(item['detections'],device='cuda')
            h=torch.tensor(item['h'],device='cuda');hn=norm(h,normal['h']);shape=tuple(item['shape']);ishape=tuple(item['input_shape']);ih,iw=ishape
            pp=F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0].flatten(1).T
            raw=F.interpolate((c@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]),ishape,mode='bilinear')[0] if len(c) else c.new_empty((0,ih,iw))
            ordinary=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]
            masks={q['id']:gt.annToMask(q).astype(bool) for q in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
            for q in gt.imgToAnns[iid]:
                if q.get('iscrowd',0):crowd|=masks[q['id']]
                else:union|=masks[q['id']]
            target={int(j):int(aid) for j,aid in zip(item['prediction_indices'],item['annotation_ids'])}
            for j in range(len(c)):
                bb=boxes[j];x1=max(0,math.ceil(float(bb[0])));x2=max(x1,min(iw,math.ceil(float(bb[2]))))
                y1=max(0,math.ceil(float(bb[1])));y2=max(y1,min(ih,math.ceil(float(bb[3]))))
                yy,xx=torch.meshgrid(torch.arange(y1,y2,device='cuda'),torch.arange(x1,x2,device='cuda'),indexing='ij')
                pos=(yy*iw+xx).flatten();points=torch.stack([xx.flatten(),yy.flatten()],1).float()
                scalar=build_features(raw.flatten(1)[:,pos],boxes,det,j,points,'neighbor');pn=norm(pp[pos],normal['p'])
                z=raw[j].flatten()[pos];values={('original',-1):z>0}
                for seed,model in models.items():
                    r=torch.empty_like(z)
                    for first in range(0,len(z),16384):
                        r[first:first+16384]=model(scalar[first:first+16384][None],pn[first:first+16384][None],hn[j:j+1],'rich_neighbor')[0]
                    slope,bias=ab[seed]
                    values['full',seed]=z+r>0;values['affine',seed]=(1+slope)*z+bias>0
                    values['remainder',seed]=z+r-slope*z-bias>0
                binaries=torch.zeros((len(arms),ih*iw),device='cuda',dtype=torch.uint8)
                for k,key in enumerate(arms):binaries[k,pos]=values[key].byte()
                binaries=binaries.reshape(len(arms),ih,iw);restored=ops.scale_masks(binaries[:,None],shape)[:,0]>.5
                for k,(mode,seed) in enumerate(arms):
                    arm=f'{mode}_s{seed}';pred=restored[k].cpu().numpy()
                    if bool(binaries[k].any()):
                        rle=mu.encode(np.asfortranarray(pred.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        preds[arm].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in target:
                        aid=target[j];aa=gt.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());same=np.zeros(shape,bool)
                        for q in ordinary:
                            if q['id']!=aid and q['category_id']==aa['category_id']:same|=masks[q['id']]
                        inter=int((pred&masks[aid]).sum());union0=int((pred|masks[aid]).sum());vp=pred&~crowd
                        spatial.append(dict(arm=arm,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],iou=inter/union0 if union0 else 1.,
                            coverage=float((vp&own).sum()/area) if area else None,
                            neighbor=float((vp&same&~own).sum()/area) if area else None,
                            background=float((vp&~union).sum()/area) if area else None))
            if number%25==0 or number==len(ids):progress('decode',images=number,total=len(ids))
    (out/'predictions').mkdir();summary=[];gts=[];pairs=[]
    for arm,pp in preds.items():
        with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        # Compare full learned and original predictions themselves, not just headline metrics.
        mode,seed=arm.rsplit('_s',1)
        if mode in ['original','full']:
            oldname=f'{"original" if mode=="original" else "rich_neighbor"}_s{seed}_d0.json.gz'
            with gzip.open(a.trained/'predictions'/oldname,'rt',encoding='utf-8') as f:old=json.load(f)
            if old!=pp:raise RuntimeError('Exact original/full S026 prediction replay failed: '+arm)
        r,gg,pr=evaluate(gt,meta,ids,pp,arm)
        if len(gg)!=len(meta):raise RuntimeError('Task denominator changed')
        r['gap']=r['r75_low']-r['r75_high'];summary.append(r);gts.extend(gg);pairs.extend(pr);progress('official',**r)
    csv_save(out/'task_summary.csv',summary);csv_save(out/'gt_recovery.csv',gts);csv_save(out/'pair_recovery.csv',pairs);csv_save(out/'spatial.csv',spatial)
    progress('COMPLETE')
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,exact_prediction_replay='PASS',
        images=len(ids),ordinary_gt=len(meta),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
