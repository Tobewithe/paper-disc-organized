"""S034: retain original pixel ordering with the learned mask's predicted area.

Every K is predicted by saved S032 heads, never read from GT. No head fitting,
test-threshold search or box changes. Stable order resolves equal logit values.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,gzip,io,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;out=a.out;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_protocol=json.loads((src/'protocol.json').read_text());cache=Path(source_protocol['cache'])
    source_receipt={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    receipt={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    images=json.loads((cache/'selection.json').read_text())['transfer']
    protocol=dict(experiment='S034_EQUAL_PREDICTED_AREA',source=str(src.resolve()),cache=str(cache.resolve()),images=images,
        network_training=False,seeds=[0,1,2],modes=['original','learned','original_rank'],
        arms='Original frozen mask; S032 saved rawCOCO head mask; original logit ranking with same predicted K as that head.',
        area='For each original prediction slot, K=sum(newmask>0) inside original predicted crop in640input. '
             'Rank original official-order upsampledlogits only inside same crop. Keep exactly K. K=0/full explicitlyhandled.',
        ties='Stable descending torch.argsort of originallogits over ascending flattened inputpixelindex. No epsilon/tie jitter.',
        invariant='Allboxes/classes/scores/sourceidentity unchanged. NoGT enters K,ranking,ties,inference or parameterselection. '
            'Emptyoutputslots same learned/control at640. K matching in640 does not imply exactoriginalresolutionarea; report changes.',
        evaluation='All300existingtransfertrain2017images, including zeroGTimages, officialCOCOeval allGT first thenICI. '
            'Sameimage alreadyexplored. Fixedbbox50spatialdiagnostic separatefromallGTtaskmatching.',
        scope='Conditional area-transfer diagnostic depends on learnedmodel for K,not a standalone cheaperbaseline. '
            'Does not prove scalarcalibration equivalence or mediation fraction. NoGT areaoracle, test-tuning, newmethod or val claim.',
        stopping='Report all3seeds andfullmetrics; no choosing control byGT. Exactoriginal/learnedJSONreplay required. '
            'If samearea originalranking reproduces gains, nextroute needs more than area changes; otherwise inspect ranking contribution.',
        sources=dict(source_receipt=sha(src/'COMPLETE.json'),cache_receipt=sha(cache/'COMPLETE.json'),script=sha(__file__)))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    if sha(src/'normalizer.pt')!=source_receipt['normalizer.pt']:raise RuntimeError('Changed normalizer')
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    models={}
    for seed in [0,1,2]:
        run=src/f'raw_coco_s{seed}';cp=run/'checkpoints/epoch015.pt'
        if sha(cp)!=json.loads((run/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Changed checkpoint')
        model=GlobalHead().cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['model']);models[seed]=model.eval()
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=receipt['conversion_input/instances_probe.json']:raise RuntimeError('Changed GT subset')
    data=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(info=data.get('info',{}),categories=data['categories'],
        images=[r for r in data['images'] if r['id'] in images],annotations=[r for r in data['annotations'] if r['image_id'] in images])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del data
    meta={r['id']:{'ici_same':ici(r,[q for q in gt.imgToAnns[r['image_id']] if not q.get('iscrowd',0)])}
          for r in gt.anns.values() if not r.get('iscrowd',0)}
    categories=sorted(gt.cats);keys=[('original',-1)]+[(m,s) for s in [0,1,2] for m in ['learned','original_rank']]
    predictions={f'{m}_s{s}_d0':[] for m,s in keys};spatial=[];areas=[];parity=[];xor=0
    for number,iid in enumerate(images,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Changed NPZ')
        item=read_np(path);c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
        det=cuda(item['detections']).float();shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        mapping={int(j):int(t) for t,j in zip(item['annotation_ids'],item['prediction_indices'])}
        masks={ann['id']:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        with torch.inference_mode():
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalization)
            original=ops.process_mask(p,c,boxes,ishape,upsample=True)
            raw=F.interpolate((c@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0] if len(c) else c.new_empty((0,ih,iw))
            diff=int(torch.count_nonzero(ops.crop_mask((raw>0).byte(),boxes)!=original));xor+=diff
            if diff:raise RuntimeError('Original upsample order replay changed')
            learned={s:ops.process_mask(p,c+model(x),boxes,ishape,upsample=True) for s,model in models.items()}
            for j in range(len(c)):
                box=boxes[j];x1=max(0,min(iw,math.ceil(float(box[0]))));x2=max(x1,min(iw,math.ceil(float(box[2]))))
                y1=max(0,min(ih,math.ceil(float(box[1]))));y2=max(y1,min(ih,math.ceil(float(box[3]))))
                yy,xx=torch.meshgrid(torch.arange(y1,y2,device='cuda'),torch.arange(x1,x2,device='cuda'),indexing='ij')
                positions=(yy*iw+xx).flatten();values=raw[j].flatten()[positions]
                if not torch.isfinite(values).all():raise RuntimeError('Nonfinite ranking values')
                order=torch.argsort(values,descending=True,stable=True);sorted_values=values[order]
                binaries={('original',-1):original[j]};metadata={}
                for seed in [0,1,2]:
                    learned_mask=learned[seed][j];k=int(learned_mask.sum())
                    if k>len(positions):raise RuntimeError('Predicted K outside originalcrop')
                    chosen=positions[order[:k]];control=torch.zeros(ih*iw,device='cuda',dtype=torch.uint8);control[chosen]=1;control=control.reshape(ih,iw)
                    if int(control.sum())!=k:raise RuntimeError('Input area mismatch')
                    binaries['learned',seed]=learned_mask;binaries['original_rank',seed]=control
                    metadata[seed]=dict(k=k,crop_pixels=len(positions),boundary_tie=bool(0<k<len(order) and sorted_values[k-1]==sorted_values[k]),
                        input_mask_xor=int(torch.count_nonzero(control!=learned_mask)),
                        original_k=int(original[j].sum()))
                restored=ops.scale_masks(torch.stack([binaries[key] for key in keys])[:,None],shape)[:,0]>.5
                key_to_index={key:n for n,key in enumerate(keys)}
                for seed in [0,1,2]:
                    ma=restored[key_to_index['learned',seed]];mb=restored[key_to_index['original_rank',seed]]
                    sa=int(ma.sum());sb=int(mb.sum());un=int((ma|mb).sum())
                    areas.append(dict(image_id=iid,prediction_index=j,source_index=int(item['source_index'][j]),seed=seed,
                        annotation_id=mapping.get(j,''),**metadata[seed],learned_original_area=sa,control_original_area=sb,
                        original_area_delta=sb-sa,original_area_delta_relative=(sb-sa)/max(1,sa),
                        original_mask_xor=int(torch.count_nonzero(ma!=mb)),original_mask_iou=int((ma&mb).sum())/un if un else 1.))
                for n,(mode,seed) in enumerate(keys):
                    pred=restored[n].cpu().numpy();name=f'{mode}_s{seed}_d0'
                    if bool(binaries[mode,seed].any()):
                        rle=mu.encode(np.asfortranarray(pred.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        predictions[name].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in mapping:
                        aid=mapping[j];ann=gt.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());vp=pred&~crowd;same=np.zeros(shape,bool)
                        for aa in gt.imgToAnns[iid]:
                            if not aa.get('iscrowd',0) and aa['id']!=aid and aa['category_id']==ann['category_id']:same|=masks[aa['id']]
                        un=int((pred|masks[aid]).sum())
                        spatial.append(dict(arm=name,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],
                            iou=int((pred&masks[aid]).sum())/un if un else 1.,
                            coverage=int((vp&own).sum())/area if area else None,
                            neighbor=int((vp&same&~own).sum())/area if area else None,
                            background=int((vp&~union).sum())/area if area else None))
        if number%20==0:progress('decode',images=number,total=len(images),area_rows=len(areas))
    (out/'predictions').mkdir();task=[];gtrows=[];pairs=[]
    for arm,pp in predictions.items():
        with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        if arm.startswith('original_s') or arm.startswith('learned_s'):
            priorname=arm.replace('learned_','raw_coco_');priorpath=src/'predictions'/f'{priorname}.json.gz'
            if sha(priorpath)!=source_receipt[f'predictions/{priorname}.json.gz']:raise RuntimeError('Prior prediction file changed')
            with gzip.open(priorpath,'rt',encoding='utf-8') as f:old=json.load(f)
            if old!=pp:raise RuntimeError('Full prediction exactreplay failed '+arm)
            parity.append(dict(arm=arm,prior=priorname,predictions=len(pp),exact=True))
        if not pp:raise RuntimeError('All empty output requires explicit evaluator')
        row,rr,pr=evaluate(gt,meta,images,pp,arm)
        if len(rr)!=len(meta):raise RuntimeError('GT count changed')
        row['gap']=row['r75_low']-row['r75_high'];task.append(row);gtrows.extend(rr);pairs.extend(pr);progress('task',**row)
    for seed in [0,1,2]:
        aa=predictions[f'learned_s{seed}_d0'];bb=predictions[f'original_rank_s{seed}_d0']
        if [(x['image_id'],x['category_id'],x['score']) for x in aa]!=[(x['image_id'],x['category_id'],x['score']) for x in bb]:raise RuntimeError('Kept identity/score differs')
    for name,rows in [('task_summary',task),('gt_recovery',gtrows),('pair_recovery',pairs),('spatial',spatial),('area_witness',areas)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'prediction_parity.json',parity)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,images=len(images),ordinary_gt=len(meta),
        area_records=len(areas),original_pixel_xor=xor,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    progress('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
