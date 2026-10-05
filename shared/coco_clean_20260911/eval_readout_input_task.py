"""Official whole-image COCO matching, then instance-density stratification.

Uses ALL original retained predictions and ALL ordinary GT on the frozen image
cohort. Never evaluate only matched/failed/high-density instances in COCOeval.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import contextlib
import csv
import gzip
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from ultralytics.utils import ops
from readout_input_probe import sha,write_json
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate


def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(exist_ok=False)
    (args.out/'predictions').mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((args.cache/'protocol.json').read_text())
    ids=json.loads((args.cache/'selection.json').read_text())['transfer']
    ann=Path(config['data_root'])/'annotations/instances_train2017.json'
    assert sha(ann)==config['annotation_sha256']
    # Load the exact raw JSON, then restrict images only, never GT by density.
    with contextlib.redirect_stdout(io.StringIO()):
        full=COCO(str(ann));gt=COCO()
        gt.dataset=dict(info=full.dataset.get('info',{}),images=[full.imgs[i] for i in ids],
           categories=list(full.cats.values()),annotations=[a for i in ids for a in full.imgToAnns[i]])
        gt.createIndex()
    del full
    meta={}
    for iid in ids:
        ordinary=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
        for a in ordinary:meta[a['id']]={'ici_same':ici(a,ordinary)}
    with np.load(args.evaluation/'coefficients'/f'{ids[0]}.npz') as q:
        arms=list(q.files)
    predictions={a:[] for a in arms};boxpred=[];start=time.monotonic();categories=sorted(gt.cats)
    for number,iid in enumerate(ids,1):
        with np.load(args.cache/'images'/f'{iid}.npz') as q:item={k:q[k] for k in q.files}
        with np.load(args.evaluation/'coefficients'/f'{iid}.npz') as q:coeffs={k:q[k] for k in q.files}
        assert set(coeffs)==set(arms)
        proto=torch.tensor(item['proto'],device='cuda');boxes=torch.tensor(item['boxes'],device='cuda')
        det=item['detections'];shape=tuple(item['shape']);input_shape=tuple(item['input_shape'])
        for d in det:
            boxpred.append(dict(image_id=iid,category_id=categories[int(d[5])],score=float(d[4]),
                 bbox=[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])]))
        with torch.inference_mode():
            for arm in arms:
                c=torch.tensor(coeffs[arm],device='cuda')
                assert len(c)==len(det)
                for first in range(0,len(c),32):
                    binary=ops.process_mask(proto,c[first:first+32],boxes[first:first+32],input_shape,upsample=True)
                    restored=ops.scale_masks(binary[:,None],shape)[:,0]>.5
                    nonempty=binary.flatten(1).any(1)
                    for k in range(len(binary)):
                        if not bool(nonempty[k]):continue
                        j=first+k;rle=mu.encode(np.asfortranarray(restored[k].cpu().numpy().astype(np.uint8)))
                        rle['counts']=rle['counts'].decode('ascii')
                        predictions[arm].append(dict(image_id=iid,category_id=categories[int(det[j,5])],
                            score=float(det[j,4]),segmentation=rle))
        if number%25==0 or number==len(ids):
            pr=dict(stage='task_decode',images=number,total=len(ids),seconds=time.monotonic()-start)
            write_json(args.out/'progress.json',pr);print(json.dumps(pr),flush=True)
    summaries=[];gtrows=[];pairrows=[]
    for arm,pp in predictions.items():
        with gzip.open(args.out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        summary,gg,pairs=evaluate(gt,meta,ids,pp,arm)
        assert len(gg)==len(meta),'Whole ordinary GT denominator changed'
        for label in ['low','middle','high']:
            rr=[r for r in gg if (r['ici']<=1e-10 if label=='low' else
                 1e-10<r['ici']<=.5+1e-10 if label=='middle' else r['ici']>.5+1e-10)]
            summary['r75_'+label+'_threeway']=float(np.mean([r['hit75'] for r in rr])) if rr else None
            summary['n_'+label+'_threeway']=len(rr)
        summary['nonhigh_minus_high_gap']=summary['r75_low']-summary['r75_high'] if summary['r75_high'] is not None else None
        summaries.append(summary);gtrows.extend(gg);pairrows.extend(pairs)
        print(json.dumps(summary),flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(boxpred);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    save_csv(args.out/'task_summary.csv',summaries);save_csv(args.out/'gt_recovery.csv',gtrows);save_csv(args.out/'pair_recovery.csv',pairrows)
    write_json(args.out/'protocol.json',dict(scope='300 heldout train2017 images, all ordinary GT and original retained predictions. '
       'Official COCOeval full-image matching before ICI strata. Not val2017 benchmark, not matched-precision Recall, not pristine pretrained holdout.',
       cache_sha256=sha(args.cache/'COMPLETE.json'),coefficient_evaluation_sha256=sha(args.evaluation/'COMPLETE.json'),
       annotation_sha256=sha(ann),images=ids,arms=arms,original_box_ap=float(ev.stats[0]),
       empty_mask_rule='Modified empty predictions dropped consistently with original stock mask path; boxes/scores of existing slots frozen. '
       'Reported box AP on original box slots, not mask-filtered box lists.',
       density='low ICI<=1e-10; middle1e-10<ICI<=.5+1e-10; high>.5+1e-10. Original helper low means nonhigh; explicit threeway columns avoid mixing.'))
    write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),ordinary_gt=len(meta),arms=len(arms),
        seconds=time.monotonic()-start,hashes={str(p.relative_to(args.out)):sha(p) for p in args.out.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
