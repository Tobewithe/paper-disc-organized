"""Published Boundary IoU API applied to unchanged saved COCO predictions."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import gc
import gzip
import json
from pathlib import Path
import sys
import time

import cv2
cv2.setNumThreads(1)
import numpy as np


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main(args):
    sys.path.insert(0,str(args.vendor))
    from boundary_iou.coco_instance_api.coco import COCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval
    config=json.loads(args.models.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();coco=COCO(str(args.annotations));assert len(coco.imgs)==5000
    results={};keys=['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL']
    for name,source in config.items():
        save(out/'progress.json',{'stage':'load','model':name,'completed':list(results),'elapsed_s':time.monotonic()-start})
        print('MODEL',name,flush=True)
        opener=gzip.open if source.endswith('.gz') else open
        with opener(source,'rt',encoding='utf-8') as f:pred=[json.loads(line) for line in f]
        assert pred and all('bbox' not in p for p in pred),'Require mask-derived area, not supplied bbox.'
        assert len(set(p['image_id'] for p in pred))<=5000
        # Match the original plain mask AP before computing the additional metric.
        if name=='baseline':
            dt=coco.loadRes(pred);ev=COCOeval(coco,dt,'segm');ev.evaluate();ev.accumulate();ev.summarize()
            ordinary=dict(zip(keys,map(float,ev.stats)))
            assert abs(ordinary['AP']-.43518455)<1e-6,ordinary
            save(out/'BASELINE_MASK_IDENTITY.json',ordinary)
            del ev,dt;gc.collect()
            coco.get_boundary=True;coco.dilation_ratio=.02
            save(out/'progress.json',{'stage':'ground_truth_boundary','model':name,'elapsed_s':time.monotonic()-start})
            coco.createIndex()
        save(out/'progress.json',{'stage':'prediction_boundary','model':name,'completed':list(results),'elapsed_s':time.monotonic()-start})
        # COCO.loadRes above mutates baseline records by adding numpy bbox arrays.
        # Reloading these takes the legacy bbox branch and would also change area semantics.
        # Restore the mask-only result representation for the boundary evaluation.
        for row in pred:row.pop('bbox',None)
        dt=coco.loadRes(pred)
        ev=COCOeval(coco,dt,iouType='boundary',dilation_ratio=.02)
        ev.evaluate();ev.accumulate();ev.summarize()
        metrics=dict(zip(keys,map(float,ev.stats)))
        results[name]={'boundary_metrics':metrics,'predictions':len(pred),'source':source}
        save(out/'RESULTS.json',results)
        print(json.dumps({'model':name,'boundary_metrics':metrics,'elapsed_s':time.monotonic()-start}),flush=True)
        del ev,dt,pred;gc.collect()
    save(out/'COMPLETE.json',{'images':5000,'dilation_ratio':.02,'models':list(config),
         'upstream':json.loads((args.vendor/'PROVENANCE.json').read_text()),'elapsed_s':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['out','vendor','models','annotations']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
