"""Finish interrupted COCO scoring from closed prediction caches; no training/inference."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def save(path,data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main(args):
    args.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    source_progress=json.loads((args.source/'progress.json').read_text())
    assert source_progress['images']==source_progress['total']==5000
    coco=COCO(str(args.annotations));ids=sorted(coco.imgs);assert len(ids)==5000
    base=set(json.loads(args.reference.read_text())['baseline'])
    if args.arm:
        assert args.selection is not None
        models={args.arm:args.selection}
    else:
        models=json.loads(args.models.read_text())
    results={};matches={'baseline':sorted(base)}
    for name,folder in models.items():
        path=args.source/f'predictions_{name}.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf-8') as f:pred=[json.loads(line) for line in f]
        # Reading to EOF checks gzip integrity; source log attests all images processed.
        assert pred and all('bbox' not in p and p['image_id'] in coco.imgs for p in pred)
        selection=json.loads((Path(folder)/'SELECTION.json').read_text())
        ev=COCOeval(coco,coco.loadRes(pred),'segm');ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
        ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
        for item in ev.evalImgs:
            if item is not None and item['aRng']==ev.params.areaRng[0]:
                good.update(int(g) for j,g in enumerate(item['gtIds']) if not item['gtIgnore'][j] and item['gtMatches'][ti,j]>0)
        metrics=dict(zip(['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'],map(float,ev.stats)))
        results[name]={'metrics':metrics,'alpha':selection['alpha'],'parameters':selection['parameters'],
            'matched75':len(good),'repaired75':len(good-base),'damaged75':len(base-good),'predictions':len(pred),
            'prediction_images':len({p['image_id'] for p in pred}),'prediction_source':str(path)}
        matches[name]=sorted(good)
        save(args.out/'RESULTS.json',results);save(args.out/'MATCHED_GT75.json',matches)
        save(args.out/'progress.json',{'completed_arms':list(results),'elapsed_s':time.monotonic()-start})
        if name=='global_scalar' and (args.source/'RESULTS.json').exists():
            expected=json.loads((args.source/'RESULTS.json').read_text())[name]['metrics']
            assert max(abs(metrics[k]-expected[k]) for k in metrics)<1e-10
        del ev,pred
    save(args.out/'COMPLETE.json',{'images':5000,'arms':list(results),'elapsed_s':time.monotonic()-start,
         'source_run':args.source.name,'scope':'CPU scoring of pre-existing predictions; no new inference/training'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','annotations','reference','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',type=Path)
    p.add_argument('--arm');p.add_argument('--selection',type=Path)
    main(p.parse_args())
