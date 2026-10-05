"""Canonical COCO scoring, preserving mask-derived areas and GT recall identities."""
import argparse
import gc
import json
import time
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    while not (a.source/'COMPLETE.json').exists():
        status=json.loads((a.source/'run.json').read_text()).get('status')
        if status in ('failed','error','cancelled','interrupted'):
            raise RuntimeError(f'Training/evaluation source {status}')
        if time.monotonic()-start>6*3600:raise TimeoutError('Upstream not complete within six hours')
        save(a.out/'progress.json',dict(stage='waiting_for_predictions',source=str(a.source)))
        time.sleep(5)
    coco=COCO(str(a.annotations));split=json.loads((a.source/'SPLIT.json').read_text());ids=split['val_ids']
    names=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL')
    results={};matched_by_mode={};baseline_good=None
    ordinary={ann['id']:ann for image_id in ids for ann in coco.imgToAnns[image_id]
              if not ann.get('iscrowd',0) and not ann.get('ignore',0)}
    def size(ann):return 'small' if ann['area']<32**2 else 'medium' if ann['area']<96**2 else 'large'
    for mode in ('baseline','scalar','coeff_bias','local4','coeff_local4'):
        preds=[{k:p[k] for k in ('image_id','category_id','score','segmentation')}
               for line in (a.source/f'predictions_{mode}.jsonl').open() for p in [json.loads(line)]]
        dt=coco.loadRes(preds);ev=COCOeval(coco,dt,'segm');ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
        ti=min(range(len(ev.params.iouThrs)),key=lambda j:abs(ev.params.iouThrs[j]-.75));good=set()
        for entry in ev.evalImgs:
            if entry is None or entry['aRng']!=ev.params.areaRng[0]:continue
            good.update(int(gt) for j,gt in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti,j]>0)
        good &= ordinary.keys()
        if mode=='baseline':baseline_good=good
        matched_by_mode[mode]=sorted(good)
        results[mode]=dict(metrics=dict(zip(names,map(float,ev.stats))),predictions=len(preds),
            mask75_matched_gt=len(good),mask75_repaired=len(good-baseline_good),mask75_damaged=len(baseline_good-good),
            size_recall75={g:dict(gt=sum(size(ann)==g for ann in ordinary.values()),
                                 matched=sum(size(ordinary[i])==g for i in good)) for g in ('small','medium','large')})
        save(a.out/'RESULTS.json',results);save(a.out/'MATCHED_GT75.json',matched_by_mode)
        save(a.out/'progress.json',dict(stage='scored',mode=mode,mask_ap=ev.stats[0]*100))
        print(json.dumps(results[mode]),flush=True)
        del preds,dt,ev;gc.collect()
    save(a.out/'COMPLETE.json',dict(images=len(ids),ordinary_gt=len(ordinary),source=str(a.source),
         result='RESULTS.json',protocol='Official COCOeval segm; bbox omitted from loadRes to preserve mask area.'))


if __name__=='__main__':main()
