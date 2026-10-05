"""One inference pass; official unique matching with all competitors, then subset GT recall."""
import argparse,contextlib,csv,gc,gzip,io,json,shutil
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.data.converter import coco80_to_coco91_class
import coco_metrics as cm
from recording import atomic_json,now

class Validator(SegmentationValidator):
    def init_metrics(self,model):
        super().init_metrics(model)
        assert bool(self.end2end) is True
        self.class_map=coco80_to_coco91_class();self.is_coco=False

def evaluate(gt,dt,task,ids):
    with contextlib.redirect_stdout(io.StringIO()):
        ev=COCOeval(gt,dt,task);ev.params.imgIds=sorted(ids);ev.evaluate();ev.accumulate();ev.summarize()
    return ev

def main():
    p=argparse.ArgumentParser()
    for key in ['study','out','weight']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',required=True);a=p.parse_args();cfg=json.loads((a.study/'protocol.json').read_text())
    manifest=json.loads((a.study/'data/manifest.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(Path(cfg['coco_root'])/'annotations/instances_val2017.json'))
    model=YOLO(str(a.weight));r=model.val(validator=Validator,data=str(a.study/'data/evaluate.yaml'),
        imgsz=640,batch=8,workers=cfg['workers'],device=0,conf=.001,iou=.7,max_det=300,end2end=True,
        save_json=True,plots=False,project=str(a.out/'exports'),name='one2one',exist_ok=False,verbose=False)
    export=Path(r.save_dir)/'predictions.json';pred=json.loads(export.read_text());del model;gc.collect()
    assert set(x['image_id'] for x in pred)<=set(manifest['union_image_ids'])
    with (a.study/'data/panel.csv').open(encoding='utf-8-sig',newline='') as f:panel=list(csv.DictReader(f))
    summaries=[];pergt=[];statsnames='ap ap50 ap75 aps apm apl ar1 ar10 ar100 ar_small ar_medium ar_large'.split()
    for task in ['bbox','segm']:
        processed=cm.task_predictions(pred,{k:k for k in gt.cats},task,gt)
        with contextlib.redirect_stdout(io.StringIO()):dt=gt.loadRes(processed)
        ev=evaluate(gt,dt,task,manifest['union_image_ids']);bygt={}
        for item in ev.evalImgs:
            if item is None or list(item['aRng'])!=[0,1e10] or item['maxDet']!=100:continue
            for j,gid in enumerate(item['gtIds']):
                if item['gtIgnore'][j]:continue
                row=dict(arm=a.arm,task=task,image_id=item['image_id'],annotation_id=int(gid),category_id=item['category_id'],
                    area=gt.anns[int(gid)]['area'],**{f'r{int(round(t*100))}':int(item['gtMatches'][i,j]>0) for i,t in enumerate(ev.params.iouThrs)})
                bygt[int(gid)]=row;pergt.append(row)
        for cohort in ['raw_geometry_small','matched_small_control']:
            rows=[bygt[int(x['annotation_id'])] for x in panel if x['cohort']==cohort]
            assert len(rows)==cfg['panel_pairs']
            summaries.append(dict(arm=a.arm,task=task,scope=cohort,n=len(rows),
                **{k:float(np.mean([x[k] for x in rows])) for k in ['r50','r75']}))
        rand=evaluate(gt,dt,task,manifest['random_image_ids'])
        micro=[v for v in bygt.values() if v['image_id'] in manifest['random_image_ids']]
        small=[v for v in micro if v['area']<32**2]
        summaries.append(dict(arm=a.arm,task=task,scope='random_val',n=len(micro),images=len(manifest['random_image_ids']),
            **{k:float(v) for k,v in zip(statsnames,rand.stats)},
            r50=float(np.mean([x['r50'] for x in micro])),r75=float(np.mean([x['r75'] for x in micro])),
            small_n=len(small),small_r50=float(np.mean([x['r50'] for x in small])),small_r75=float(np.mean([x['r75'] for x in small]))))
        print('EVALUATED',json.dumps(summaries[-3:]),flush=True)
    cm.write_csv(a.out/'per_gt.csv',pergt)
    atomic_json(a.out/'metrics.json',summaries)
    with export.open('rb') as src,gzip.open(export.with_suffix('.json.gz'),'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
    export.unlink()
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',at=now(),arm=a.arm,branch='one2one',
        inference_images=manifest['union_images'],random_images=cfg['random_val_images'],panel_targets=2*cfg['panel_pairs'],
        checkpoint='last',protocol='original COCOeval; maxDets=100 per class; conf=.001; all GT competitors; original RLE areas; subsets only after matching',
        limitation='one-epoch one-seed screen, random 256 images, no full COCO claim'))
if __name__=='__main__':main()
