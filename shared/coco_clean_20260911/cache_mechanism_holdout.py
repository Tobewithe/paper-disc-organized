"""Cache fixed official predictions on GT-selected images disjoint from discovery."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from frozen_mechanism_probe import ROOT,SEED,rank,box_iou,Capture,ownership,sha,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exclude',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--images',type=int,default=300);a=ap.parse_args()
    out=a.out.resolve();out.mkdir(exist_ok=False,parents=True);(out/'tensors').mkdir();excluded={x['image_id'] for x in json.loads(a.exclude.read_text())}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    pools={'high':[],'low':[]}
    for iid in sorted(gt.imgs):
        if iid in excluded:continue
        anns=[x for x in gt.imgToAnns[iid] if not x.get('iscrowd',0)];pairs=[]
        for i,x in enumerate(anns):
            for y in anns[i+1:]:
                if x['category_id']!=y['category_id']:continue
                overlap=box_iou(x['bbox'],y['bbox'])
                if overlap<=.05:continue
                aa,bb=sorted([x['id'],y['id']]);high=max(float(meta[t]['ici_same']) for t in [aa,bb])>.5
                pairs.append(dict(image_id=iid,annotation_a=aa,annotation_b=bb,gt_box_iou=overlap,pair_group='high' if high else 'low'))
        if pairs:
            p=min(pairs,key=lambda p:rank(f"{iid}:{p['annotation_a']}:{p['annotation_b']}"));pools[p['pair_group']].append(p)
    chosen=[]
    for group,count in [('high',2*a.images//3),('low',a.images-2*a.images//3)]:
        ordered=sorted(pools[group],key=lambda p:rank(p['image_id']));assert len(ordered)>=count;chosen.extend(ordered[:count])
    chosen.sort(key=lambda p:rank(p['image_id']));assert not excluded.intersection(p['image_id'] for p in chosen)
    write_json(out/'selection.json',chosen)
    write_json(out/'protocol.json',dict(training=False,script_sha256=sha(__file__),excluded_sha256=sha(a.exclude),weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),
        frozen_candidate='BCE gradient with (G+0.01 trace(G)/K I)^-1; compare same 0.05 logit-RMS budget; all other objectives retained as controls',
        confirmatory_scope='Disjoint from current 300 discovery images only; COCO val has been used earlier in project and is not untouched publication test data.',
        images=a.images,selection='Same GT pair and image hashes as discovery, exclude discovery IDs before selecting 2:1 high/low.'))
    draws=np.random.default_rng(SEED+1).integers(a.images,size=(2000,a.images));np.savez_compressed(out/'bootstrap_draws.npz',image_ids=[x['image_id'] for x in chosen],draws=draws)
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);statuses=[];start=time.monotonic()
    for n,pair in enumerate(chosen,1):
        iid=pair['image_id'];image=ROOT/'official_yolo/images/val2017'/gt.imgs[iid]['file_name']
        with torch.no_grad():
            model.predict(str(image),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
            capture={k:v.clone() if torch.is_tensor(v) else v for k,v in model.predictor.capture.items()};mapped=ownership(gt,iid,capture['detections'])
        ids=[pair['annotation_a'],pair['annotation_b']];status=dict(**pair,matched_a=ids[0] in mapped,matched_b=ids[1] in mapped)
        if all(x in mapped for x in ids):
            indices=[mapped[x] for x in ids]
            np.savez_compressed(out/'tensors'/f'{iid}.npz',proto=capture['proto'].cpu().numpy(),coeff=capture['coeff'][indices].cpu().numpy(),boxes=capture['boxes'][indices].cpu().numpy(),
                shape=capture['shape'],input_shape=capture['input_shape'],annotation_ids=ids,detections=capture['detections'][indices].cpu().numpy())
            status['status']='ok'
        else:status['status']='unmatched'
        statuses.append(status)
        if n%50==0:print(json.dumps(dict(completed=n,total=a.images,valid_pairs=sum(s['status']=='ok' for s in statuses),seconds=round(time.monotonic()-start,1))),flush=True)
    write_json(out/'statuses.json',statuses);write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,images=a.images,valid_pairs=sum(s['status']=='ok' for s in statuses),seconds=time.monotonic()-start))

if __name__=='__main__':main()
