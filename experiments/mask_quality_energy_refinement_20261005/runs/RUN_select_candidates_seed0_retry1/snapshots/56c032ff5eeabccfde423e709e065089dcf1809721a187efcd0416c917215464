from __future__ import annotations
import argparse, gzip, json, math, os, sys, traceback
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import ops
from pycocotools.coco import COCO
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_runtime import load_asset, load_index, dump, resolve_runtime_config

def iou(a,b):
    inter=(a&b).sum().item(); union=(a|b).sum().item()
    return float(inter/max(union,1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); cfg=resolve_runtime_config(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('candidate selection requires authorized CUDA server')
    device=torch.device('cuda'); idx=load_index(cfg); coco=COCO(cfg['annotations_train'])
    cat_ids=sorted(coco.getCatIds()); cat_to_idx={int(c):i for i,c in enumerate(cat_ids)}
    pools={'failure':[],'success':[]}; per_image={}; scanned=0; candidate_total=0
    for entry in idx['fit']:
        iid=int(entry['image_id']);
        if int(entry.get('n',0))<=0: continue
        x=load_asset(cfg,iid,verify=True); scanned+=1
        proto=x['proto'].float().to(device); c=x['c0'].float().to(device); boxes=x['boxes'].float().to(device);
        pred=ops.process_mask(proto,c,boxes,(640,640),upsample=True).bool()
        for k,row in enumerate(x['rows']):
            candidate_total+=1; ann=coco.anns[int(row['annotation_id'])]
            cls_ok=int(row['predicted_class_id'])==cat_to_idx[int(ann['category_id'])]
            biou=float(row['box_iou'])
            gt=(x['masks'].to(device)==(int(x['owners'][k])+1))
            miou=iou(pred[k],gt)
            if not cls_ok or biou < .75: continue
            kind='failure' if miou < .75 else 'success'
            pools[kind].append({'split':'fit','image_id':iid,'annotation_id':int(row['annotation_id']),'branch':row['branch'],'raw_id':int(row['raw_id']),'pyramid_level':int(row['pyramid_level']),'target_gt_idx':int(row['target_gt_idx']),'row_index':k,'box_iou':biou,'mask_iou':miou,'predicted_class_id':int(row['predicted_class_id']),'gt_class_id':int(cat_to_idx[int(ann['category_id'])])})
        if scanned%250==0: print(json.dumps({'images':scanned,'candidates':candidate_total,'failure':len(pools['failure']),'success':len(pools['success'])}),flush=True)
    for v in pools.values(): v.sort(key=lambda r:(r['image_id'],r['annotation_id'],r['raw_id'],r['pyramid_level']))
    max_fail=int(cfg.get('max_failure',30000)); max_succ=int(cfg.get('max_success',30000))
    selected={'failure':pools['failure'][:max_fail],'success':pools['success'][:max_succ]}
    if len(selected['failure']) < int(cfg.get('target_failure_min',20000)) or len(selected['success']) < int(cfg.get('target_success_min',20000)):
        raise RuntimeError(f'locked target not met: { {k:len(v) for k,v in selected.items()} }')
    by_image={}
    for kind,rows in selected.items():
        for r in rows: by_image.setdefault(str(r['image_id']),[]).append({**r,'kind':kind})
    payload={'schema':'qcr-candidate-manifest-v1','split_sha256':__import__('hashlib').sha256(Path(cfg['split']).read_bytes()).hexdigest(),'counts':{'fit_images_scanned':scanned,'raw_candidates':candidate_total,'failure_pool':len(pools['failure']),'success_pool':len(pools['success']),'failure_selected':len(selected['failure']),'success_selected':len(selected['success']),'images_with_selected':len(by_image)},'candidates':selected,'by_image':by_image}
    dump(out,payload)
    dump(out.with_name('CANDIDATE_MANIFEST_SUMMARY.json'),payload['counts'])
if __name__=='__main__':
    try: main()
    except BaseException as e:
        print(traceback.format_exc(),file=sys.stderr); raise
