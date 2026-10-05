"""Freeze outcome groups and baseline prediction identity before field interventions."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu


def main():
    ap=argparse.ArgumentParser()
    for k in ['study','annotations','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    local=a.study/'runs/RUN_19247caa79364ca5b743a4d2a2f1e8d7'
    scalar=a.study/'runs/RUN_690c1580277a44679752ff727705f2e6'
    lm=json.loads((local/'MATCHED_GT75.json').read_text());sm=json.loads((scalar/'MATCHED_GT75.json').read_text())
    b=set(lm['baseline']);l=set(lm['local4_center_s0']);s=set(sm['scalar_selected']);assert b==set(sm['baseline'])
    target=(b^l)|(b^s);coco=COCO(str(a.annotations));images={coco.anns[i]['image_id'] for i in target}
    groups={}
    for aid in target:
        if aid not in b:
            groups[aid]='both_repair' if aid in l&s else 'local_only_repair' if aid in l else 'scalar_only_repair'
        else:groups[aid]='both_damage' if aid not in l|s else 'local_only_damage' if aid not in l else 'scalar_only_damage'
    rows={};predictions=defaultdict(list)
    with gzip.open(local/'predictions_baseline.jsonl.gz','rt') as f:
        for line in f:
            p=json.loads(line)
            if p['image_id'] in images:predictions[p['image_id']].append(p)
    for image_id in sorted(images):
        for ann in coco.imgToAnns[image_id]:
            aid=ann['id']
            if aid not in target:continue
            candidates=[p for p in predictions[image_id] if p['category_id']==ann['category_id']]
            row={'annotation_id':aid,'image_id':image_id,'category_id':ann['category_id'],'area':ann['area'],
                 'group':groups[aid],'official_matched':{'baseline':aid in b,'local':aid in l,'scalar':aid in s}}
            if candidates:
                ious=mu.iou([p['segmentation'] for p in candidates],[coco.annToRLE(ann)],[0])[:,0]
                # Retain original score-sorted output order for exact ties.
                ix=int(np.argmax(ious));chosen=candidates[ix]
                row.update(raw_id=chosen['raw_id'],score=chosen['score'],baseline_mask_iou=float(ious[ix]),reference={'baseline':chosen['segmentation']})
            else:row['missing_baseline_same_class']=True
            rows[aid]=row
    wanted=defaultdict(list)
    for aid,r in rows.items():
        if 'raw_id' in r:wanted[(r['image_id'],r['category_id'],r['raw_id'])].append(aid)
    for name,path in [('local',local/'predictions_local4_center_s0.jsonl.gz'),('scalar',scalar/'predictions_scalar_selected.jsonl.gz')]:
        with gzip.open(path,'rt') as f:
            for line in f:
                p=json.loads(line);key=(p['image_id'],p['category_id'],p['raw_id'])
                for aid in wanted.get(key,[]):
                    assert abs(rows[aid]['score']-p['score'])<1e-6
                    rows[aid]['reference'][name]=p['segmentation']
    with gzip.open(a.out/'cohort.jsonl.gz','wt',encoding='utf-8') as f:
        for aid in sorted(rows):f.write(json.dumps(rows[aid])+'\n')
    result={'gt':len(rows),'images':len(images),'groups':dict(Counter(groups.values())),
            'missing_reference':sum('raw_id' not in r for r in rows.values()),'elapsed_s':time.monotonic()-start}
    (a.out/'SUMMARY.json').write_text(json.dumps(result,indent=2));(a.out/'COMPLETE.json').write_text(json.dumps(result));print(json.dumps(result))


if __name__=='__main__':main()
