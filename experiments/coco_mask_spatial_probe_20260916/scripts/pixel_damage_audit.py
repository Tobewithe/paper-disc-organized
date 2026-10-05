"""Full-val repair/damage pixel analysis at a fixed original prediction identity.

The diagnostic representative is the original same-class output with maximum
mask IoU to each GT. This is an oracle reference, not the COCO score-order match.
"""
import argparse
from collections import defaultdict,Counter
import json
import gzip
from pathlib import Path
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu


def key(p):
    return (p['image_id'],p['category_id'],p['score'],p['raw_id']) if 'raw_id' in p else (p['image_id'],p['category_id'],p['score'],*p['bbox'])
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    for k in ('source','score','annotations','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',default='coeff_local4')
    p.add_argument('--group-mode',default=None,help='Hold target GT transition groups from another mode fixed.')
    p.add_argument('--baseline-source',type=Path,help='Read frozen baseline predictions from a different completed run.')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    coco=COCO(str(a.annotations));matched=json.loads((a.score/'MATCHED_GT75.json').read_text())
    b=set(matched['baseline']);c=set(matched[a.group_mode or a.mode]);groups={'damaged':b-c,'repaired':c-b}
    target=groups['damaged']|groups['repaired'];image_ids={coco.anns[i]['image_id'] for i in target}
    pred={}
    for mode in ('baseline',a.mode):
        by=defaultdict(list)
        source=a.baseline_source if mode=='baseline' and a.baseline_source else a.source
        file=source/f'predictions_{mode}.jsonl'
        stream=file.open() if file.exists() else gzip.open(str(file)+'.gz','rt')
        for line in stream:
            r=json.loads(line)
            if r['image_id'] in image_ids:by[r['image_id']].append(r)
        pred[mode]=by
        stream.close()
    rows=[]
    for num,image_id in enumerate(sorted(image_ids)):
        anns=[a for a in coco.imgToAnns[image_id] if a['id'] in target]
        originals=pred['baseline'][image_id];variants={key(x):x for x in pred[a.mode][image_id]}
        assert len(variants)==len(pred[a.mode][image_id])
        all_anns=[v for v in coco.imgToAnns[image_id] if not v.get('iscrowd',0) and not v.get('ignore',0)]
        gt_masks={v['id']:coco.annToMask(v).astype(bool) for v in all_anns}
        union=np.logical_or.reduce(list(gt_masks.values()))
        for ann in anns:
            gt=gt_masks[ann['id']];same=[x for x in originals if x['category_id']==ann['category_id']]
            if not same:
                rows.append(dict(annotation_id=ann['id'],image_id=image_id,group='damaged' if ann['id'] in groups['damaged'] else 'repaired',missing_original_candidate=True));continue
            grle=mu.encode(np.asfortranarray(gt.astype('uint8')))
            ious=mu.iou([x['segmentation'] for x in same],[grle],[0])[:,0];chosen=same[int(ious.argmax())]
            other=variants.get(key(chosen));mask0=mu.decode(chosen['segmentation']).astype(bool)
            mask1=mu.decode(other['segmentation']).astype(bool) if other else np.zeros_like(gt)
            neigh=union&~gt
            def measures(mask):
                tp=int((mask&gt).sum());fp=int((mask&~gt).sum());fn=int((~mask&gt).sum())
                return dict(tp=tp,fp=fp,fn=fn,iou=tp/max(tp+fp+fn,1),coverage=tp/max(tp+fn,1),purity=tp/max(tp+fp,1),
                            neighbor_fp=int((mask&neigh).sum()),background_fp=int((mask&~union).sum()))
            m0=measures(mask0);m1=measures(mask1);added=mask1&~mask0;removed=mask0&~mask1
            rows.append(dict(annotation_id=ann['id'],image_id=image_id,category_id=ann['category_id'],group='damaged' if ann['id'] in groups['damaged'] else 'repaired',
                area_group='small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large',
                fill=float(gt.sum()/max(ann['bbox'][2]*ann['bbox'][3],1)),baseline=m0,corrected=m1,
                added_target=int((added&gt).sum()),removed_target=int((removed&gt).sum()),added_background=int((added&~union).sum()),
                removed_background=int((removed&~union).sum()),added_neighbor=int((added&neigh).sum()),removed_neighbor=int((removed&neigh).sum())))
        if num%100==0:print(json.dumps(dict(images=num+1,total=len(image_ids))),flush=True)
    with (a.out/'instances.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result={}
    for group in groups:
        rs=[r for r in rows if r['group']==group and 'baseline' in r]
        result[group]=dict(total_gt=len(groups[group]),fixed_references=len(rs),
            retained_good_mask75=sum(r['corrected']['iou']>=.75 for r in rs),
            delta_iou_points=100*float(np.mean([r['corrected']['iou']-r['baseline']['iou'] for r in rs])),
            delta_coverage_points=100*float(np.mean([r['corrected']['coverage']-r['baseline']['coverage'] for r in rs])),
            delta_purity_points=100*float(np.mean([r['corrected']['purity']-r['baseline']['purity'] for r in rs])),
            lost_target=sum(r['removed_target']>r['added_target'] for r in rs),
            increased_fp=sum(r['corrected']['fp']>r['baseline']['fp'] for r in rs),
            target_loss_and_fp_increase=sum(r['removed_target']>r['added_target'] and r['corrected']['fp']>r['baseline']['fp'] for r in rs),
            size={g:sum(r['area_group']==g for r in rs) for g in ('small','medium','large')},
            normalized_pixel_change={k:float(np.mean([r[k]/max(r['baseline']['tp']+r['baseline']['fn'],1) for r in rs])) for k in
                ('added_target','removed_target','added_background','removed_background','added_neighbor','removed_neighbor')})
    result['limitations']=['Fixed identity is GT-oracle best original same-class mask, not original COCO assignment.','GT transition groups use COCO matching; competition can change matched identities.','Pixel changes do not identify the network root cause.']
    save(a.out/'SUMMARY.json',result);save(a.out/'COMPLETE.json',dict(gt=len(rows),images=len(image_ids),elapsed_s=time.monotonic()-start));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
