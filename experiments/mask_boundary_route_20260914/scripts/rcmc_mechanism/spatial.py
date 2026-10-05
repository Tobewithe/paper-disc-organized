"""Describe mutually exclusive FP sources and deleted pixels on frozen output slots."""
from pathlib import Path
import argparse
import csv
import json
import time
from collections import defaultdict
from common import setup, atomic, load_slots, load_predictions, image_regions, gt_regions, mask_counts, progress, choose_panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--limit-images',type=int,default=0);args=parser.parse_args()
    p,out=setup(args.protocol,args.output)
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    coco=COCO(p['annotations']);rows=load_slots(p);panel=choose_panel(rows,coco,p)
    atomic(out/'panel.json',panel)
    _,base=load_predictions(Path(p['bank'])/'predictions_official_zero.json')
    _,trial=load_predictions(Path(p['bank'])/'predictions_smooth_gated.json')
    by_image=defaultdict(list)
    for row in rows:by_image[row['image_id']].append(row)
    all_ids=json.loads((Path(p['bank'])/'image_ids.json').read_text())
    ids=all_ids[:args.limit_images] if args.limit_images else all_ids
    accum=defaultdict(lambda:defaultdict(float));checks=0;start=time.perf_counter()
    f=(out/'spatial_instances.csv').open('w',newline='',encoding='utf-8');writer=None
    for ordinal,iid in enumerate(ids):
        if not by_image[iid]:continue
        regions=image_regions(coco,iid)
        for row in by_image[iid]:
            key=(iid,row['candidate_index']);target,labels=gt_regions(coco,row['annotation_id'],regions)
            b=mu.decode(base[key]['segmentation']).astype(bool)
            t=mu.decode(trial[key]['segmentation']).astype(bool) if key in trial else np.zeros_like(b)
            assert not np.any(t & ~b), ('not nested',key)
            r=t if row['calibrated'] else b
            doutside=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            dinside=cv2.distanceTransform(target.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            yy,xx=np.ogrid[:target.shape[0],:target.shape[1]]
            gx,gy,gw,gh=coco.anns[row['annotation_id']]['bbox']
            gt_rect=(xx>=gx)&(xx<gx+gw)&(yy>=gy)&(yy<gy+gh)
            bx1,by1,bx2,by2=row['box']
            boxdist=np.minimum(np.minimum(xx-bx1,bx2-xx),np.minimum(yy-by1,by2-yy))
            for name,mask,expected in [('baseline',b,row['baseline_iou']),('fixed_trial',t,None),('response',r,row['gated_iou'])]:
                values=mask_counts(mask,target,labels)
                if expected is not None:
                    assert abs(values['iou']-expected)<1e-7,(key,name,values['iou'],expected)
                    checks+=1
                removed=b & ~mask;wrong=mask & ~target
                removed_counts=mask_counts(removed,target,labels)
                values.update({f'removed_{k}':removed_counts[k] for k in ('tp','fp','background_fp','same_neighbor_fp','different_neighbor_fp','mixed_neighbor_fp','crowd_ignore_fp')})
                for width in (1,2,4):
                    values[f'fp_gt_boundary_{width}px']=int((wrong & (doutside<=width)).sum())
                    values[f'removed_tp_interior_gt_{width}px']=int((removed & target & (dinside>width)).sum())
                    values[f'fp_pred_box_band_{width}px']=int((wrong & (boxdist>=0)&(boxdist<=width)).sum())
                values['fp_distance_gt_mean']=float(doutside[wrong].mean()) if wrong.any() else None
                values['fp_distance_gt_scale_mean']=values['fp_distance_gt_mean']/max(np.sqrt(target.sum()),1) if wrong.any() else None
                values['fp_outside_continuous_gt_bbox_proxy']=int((wrong & ~gt_rect).sum())
                _,components=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
                touching=np.unique(components[mask & target]);touching=touching[touching!=0]
                values['fp_components_without_any_target']=int((wrong & ~np.isin(components,touching)).sum())
                outcome='repair' if row['baseline_iou']<.75<=values['iou'] else 'damage' if values['iou']<.75<=row['baseline_iou'] else 'stable_success' if row['baseline_iou']>=.75 else 'stable_failure'
                record={k:row[k] for k in ('image_id','annotation_id','candidate_index','category_id','group','box_iou')}
                record.update(variant=name,outcome=outcome,**values)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader()
                writer.writerow(record)
                for grouping in (row['group'],'all_matched'):
                    a=accum[(grouping,name)];a['n']+=1;a['gt_pixels']+=values['gt_pixels']
                    a['repairs']+=int(outcome=='repair');a['damages']+=int(outcome=='damage')
                    for field in ('iou','coverage','fp','background_fp','same_neighbor_fp','different_neighbor_fp','mixed_neighbor_fp','crowd_ignore_fp','removed_tp','removed_fp','fp_outside_continuous_gt_bbox_proxy','fp_components_without_any_target'):
                        a[field+'_sum']+=values[field]
                    a['fp_per_gt_sum']+=values['fp']/max(values['gt_pixels'],1)
                    a['removed_tp_per_gt_sum']+=values['removed_tp']/max(values['gt_pixels'],1)
        if (ordinal+1)%25==0 or ordinal==len(ids)-1:
            f.flush();progress(out,'spatial',images=ordinal+1,total=len(ids),elapsed_seconds=time.perf_counter()-start)
    f.close()
    summary={}
    for (group,variant),a in accum.items():
        v=dict(a)
        for key in ('iou','coverage','fp_per_gt','removed_tp_per_gt'):
            v[key+'_mean']=a[key+'_sum']/a['n']
        for key in ('background_fp','same_neighbor_fp','different_neighbor_fp','mixed_neighbor_fp','crowd_ignore_fp'):
            v[key+'_share_of_fp']=a[key+'_sum']/a['fp_sum'] if a['fp_sum'] else None
        summary.setdefault(group,{})[variant]=v
    result={'groups':summary,'images':len(ids),'checked_iou_values':checks,'panel':{k:panel[k] for k in ('targets','controls','unmatched_target_ids')},'elapsed_seconds':time.perf_counter()-start,
            'scope':'fixed final-output matched slots, not score-independent raw taxonomy',
            'limitations':['FP source uses annotated COCO masks; unannotated background is not proof of physical background.','GT bbox proxy is not the actual training loss support.','Crowd pixels are separate in FP accounting; original baseline IoU/cohort definitions are unchanged.','Pixel quantities do not prove an upstream cause.']}
    atomic(out/'SUMMARY.json',result);progress(out,'completed',images=len(ids),target_failure=summary.get('target_failure'))


if __name__=='__main__':main()
