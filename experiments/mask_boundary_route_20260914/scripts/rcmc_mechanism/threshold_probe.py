"""Frozen-output threshold response with pre-top-k raw tensors captured separately."""
from pathlib import Path
import argparse
import csv
import json
import time
from collections import defaultdict
from common import setup, atomic, load_predictions, image_regions, gt_regions, mask_counts, progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--panel',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--limit-images',type=int,default=0)
    args=parser.parse_args();p,out=setup(args.protocol,args.output)
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops
    from mask_calibration import input_logits,export_masks,apply_threshold
    from risk_calibration import prediction_features
    from portable_risk import load_estimator
    coco=COCO(p['annotations']);panel=json.loads(Path(args.panel).read_text());by_image=defaultdict(list)
    for row in panel['rows']:by_image[row['image_id']].append(row)
    ids=sorted(by_image);ids=ids[:args.limit_images] if args.limit_images else ids
    _,basebank=load_predictions(Path(p['bank'])/'predictions_official_zero.json')
    model=YOLO(p['weights']);head=model.model.model[-1]
    assert head.end2end is True, 'Expected the weight default one-to-one branch; do not change it silently'
    estimator=load_estimator(p['response_model']);captured={};all_summary=[];start=time.perf_counter()
    (out/'raw').mkdir(exist_ok=True)
    f=(out/'threshold_pixels.csv').open('w',newline='',encoding='utf-8');writer=None
    def capture(module,inputs,output):
        raw=output[1]['one2one']
        decoded=module._inference(raw)[0].T.detach()
        _,_,indices=module.get_topk_index(decoded[None,:,4:4+module.nc],module.max_det)
        captured.clear();captured.update(decoded=decoded,indices=indices[0,:,0],
            proto=output[0][1][0].detach(),post=output[0][0][0].detach(),
            feature_shapes=[list(x.shape) for x in inputs[0]])
    hook=head.register_forward_hook(capture)

    class ProbePredictor(SegmentationPredictor):
        def construct_result(self,pred,img,orig_img,img_path,proto):
            nonlocal writer
            iid=int(Path(img_path).stem);original_shape=orig_img.shape[:2];shape=img.shape[2:]
            post=captured['post'];keep=(post[:,4]>.001)
            post=post[keep][:300];rawids=captured['indices'][keep][:300]
            assert pred.shape==post.shape and torch.allclose(pred,post,rtol=0,atol=1e-6),'postprocess identity mismatch'
            dec=captured['decoded'];np.savez_compressed(out/'raw'/f'{iid:012d}.npz',
                boxes_input=dec[:,:4].cpu().numpy(),scores=dec[:,4:4+head.nc].cpu().numpy(),
                coefficients=dec[:,4+head.nc:].cpu().numpy(),proto=proto.cpu().numpy(),
                raw_ids=np.arange(len(dec),dtype=np.int32),final_raw_ids=rawids.cpu().numpy(),
                input_shape=np.asarray(shape),original_shape=np.asarray(original_shape),
                feature_shapes=np.asarray(captured['feature_shapes']),class_names=np.asarray([model.names[k] for k in range(head.nc)]))
            regions=image_regions(coco,iid)
            for row in by_image[iid]:
                candidate=row['candidate_index'];one=pred[candidate:candidate+1]
                assert int(one[0,5]) in model.names
                box_original=ops.scale_boxes(shape,one[:,:4].clone(),orig_img.shape)[0].cpu().numpy()
                assert np.max(np.abs(box_original-np.asarray(row['box'])))<1e-3,('box mismatch',iid,candidate)
                logits=input_logits(proto,one[:,6:],one[:,:4],shape)
                binary=(logits>0).byte();baseline=export_masks(binary,original_shape)
                baseline_np=baseline[0].cpu().numpy().astype(bool)
                reference=mu.decode(basebank[(iid,candidate)]['segmentation']).astype(bool)
                mismatch=int(np.count_nonzero(baseline_np!=reference))
                assert mismatch==0,('baseline pixel mismatch',iid,candidate,mismatch)
                target,labels=gt_regions(coco,row['annotation_id'],regions)
                area=baseline.sum((1,2)).float();trial=apply_threshold(logits,area,'smooth');trial_export=export_masks(trial,original_shape)
                features=prediction_features(binary,baseline,trial_export,'response')
                benefit=float(estimator.predict(features)[0]);use=benefit>0 and bool(trial_export.any())
                assert use==bool(row['calibrated']),('gate parity mismatch',iid,candidate,benefit)
                result={'image_id':iid,'annotation_id':row['annotation_id'],'candidate_index':candidate,
                        'raw_id':int(rawids[candidate]),'role':row['panel_role'],'partition':row['panel_partition'],
                        'baseline_iou':row['baseline_iou'],'baseline_coverage':row['baseline_recall'],
                        'baseline_parity_pixels':mismatch,'predicted_gain':benefit,'raw_count':len(dec)}
                records=[]
                thresholds=p['thresholds'];tt=torch.tensor(thresholds,device=logits.device,dtype=logits.dtype)
                masks=export_masks((logits>tt[:,None,None]).byte(),original_shape).cpu().numpy().astype(bool)
                for i,threshold in enumerate(thresholds):
                    counts=mask_counts(masks[i],target,labels)
                    counts.update(threshold=float(threshold),variant='threshold',empty=not bool(masks[i].any()))
                    records.append(counts)
                for name,mask in [('smooth',trial_export[0].cpu().numpy().astype(bool)),
                                  ('response',trial_export[0].cpu().numpy().astype(bool) if use else baseline_np)]:
                    counts=mask_counts(mask,target,labels)
                    counts.update(threshold=.75/(1+(float(area[0])/2304)**2) if name=='smooth' else None,
                                  variant=name,empty=not bool(mask.any()))
                    records.append(counts)
                assert abs(records[-1]['iou']-row['gated_iou'])<1e-7
                for counts in records:
                    line={k:result[k] for k in ('image_id','annotation_id','candidate_index','raw_id','role','partition')}
                    line.update(**counts)
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(line));writer.writeheader()
                    writer.writerow(line)
                baseline_counts=records[0];grid=records[:len(thresholds)]
                result['best_grid']=max(grid,key=lambda x:(x['iou'],-x['threshold']))
                result['grid_max_at_right_endpoint']=result['best_grid']['threshold']==thresholds[-1]
                result['smooth']=records[-2];result['response']=records[-1]
                for budget in p['coverage_loss_budgets']:
                    feasible=[x for x in grid if x['coverage']>=baseline_counts['coverage']-budget-1e-12]
                    best=max(feasible,key=lambda x:(x['iou'],-x['threshold']))
                    maxfp=min(feasible,key=lambda x:(x['fp'],x['threshold']))
                    result[f'coverage_loss_{budget}']={'best_iou':best['iou'],'threshold':best['threshold'],
                        'repair':baseline_counts['iou']<.75<=best['iou'],
                        'max_fp_removed_fraction':(baseline_counts['fp']-maxfp['fp'])/baseline_counts['fp'] if baseline_counts['fp'] else None,
                        'max_fp_removal_threshold':maxfp['threshold'],'iou_at_max_fp_removal':maxfp['iou']}
                all_summary.append(result)
            f.flush();atomic(out/'instances_partial.json',all_summary)
            progress(out,'threshold_panel',images=len(list((out/'raw').glob('*.npz'))),total=len(ids),instances=len(all_summary),elapsed_seconds=time.perf_counter()-start)
            returned=pred[:,:6].clone();returned[:,:4]=ops.scale_boxes(shape,returned[:,:4],orig_img.shape)
            return Results(orig_img,path=img_path,names=self.model.names,boxes=returned,masks=None)

    pathlist=out/'image_paths.txt';pathlist.write_text('\n'.join(str(Path(p['images'])/coco.imgs[i]['file_name']) for i in ids),encoding='utf-8')
    try:
        for _ in model.predict(source=str(pathlist),predictor=ProbePredictor,imgsz=640,conf=.001,max_det=300,
            batch=1,half=False,retina_masks=False,device=0,verbose=False,stream=True,save=False):pass
    finally:hook.remove();f.close()
    atomic(out/'instances.json',all_summary)
    summary={}
    for role in ('target','control'):
        rr=[r for r in all_summary if r['role']==role]
        group={'n':len(rr),'grid_oracle_repairs':sum(r['baseline_iou']<.75<=r['best_grid']['iou'] for r in rr),
               'response_repairs':sum(r['baseline_iou']<.75<=r['response']['iou'] for r in rr),
               'response_damage':sum(r['response']['iou']<.75<=r['baseline_iou'] for r in rr),
               'grid_endpoint_selected':sum(r['grid_max_at_right_endpoint'] for r in rr)}
        for budget in p['coverage_loss_budgets']:
            key=f'coverage_loss_{budget}'
            group[key]={'repaired':sum(r[key]['repair'] for r in rr),
                        'mean_iou_gain':float(np.mean([r[key]['best_iou']-r['baseline_iou'] for r in rr])) if rr else None}
        summary[role]=group
    atomic(out/'SUMMARY.json',{'groups':summary,'images':len(ids),'instances':len(all_summary),'baseline_pixel_parity':True,
        'environment':p['environment'],'elapsed_seconds':time.perf_counter()-start,
        'raw_capture':'complete pre-top-k boxes, all class scores, coefficients and prototypes captured per image; five-state geometry not yet evaluated',
        'limitations':['GT chosen grid thresholds are diagnostic oracles, not deployable improvements.','Panel has baseline-stratified sampling, not natural population prevalence.','Current panel partitions were previously explored validation images.']})
    progress(out,'completed',images=len(ids),instances=len(all_summary),groups=summary)


if __name__=='__main__':main()
