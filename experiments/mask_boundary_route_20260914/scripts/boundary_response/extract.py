"""Revisit fixed train2017 slots; capture spatial responses without refitting YOLO."""
import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict
from common import setup,atomic,progress


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();p,out=setup(args.protocol,args.output)
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    from data_access import load_bank
    from mask_calibration import input_logits,export_masks
    from local_features import local_features
    bank=load_bank(p['bank']);matched=bank['matched'];n=len(matched);k=len(p['actions'])
    lookup={(r['image_id'],r['candidate_index']):i for i,r in enumerate(matched)}
    byimage=defaultdict(list)
    for i,r in enumerate(matched):byimage[r['image_id']].append(i)
    x=np.zeros((n,k,5));local=np.zeros((n,k,len(p['local_features'])));tau=np.zeros((n,k))
    labels=np.zeros((n,k,3));eligible=np.zeros((n,k),bool);expected_area=np.zeros((n,k))
    action_index={a['name']:j for j,a in enumerate(p['actions'])};seen=np.zeros((n,k),bool)
    for s in csv.DictReader((Path(p['bank'])/'instance_records.csv').open(newline='',encoding='utf-8')):
        if s['variant'] not in action_index:continue
        i=lookup[(int(s['image_id']),int(s['candidate_index']))];j=action_index[s['variant']];r=matched[i]
        assert not seen[i,j] and int(s['annotation_id'])==r['annotation_id'];seen[i,j]=True
        a=float(s['removed_tp']);b=float(s['removed_fp']);g=r['gt_area'];area=bank['area'][r['index']]
        expected_area[i,j]=area-a-b;eligible[i,j]=expected_area[i,j]>.5
        labels[i,j]=[float(s['iou'])-r['baseline_iou'],a/g,b/g] if eligible[i,j] else [0,0,0]
        x[i,j]=bank['features'][r['index']];x[i,j,4]=(a+b)/area
    assert seen.all()
    coco=COCO(p['annotations']);ids=sorted(bank['ids']);paths=[str(Path(p['images'])/coco.imgs[i]['file_name']) for i in ids]
    assert all(Path(s).is_file() for s in paths)
    source=out/'image_paths.txt';source.write_text('\n'.join(paths),encoding='utf-8')
    # Pixel parity at six predeclared evenly spaced images, not outcome-selected.
    check_ids=set(ids[::max(len(ids)//6,1)][:6]);baseline_rle={}
    predictions=json.loads((Path(p['bank'])/'predictions_official_zero.json').read_text())
    for r in predictions:
        if r['image_id'] in check_ids:baseline_rle[(r['image_id'],r['candidate_index'])]=r['segmentation']
    del predictions
    model=YOLO(p['weights']);assert model.model.model[-1].end2end is True
    catbyname={c['name']:c['id'] for c in coco.loadCats(coco.getCatIds())}
    catmap={i:catbyname[name] for i,name in model.names.items()}
    done=0;checked=0;start=time.perf_counter();captured=np.zeros(n,bool)
    class Extractor(SegmentationPredictor):
        def construct_result(self,pred,img,orig_img,img_path,proto):
            nonlocal done,checked
            iid=int(Path(img_path).stem);indices=byimage[iid];shape=img.shape[2:];orig=orig_img.shape[:2]
            # Preserve the original decoder's GEMM batch shapes. Changing only
            # coefficient row selection can change rounding at mask thresholds.
            for off in range(0,len(pred),24):
                ri=[i for i in indices if off<=matched[i]['candidate_index']<off+24]
                if not ri:continue
                ci=[matched[i]['candidate_index'] for i in ri]
                one=pred[ci];boxes=ops.scale_boxes(shape,one[:,:4].clone(),orig).cpu().numpy()
                for ii,ix in enumerate(ri):
                    cand=bank['rows'][matched[ix]['index']]
                    assert np.max(np.abs(boxes[ii]-[float(cand[a]) for a in ['box_x1','box_y1','box_x2','box_y2']]))<1e-3
                    assert catmap[int(one[ii,5])]==matched[ix]['category_id']
                    assert abs(float(one[ii,4])-float(cand['score']))<1e-6
                batch=pred[off:off+24]
                batch_logits=input_logits(proto,batch[:,6:],batch[:,:4],shape)
                logits=batch_logits[[i-off for i in ci]];base=(logits>0).byte()
                exported=export_masks(base,orig);area=exported.sum((1,2)).float()
                assert np.max(np.abs(area.cpu().numpy()-bank['area'][[matched[i]['index'] for i in ri]]))<.5
                if iid in check_ids:
                    for j,ix in enumerate(ri):
                        ref=mu.decode(baseline_rle[(iid,matched[ix]['candidate_index'])])
                        assert np.array_equal(exported[j].cpu().numpy(),ref)
                        checked+=1
                for j,action in enumerate(p['actions']):
                    threshold=.75/(1+(area/2304).square()) if j==0 else torch.full_like(area,float(action['tau']))
                    trial=(logits>threshold[:,None,None]).byte();exp_trial=export_masks(trial,orig)
                    observed_area=exp_trial.sum((1,2)).cpu().numpy()
                    error=np.abs(observed_area-expected_area[ri,j])
                    assert error.max()<1e-5,dict(image_id=iid,action=action['name'],candidates=ci,
                        observed=observed_area.tolist(),expected=expected_area[ri,j].tolist(),max_error=float(error.max()))
                    tau[ri,j]=threshold.cpu().numpy();local[ri,j]=local_features(logits,base,trial)
                captured[ri]=True
            done+=1
            if done%100==0 or done==len(ids):progress(out,'extract',images=done,total=len(ids),matched=int(captured.sum()),elapsed_seconds=time.perf_counter()-start)
            return Results(orig_img,path=img_path,names=self.model.names,boxes=None,masks=None)
    for _ in model.predict(source=str(source),predictor=Extractor,imgsz=640,conf=.001,max_det=300,
        batch=1,half=False,retina_masks=False,device=0,verbose=False,stream=True,save=False):pass
    assert captured.all() and np.isfinite(local).all()
    np.savez_compressed(out/'features.npz',shape=x,local=local,tau=tau,labels=labels,eligible=eligible,
        keys=np.asarray([(r['image_id'],r['candidate_index'],r['annotation_id']) for r in matched],dtype=np.int64))
    atomic(out/'instances.json',matched)
    result=dict(images=len(ids),matched=len(matched),actions=k,local_features=p['local_features'],
        checked_pixel_masks=checked,all_action_area_parity=True,all_slot_box_class_score_parity=True,
        elapsed_seconds=time.perf_counter()-start,scope=p['scope'])
    atomic(out/'SUMMARY.json',result);progress(out,'completed',**result)


if __name__=='__main__':main()
