"""One-to-many FP32 COCO evaluation, using official NMS and native mask decoding."""
import argparse, gc, gzip, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def scores(coco,pred,kind,ids):
    if pred:
        detections=coco.loadRes(pred)
    else:
        detections=COCO();detections.dataset={'images':list(coco.imgs.values()),'categories':list(coco.cats.values()),'annotations':[]};detections.createIndex()
    ev=COCOeval(coco,detections,kind);ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    good=set();ti=int(np.argmin(abs(ev.params.iouThrs-.75)))
    for item in ev.evalImgs:
        if item is not None and item['aRng']==ev.params.areaRng[0]:
            good.update(int(g) for j,g in enumerate(item['gtIds']) if not item['gtIgnore'][j] and item['gtMatches'][ti,j]>0)
    ordinary={int(g) for g,a in coco.anns.items() if a['image_id'] in set(ids) and not a.get('iscrowd',0) and not a.get('ignore',0)}
    metrics=dict(zip(('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'),[100*float(x) for x in ev.stats]))
    del detections,ev;gc.collect()
    return metrics,sorted(good&ordinary)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--mode',choices=['R','A','B','C'],required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=a.root;out=Path(os.environ['RESEARCH_RUN_DIRECTORY']);start=time.monotonic()
    cfg=json.loads((root/'PROTOCOL.json').read_text());manifest=json.loads((root/'deployment_ids.json').read_text())
    data=Path(cfg['data']);coco=COCO(str(data/'annotations/instances_val2017.json'));ids=sorted(coco.imgs)
    if a.limit:ids=ids[:a.limit]
    assert len(ids)==5000 or a.limit
    cats=sorted(coco.cats);model=YOLO(str(a.weights)).model.cuda().float().eval();head=model.model[-1]
    if a.mode!='R':
        cp=root/'runs'/manifest['run_ids']['train_'+a.mode]/f"epoch{cfg['epochs']}.pt"
        ck=torch.load(cp,map_location='cpu',weights_only=False);head.cv4.load_state_dict(ck['ema_cv4']);del ck
    head.end2end=False
    transform=LetterBox((640,640),auto=False,stride=32)
    identity=hashlib.sha256();geometry={};timings=[]
    with gzip.open(out/'predictions.jsonl.gz','wt',compresslevel=1) as masks_file,gzip.open(out/'boxes.jsonl.gz','wt',compresslevel=1) as boxes_file:
        for ni,iid in enumerate(ids):
            im=cv2.imread(str(data/'images/val2017'/coco.imgs[iid]['file_name']));assert im is not None
            orig=im.shape[:2];params=transform.get_params({'img':im});x=transform.apply_image({'img':im},params)['img']
            ratio_pad=(params['ratio'],(params['left'],params['top']))
            inp=torch.from_numpy(np.ascontiguousarray(x[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            torch.cuda.synchronize();t=time.monotonic()
            decoded,raw=model(inp);prediction,proto=decoded
            dets,indices=non_max_suppression(prediction.clone(),conf_thres=.001,iou_thres=.7,multi_label=True,nc=80,max_det=300,end2end=False,return_idxs=True,max_time_img=5.)
            det,rids=dets[0],indices[0]
            identity.update(str(iid).encode());identity.update(det[:,:6].cpu().numpy().tobytes());identity.update(rids.cpu().numpy().tobytes())
            boxes=ops.scale_boxes(inp.shape[2:],det[:,:4].clone(),orig,ratio_pad=ratio_pad)
            xywh=boxes.clone();xywh[:,2:]-=xywh[:,:2]
            image_masks=[]
            # Decode the complete retained batch consistently, matching native COCO export.
            if len(det):
                native=ops.process_mask_native(proto[0],det[:,6:],det[:,:4],inp.shape[2:])
                masks=ops.scale_masks(native[None],orig,ratio_pad=ratio_pad)[0].byte().cpu().numpy()
            else:masks=[]
            torch.cuda.synchronize();timings.append(time.monotonic()-t)
            for j,mask in enumerate(masks):
                rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                common={'image_id':iid,'category_id':cats[int(det[j,5])],'score':float(det[j,4]),'raw_id':int(rids[j])}
                masks_file.write(json.dumps({**common,'segmentation':rle})+'\n')
                boxes_file.write(json.dumps({**common,'bbox':xywh[j].tolist()})+'\n');image_masks.append(rle)
            # A fixed-output geometry panel, not a full-raw candidate-capability classification.
            gt=[v for v in coco.imgToAnns[iid] if not v.get('iscrowd',0) and not v.get('ignore',0)]
            gt_rle=[coco.annToRLE(v) for v in gt]
            ious=mu.iou(image_masks,gt_rle,[0]*len(gt)) if image_masks and gt else np.zeros((len(image_masks),len(gt)))
            box_ious=mu.iou(xywh.cpu().numpy().tolist(),[v['bbox'] for v in gt],[0]*len(gt)) if image_masks and gt else np.zeros_like(ious)
            gt_areas=mu.area(gt_rle).tolist() if gt else []
            pred_areas=mu.area(image_masks).tolist() if image_masks else []
            geometry[str(iid)]={'gt_ids':[v['id'] for v in gt],'mask_iou':ious.astype(np.float32).tolist(),
                                 'box_iou':box_ious.astype(np.float32).tolist(),'raw_ids':rids.cpu().tolist(),
                                 'classes':[cats[int(v)] for v in det[:,5]],'gt_areas':gt_areas,'pred_areas':pred_areas}
            if ni%100==0 or ni+1==len(ids):
                masks_file.flush();boxes_file.flush()
                p={'mode':a.mode,'images':ni+1,'total':len(ids),'elapsed_s':time.monotonic()-start};save(out/'progress.json',p);print(json.dumps(p),flush=True)
            del inp,decoded,raw,prediction,proto,dets,det,masks
    with gzip.open(out/'fixed_output_geometry.json.gz','wt',compresslevel=1) as f:json.dump(geometry,f)
    del geometry;gc.collect()
    with gzip.open(out/'predictions.jsonl.gz','rt') as f:pred=[json.loads(x) for x in f]
    metrics,matched=scores(coco,pred,'segm',ids);del pred;gc.collect()
    with gzip.open(out/'boxes.jsonl.gz','rt') as f:pred=[json.loads(x) for x in f]
    boxmetrics,boxmatched=scores(coco,pred,'bbox',ids)
    result={'mode':a.mode,'images':len(ids),'mask':metrics,'box':boxmetrics,'matched75':len(matched),
            'detection_identity_sha256':identity.hexdigest(),'mean_forward_decode_seconds':float(np.mean(timings[5:])),
            'evaluation':cfg['evaluation'],'epoch':cfg['epochs'] if a.mode!='R' else None}
    save(out/'RESULTS.json',result);save(out/'MATCHED_GT75.json',{'mask':matched,'box':boxmatched})
    save(out/'COMPLETE.json',{'status':'completed','mode':a.mode,'images':len(ids),'elapsed_s':time.monotonic()-start});print(json.dumps(result),flush=True)


if __name__=='__main__':main()
