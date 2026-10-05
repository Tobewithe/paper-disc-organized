import sys
from pathlib import Path
import json
import csv
import torch
import numpy as np
import cv2
from unittest.mock import patch
import pycocotools.mask as mask_util

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model
from ultralytics.utils.ops import process_mask_native

def box_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def main():
    # 1. Load failed GTs (the 270)
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    target_gts = {}
    with csv_path.open('r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['stage'] == 'raw' and r['raw_strict_mask_absent'] == '1' and r['support_available'] == '1':
                if int(r['image_id']) not in target_gts:
                    target_gts[int(r['image_id'])] = []
                target_gts[int(r['image_id'])].append({
                    'id': int(r['annotation_id'])
                })

    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    anns_by_id = {ann['id']: ann for ann in manifest['annotations']}
    
    # LOAD NEW MODEL
    new_weights = Path('C:/Dpan/codexproject/paper-disc/runs/segment/experiments/ccl_finetune/run_ccl_15e/weights/best.pt')
    model = load_model(new_weights, 'cuda')
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    recovered = 0
    total = sum(len(v) for v in target_gts.values())
    
    for img_id, gts in target_gts.items():
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        h, w = img_info['height'], img_info['width']
        
        stash = []
        orig_process = process_mask_native
        
        def patched_process_oracle(protos, masks_in, bboxes, shape):
            new_bboxes = bboxes.clone()
            np_bboxes = bboxes.detach().cpu().numpy()
            
            for gt in gts:
                ann = anns_by_id[gt['id']]
                gx, gy, gw, gh = ann['bbox']
                gt_box = [gx, gy, gx+gw, gy+gh]
                
                best_iou = 0
                best_idx = -1
                for i, pb in enumerate(np_bboxes):
                    iou = box_iou(gt_box, pb)
                    if iou > 0.5 and iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_idx != -1:
                    new_bboxes[best_idx, 0] = gt_box[0]
                    new_bboxes[best_idx, 1] = gt_box[1]
                    new_bboxes[best_idx, 2] = gt_box[2]
                    new_bboxes[best_idx, 3] = gt_box[3]
            
            return orig_process(protos, masks_in, new_bboxes, shape)

        with patch('ultralytics.utils.ops.process_mask_native', patched_process_oracle):
            results = model.predict(source=str(img_path), conf=0.05, retina_masks=True, verbose=False, save=False)
            
        if not results[0].masks:
            continue
            
        pred_masks = results[0].masks.data.cpu().numpy()
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for gt in gts:
            ann = anns_by_id[gt['id']]
            gx, gy, gw, gh = ann['bbox']
            gt_box = [gx, gy, gx+gw, gy+gh]
            rle = mask_util.frPyObjects(ann['segmentation'], h, w)
            gt_mask = mask_util.decode(rle)
            if len(gt_mask.shape) == 3: gt_mask = np.any(gt_mask, axis=2)
            gt_area = gt_mask.sum()
            
            best_coverage = 0
            best_purity = 0
            
            for i, pbox in enumerate(pred_boxes):
                iou = box_iou(gt_box, pbox)
                if iou > 0.5: 
                    pmask = pred_masks[i] > 0.5
                    pmask = cv2.resize(pmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    inter = np.logical_and(pmask, gt_mask).sum()
                    pred_area = pmask.sum()
                    cov = inter / gt_area if gt_area > 0 else 0
                    pur = inter / pred_area if pred_area > 0 else 0
                    if cov > best_coverage:
                        best_coverage = cov
                        best_purity = pur
            
            if best_coverage >= 0.75 and best_purity >= 0.75:
                recovered += 1

    print(f"\n--- CCL Oracle Box Crop Result ---")
    print(f"Total Doomed Cases Tested: {total}")
    print(f"Recovered (Cov>=0.75 & Pur>=0.75): {recovered}")
    print(f"Still Unrecoverable: {total - recovered}")

if __name__ == '__main__':
    main()
