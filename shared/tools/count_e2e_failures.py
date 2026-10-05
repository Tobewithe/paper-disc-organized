import json
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import pycocotools.mask as mask_util
import sys

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

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
    # 1. Load the 270 explicitly tracked "doomed dense" GTs
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    doomed_gts = set()
    with csv_path.open('r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['stage'] == 'raw' and r['raw_strict_mask_absent'] == '1' and r['support_available'] == '1':
                doomed_gts.add(int(r['annotation_id']))
                
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    anns_by_img = {}
    total_gts = len(manifest['annotations'])
    
    for ann in manifest['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    weights = Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt')
    model = load_model(weights, 'cuda')
    
    total_e2e_failed = 0
    doomed_still_failed = 0
    
    # Evaluate all test images
    for img_id, gts in anns_by_img.items():
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        h, w = img_info['height'], img_info['width']
        
        results = model.predict(source=str(img_path), conf=0.25, retina_masks=True, save=False, verbose=False)
        res = results[0]
        
        pred_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
        pred_masks = res.masks.data.cpu().numpy() if res.masks else []
        
        for gt in gts:
            gx, gy, gw, gh = gt['bbox']
            gt_box = [gx, gy, gx+gw, gy+gh]
            rle = mask_util.frPyObjects(gt['segmentation'], h, w)
            gt_mask = mask_util.decode(rle)
            if len(gt_mask.shape) == 3: gt_mask = np.any(gt_mask, axis=2)
            gt_area = gt_mask.sum()
            
            best_cov = 0
            best_pur = 0
            
            for i, pbox in enumerate(pred_boxes):
                iou = box_iou(gt_box, pbox)
                if iou > 0.1: # loose check to find any overlapping prediction
                    pmask = pred_masks[i] > 0.5
                    pmask = cv2.resize(pmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    inter = np.logical_and(pmask, gt_mask).sum()
                    pred_area = pmask.sum()
                    
                    cov = inter / gt_area if gt_area > 0 else 0
                    pur = inter / pred_area if pred_area > 0 else 0
                    
                    if cov + pur > best_cov + best_pur: # maximize joint quality
                        best_cov = cov
                        best_pur = pur
            
            # Strict Criteria: cov >= 0.75 and pur >= 0.75
            is_failed = best_cov < 0.75 or best_pur < 0.75
            
            if is_failed:
                total_e2e_failed += 1
                if gt['id'] in doomed_gts:
                    doomed_still_failed += 1

    print("\n--- E2E Failure Stats (Finetuned Baseline) ---")
    print(f"Total GT instances in Faro Test Set: {total_gts}")
    print(f"Total E2E Failed Instances: {total_e2e_failed} ({(total_e2e_failed/total_gts)*100:.1f}%)")
    print("-" * 45)
    print(f"Original 'Doomed Dense' cases: {len(doomed_gts)}")
    print(f"Doomed cases that STILL FAIL in E2E: {doomed_still_failed} ({(doomed_still_failed/len(doomed_gts))*100:.1f}%)")

if __name__ == '__main__':
    main()
