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

def calc_ici(target_box, all_boxes):
    tx1, ty1, tw, th = target_box
    tx2, ty2 = tx1+tw, ty1+th
    t_area = max(0, tw * th)
    if t_area == 0: return 0
    ici = 0
    for b in all_boxes:
        if b == target_box: continue
        bx1, by1, bw, bh = b
        bx2, by2 = bx1+bw, by1+bh
        inter_x1 = max(tx1, bx1)
        inter_y1 = max(ty1, by1)
        inter_x2 = min(tx2, bx2)
        inter_y2 = min(ty2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        ici += inter_area / t_area
    return ici

def evaluate_model(model, imgs_by_id, anns_by_img, img_root, all_gts_info):
    failed_counts = { 'Bin 1 (0-0.2)': 0, 'Bin 2 (0.2-0.5)': 0, 'Bin 3 (0.5-1.0)': 0, 'Bin 4 (>1.0)': 0 }
    
    for img_id, gts in anns_by_img.items():
        img_info = imgs_by_id[img_id]
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
            
            best_cov, best_pur = 0, 0
            for i, pbox in enumerate(pred_boxes):
                if box_iou(gt_box, pbox) > 0.05:
                    pmask = pred_masks[i] > 0.5
                    pmask = cv2.resize(pmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    inter = np.logical_and(pmask, gt_mask).sum()
                    pred_area = pmask.sum()
                    cov = inter / gt_area if gt_area > 0 else 0
                    pur = inter / pred_area if pred_area > 0 else 0
                    if cov + pur > best_cov + best_pur:
                        best_cov, best_pur = cov, pur
            
            if best_cov < 0.75 or best_pur < 0.75:
                ici = all_gts_info[gt['id']]
                if ici <= 0.2: failed_counts['Bin 1 (0-0.2)'] += 1
                elif ici <= 0.5: failed_counts['Bin 2 (0.2-0.5)'] += 1
                elif ici <= 1.0: failed_counts['Bin 3 (0.5-1.0)'] += 1
                else: failed_counts['Bin 4 (>1.0)'] += 1
                
    return failed_counts

def main():
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    imgs_by_id = {img['id']: img for img in manifest['images']}
    anns_by_img = {}
    for ann in manifest['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)
        
    all_gts_info = {}
    ici_distribution = { 'Bin 1 (0-0.2)': 0, 'Bin 2 (0.2-0.5)': 0, 'Bin 3 (0.5-1.0)': 0, 'Bin 4 (>1.0)': 0 }
    
    for img_id, gts in anns_by_img.items():
        all_boxes = [g['bbox'] for g in gts]
        for gt in gts:
            ici = calc_ici(gt['bbox'], all_boxes)
            all_gts_info[gt['id']] = ici
            if ici <= 0.2: ici_distribution['Bin 1 (0-0.2)'] += 1
            elif ici <= 0.5: ici_distribution['Bin 2 (0.2-0.5)'] += 1
            elif ici <= 1.0: ici_distribution['Bin 3 (0.5-1.0)'] += 1
            else: ici_distribution['Bin 4 (>1.0)'] += 1
            
    print("--- Faro Test Set ICI Distribution ---")
    for bin_name, count in ici_distribution.items():
        print(f"{bin_name}: {count} instances")
        
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    print("\nLoading Baseline Model...")
    baseline = load_model(Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt'), 'cuda')
    print("Evaluating Baseline E2E Failures...")
    baseline_fails = evaluate_model(baseline, imgs_by_id, anns_by_img, img_root, all_gts_info)
    
    print("\nLoading CCL Model...")
    ccl = load_model(Path('runs/segment/experiments/ccl_true_finetune/run_ccl_true_15e/weights/best.pt'), 'cuda')
    print("Evaluating CCL E2E Failures...")
    ccl_fails = evaluate_model(ccl, imgs_by_id, anns_by_img, img_root, all_gts_info)
    
    print("\n=========================================================")
    print(f"{'ICI Bin':<18} | {'Total GTs':<10} | {'Baseline Fails':<15} | {'CCL Fails':<10}")
    print("-" * 57)
    for bin_name in ici_distribution.keys():
        total = ici_distribution[bin_name]
        bf = baseline_fails[bin_name]
        cf = ccl_fails[bin_name]
        print(f"{bin_name:<18} | {total:<10} | {bf:<15} | {cf:<10}")
    print("=========================================================")

if __name__ == '__main__':
    main()
