import json
import csv
import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import sys
from ultralytics.utils.ops import process_mask_native

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
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    failed_gts = set()
    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['stage'] == 'raw' and r['raw_strict_mask_absent'] == '1' and r['support_available'] == '1':
                failed_gts.add((int(r['image_id']), int(r['annotation_id'])))
                
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    
    gts_by_image = defaultdict(list)
    for ann in manifest['annotations']:
        gt_bbox = ann['bbox'] # x, y, w, h
        gt_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]]
        is_failed = (ann['image_id'], ann['id']) in failed_gts
        gts_by_image[ann['image_id']].append({
            'id': ann['id'],
            'box': gt_xyxy,
            'is_failed': is_failed
        })
        
    target_image_ids = [img_id for img_id, gts in gts_by_image.items() if any(g['is_failed'] for g in gts)]
    
    # Load NEW MODEL
    new_weights = Path('C:/Dpan/codexproject/paper-disc/runs/segment/experiments/ccl_finetune/run_ccl_15e/weights/best.pt')
    model = load_model(new_weights, 'cuda')
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    dense_similarities = []
    
    for img_id in target_image_ids[:50]:
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        
        stash = []
        orig_process = process_mask_native
        def patched_process(protos, masks_in, bboxes, shape):
            stash.append({
                'masks_in': masks_in.detach().cpu(),
                'bboxes': bboxes.detach().cpu()
            })
            return orig_process(protos, masks_in, bboxes, shape)
            
        with patch('ultralytics.utils.ops.process_mask_native', patched_process):
            model.predict(source=str(img_path), conf=0.05, retina_masks=True, verbose=False, save=False)
            
        if not stash: continue
        data = stash[0]
        bboxes = data['bboxes'].numpy()
        masks_in = data['masks_in'].numpy()
        
        gts = gts_by_image[img_id]
        matched_preds = {}
        for gt in gts:
            best_iou = 0
            best_idx = -1
            for i, pbox in enumerate(bboxes):
                iou = box_iou(gt['box'], pbox)
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx != -1:
                matched_preds[gt['id']] = best_idx
                
        gt_ids = list(matched_preds.keys())
        for i in range(len(gt_ids)):
            for j in range(i+1, len(gt_ids)):
                gt1 = next(g for g in gts if g['id'] == gt_ids[i])
                gt2 = next(g for g in gts if g['id'] == gt_ids[j])
                
                inter_iou = box_iou(gt1['box'], gt2['box'])
                if inter_iou > 0.05:
                    idx1 = matched_preds[gt_ids[i]]
                    idx2 = matched_preds[gt_ids[j]]
                    c1 = masks_in[idx1]
                    c2 = masks_in[idx2]
                    sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                    dense_similarities.append(float(sim))

    print("--- CCL Model Evaluation ---")
    if dense_similarities:
        print(f"New Dense Mean Cosine Similarity (N={len(dense_similarities)}): {np.mean(dense_similarities):.4f}")
    else:
        print("No dense pairs found.")

if __name__ == '__main__':
    main()
