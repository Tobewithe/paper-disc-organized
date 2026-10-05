import torch
import cv2
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
import sys

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
    # 1. We know image ID 88 (file: 03_faro_1_64.jpg) has failed dense cases.
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    
    img_info = next(img for img in manifest['images'] if img['id'] == 88)
    img_path = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images') / Path(img_info['file_name']).name
    
    # 2. Load the Fine-Tuned Model (which supposedly 'solved' it)
    weights = Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt')
    model = load_model(weights, 'cuda')
    
    stash = {}
    orig_process = process_mask_native
    def patched_process(protos, masks_in, bboxes, shape):
        # Calculate UNCROPPED raw mask (the fundamental representation)
        c, mh, mw = protos.shape
        raw_masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        raw_masks = torch.sigmoid(raw_masks).detach().cpu().numpy()
        
        stash['raw_masks'] = raw_masks
        stash['bboxes'] = bboxes.detach().cpu().numpy()
        stash['masks_in'] = masks_in.detach().cpu().numpy()
        
        return orig_process(protos, masks_in, bboxes, shape)
        
    with patch('ultralytics.utils.ops.process_mask_native', patched_process):
        model.predict(source=str(img_path), conf=0.25, retina_masks=True, verbose=False, save=False)
        
    raw_masks = stash['raw_masks']
    bboxes = stash['bboxes']
    coeffs = stash['masks_in']
    
    # Find two densely overlapping boxes
    n = len(bboxes)
    best_pair = None
    best_iou = 0
    for i in range(n):
        for j in range(i+1, n):
            iou = box_iou(bboxes[i], bboxes[j])
            if iou > best_iou and iou < 0.9: # High overlap but not duplicate
                best_iou = iou
                best_pair = (i, j)
                
    if best_pair:
        i, j = best_pair
        print(f"Found Overlapping Pair: {i} and {j} with Box IoU = {best_iou:.4f}")
        
        sim = np.dot(coeffs[i], coeffs[j]) / (np.linalg.norm(coeffs[i]) * np.linalg.norm(coeffs[j]))
        print(f"Coefficient Cosine Similarity: {sim:.4f}")
        
        # Check raw mask similarity (before bounding box crop!)
        rm1 = (raw_masks[i] > 0.5).astype(np.float32)
        rm2 = (raw_masks[j] > 0.5).astype(np.float32)
        
        inter = np.logical_and(rm1, rm2).sum()
        union = np.logical_or(rm1, rm2).sum()
        mask_iou = inter / union if union > 0 else 0
        
        print(f"RAW UNCROPPED Mask IoU between the two 'distinct' instances: {mask_iou:.4f}")
        
        # Save visualization
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        rm1_resized = cv2.resize(raw_masks[i], (w, h))
        rm2_resized = cv2.resize(raw_masks[j], (w, h))
        
        heatmap1 = cv2.applyColorMap(np.uint8(rm1_resized * 255), cv2.COLORMAP_JET)
        heatmap2 = cv2.applyColorMap(np.uint8(rm2_resized * 255), cv2.COLORMAP_JET)
        
        vis1 = cv2.addWeighted(img, 0.5, heatmap1, 0.5, 0)
        vis2 = cv2.addWeighted(img, 0.5, heatmap2, 0.5, 0)
        
        # Draw bounding boxes
        box1 = bboxes[i].astype(int)
        box2 = bboxes[j].astype(int)
        cv2.rectangle(vis1, (box1[0], box1[1]), (box1[2], box1[3]), (0,255,0), 2)
        cv2.rectangle(vis2, (box2[0], box2[1]), (box2[2], box2[3]), (0,255,0), 2)
        
        cv2.imwrite('experiments/raw_blob_1.jpg', vis1)
        cv2.imwrite('experiments/raw_blob_2.jpg', vis2)
        print("Saved raw blob visualizations.")

if __name__ == '__main__':
    main()
