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
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    
    img_info = next(img for img in manifest['images'] if img['id'] == 88)
    img_path = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images') / Path(img_info['file_name']).name
    
    # Load the TRUE CCL Finetuned Model
    weights = Path('runs/segment/experiments/ccl_true_finetune/run_ccl_true_15e/weights/best.pt')
    model = load_model(weights, 'cuda')
    
    stash = {}
    orig_process = process_mask_native
    def patched_process(protos, masks_in, bboxes, shape):
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
    
    n = len(bboxes)
    best_pair = None
    best_iou = 0
    # Try to find a highly overlapping pair
    for i in range(n):
        for j in range(i+1, n):
            iou = box_iou(bboxes[i], bboxes[j])
            if iou > best_iou and iou < 0.9: 
                best_iou = iou
                best_pair = (i, j)
                
    if best_pair:
        i, j = best_pair
        print(f"--- TRUE CCL MODEL UNDERLYING RAW TENSORS ---")
        print(f"Found Overlapping Pair: {i} and {j} with Box IoU = {best_iou:.4f}")
        
        sim = np.dot(coeffs[i], coeffs[j]) / (np.linalg.norm(coeffs[i]) * np.linalg.norm(coeffs[j]))
        print(f"NEW Coefficient Cosine Similarity: {sim:.4f}")
        
        rm1 = (raw_masks[i] > 0.5).astype(np.float32)
        rm2 = (raw_masks[j] > 0.5).astype(np.float32)
        
        inter = np.logical_and(rm1, rm2).sum()
        union = np.logical_or(rm1, rm2).sum()
        mask_iou = inter / union if union > 0 else 0
        
        print(f"NEW RAW UNCROPPED Mask IoU: {mask_iou:.4f}")

if __name__ == '__main__':
    main()
