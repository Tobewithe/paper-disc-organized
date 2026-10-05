import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys
import urllib.request
import json

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from ultralytics import YOLO
from ultralytics.utils.ops import process_mask_native

def box_intersection(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2
    
    inter_x1 = max(x1_1, x2_1)
    inter_y1 = max(y1_1, y2_1)
    inter_x2 = min(x1_2, x2_2)
    inter_y2 = min(y1_2, y2_2)
    
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

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

def analyze_coco_leakage(img_url, img_path, gt1, gt2):
    if not os.path.exists(img_path):
        urllib.request.urlretrieve(img_url, img_path)
        
    model = YOLO('yolov8m-seg.pt')
    
    stash = {}
    orig_process = process_mask_native
    def patched_process(protos, masks_in, bboxes, shape):
        c, mh, mw = protos.shape
        raw_masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        raw_masks = torch.sigmoid(raw_masks).detach().cpu().numpy()
        stash['raw_masks'] = raw_masks
        stash['bboxes'] = bboxes.detach().cpu().numpy()
        stash['coeffs'] = masks_in.detach().cpu().numpy()
        return orig_process(protos, masks_in, bboxes, shape)
        
    with patch('ultralytics.utils.ops.process_mask_native', patched_process):
        model.predict(source=str(img_path), conf=0.1, retina_masks=True, verbose=False, save=False)
        
    bboxes = stash['bboxes']
    coeffs = stash['coeffs']
    
    # YOLO boxes are [x1, y1, x2, y2]
    # gt boxes are [x1, y1, w, h] -> convert to [x1, y1, x2, y2]
    gt1_b = [gt1[0], gt1[1], gt1[0]+gt1[2], gt1[1]+gt1[3]]
    gt2_b = [gt2[0], gt2[1], gt2[0]+gt2[2], gt2[1]+gt2[3]]
    
    best_idx1, best_iou1 = -1, 0
    best_idx2, best_iou2 = -1, 0
    for i, b in enumerate(bboxes):
        iou1 = box_iou(b, gt1_b)
        if iou1 > best_iou1: best_idx1, best_iou1 = i, iou1
        iou2 = box_iou(b, gt2_b)
        if iou2 > best_iou2: best_idx2, best_iou2 = i, iou2
        
    print(f"Matched GT1 to Box {best_idx1} (IoU={best_iou1:.2f})")
    print(f"Matched GT2 to Box {best_idx2} (IoU={best_iou2:.2f})")
    
    if best_idx1 != -1 and best_idx2 != -1 and best_idx1 != best_idx2:
        c1, c2 = coeffs[best_idx1], coeffs[best_idx2]
        sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
        return sim
    return None

def main():
    anno_file = 'datasets/coco/annotations/instances_val2017.json'
    with open(anno_file, 'r') as f:
        coco = json.load(f)
        
    # Find the worst pair in image 492077
    img_id = 492077
    anns = [a for a in coco['annotations'] if a['image_id'] == img_id and a.get('iscrowd', 0) == 0 and a['category_id'] == 1]
    
    best_pair = None
    max_ici = 0
    
    for i in range(len(anns)):
        a1 = anns[i]
        area1 = a1['bbox'][2] * a1['bbox'][3]
        if area1 < 100: continue
        
        sum_inter = 0
        for j in range(len(anns)):
            if i == j: continue
            a2 = anns[j]
            inter = box_intersection(a1['bbox'], a2['bbox'])
            if inter > 0:
                sum_inter += inter
                if inter / area1 > max_ici:
                    max_ici = inter / area1
                    best_pair = (a1['bbox'], a2['bbox'])
                    
    print(f"Image 492077 Max ICI: {max_ici:.2f}")
    
    img_url = 'http://images.cocodataset.org/val2017/000000492077.jpg'
    img_path = 'datasets/coco/000000492077.jpg'
    
    sim = analyze_coco_leakage(img_url, img_path, best_pair[0], best_pair[1])
    print(f"COCO YOLOv8m-seg Baseline Coeff Cosine Sim: {sim:.4f}")

if __name__ == '__main__':
    main()
