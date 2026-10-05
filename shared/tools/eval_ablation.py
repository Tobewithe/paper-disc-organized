import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
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

def extract_coeff_sim(model_path, img_path, gt1, gt2):
    model = load_model(Path(model_path), 'cuda')
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
    
    best_idx1, best_iou1 = -1, 0
    best_idx2, best_iou2 = -1, 0
    for i, b in enumerate(bboxes):
        iou1 = box_iou(b, gt1)
        if iou1 > best_iou1: best_idx1, best_iou1 = i, iou1
        iou2 = box_iou(b, gt2)
        if iou2 > best_iou2: best_idx2, best_iou2 = i, iou2
        
    if best_idx1 != -1 and best_idx2 != -1:
        c1, c2 = coeffs[best_idx1], coeffs[best_idx2]
        sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
        return sim
    return None

def main():
    weight_path = Path('runs/segment/experiments/ablation_study/ccl_w0.1_m0.1-2/weights/best.pt')
    if not weight_path.exists():
        print(f"Weight not found: {weight_path}")
        return
        
    print(f"Evaluating Model: {weight_path.parent.parent.name}")
    
    # 1. Evaluate mAP
    test_yaml = 'experiments/ccl_true_finetune/faro_test.yaml'
    model = load_model(weight_path, 'cuda')
    metrics = model.val(data=test_yaml, split='val', imgsz=640, device='cuda', verbose=False)
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    
    # 2. Evaluate Coeff Sim
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    img_info = next(img for img in manifest['images'] if img['id'] == 144)
    img_path = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images') / Path(img_info['file_name']).name
    gts = [a for a in manifest['annotations'] if a['image_id'] == 144]
    
    best_pair = None
    best_iou = 0
    for i in range(len(gts)):
        for j in range(i+1, len(gts)):
            g1 = gts[i]['bbox']
            b1 = [g1[0], g1[1], g1[0]+g1[2], g1[1]+g1[3]]
            g2 = gts[j]['bbox']
            b2 = [g2[0], g2[1], g2[0]+g2[2], g2[1]+g2[3]]
            iou = box_iou(b1, b2)
            if iou > best_iou and iou < 0.9:
                best_iou = iou
                best_pair = (b1, b2)
                
    gt1, gt2 = best_pair
    sim = extract_coeff_sim(weight_path, img_path, gt1, gt2)
    print(f"Coeff Cosine Sim: {sim:.4f}")

if __name__ == '__main__':
    main()

