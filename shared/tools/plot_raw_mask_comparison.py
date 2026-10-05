import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
import sys
import matplotlib.pyplot as plt

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

def extract_raw_masks(model_path, img_path, gt1, gt2):
    model = load_model(Path(model_path), 'cuda')
    stash = {}
    orig_process = process_mask_native
    def patched_process(protos, masks_in, bboxes, shape):
        c, mh, mw = protos.shape
        raw_masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        raw_masks = torch.sigmoid(raw_masks).detach().cpu().numpy()
        stash['raw_masks'] = raw_masks
        stash['bboxes'] = bboxes.detach().cpu().numpy()
        return orig_process(protos, masks_in, bboxes, shape)
        
    with patch('ultralytics.utils.ops.process_mask_native', patched_process):
        model.predict(source=str(img_path), conf=0.1, retina_masks=True, verbose=False, save=False)
        
    bboxes = stash['bboxes']
    raw_masks = stash['raw_masks']
    
    best_idx1, best_iou1 = -1, 0
    best_idx2, best_iou2 = -1, 0
    for i, b in enumerate(bboxes):
        if box_iou(b, gt1) > best_iou1: best_idx1, best_iou1 = i, box_iou(b, gt1)
        if box_iou(b, gt2) > best_iou2: best_idx2, best_iou2 = i, box_iou(b, gt2)
        
    return raw_masks[best_idx1], raw_masks[best_idx2]

def main():
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
    
    b_rm1, b_rm2 = extract_raw_masks('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt', img_path, gt1, gt2)
    c_rm1, c_rm2 = extract_raw_masks('runs/segment/experiments/ccl_true_finetune/run_ccl_true_15e/weights/best.pt', img_path, gt1, gt2)
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    b_rm1 = cv2.resize(b_rm1, (w, h))
    b_rm2 = cv2.resize(b_rm2, (w, h))
    c_rm1 = cv2.resize(c_rm1, (w, h))
    c_rm2 = cv2.resize(c_rm2, (w, h))
    
    tx, ty = min(gt1[0], gt2[0]), min(gt1[1], gt2[1])
    bx, by = max(gt1[2], gt2[2]), max(gt1[3], gt2[3])
    pad = 50
    cx1, cy1 = max(0, int(tx)-pad), max(0, int(ty)-pad)
    cx2, cy2 = min(w, int(bx)+pad), min(h, int(by)+pad)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Raw Uncropped Mask Activation (Before Bounding Box Crop)', fontsize=16)
    
    def plot_mask(ax, m, title, color_idx):
        ax.imshow(img[cy1:cy2, cx1:cx2])
        m_color = np.zeros((*m.shape, 4))
        m_color[..., color_idx] = 1 
        m_color[..., 3] = m * 0.7
        ax.imshow(m_color[cy1:cy2, cx1:cx2])
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        
    plot_mask(axes[0,0], b_rm1, "Baseline: Pig A (Red)", 0)
    plot_mask(axes[0,1], b_rm2, "Baseline: Pig B (Blue)", 2)
    plot_mask(axes[1,0], c_rm1, "CCL Model: Pig A (Red)", 0)
    plot_mask(axes[1,1], c_rm2, "CCL Model: Pig B (Blue)", 2)
    
    out_dir = Path('C:/Users/Administrator/.gemini/antigravity/brain/08659641-44b7-4544-8b51-2274a96336ad/images')
    out_path = out_dir / 'raw_mask_comparison.png'
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
