import json
import csv
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
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
    # 1. Load failed GTs
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
    
    model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cuda')
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    # Hooks
    features = {}
    def hook_b3(m, i, o):
        features['b3'] = o.detach().cpu()
    def hook_p3(m, i, o):
        features['p3'] = o.detach().cpu()
        
    h1 = model.model.model[4].register_forward_hook(hook_b3)
    h2 = model.model.model[16].register_forward_hook(hook_p3)
    
    dense_b3_sims, dense_p3_sims = [], []
    distant_b3_sims, distant_p3_sims = [], []
    
    for img_id in target_image_ids[:100]:
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        
        # We need to know original image dimensions to scale coordinates properly
        # model.predict resizes the image. We can get the tensor shape from the feature map.
        features.clear()
        results = model.predict(source=str(img_path), conf=0.05, retina_masks=True, verbose=False, save=False)
        if 'b3' not in features: continue
        
        # Coordinate mapping: find the original image shape
        orig_h, orig_w = img_info['height'], img_info['width']
        # The inference image is letterboxed to 640x640 typically. 
        # Actually, Ultralytics Results object has speed and orig_img, let's use the feature map size directly 
        # assuming letterboxing scales the longest edge to 640.
        scale = 640.0 / max(orig_h, orig_w)
        pad_w = (640 - orig_w * scale) / 2
        pad_h = (640 - orig_h * scale) / 2
        
        def get_feature_vector(feature_map, box):
            # box is [x1, y1, x2, y2] in orig coords
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            
            # Map to 640x640
            cx_scaled = cx * scale + pad_w
            cy_scaled = cy * scale + pad_h
            
            # Map to feature map (stride 8)
            fm_x = int(cx_scaled / 8.0)
            fm_y = int(cy_scaled / 8.0)
            
            # Clamp
            _, C, H, W = feature_map.shape
            fm_x = max(0, min(fm_x, W-1))
            fm_y = max(0, min(fm_y, H-1))
            
            return feature_map[0, :, fm_y, fm_x].numpy()

        gts = gts_by_image[img_id]
        
        for i in range(len(gts)):
            for j in range(i+1, len(gts)):
                gt1 = gts[i]
                gt2 = gts[j]
                
                inter_iou = box_iou(gt1['box'], gt2['box'])
                
                v1_b3 = get_feature_vector(features['b3'], gt1['box'])
                v2_b3 = get_feature_vector(features['b3'], gt2['box'])
                sim_b3 = np.dot(v1_b3, v2_b3) / (np.linalg.norm(v1_b3) * np.linalg.norm(v2_b3) + 1e-8)
                
                v1_p3 = get_feature_vector(features['p3'], gt1['box'])
                v2_p3 = get_feature_vector(features['p3'], gt2['box'])
                sim_p3 = np.dot(v1_p3, v2_p3) / (np.linalg.norm(v1_p3) * np.linalg.norm(v2_p3) + 1e-8)
                
                if inter_iou > 0.05:
                    dense_b3_sims.append(float(sim_b3))
                    dense_p3_sims.append(float(sim_p3))
                elif inter_iou == 0:
                    distant_b3_sims.append(float(sim_b3))
                    distant_p3_sims.append(float(sim_p3))
                    
    h1.remove()
    h2.remove()
    
    print("Experiment 3: Backbone vs Neck Feature Spillover")
    print(f"Dense/Touching Pairs (N={len(dense_b3_sims)}):")
    print(f"  - Backbone (Layer 4)  Mean Similarity: {np.mean(dense_b3_sims):.4f}")
    print(f"  - Neck FPN (Layer 16) Mean Similarity: {np.mean(dense_p3_sims):.4f}")
    print(f"  - Pollution Delta: {np.mean(dense_p3_sims) - np.mean(dense_b3_sims):.4f}")
    print()
    print(f"Distant Pairs (N={len(distant_b3_sims)}):")
    print(f"  - Backbone (Layer 4)  Mean Similarity: {np.mean(distant_b3_sims):.4f}")
    print(f"  - Neck FPN (Layer 16) Mean Similarity: {np.mean(distant_p3_sims):.4f}")

if __name__ == '__main__':
    main()
