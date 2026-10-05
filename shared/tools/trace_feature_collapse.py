import json
import csv
import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model
from ultralytics.utils.ops import process_mask_native

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
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
    out_dir = Path('experiments/feature_trace_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    dense_similarities = []
    distant_similarities = []
    
    for img_id in target_image_ids[:50]:
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        
        stash = []
        orig_process = process_mask_native
        def patched_process(protos, masks_in, bboxes, shape):
            stash.append({
                'protos': protos.detach().cpu(),
                'masks_in': masks_in.detach().cpu(),
                'bboxes': bboxes.detach().cpu(),
                'shape': shape
            })
            return orig_process(protos, masks_in, bboxes, shape)
            
        with patch('ultralytics.utils.ops.process_mask_native', patched_process):
            model.predict(source=str(img_path), conf=0.05, retina_masks=True, verbose=False, save=False)
            
        if not stash: continue
        data = stash[0]
        bboxes = data['bboxes'].numpy()
        masks_in = data['masks_in'].numpy()
        protos = data['protos'].numpy()
        
        gts = gts_by_image[img_id]
        matched_preds = {}
        for gt in gts:
            best_iou = 0
            best_idx = -1
            for i, pbox in enumerate(bboxes):
                iou = box_iou(gt['box'], pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou > 0.5:
                matched_preds[gt['id']] = best_idx
                
        gt_ids = list(matched_preds.keys())
        for i in range(len(gt_ids)):
            for j in range(i+1, len(gt_ids)):
                gt1 = next(g for g in gts if g['id'] == gt_ids[i])
                gt2 = next(g for g in gts if g['id'] == gt_ids[j])
                
                inter_iou = box_iou(gt1['box'], gt2['box'])
                
                idx1 = matched_preds[gt_ids[i]]
                idx2 = matched_preds[gt_ids[j]]
                c1 = masks_in[idx1]
                c2 = masks_in[idx2]
                
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                
                if inter_iou > 0.05:
                    dense_similarities.append(float(sim))
                elif inter_iou == 0:
                    distant_similarities.append(float(sim))
                    
        if len(list(out_dir.glob("pca_protos_*.png"))) < 3:
            c, mh, mw = protos.shape
            flat_protos = protos.reshape(c, -1).T
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(flat_protos)
            
            pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
            pca_img = (pca_features * 255).astype(np.uint8).reshape(mh, mw, 3)
            
            h, w = data['shape']
            pca_img = cv2.resize(pca_img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            orig_img = cv2.imread(str(img_path))
            combined = np.hstack((orig_img, pca_img))
            cv2.imwrite(str(out_dir / f"pca_protos_img{img_id}.png"), combined)

    print("Experiment 1: Coefficient Similarity Analysis")
    print(f"Dense/Touching Pairs (N={len(dense_similarities)}): Mean Cosine Similarity = {np.mean(dense_similarities):.4f}")
    if distant_similarities:
        print(f"Distant Pairs (N={len(distant_similarities)}): Mean Cosine Similarity = {np.mean(distant_similarities):.4f}")
    else:
        print("Distant Pairs: N/A")

if __name__ == '__main__':
    main()
