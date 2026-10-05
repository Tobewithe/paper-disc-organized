import argparse
import csv
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as mask_utils

import sys
sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model
from ultralytics.utils.ops import process_mask_native, scale_masks, crop_mask
from unittest.mock import patch

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1. Load 270 GTs
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    failed_gts = []
    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['stage'] == 'raw' and r['raw_strict_mask_absent'] == '1' and r['support_available'] == '1':
                failed_gts.append({'image_id': int(r['image_id']), 'annotation_id': int(r['annotation_id'])})
    
    print(f'Loaded {len(failed_gts)} target GTs.')
    failed_gts_by_image = defaultdict(list)
    for g in failed_gts:
        failed_gts_by_image[g['image_id']].append(g['annotation_id'])
        
    # 2. Load manifest
    manifest_path = Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json')
    manifest = read_json(manifest_path)
    images = {img['id']: img for img in manifest['images']}
    anns = {(ann['image_id'], ann['id']): ann for ann in manifest['annotations']}
    
    # 3. Load model
    model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cuda')
    
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    results_out = []
    
    # 4. Process each image
    for image_id, target_ann_ids in failed_gts_by_image.items():
        img_info = images[image_id]
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
            model.predict(source=str(img_path), conf=0.01, retina_masks=True, verbose=False, save=False)
            
        if not stash:
            continue
            
        data = stash[0]
        protos = data['protos'].cuda()
        masks_in = data['masks_in'].cuda()
        orig_shape = data['shape']
        
        c, mh, mw = protos.shape
        h, w = orig_shape
        if masks_in.shape[0] == 0:
            continue
            
        coeffs = masks_in @ protos.float().view(c, -1)
        step = max(1, 32_000_000 // (h * w))
        raw_masks = []
        for i in range(0, coeffs.shape[0], step):
            m = scale_masks(coeffs[i : i + step].view(-1, mh, mw)[None], orig_shape)[0].gt_(0.0).byte()
            raw_masks.append(m)
        raw_masks = torch.cat(raw_masks).cpu() # [N, H, W]
        
        for ann_id in target_ann_ids:
            ann = anns[(image_id, ann_id)]
            gt_bbox = ann['bbox'] # x, y, w, h
            gt_box_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]]
            
            rle = mask_utils.merge(mask_utils.frPyObjects(ann['segmentation'], h, w))
            gt_mask = mask_utils.decode(rle).astype(bool)
            gt_area = gt_mask.sum()
            
            gt_boxes_tensor = torch.tensor([gt_box_xyxy] * raw_masks.shape[0])
            cropped_masks = crop_mask(raw_masks, gt_boxes_tensor).numpy()
            
            best_cov = 0
            best_pur = 0
            best_idx = -1
            
            for i in range(cropped_masks.shape[0]):
                cand_mask = cropped_masks[i].astype(bool)
                intersection = (cand_mask & gt_mask).sum()
                pred_area = cand_mask.sum()
                
                cov = float(intersection / gt_area) if gt_area > 0 else 0
                pur = float(intersection / pred_area) if pred_area > 0 else 0
                
                if cov >= 0.75 and pur >= 0.75:
                    if cov > best_cov:
                        best_cov = cov
                        best_pur = pur
                        best_idx = i
                elif cov > best_cov and best_idx == -1:
                    best_cov = cov
                    best_pur = pur
                    
            recovered = (best_idx != -1)
            results_out.append({
                'image_id': image_id,
                'annotation_id': ann_id,
                'recovered': recovered,
                'best_cov': best_cov,
                'best_pur': best_pur
            })
            
    rec = sum(1 for r in results_out if r['recovered'])
    print(f'Total processed: {len(results_out)}')
    print(f'Recovered with Oracle GT Box: {rec}')
    print(f'Still failing (Feature/Prototype Collapse): {len(results_out) - rec}')
    
    Path('experiments/faro_oracle_crop_results.json').write_text(json.dumps(results_out, indent=2))

if __name__ == '__main__':
    main()
