import json
import csv
import cv2
import numpy as np
from pathlib import Path
import pycocotools.mask as mask_util
import random

def main():
    # 1. Load the 270 failed GTs
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    failed_gts = []
    with csv_path.open('r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['stage'] == 'raw' and r['raw_strict_mask_absent'] == '1' and r['support_available'] == '1':
                failed_gts.append({'img_id': int(r['image_id']), 'ann_id': int(r['annotation_id'])})
                
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    anns_by_id = {ann['id']: ann for ann in manifest['annotations']}
    
    # Group all annotations by image
    anns_by_img = {}
    for ann in manifest['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    out_dir = Path('C:/Users/Administrator/.gemini/antigravity/brain/08659641-44b7-4544-8b51-2274a96336ad/images/failed_crowds')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomly select 6 failed cases
    random.seed(42)
    sample_gts = random.sample(failed_gts, 6)
    
    output_files = []
    for idx, f_gt in enumerate(sample_gts):
        img_id = f_gt['img_id']
        ann_id = f_gt['ann_id']
        
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        target_ann = anns_by_id[ann_id]
        tx, ty, tw, th = [int(v) for v in target_ann['bbox']]
        
        # Define crop region (expand by 150 pixels)
        pad = 150
        cx1, cy1 = max(0, tx - pad), max(0, ty - pad)
        cx2, cy2 = min(w, tx + tw + pad), min(h, ty + th + pad)
        
        crop_img = img[cy1:cy2, cx1:cx2].copy()
        crop_h, crop_w = crop_img.shape[:2]
        
        overlay = crop_img.copy()
        
        # Draw all neighboring GTs
        for ann in anns_by_img[img_id]:
            ax, ay, aw, ah = [int(v) for v in ann['bbox']]
            # Check intersection with crop
            if ax < cx2 and ax+aw > cx1 and ay < cy2 and ay+ah > cy1:
                rle = mask_util.frPyObjects(ann['segmentation'], h, w)
                m = mask_util.decode(rle)
                if len(m.shape) == 3: m = np.any(m, axis=2)
                
                m_crop = m[cy1:cy2, cx1:cx2]
                
                if ann['id'] == ann_id:
                    # Target failed GT in RED
                    color = (0, 0, 255) 
                    alpha = 0.6
                else:
                    # Neighbors in BLUE/GREEN
                    color = (255, 100, 0)
                    alpha = 0.3
                    
                overlay[m_crop > 0] = overlay[m_crop > 0] * (1 - alpha) + np.array(color) * alpha
                
                # Draw bounding box for target
                if ann['id'] == ann_id:
                    bx1, by1 = max(0, ax - cx1), max(0, ay - cy1)
                    bx2, by2 = min(crop_w, ax + aw - cx1), min(crop_h, ay + ah - cy1)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    
        out_path = out_dir / f"crowd_{idx}_img{img_id}_ann{ann_id}.jpg"
        cv2.imwrite(str(out_path), overlay)
        output_files.append(str(out_path))
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
