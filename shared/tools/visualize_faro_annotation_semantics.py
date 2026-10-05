import csv
import json
import random
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    out_dir = Path('research-wiki/graph/faro_annotation_semantics')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json')
    if not manifest_path.exists():
        print(f'Manifest not found: {manifest_path}')
        return
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    
    images_by_id = {img['id']: img for img in manifest['images']}
    anns_by_id = {(ann['image_id'], ann['id']): ann for ann in manifest['annotations']}
    
    # Load rows
    csv_path = Path('experiments/faro_box_support_20260907_v1/per_gt_stage.csv')
    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows_270 = []
        rows_37 = []
        for r in reader:
            if r['stage'] != 'raw': continue
            if r['raw_strict_mask_absent'] == '1':
                if r['support_available'] == '1':
                    rows_270.append(r)
                else:
                    rows_37.append(r)
                    
    random.seed(42)
    sample_270 = random.sample(rows_270, min(10, len(rows_270)))
    sample_37 = random.sample(rows_37, min(10, len(rows_37)))
    
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    def process_samples(samples, prefix):
        for i, r in enumerate(samples):
            img_id = int(r['image_id'])
            ann_id = int(r['annotation_id'])
            img_info = images_by_id[img_id]
            ann_info = anns_by_id[(img_id, ann_id)]
            
            # Load image
            img_path = img_root / Path(img_info['file_name']).name
            if not img_path.exists():
                print(f'Image not found: {img_path}')
                continue
            with Image.open(img_path).convert('RGBA') as img:
                draw = ImageDraw.Draw(img, 'RGBA')
                
                # Draw all other GTs in dim red
                for other_ann in manifest['annotations']:
                    if other_ann['image_id'] == img_id and other_ann['id'] != ann_id:
                        for poly in other_ann['segmentation']:
                            draw.polygon(poly, outline=(255, 0, 0, 100), fill=(255, 0, 0, 30))
                
                # Draw target GT in bright green
                for poly in ann_info['segmentation']:
                    draw.polygon(poly, outline=(0, 255, 0, 255), fill=(0, 255, 0, 100))
                    
                # Crop around the target GT to see clearly
                bbox = ann_info['bbox'] # [x, y, w, h]
                cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                size = max(bbox[2], bbox[3]) * 2.5
                crop_box = (int(cx - size/2), int(cy - size/2), int(cx + size/2), int(cy + size/2))
                cropped = img.crop(crop_box)
                
                out_path = out_dir / f'{prefix}_{i:02d}_img{img_id}_ann{ann_id}.png'
                cropped.convert('RGB').save(out_path)
                print(f'Saved {out_path}')

    process_samples(sample_270, 'feasible270')
    process_samples(sample_37, 'infeasible37')

if __name__ == '__main__':
    main()
