import json
import csv
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def main():
    # Target crops from before
    targets = [
        {'img_id': 33, 'ann_id': 313},
        {'img_id': 7, 'ann_id': 74},
        {'img_id': 92, 'ann_id': 757},
        {'img_id': 86, 'ann_id': 705},
        {'img_id': 79, 'ann_id': 614},
        {'img_id': 49, 'ann_id': 409}
    ]
    
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    anns_by_id = {ann['id']: ann for ann in manifest['annotations']}
    
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    weights = Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt')
    model = load_model(weights, 'cuda')
    
    out_dir = Path('C:/Users/Administrator/.gemini/antigravity/brain/08659641-44b7-4544-8b51-2274a96336ad/images/finetuned_e2e_crops')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t in targets:
        img_id = t['img_id']
        ann_id = t['ann_id']
        
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        
        # Get target crop coordinates
        target_ann = anns_by_id[ann_id]
        tx, ty, tw, th = [int(v) for v in target_ann['bbox']]
        pad = 150
        h, w = img_info['height'], img_info['width']
        cx1, cy1 = max(0, tx - pad), max(0, ty - pad)
        cx2, cy2 = min(w, tx + tw + pad), min(h, ty + th + pad)
        
        # Run prediction
        results = model.predict(source=str(img_path), conf=0.25, retina_masks=True, save=False, verbose=False)
        res = results[0]
        
        # Plotting
        plotted = res.plot()
        
        # Crop the plotted image
        crop_plotted = plotted[cy1:cy2, cx1:cx2]
        
        out_path = out_dir / f"e2e_crop_img{img_id}_ann{ann_id}.jpg"
        cv2.imwrite(str(out_path), crop_plotted)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
