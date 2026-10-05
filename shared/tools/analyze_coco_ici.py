import os
import urllib.request
import zipfile
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def box_intersection(box1, box2):
    # box format: [x, y, width, height]
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2
    
    inter_x1 = max(x1_1, x2_1)
    inter_y1 = max(y1_1, y2_1)
    inter_x2 = min(x1_2, x2_2)
    inter_y2 = min(y1_2, y2_2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    return inter_w * inter_h

def main():
    anno_dir = Path('datasets/coco/annotations')
    anno_file = anno_dir / 'instances_val2017.json'
    
    if not anno_file.exists():
        print("Downloading COCO annotations...")
        anno_dir.mkdir(parents=True, exist_ok=True)
        zip_path = anno_dir / 'annotations_trainval2017.zip'
        urllib.request.urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('datasets/coco/')
        os.remove(zip_path)
        
    print("Loading annotations...")
    with open(anno_file, 'r') as f:
        coco = json.load(f)
        
    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 0: # Only count explicit instance annotations
            img_to_anns[ann['image_id']].append(ann)
            
    print("Computing ICI...")
    ici_scores = []
    hard_images = []
    
    for img_id, anns in img_to_anns.items():
        if len(anns) < 2: continue
        
        # Group by category to only compute ICI among same-class objects
        cat_to_anns = defaultdict(list)
        for a in anns:
            cat_to_anns[a['category_id']].append(a)
            
        img_max_ici = 0
        best_pair = None
        best_cat = None
        
        for cat_id, cat_anns in cat_to_anns.items():
            if len(cat_anns) < 2: continue
            
            for i in range(len(cat_anns)):
                a1 = cat_anns[i]
                area1 = a1['bbox'][2] * a1['bbox'][3]
                if area1 < 100: continue # ignore tiny objects
                
                sum_inter = 0
                for j in range(len(cat_anns)):
                    if i == j: continue
                    a2 = cat_anns[j]
                    inter = box_intersection(a1['bbox'], a2['bbox'])
                    if inter > 0:
                        sum_inter += inter
                        # track pair if this is the dominant overlap
                        if inter / area1 > img_max_ici:
                            img_max_ici = inter / area1
                            best_pair = (a1['id'], a2['id'])
                            best_cat = cat_id
                            
                if area1 > 0:
                    ici = sum_inter / area1
                    ici_scores.append(ici)
                    
        if img_max_ici > 0.5:
            hard_images.append({
                'img_id': img_id,
                'max_ici': img_max_ici,
                'category_id': best_cat,
                'pair': best_pair
            })
            
    hard_images.sort(key=lambda x: x['max_ici'], reverse=True)
    
    ici_scores = np.array(ici_scores)
    print(f"Total valid instances evaluated: {len(ici_scores)}")
    print(f"Instances with ICI > 0.3: {np.sum(ici_scores > 0.3)}")
    print(f"Instances with ICI > 0.5: {np.sum(ici_scores > 0.5)}")
    print(f"Images with severe crowding (max_ICI > 0.5): {len(hard_images)}")
    
    print("\nTop 5 Hardest Images in COCO (Same-Class Overlap):")
    for i in range(min(5, len(hard_images))):
        img_meta = next(img for img in coco['images'] if img['id'] == hard_images[i]['img_id'])
        cat_meta = next(cat for cat in coco['categories'] if cat['id'] == hard_images[i]['category_id'])
        print(f"Image ID: {hard_images[i]['img_id']} | Category: {cat_meta['name']} | Max ICI: {hard_images[i]['max_ici']:.2f}")
        print(f"  URL: {img_meta['coco_url']}")

if __name__ == '__main__':
    main()
