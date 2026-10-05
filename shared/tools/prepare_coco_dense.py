import json
import os
from pathlib import Path
from collections import defaultdict

def box_intersection(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2
    return max(0, min(x1_2, x2_2) - max(x1_1, x2_1)) * max(0, min(y1_2, y2_2) - max(y1_1, y2_1))

def filter_dense_images(anno_file, threshold=0.5):
    print(f"Loading {anno_file}...")
    with open(anno_file, 'r') as f:
        coco = json.load(f)
        
    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 0:
            img_to_anns[ann['image_id']].append(ann)
            
    dense_img_ids = set()
    total_imgs = len(img_to_anns)
    
    print("Computing ICI to filter dense images...")
    for idx, (img_id, anns) in enumerate(img_to_anns.items()):
        if idx % 10000 == 0:
            print(f"Processed {idx}/{total_imgs} images...")
            
        if len(anns) < 2: continue
        cat_to_anns = defaultdict(list)
        for a in anns:
            cat_to_anns[a['category_id']].append(a)
            
        img_max_ici = 0
        for cat_id, cat_anns in cat_to_anns.items():
            for i in range(len(cat_anns)):
                a1 = cat_anns[i]
                area1 = a1['bbox'][2] * a1['bbox'][3]
                if area1 < 100: continue
                sum_inter = 0
                for j in range(len(cat_anns)):
                    if i == j: continue
                    inter = box_intersection(a1['bbox'], cat_anns[j]['bbox'])
                    sum_inter += inter
                if sum_inter / area1 > img_max_ici:
                    img_max_ici = sum_inter / area1
                    
        if img_max_ici > threshold:
            dense_img_ids.add(img_id)
            
    # Generate relative paths for YOLO
    lines = []
    for img in coco['images']:
        if img['id'] in dense_img_ids:
            # YOLO expects paths relative to dataset root
            lines.append(f"./images/{Path(anno_file).stem.replace('instances_', '')}/{img['file_name']}\n")
            
    return lines

def main():
    # Make sure to run this where datasets/coco exists
    coco_root = Path("datasets/coco")
    
    train_lines = filter_dense_images(coco_root / "annotations/instances_train2017.json", 0.5)
    val_lines = filter_dense_images(coco_root / "annotations/instances_val2017.json", 0.5)
    
    with open(coco_root / "train_dense.txt", "w") as f:
        f.writelines(train_lines)
    with open(coco_root / "val_dense.txt", "w") as f:
        f.writelines(val_lines)
        
    print(f"Selected {len(train_lines)} Train images and {len(val_lines)} Val images.")
    
    # Create the custom YAML
    yaml_content = f"""
path: ./datasets/coco
train: train_dense.txt
val: val_dense.txt

# Classes
names:
  0: person
  1: bicycle
  2: car
  # ... (add standard 80 COCO classes here, or just copy from standard coco.yaml)
"""
    with open("coco_dense.yaml", "w") as f:
        f.write(yaml_content)
    print("Generated coco_dense.yaml!")

if __name__ == '__main__':
    main()
