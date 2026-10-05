"""Prepare 1,000 top crowded images from local COCO train2017 and run official segment conversion."""
import json
import os
import shutil
from pathlib import Path
import pandas as pd
from ultralytics.data.converter import convert_coco

def main():
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    census_csv = root / "census/train2017_images.csv"
    train_img_dir = Path(r"C:\Dpan\document\model_datasets\datasets\coco\downloads\train2017\train2017")
    train_ann_json = Path(r"C:\Dpan\document\model_datasets\datasets\coco\annotations\instances_train2017.json")
    
    out_dir = root / "data/pilot_1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading census from {census_csv}...")
    df = pd.read_csv(census_csv)
    
    # Filter for valid crowding:
    # Sort by same_class_pairs_box_iou_gt05 descending, then max_ici_same descending
    df_sorted = df.sort_values(
        by=["same_class_pairs_box_iou_gt05", "max_ici_same", "n_instances"],
        ascending=[False, False, False]
    )
    
    selected_1000 = df_sorted.head(1000)
    selected_img_ids = set(selected_1000["image_id"].tolist())
    print(f"Selected 1000 images. Min pairs: {selected_1000['same_class_pairs_box_iou_gt05'].min()}, Max pairs: {selected_1000['same_class_pairs_box_iou_gt05'].max()}")
    
    # Save selection metadata
    selected_1000.to_csv(out_dir / "pilot_1000_images.csv", index=False)
    
    # Read full annotations JSON and filter
    print(f"Filtering {train_ann_json} for 1000 selected images...")
    with open(train_ann_json, "r") as f:
        full_coco = json.load(f)
        
    filtered_images = [im for im in full_coco["images"] if im["id"] in selected_img_ids]
    filtered_annotations = [ann for ann in full_coco["annotations"] if ann["image_id"] in selected_img_ids]
    
    print(f"Filtered images: {len(filtered_images)}, annotations: {len(filtered_annotations)}")
    assert len(filtered_images) == 1000
    
    filtered_coco = {
        "info": full_coco.get("info", {}),
        "licenses": full_coco.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": full_coco["categories"]
    }
    
    # Write conversion input
    conv_input_dir = out_dir / "conversion_input"
    conv_input_dir.mkdir(parents=True, exist_ok=True)
    target_json = conv_input_dir / "instances_train2017.json"
    with open(target_json, "w") as f:
        json.dump(filtered_coco, f)
    print(f"Wrote conversion input JSON to {target_json}")
    
    # Run official converter
    conv_output_dir = out_dir / "conversion_output"
    if conv_output_dir.exists():
        shutil.rmtree(conv_output_dir)
        
    print(f"Running official convert_coco(use_segments=True)...")
    convert_coco(labels_dir=str(conv_input_dir), save_dir=str(conv_output_dir), use_segments=True)
    
    # Link or copy images into dataset structure
    images_dir = out_dir / "images/train2017"
    labels_dir = out_dir / "labels/train2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    converted_labels_src = conv_output_dir / "labels/train2017"
    label_files = list(converted_labels_src.glob("*.txt"))
    print(f"Converted label files: {len(label_files)}")
    
    # Copy label files
    for lf in label_files:
        shutil.copy2(lf, labels_dir / lf.name)
        
    # Symlink or copy images
    print("Linking/copying 1000 image files...")
    train_txt_lines = []
    for im in filtered_images:
        fn = im["file_name"]
        src_img = train_img_dir / fn
        dst_img = images_dir / fn
        if not dst_img.exists():
            # Try symlink or hardlink
            try:
                os.link(src_img, dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)
        train_txt_lines.append(str(dst_img.resolve()) + "\n")
        
    train_txt_path = out_dir / "train2017.txt"
    train_txt_path.write_text("".join(train_txt_lines))
    print(f"Wrote train image paths to {train_txt_path}")
    
    # Create dataset YAML
    cat_names = {i: c["name"] for i, c in enumerate(sorted(full_coco["categories"], key=lambda x: x["id"]))}
    val_txt = root / "val_dense_1576.txt"
    
    # If val_dense_1576.txt doesn't exist, create it
    val_img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    val_imgs = sorted(list(val_img_dir.glob("*.jpg")))
    val_txt.write_text("".join(str(p.resolve()) + "\n" for p in val_imgs))
    
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": str(train_txt_path.resolve()),
        "val": str(val_txt.resolve()),
        "names": cat_names
    }
    
    import yaml
    yaml_path = out_dir / "coco_pilot_1000.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)
    print(f"\nSUCCESS! Pilot dataset ready at {yaml_path}")

if __name__ == "__main__":
    main()
