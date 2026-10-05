import json, os
for p in [r"D:\coco_wire\data\annotations\instances_train2017.json", r"D:\coco_wire\data\annotations\instances_val2017.json"]:
    d=json.load(open(p, encoding="utf-8"))
    print(os.path.basename(p), "images", len(d.get("images",[])), "annotations", len(d.get("annotations",[])), "categories", len(d.get("categories",[])))

