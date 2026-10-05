from ultralytics.utils import YAML
p=r"D:\coco_wire\research\mask_quality_boundary_branch_20261004\data\coco_val5k.yaml"
d=YAML.load(p)
print(d["path"])
print(d["val"])

