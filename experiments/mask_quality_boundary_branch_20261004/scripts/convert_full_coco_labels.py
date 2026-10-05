import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ultralytics.data.converter import convert_coco
convert_coco(
    r"D:\coco_wire\research\mask_quality_boundary_branch_20261004\data\coco_annotations_full",
    r"D:\coco_wire\research\mask_quality_boundary_branch_20261004\data\coco_full_standard_20261005_v2",
    use_segments=True,
    cls91to80=True,
)
print("COCO_FULL_LABEL_CONVERSION_COMPLETE", flush=True)

