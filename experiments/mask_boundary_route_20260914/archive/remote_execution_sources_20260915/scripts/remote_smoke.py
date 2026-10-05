import os
import sys

sys.path.insert(0, "D:/coco_wire/py")

import torch
import ultralytics

print("ultralytics", ultralytics.__version__, ultralytics.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("images", len(os.listdir("D:/coco_wire/data/images/val2017")))
print("annotation_bytes", os.path.getsize("D:/coco_wire/data/annotations/instances_val2017.json"))

from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops
import inspect

print("construct_result_signature", inspect.signature(SegmentationPredictor.construct_result))
print("construct_result_source")
print(inspect.getsource(SegmentationPredictor.construct_result))
print("process_mask_signature", inspect.signature(ops.process_mask))
print("process_mask_source")
print(inspect.getsource(ops.process_mask))
print("scale_image_source")
print(inspect.getsource(ops.scale_image))

model = YOLO("D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt")
print("head", type(model.model.model[-1]).__name__)
print("end2end", getattr(model.model.model[-1], "end2end", None))

result = model.predict(
    source="D:/coco_wire/data/images/val2017/000000000139.jpg",
    imgsz=640,
    device=0,
    conf=0.001,
    max_det=300,
    verbose=False,
    save=False,
)[0]
print("detections", len(result.boxes), "mask_shape", tuple(result.masks.data.shape) if result.masks is not None else None)
print("orig_shape", result.orig_img.shape[:2], "boxes_shape", tuple(result.boxes.xyxy.shape))

try:
    import pycocotools
    print("pycocotools", pycocotools.__file__)
except Exception as exc:
    print("pycocotools_error", repr(exc))
