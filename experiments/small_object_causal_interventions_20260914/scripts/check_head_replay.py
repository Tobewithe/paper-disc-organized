import contextlib,io,sys
from pathlib import Path
import cv2,torch
from ultralytics import YOLO

root=Path(__file__).resolve().parents[3];shared=root/'shared/coco_clean_20260911';sys.path.insert(0,str(shared))
from structure_candidate_trace import TraceCapture
model=YOLO(str(root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'))
image=cv2.imread(str(shared/'local_readout_runtime_20260912/data/images/val2017/000000000139.jpg'))
with torch.inference_mode():model.predict(image,predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,device=0,verbose=False,end2end=False)
head=model.predictor.model.model.model[-1];captured={}
h=head.register_forward_pre_hook(lambda m,a:captured.update(x=tuple(v.detach().clone() for v in a[0])))
with torch.inference_mode():model.predict(image,predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,device=0,verbose=False,end2end=False)
h.remove();raw=model.predictor.dense
with torch.inference_mode():manual=head._inference(head.forward_head(list(captured['x']),**head.one2many))
print(type(head),raw.shape,manual.shape,float((raw-manual).abs().max()))
