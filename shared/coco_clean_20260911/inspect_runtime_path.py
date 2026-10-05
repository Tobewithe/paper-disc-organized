"""Read-only runtime branch witness for the exact diagnostic predictor."""
import inspect,json,contextlib,io
from pathlib import Path
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms
from frozen_mechanism_probe import ROOT,Capture,sha,write_json

def state(model):
    head=model.model[-1]
    return dict(head_type=type(head).__name__,head_end2end=bool(head.end2end),has_one2one_cv2=getattr(head,'one2one_cv2',None) is not None,
        yaml_end2end=model.yaml.get('end2end'),nm=head.nm,strides=head.stride.cpu().tolist(),head_inputs=head.f,
        class_head=str(head.cv3),box_head=str(head.cv2),mask_head=str(head.cv4),prototype=str(head.proto))

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False)
    before=state(model.model)
    traces={};hooks=[]
    for index in [4,6,10,16,19,22]:
        def hook(module,args,output,index=index):traces[str(index)]=list(output.shape) if torch.is_tensor(output) else str(type(output))
        hooks.append(model.model.model[index].register_forward_hook(hook))
    source=ROOT/'diagnostics/full_val_cache_20260911/val/1000.npz'
    item=np.load(source)
    with contextlib.redirect_stdout(io.StringIO()):
        model.predict(str(ROOT/'data/images/val2017/000000001000.jpg'),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
    cap=model.predictor.capture
    error={}
    for key in ['coeff','boxes','detections','proto']:
        got=cap[key].cpu().numpy();expected=item[key]
        error[key]=dict(shape=list(got.shape),expected_shape=list(expected.shape),max_abs_error=float(np.max(np.abs(got-expected))) if got.shape==expected.shape and got.size else None)
    report=dict(ultralytics=ultralytics.__version__,torch=torch.__version__,before_predict=before,after_predict=state(model.model),predictor_backend_end2end=bool(model.predictor.model.end2end),
        predict_args={k:getattr(model.predictor.args,k,None) for k in ['end2end','agnostic_nms','conf','iou','max_det','imgsz','rect']},
        layer_shapes=traces,cache_replay=error,checkpoint_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),cache_sha256=sha(source),
        source=dict(detect_forward=inspect.getsource(Detect.forward),head_inference=inspect.getsource(Detect._inference),detect_postprocess=inspect.getsource(DetectionPredictor.postprocess),mask_construct=inspect.getsource(SegmentationPredictor.construct_result),nms=inspect.getsource(nms.non_max_suppression)))
    out=ROOT/'audits/RUNTIME_BRANCH_WITNESS_FP32_20260911.json';write_json(out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source','before_predict','after_predict']}))
    print(json.dumps({k:{kk:vv for kk,vv in report[k].items() if kk not in ['class_head','box_head','mask_head','prototype']} for k in ['before_predict','after_predict']}))
    for h in hooks:h.remove()

if __name__=='__main__':main()
