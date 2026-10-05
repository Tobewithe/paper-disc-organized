import argparse, json, os
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ultralytics import YOLO
from ultralytics.utils import YAML

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--mode", choices=["default","one2many"], required=True)
    ap.add_argument("--native", action="store_true")
    a=ap.parse_args()
    model=YOLO(a.weights)
    head=model.model.model[-1]
    if a.mode=="one2many":
        head.end2end=False
    data=YAML.load(a.data)
    root=Path(data["path"]).resolve()
    print({"ultralytics":__import__("ultralytics").__version__,
           "mode":a.mode,"head_end2end":bool(head.end2end),
           "weights":a.weights,"native":a.native,"val":str(root/"images"/"val2017")},flush=True)
    metrics=model.val(data=a.data,imgsz=640,batch=2,workers=0,device=0,
                      project=a.project,name=a.name,plots=False,verbose=False,
                      split="val",cache=False,save_json=a.native)
    payload={"images":5000,"mode":a.mode,"native":a.native,"head_end2end":bool(head.end2end),
             "box_map50_95":float(metrics.box.map),"box_map50":float(metrics.box.map50),
             "mask_map50_95":float(metrics.seg.map),"mask_map50":float(metrics.seg.map50)}
    out=Path(a.metrics_out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload),flush=True)

if __name__=="__main__":
    main()

