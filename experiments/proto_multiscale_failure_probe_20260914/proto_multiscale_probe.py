"""Object-conditioned causal ablation of YOLO26 Proto26 multi-scale fusion.

Detection candidates, boxes and coefficients stay fixed. Only the feature sum entering Proto26 is
changed by leaving out one or more P3/P4/P5 components. Matched success cases quantify generic
distribution-shift damage from branch removal; failure-specific benefit is the relevant signal.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.nn.modules import Proto
from ultralytics.utils import ops


COCO80 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def selection(rows: list[dict], n: int, seed: int) -> list[dict]:
    predicates = {
        "historical_boxgood_maskbad": lambda r: r["failure_scope"] == "box_good_support_sufficient_mask_bad",
        "historical_success": lambda r: r["failure_scope"] == "box_good_support_sufficient_mask_good" and r["task_mask75"] == "hit",
    }
    rng = random.Random(seed); used: set[int] = set(); out: list[dict] = []
    for cohort, pred in predicates.items():
        pools: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if pred(r) and int(r["image_id"]) not in used:
                pools[r["area_bin"]].append(r)
        for p in pools.values(): rng.shuffle(p)
        picked: list[dict] = []
        while len(picked) < n:
            changed = False
            for size in ("small", "medium", "large"):
                p = pools[size]
                while p and int(p[-1]["image_id"]) in used: p.pop()
                if p and len(picked) < n:
                    r = p.pop(); used.add(int(r["image_id"])); picked.append(r); changed = True
            if not changed: break
        if len(picked) < n: raise RuntimeError(f"Only {len(picked)} for {cohort}")
        for r in picked:
            out.append({"selection_cohort": cohort, "image_id": int(r["image_id"]),
                        "annotation_id": int(r["annotation_id"]), "category_id": int(r["category_id"]),
                        "area_bin": r["area_bin"], "density_e4": r["density_e4"]})
    return out


def box_iou(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    lt = np.maximum(boxes[:, :2], target[:2]); rb = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(rb-lt, 0).prod(1)
    aa = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(1)
    ta = np.maximum(target[2:]-target[:2], 0).prod()
    return inter / np.maximum(aa + ta - inter, 1e-12)


def decode(proto: torch.Tensor, coeff: torch.Tensor, box: torch.Tensor, input_shape, original_shape) -> np.ndarray:
    mask = ops.process_mask(proto, coeff[None], box[None], input_shape, upsample=True)
    mask = ops.scale_masks(mask[:, None], original_shape)[:, 0]
    return (mask[0] > 0.5).detach().cpu().numpy()


def measure(pred: np.ndarray, target: np.ndarray, same: np.ndarray, others: np.ndarray) -> dict:
    inter = int((pred & target).sum()); pa = int(pred.sum()); ga = int(target.sum())
    fp = pred & ~target
    return {
        "mask_iou": inter / max(pa + ga - inter, 1), "coverage": inter / max(ga, 1),
        "same_neighbor_fp": int((fp & same).sum()) / max(ga, 1),
        "other_instance_fp": int((fp & others & ~same).sum()) / max(ga, 1),
        "background_fp": int((fp & ~others).sum()) / max(ga, 1),
    }


def bootstrap(values: list[float], seed: int, reps: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--per-cohort", type=int, default=160); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0); a=ap.parse_args()
    out=Path(__file__).resolve().parent; root=out.parents[1]; shared=root/"shared/coco_clean_20260911"; sys.path.insert(0,str(shared))
    from structure_candidate_trace import TraceCapture
    ann_path=root/"assets/datasets/coco/annotations/instances_val2017.json"; images=shared/"local_readout_runtime_20260912/data/images/val2017"; weight=root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"
    chosen=selection(read_csv(root/"diagnostics/coco_failure_dimensions_20260913/instances_improved.csv"),a.per_cohort,a.seed)
    if a.limit: chosen=chosen[:a.limit]
    write_csv(out/"selection.csv",chosen)
    with contextlib.redirect_stdout(io.StringIO()): gt=COCO(str(ann_path))
    model=YOLO(str(weight)); model.model.eval().requires_grad_(False); head=model.model.model[-1]; head.end2end=False
    captured: dict[str,torch.Tensor]={}
    def prehook(module, args):
        x=args[0]; p3=x[0]; p4=torch.nn.functional.interpolate(module.feat_refine[0](x[1]),size=p3.shape[2:],mode="nearest"); p5=torch.nn.functional.interpolate(module.feat_refine[1](x[2]),size=p3.shape[2:],mode="nearest")
        components={"full":p3+p4+p5,"drop_p3":p4+p5,"drop_p4":p3+p5,"drop_p5":p3+p4,"p3_only":p3}
        captured["component_norms"]=torch.tensor([p3.norm(),p4.norm(),p5.norm()]).cpu()
        for name,feat in components.items(): captured[name]=Proto.forward(module,module.feat_fuse(feat)).detach().float().cpu()
    hook=head.proto.register_forward_pre_hook(prehook); rows=[]; start=time.time(); arms=["full","drop_p3","drop_p4","drop_p5","p3_only"]
    for number,item in enumerate(chosen,1):
        ann=gt.anns[item["annotation_id"]]; info=gt.imgs[item["image_id"]]; captured.clear()
        with torch.inference_mode(): result=model.predict(str(images/info["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)[0]
        cap=model.predictor.capture; official=cap["proto"].detach().float().cpu(); parity=float((official-captured["full"][0]).abs().max())
        if parity>1e-5: raise RuntimeError(f"Manual full proto mismatch {parity}")
        det=cap["detections"].detach().cpu().numpy(); classes=det[:,5].astype(int) if len(det) else np.empty(0,dtype=int)
        x,y,w,h=ann["bbox"]; target_box=np.array([x,y,x+w,y+h]); valid=np.array([0<=c<len(COCO80) and COCO80[c]==item["category_id"] for c in classes])
        if not valid.any(): continue
        idxs=np.flatnonzero(valid); ious=box_iou(det[idxs,:4],target_box); best=int(idxs[int(ious.argmax())]); biou=float(ious.max())
        target=gt.annToMask(ann).astype(bool); same=np.zeros_like(target); others=np.zeros_like(target)
        for oid in gt.getAnnIds(imgIds=[item["image_id"]],iscrowd=None):
            if oid==item["annotation_id"] or gt.anns[oid].get("iscrowd",0): continue
            om=gt.annToMask(gt.anns[oid]).astype(bool); others|=om
            if gt.anns[oid]["category_id"]==item["category_id"]: same|=om
        arm_metrics={}
        for arm in arms:
            pred=decode(captured[arm][0].to(cap["coeff"].device),cap["coeff"][best],cap["boxes"][best],cap["input_shape"],cap["shape"])
            arm_metrics[arm]=measure(pred,target,same,others)
        full_iou=arm_metrics["full"]["mask_iou"]
        current_state="boxgood_maskbad" if biou>=.75 and full_iou<.75 else "boxgood_maskgood" if biou>=.75 and full_iou>=.75 else "boxbad"
        norms=captured["component_norms"].numpy()
        for arm in arms:
            rows.append({**item,"current_state":current_state,"arm":arm,"box_iou":biou,"proto_parity_max":parity,"p3_norm":norms[0],"p4_norm":norms[1],"p5_norm":norms[2],**arm_metrics[arm]})
        if number%24==0 or number==len(chosen):
            write_csv(out/"per_target.csv",rows); (out/"progress.json").write_text(json.dumps({"done":number,"total":len(chosen),"elapsed_s":time.time()-start},indent=2),encoding="utf-8"); print(f"[{number}/{len(chosen)}] elapsed={time.time()-start:.1f}s",flush=True)
    hook.remove(); base={(int(r["annotation_id"]),r["current_state"]):r for r in rows if r["arm"]=="full"}; summary=[]
    for state in ("boxgood_maskbad","boxgood_maskgood"):
        for arm in arms:
            rr=[r for r in rows if r["current_state"]==state and r["arm"]==arm]
            if not rr: continue
            deltas=[float(r["mask_iou"])-float(base[(int(r["annotation_id"]),state)]["mask_iou"]) for r in rr]; mean,lo,hi=bootstrap(deltas,a.seed)
            summary.append({"current_state":state,"arm":arm,"n":len(rr),"mask_iou_mean":np.mean([float(r["mask_iou"]) for r in rr]),"delta_iou_mean":mean,"delta_iou_ci_low":lo,"delta_iou_ci_high":hi,"mask75_rate":np.mean([float(r["mask_iou"])>=.75 for r in rr]),"coverage_mean":np.mean([float(r["coverage"]) for r in rr]),"same_neighbor_fp_mean":np.mean([float(r["same_neighbor_fp"]) for r in rr]),"background_fp_mean":np.mean([float(r["background_fp"]) for r in rr])})
    write_csv(out/"summary.csv",summary); (out/"COMPLETE.json").write_text(json.dumps({"status":"complete","selected":len(chosen),"decoded_targets":len(base),"ultralytics":"8.4.100","end2end":False,"fixed":["candidate","box","coefficient"],"elapsed_s":time.time()-start},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
