"""No-GT mask rendering controls on the identical normal prediction identities.

Distinguish network errors from converting binary masks back to the original
grid. Both controls keep original classes/scores/boxes/output count. These are
decoder controls and existing ideas, not a claimed novel learned method.
"""
import argparse
import contextlib
import gc
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--evaluation",type=Path,required=True)
    ap.add_argument("--after",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    sys.path.insert(0,str(a.root/"shared/vendor/ultralytics_8_4_100"))
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mu
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(parents=True,exist_ok=True)
    # Serialize memory-intensive COCO evaluations on this 32 GB host.
    while not (a.after/"COMPLETE.json").exists():
        status=json.loads((a.after/"run.json").read_text()).get("status")
        if status in ("failed","error","interrupted","cancelled"):
            raise RuntimeError(f"Prerequisite {status}")
        time.sleep(2)
    start=time.monotonic()
    base=json.loads((a.evaluation/"SUMMARY.json").read_text())
    predictions=json.loads((a.evaluation/"predictions_with_identity.json").read_text())
    groups=defaultdict(list)
    for i,p in enumerate(predictions):
        groups[p["image_id"]].append(i)
    modes=("binary_resize_threshold05","native_logits_original_grid")
    versions={mode:[] for mode in modes}
    changed={mode:0 for mode in modes}
    with torch.inference_mode():
        for number,(image_id,pids) in enumerate(sorted(groups.items())):
            with np.load(a.raw/"images"/f"{image_id:012d}.npz") as z:
                shape=tuple(z["input_shape"].tolist());orig=tuple(z["original_shape"].tolist())
                coef=torch.as_tensor(z["coefficients"],device="cuda")
                proto=torch.as_tensor(z["proto"],device="cuda")
                boxes=torch.as_tensor(z["boxes_input"],device="cuda")
                original_boxes=torch.as_tensor(z["boxes_original"],device="cuda")
            logits=(coef.T@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            unique=sorted({predictions[i]["raw_id"] for i in pids})
            encodings={mode:{} for mode in modes}
            for k in range(0,len(unique),32):
                ix=unique[k:k+32]
                grid=F.interpolate(logits[ix][None],shape,mode="bilinear")[0]
                binary=ops.crop_mask(grid,boxes[ix]).gt(0).byte()
                scaled=ops.scale_masks(binary[None],orig)[0]
                normal=scaled.byte().cpu().numpy()
                thresholded=(scaled>.5).byte().cpu().numpy()
                # Same ordering as official process_mask_native, with the full-raw
                # cached matrix multiplication used by the primary experiment.
                native=ops.scale_masks(logits[ix][None],orig)[0].gt(0).byte()
                native=ops.crop_mask(native,original_boxes[ix]).cpu().numpy()
                for mode,masks in zip(modes,(thresholded,native)):
                    for raw_id,mask,reference in zip(ix,masks,normal):
                        rle=mu.encode(np.asfortranarray(mask));rle["counts"]=rle["counts"].decode("ascii")
                        encodings[mode][raw_id]=(rle,bool(np.any(mask!=reference)))
            for i in pids:
                p=predictions[i]
                for mode in modes:
                    rle,was_changed=encodings[mode][p["raw_id"]]
                    changed[mode]+=int(was_changed)
                    versions[mode].append(dict(image_id=image_id,category_id=p["category_id"],score=p["score"],segmentation=rle))
            if (number+1)%250==0:
                progress=dict(stage="rendering",images=number+1,elapsed_s=round(time.monotonic()-start,2))
                print(json.dumps(progress),flush=True);dump(a.out/"progress.json",progress)
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(a.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    results={}
    ordinary={ann["id"] for ann in coco.anns.values() if not ann.get("iscrowd",0) and not ann.get("ignore",0)}
    for mode in modes:
        dump(a.out/(mode+"_predictions.json"),versions[mode])
        print("COCOeval "+mode,flush=True)
        dump(a.out/"progress.json",dict(stage="cocoeval_"+mode,elapsed_s=round(time.monotonic()-start,2)))
        dt=coco.loadRes(versions[mode]);ev=COCOeval(coco,dt,"segm")
        ev.params.imgIds=sorted(coco.imgs)
        ev.evaluate();ev.accumulate();ev.summarize()
        names=("AP","AP50","AP75","APS","APM","APL","AR1","AR10","AR100","ARS","ARM","ARL")
        result=dict(zip(names,[float(v) for v in ev.stats]))
        success=set()
        for entry in ev.evalImgs:
            if entry is None or entry["aRng"]!=ev.params.areaRng[0] or entry["maxDet"]!=100:continue
            for gid,matched in zip(entry["gtIds"],entry["gtMatches"][5]):
                if gid in ordinary and matched:success.add(gid)
        result.update(delta_AP_points=100*(result["AP"]-base["coco"]["segm"]["AP"]),R75_TP_count=len(success),R75_micro=len(success)/len(ordinary))
        results[mode]=result
        dump(a.out/(mode+".json"),result)
        dump(a.out/(mode+"_mask75_gt_ids.json"),sorted(success))
        del ev,dt
        gc.collect()
    summary=dict(baseline=base["coco"]["segm"],results=results,changed_prediction_counts=changed,predictions=len(predictions),
        images=5000,source_raw=a.raw.name,source_evaluation=a.evaluation.name,
        protocol="No GT used to construct masks; fixed original prediction identities/classes/scores/boxes; only rendering changes.",
        limits=["These are rendering controls, not a new learned method.","Native original-grid decoding is an existing official option, not novel.",
          "Originally empty or filtered candidates remain absent; this isolates mask content at fixed output set.",
          "Raw geometric five-class counts remain those of the original declared decoder; not recomputed for these variants."])
    dump(a.out/"SUMMARY.json",summary)
    dump(a.out/"COMPLETE.json",dict(images=5000,elapsed_s=round(time.monotonic()-start,2)))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
