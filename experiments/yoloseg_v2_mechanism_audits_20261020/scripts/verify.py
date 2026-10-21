"""Independent mask/count validation, separate from inference and summary paths."""
from pathlib import Path
import json
import numpy as np
from pycocotools import mask as api


def verify(run, kind, base):
    run,base=Path(run),Path(base)
    rows=[json.loads(s) for s in (run/("FACTORIAL_ROWS.jsonl" if kind=="chain" else "SHIFT_ROWS.jsonl")).read_text().splitlines()]
    count=0
    if kind=="chain":
        lookup={r["pressure_image"]:r for r in rows if r["eligible"]}
        for s in (run/"HYBRID_MASKS.jsonl").read_text().splitlines():
            r=json.loads(s)
            with np.load(base/f'dataset/labels/{r["pressure_image"]:012d}.npz') as z:g=z["instance_id"]==1
            p=api.decode(r["segmentation"]).astype(bool)
            expected=lookup[r["pressure_image"]]["results"][r["box"]][r["combination"]]
            intersection=int(np.count_nonzero(p&g));union=int(np.count_nonzero(p|g))
            assert abs(intersection/max(1,union)-expected["iou"])<=1e-12
            count+=1
        assert count==8*len(lookup)
        chain=[json.loads(s) for s in (run/"CHAIN_ROWS.jsonl").read_text().splitlines()]
        assert len({(r["image_id"],r["track_id"]) for r in chain})==len(chain)
        for r in chain:
            assert r["first_failure"]==next((j+1 for j,v in enumerate(r["stages"]) if not v),None)
    else:
        lookup={(r["image_id"],r["track_id"],tuple(r["offset"])):r for r in rows}
        assert len(lookup)==len(rows)
        for s in (run/"ALIGNED_MASKS.jsonl").read_text().splitlines():
            r=json.loads(s); expected=lookup[r["image_id"],r["track_id"],tuple(r["offset"])]
            p=api.decode(r["segmentation"]).astype(bool)
            with np.load(base/f'dataset/labels/{r["image_id"]:012d}.npz') as z:g=z["instance_id"]==r["track_id"]
            g[:4]=False;g[-4:]=False;g[:,:4]=False;g[:,-4:]=False
            measured=np.count_nonzero(p&g)/max(1,np.count_nonzero(p|g))
            assert abs(measured-expected["operational_iou"])<=1e-12
            count+=1
        assert count==len(rows)*8//9
    return dict(status="passed",independent_RLE_IoU_checks=count,training_steps=0)
