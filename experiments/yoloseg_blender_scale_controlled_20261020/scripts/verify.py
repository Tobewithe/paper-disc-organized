"""Independent arithmetic, geometry and RLE checks, without model execution."""
import collections
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mask_api


def verify(run,deep=True,output=None):
    run=Path(run)
    timeline=[json.loads(x) for x in (run/"TIMELINE.jsonl").read_text().splitlines()]
    frames=[json.loads(x) for x in (run/"dataset/FRAMES.jsonl").read_text().splitlines()]
    summary=json.loads((run/"SUMMARY.json").read_text())
    index={(r["seed"],r["variant"],r["frame"],r["track_id"]):r for r in timeline}
    for r in timeline:
        q=r.get("mask_metrics",{}).get("iou",0)
        assert q==r["operational_iou"]
        assert r["segmentation_failure"] == (not r["class_correct"] or q<.75)
    for p in summary["paired_rows"]:
        a=index[(p["seed"],p["pressure"],p["frame"],1)]
        b=index[(p["seed"],p["control"],p["frame"],1)]
        assert p["failure_delta"]==int(a["segmentation_failure"])-int(b["segmentation_failure"])
        assert abs(p["iou_delta"]-(a["operational_iou"]-b["operational_iou"]))<1e-12
    for c in summary["paired_comparisons"]:
        pp=[p for p in summary["paired_rows"] if p["pressure"]==c["pressure"]]
        assert len(pp)==c["n"]
        assert abs(sum(p["failure_delta"] for p in pp)/len(pp)-c["failure_delta"])<1e-12
    for c in summary["scenes"]:
        rows=[r for r in timeline if r["scene_id"]==c["scene_id"]]
        assert c["visible_observations"]==len(rows)
        assert c["correct_class_box50"]==sum(r["class_correct"] for r in rows)
        assert c["material_bridge"]==sum(bool(r.get("mask_metrics",{}).get("material_bridge")) for r in rows)
        assert c["material_split"]==sum(bool(r.get("mask_metrics",{}).get("material_split")) for r in rows)
        previous={}; flips=transitions=presence_flips=presence_transitions=0
        for r in rows:
            prev=previous.get(r["track_id"])
            if prev and prev["frame"]==r["frame"]-1:
                presence_transitions+=1;presence_flips+=prev["matched"]!=r["matched"]
                common=prev["point_bits"].keys()&r["point_bits"].keys()
                transitions+=len(common);flips+=sum(prev["point_bits"][k]!=r["point_bits"][k] for k in common)
            previous[r["track_id"]]=r
        assert (flips,transitions,presence_flips,presence_transitions)==(
            c["point_flips"],c["point_transitions"],c["presence_flips"],c["presence_transitions"])
    masks_checked=projection_checked=point_bits_checked=0; max_error=0.
    if deep:
        predictions={r["output_detection_id"]:r for r in json.loads((run/"PREDICTIONS.json").read_text())}
        by_image=collections.defaultdict(list)
        for r in timeline:by_image[r["image_id"]].append(r)
        baseline={}; ref_amodal={}
        for f in frames:
            label=np.load(run/"dataset/labels"/f"{f['image_id']:012d}.npz")
            key=(f["seed"],f["frame"])
            if f["variant"]=="clear":
                baseline[key]={i:label[f"projection_{i}"] for i in (1,2,3)}
                ref_amodal[key]=label["amodal_1"]
            assert np.array_equal(label["amodal_1"],ref_amodal[key])
            for i in (1,2,3):
                a=label[f"projection_{i}"];b=baseline[key][i]
                error=float(np.max(np.abs((a-a.mean(0))-(b-b.mean(0)))))
                assert error<=.001
                assert np.max(np.abs(np.ptp(a,axis=0)-np.ptp(b,axis=0)))<=.001
                projection_checked+=1;max_error=max(max_error,error)
            for r in by_image[f["image_id"]]:
                gt=label["instance_id"]==r["track_id"]
                assert int(gt.sum())==r["gt_pixels"]
                if r["class_correct"]:
                    pred=mask_api.decode(predictions[r["output_detection_id"]]["segmentation"]).astype(bool)
                    iou=float(np.count_nonzero(gt&pred)/np.count_nonzero(gt|pred))
                    assert abs(iou-r["operational_iou"])<1e-12
                    for k,bit in r["point_bits"].items():
                        pi=int(k);x,y=np.floor(label["point_xy"][pi]).astype(int)
                        assert label["point_valid"][pi] and label["point_owner"][pi]==r["track_id"]
                        assert bool(pred[y,x])==bit;point_bits_checked+=1
                    masks_checked+=1
    result=dict(status="passed",rows=len(timeline),paired_rows=len(summary["paired_rows"]),
        deep=deep,masks_checked=masks_checked,point_bits_checked=point_bits_checked,
        projections_checked=projection_checked,max_centered_error_px=max_error,model_forwards=0,training_steps=0)
    (Path(output) if output else run/"INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    verify(sys.argv[1],deep=True,output=sys.argv[2] if len(sys.argv)>2 else None)
