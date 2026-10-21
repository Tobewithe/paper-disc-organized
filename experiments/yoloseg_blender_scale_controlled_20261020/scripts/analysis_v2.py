"""Pre-registered focus census and paired comparisons; no IID-frame CI."""
import collections
import json
from pathlib import Path

import cv2
import numpy as np


def counts(rows):
    n=len(rows)
    return dict(n=n,correct=sum(r["class_correct"] for r in rows),
        failures=sum(r["segmentation_failure"] for r in rows),
        operational_iou=sum(r["operational_iou"] for r in rows)/n if n else None,
        material_bridge=sum(r.get("mask_metrics",{}).get("material_bridge",False) for r in rows),
        material_split=sum(r.get("mask_metrics",{}).get("material_split",False) for r in rows))


def summarize(timeline,frames):
    groups=collections.defaultdict(list)
    for r in timeline: groups[(r["seed"],r["variant"])].append(r)
    scene_counts=[]; domain={}; pairs=[]
    seeds=sorted({r["seed"] for r in timeline})
    for (seed,variant),rows in sorted(groups.items()):
        focus=[r for r in rows if r["track_id"]==1]
        a,b=counts(rows),counts(focus)
        scene_counts.append(dict(seed=seed,variant=variant,all=a,focus=b))
        if variant=="clear":
            domain[str(seed)]=b["n"]>0 and b["correct"]/b["n"]>=.9 and a["correct"]/a["n"]>=.8
    for seed in seeds:
        for pressure,control in [("near","clear"),("front","rear"),("rear","clear")]:
            pa={r["frame"]:r for r in groups[(seed,pressure)] if r["track_id"]==1}
            co={r["frame"]:r for r in groups[(seed,control)] if r["track_id"]==1}
            for f in sorted(pa.keys()&co.keys()):
                a,b=pa[f],co[f]
                pairs.append(dict(seed=seed,pressure=pressure,control=control,frame=f,
                    failure_delta=int(a["segmentation_failure"])-int(b["segmentation_failure"]),
                    iou_delta=a["operational_iou"]-b["operational_iou"],
                    both_correct=a["class_correct"] and b["class_correct"],
                    bridge_pressure=bool(a.get("mask_metrics",{}).get("material_bridge",False)),
                    bridge_control=bool(b.get("mask_metrics",{}).get("material_bridge",False))))
    comparisons=[]
    for pressure,control in [("near","clear"),("front","rear"),("rear","clear")]:
        rows=[r for r in pairs if r["pressure"]==pressure]
        per_seed=[]
        for seed in seeds:
            rr=[r for r in rows if r["seed"]==seed]
            per_seed.append(dict(seed=seed,n=len(rr),failure_delta=sum(r["failure_delta"] for r in rr)/len(rr) if rr else None,
                                 iou_delta=sum(r["iou_delta"] for r in rr)/len(rr) if rr else None))
        both=[r for r in rows if r["both_correct"]]
        failure_delta=sum(r["failure_delta"] for r in rows)/len(rows) if rows else None
        comparisons.append(dict(pressure=pressure,control=control,n=len(rows),per_seed=per_seed,
            failure_delta=failure_delta,iou_delta=sum(r["iou_delta"] for r in rows)/len(rows) if rows else None,
            jointly_correct=len(both),bridge_pressure=sum(r["bridge_pressure"] for r in both),
            bridge_control=sum(r["bridge_control"] for r in both),
            effect_gate=(all(domain.values()) and len(seeds)==3 and failure_delta is not None and failure_delta>=.1
                         and sum(r["failure_delta"] is not None and r["failure_delta"]>0 for r in per_seed)>=2)
                        if pressure!="rear" else None))
    visibility=collections.Counter(r["state"] for f in frames for r in f["records"])
    return dict(focus_counts=scene_counts,domain_gate=domain,all_domains_passed=all(domain.values()),
        paired_comparisons=comparisons,paired_rows=pairs,gt_visibility=dict(visibility),
        inference_scope="native N only; no G or LCF comparison; zero training")


def comparison_videos(run,frames):
    directory=Path(run)/"previews"
    config={f["scene_id"]:(f["seed"],f["variant"]) for f in frames}
    for seed in sorted({f["seed"] for f in frames}):
        sid={v:s for s,(se,v) in config.items() if se==seed}
        for control,pressure in [("clear","near"),("rear","front")]:
            caps=[cv2.VideoCapture(str(directory/f"scene_{sid[v]:02d}.mp4")) for v in (control,pressure)]
            writer=cv2.VideoWriter(str(directory/f"compare_{seed}_{control}_{pressure}.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"),12,(1920,768))
            assert writer.isOpened()
            index=0
            while True:
                ar,a=caps[0].read();br,b=caps[1].read()
                assert ar==br
                if not ar:break
                frame=np.concatenate([a,b],axis=0);writer.write(frame)
                if index in (0,16,24,32,47):
                    cv2.imwrite(str(directory/f"compare_{seed}_{control}_{pressure}_f{index:02d}.jpg"),frame)
                index+=1
            writer.release()
            for cap in caps:cap.release()
