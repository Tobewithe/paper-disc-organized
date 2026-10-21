"""Descriptive smoke metrics; GT is used only after predictions are saved."""
import collections
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_api
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sources"))
from topology import Context


def boxes_iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    intersection = np.maximum(np.minimum(a[:,None,2:], b[None,:,2:])-np.maximum(a[:,None,:2], b[None,:,:2]), 0).prod(2)
    area_a = np.maximum(a[:,2:]-a[:,:2], 0).prod(1)
    area_b = np.maximum(b[:,2:]-b[:,:2], 0).prod(1)
    return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection, 1e-12)


def associate(gt, preds):
    pb = [p["bbox"][:2]+[p["bbox"][0]+p["bbox"][2],p["bbox"][1]+p["bbox"][3]] for p in preds]
    iou = boxes_iou([g["bbox"] for g in gt], pb)
    if not iou.size:
        return {}
    valid = iou >= .5
    # Maximise valid match cardinality before overlap, never use GT class/mask.
    score = valid*(max(iou.shape)+1+iou)
    rows, cols = linear_sum_assignment(-score)
    return {int(r):int(c) for r,c in zip(rows,cols) if valid[r,c]}


def evaluate(run):
    run = Path(run)
    frames = [json.loads(line) for line in (run/"dataset/FRAMES.jsonl").read_text().splitlines()]
    by_image = collections.defaultdict(list)
    for p in json.loads((run/"PREDICTIONS.json").read_text()):
        by_image[p["image_id"]].append(p)
    counters = {}; previous = {}; timeline = []
    palette = np.array([[230,90,55],[50,200,70],[60,110,240],[220,70,210],
                        [200,190,45],[65,210,215],[150,80,220],[170,150,90]], np.uint8)
    previews = run/"previews"; previews.mkdir()
    writers = {}
    for frame in frames:
        sid, iid = frame["scene_id"], frame["image_id"]
        c = counters.setdefault(sid, dict(scene_id=sid, scenario=frame["scenario"], variant=frame["variant"],
            visible_observations=0, matched_box50=0, correct_class_box50=0, wrong_class=0, missed=0,
            topology_eligible=0, material_split=0, material_bridge=0, iou_sum=0.,
            point_transitions=0, point_flips=0, presence_transitions=0, presence_flips=0, class_switches=0))
        label = np.load(run/"dataset/labels"/f"{iid:012d}.npz")
        ids = label["instance_id"]
        targets = [r for r in frame["records"] if r["visible_pixels"] > 0]
        preds = [p for p in by_image[iid] if p["score"] > .25]
        matched = associate(targets, preds)
        rgb = cv2.imread(str(run/"dataset/images"/f"{iid:012d}.png"))
        gt_view, pred_view = rgb.copy(), rgb.copy()
        state = {}; rows = []
        for gi, g in enumerate(targets):
            tid = g["track_id"]; gt = ids == tid; c["visible_observations"] += 1
            color = palette[(tid-1)%len(palette)]
            gt_view[gt] = (.45*gt_view[gt]+.55*color).astype(np.uint8)
            entry = dict(image_id=iid, scene_id=sid, frame=frame["frame"], track_id=tid,
                visibility=g["visibility"], gt_pixels=g["visible_pixels"], matched=gi in matched)
            mask = None; category = None
            if gi in matched:
                p = preds[matched[gi]]; category = p["category_id"]
                mask = mask_api.decode(p["segmentation"]).astype(bool)
                c["matched_box50"] += 1
                correct = category == 21  # COCO Cow; asserted against source categories in runner.
                c["correct_class_box50"] += int(correct); c["wrong_class"] += int(not correct)
                entry.update(output_detection_id=p["output_detection_id"], category_id=category, score=p["score"],
                             class_correct=correct)
                pred_view[mask] = (.45*pred_view[mask]+.55*color).astype(np.uint8)
                if correct:
                    neighbors = [(r["track_id"], ids == r["track_id"]) for r in targets if r["track_id"] != tid]
                    m = Context(gt, neighbors).measure(mask)
                    entry["mask_metrics"] = m
                    c["topology_eligible"] += 1; c["material_split"] += int(m["material_split"])
                    c["material_bridge"] += int(m["material_bridge"]); c["iou_sum"] += m["iou"]
            else:
                c["missed"] += 1
            point_ids = np.where((label["point_owner"] == tid) & label["point_valid"])[0]
            bits = {}
            if mask is not None and category == 21:
                for pi in point_ids:
                    x, y = np.floor(label["point_xy"][pi]).astype(int)
                    bits[int(pi)] = bool(mask[y,x])
            state[tid] = dict(present=mask is not None, category=category, bits=bits)
            if sid in previous and tid in previous[sid]:
                before = previous[sid][tid]
                c["presence_transitions"] += 1
                c["presence_flips"] += int(before["present"] != state[tid]["present"])
                if before["present"] and state[tid]["present"]:
                    c["class_switches"] += int(before["category"] != category)
                common = before["bits"].keys() & bits.keys()
                c["point_transitions"] += len(common)
                c["point_flips"] += sum(before["bits"][k] != bits[k] for k in common)
            rows.append(entry)
        previous[sid] = state
        timeline.extend(rows)
        # Also show unmatched normal predictions so false positives stay visible.
        used = set(matched.values())
        for pi, p in enumerate(preds):
            x,y,w,h = p["bbox"]
            cv2.rectangle(pred_view,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255) if pi not in used else (220,220,220),1)
            cv2.putText(pred_view,f"{p['category_id']} {p['score']:.2f}",(max(0,int(x)),max(12,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        panels = [rgb, gt_view, pred_view]
        for panel, title in zip(panels,["RGB", "Visible GT", "YOLO native score>.25"]):
            cv2.rectangle(panel,(0,0),(640,24),(0,0,0),-1)
            cv2.putText(panel,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        joined = np.concatenate(panels,axis=1)
        cv2.putText(joined,f"{frame['scenario']} {frame['variant']} f={frame['frame']}",(8,375),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if sid not in writers:
            writer = cv2.VideoWriter(str(previews/f"scene_{sid:02d}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),12,(1920,384))
            assert writer.isOpened(), "preview encoder unavailable"
            writers[sid] = writer
        writers[sid].write(joined)
        if frame["frame"] in [0, 16, 24, 32, 47]:
            assert cv2.imwrite(str(previews/f"scene_{sid:02d}_f{frame['frame']:02d}.jpg"), joined)
    for w in writers.values(): w.release()
    for c in counters.values():
        for name, numerator, denominator in [
            ("correct_class_recall", "correct_class_box50","visible_observations"),
            ("mean_matched_iou","iou_sum","topology_eligible"),
            ("split_rate","material_split","topology_eligible"),
            ("bridge_rate","material_bridge","topology_eligible"),
            ("interior_point_flip_rate","point_flips","point_transitions")]:
            c[name] = c[numerator]/c[denominator] if c[denominator] else None
    result = dict(status="engineering_completed", training_steps=0, scenes=list(counters.values()),
        inference_predictions="all native score>.001 saved", operating_threshold=.25,
        scene_domain_gate={str(k):v["correct_class_recall"] >= .8 for k,v in counters.items() if v["variant"]=="control"},
        claim_scope="descriptive synthetic engineering smoke; no causal mechanism, no method gain, no real-video generalization",
        temporal_scope="visible pure-interior persistent surface points; detection absence and wrong class excluded from mask flips and counted separately")
    (run/"TIMELINE.jsonl").write_text("".join(json.dumps(r)+"\n" for r in timeline))
    (run/"SUMMARY.json").write_text(json.dumps(result,indent=2))
    return result
