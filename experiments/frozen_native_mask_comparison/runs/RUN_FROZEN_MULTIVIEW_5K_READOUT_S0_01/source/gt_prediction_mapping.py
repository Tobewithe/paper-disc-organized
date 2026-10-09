"""Score-dependent OUTPUT audit with indivisible box/mask prediction identities.

COCOeval owns task success. This module extends its frozen Mask75 pairs only for
diagnosing unmatched objects; it never converts a diagnostic association into a TP.
associate_instances is NOT raw, score-independent geometry matching: its input
pool and pairing order depend on the evaluator and prediction scores. Use it only
to explain exported-output results. pixel_metrics itself is score-independent.
Matrices have shape (number of predictions, number of GTs).
"""
from __future__ import annotations

import numpy as np


def associate_instances(*, gt_ids, gt_categories, prediction_ids, prediction_categories,
                        scores, box_ious, mask_ious, task_matches, eligible_prediction_ids,
                        ignored_prediction_ids=(), ignored_gt_ids=(), task_threshold=.75,
                        association_threshold=.50):
    gt_ids=list(gt_ids); prediction_ids=list(prediction_ids)
    if len(set(gt_ids))!=len(gt_ids) or len(set(prediction_ids))!=len(prediction_ids):
        raise ValueError("GT and prediction IDs must each be unique")
    gindex={v:i for i,v in enumerate(gt_ids)}; pindex={v:i for i,v in enumerate(prediction_ids)}
    b=np.asarray(box_ious,dtype=float); m=np.asarray(mask_ious,dtype=float)
    shape=(len(prediction_ids),len(gt_ids))
    if b.shape!=shape or m.shape!=shape:
        raise ValueError(f"IoU matrices must both have shape {shape}")
    if len(gt_categories)!=len(gt_ids) or len(prediction_categories)!=len(prediction_ids) or len(scores)!=len(prediction_ids):
        raise ValueError("Category/score arrays must preserve original identities")
    if not np.isfinite(b).all() or not np.isfinite(m).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite predictions/metrics are evaluation errors, not misses")
    if ((b<0)|(b>1)|(m<0)|(m>1)).any():
        raise ValueError("IoUs must be in [0,1]")
    eligible=set(eligible_prediction_ids); ignored_p=set(ignored_prediction_ids); ignored_g=set(ignored_gt_ids)
    if not eligible<=set(prediction_ids) or not ignored_p<=eligible or not ignored_g<=set(gt_ids):
        raise ValueError("Evaluation status contains unknown or inconsistent IDs")
    if not 0<association_threshold<task_threshold<=1:
        raise ValueError("Association threshold must be below the task threshold")
    pairs={};used=set();primary=dict(task_matches)
    for gid,pid in primary.items():
        if gid not in gindex or pid not in pindex or gid in ignored_g or pid not in eligible or pid in ignored_p or pid in used:
            raise ValueError("Task pairs must be eligible, nonignored and one-to-one")
        gi,pi=gindex[gid],pindex[pid]
        if gt_categories[gi]!=prediction_categories[pi] or m[pi,gi]+1e-12<task_threshold:
            raise ValueError("Task pair disagrees with its class or mask IoU")
        pairs[gid]=(pid,"task_mask_match");used.add(pid)

    # Tie handling: prediction input order is stable; equal GT IoU uses the later
    # GT in the preserved input order, matching COCO's >= replacement convention.
    order=np.argsort(-np.asarray(scores,dtype=float),kind="mergesort")
    for matrix,basis in ((m,"diagnostic_mask50"),(b,"diagnostic_box50")):
        for pi in order:
            pid=prediction_ids[pi]
            if pid not in eligible or pid in ignored_p or pid in used:
                continue
            candidates=[gi for gi,gid in enumerate(gt_ids) if gid not in pairs and gid not in ignored_g
                        and prediction_categories[pi]==gt_categories[gi] and matrix[pi,gi]>=association_threshold]
            if not candidates:
                continue
            gi=max(candidates,key=lambda j:(matrix[pi,j],j));gid=gt_ids[gi]
            if m[pi,gi]+1e-12>=task_threshold:
                raise ValueError("Free same-class Mask75 edge contradicts supplied task matches; inspect evaluator status")
            pairs[gid]=(pid,basis);used.add(pid)

    gt_rows=[]
    for gi,gid in enumerate(gt_ids):
        pair=pairs.get(gid)
        pid,basis=pair if pair else (None,"ignored_gt" if gid in ignored_g else "unassociated")
        pi=pindex[pid] if pid is not None else None
        bi=float(b[pi,gi]) if pi is not None else None
        mi=float(m[pi,gi]) if pi is not None else None
        if gid in ignored_g:
            state="ignored"
        elif pair is None:
            state="unassociated"
        else:
            state=("box_good" if bi>=task_threshold else "box_bad")+"_"+("mask_good" if mi>=task_threshold else "mask_bad")
        gt_rows.append(dict(annotation_id=gid,prediction_id=pid,association_basis=basis,
                            task_mask_success=None if gid in ignored_g else gid in primary,
                            box_iou=bi,mask_iou=mi,state=state))
    association_owner={pid:gid for gid,(pid,_) in pairs.items()}
    task_owner={pid:gid for gid,pid in primary.items()}
    pred_rows=[]
    for pi,pid in enumerate(prediction_ids):
        status=("outside_evaluation_budget" if pid not in eligible else "ignored" if pid in ignored_p else
                "true_positive" if pid in task_owner else "false_positive_at_task_threshold")
        duplicates=[gid for gid in primary if prediction_categories[pi]==gt_categories[gindex[gid]]
                    and m[pi,gindex[gid]]>=task_threshold] if status=="false_positive_at_task_threshold" else []
        pred_rows.append(dict(prediction_id=pid,task_status=status,task_gt_id=task_owner.get(pid),
                              associated_gt_id=association_owner.get(pid),duplicate_related_gt_ids=duplicates))
    return dict(gt_rows=gt_rows,prediction_rows=pred_rows)


def pixel_metrics(prediction_mask, gt_mask):
    """Original image grid only; use this ONE prediction's mask and box together."""
    p=np.asarray(prediction_mask,dtype=bool);g=np.asarray(gt_mask,dtype=bool)
    if p.shape!=g.shape or p.ndim!=2:
        raise ValueError("Masks must use the same original-image H x W grid")
    tp=int((p & g).sum());fp=int((p & ~g).sum());fn=int((~p & g).sum());ga=tp+fn;pa=tp+fp
    return dict(tp=tp,fp=fp,fn=fn,gt_pixel_area=ga,prediction_pixel_area=pa,
                mask_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                target_coverage=tp/ga if ga else None,prediction_purity=tp/pa if pa else None,
                fp_per_gt_area=fp/ga if ga else None,fn_per_gt_area=fn/ga if ga else None)
