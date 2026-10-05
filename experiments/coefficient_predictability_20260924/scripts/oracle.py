"""Finite per-instance native-BCE coefficient oracle and separability controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def loss(c, row):
    return F.binary_cross_entropy_with_logits(row["p"] @ c, row["y"]) * row["factor"]


def optimize(row, init):
    c = init.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([c], lr=1, max_iter=100, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)
    calls = 0
    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        objective = loss(c, row)
        objective.backward()
        return objective
    optimizer.step(closure)
    objective = loss(c, row)
    grad = torch.autograd.grad(objective, c)[0]
    return c.detach(), dict(loss=float(objective.detach()), gradient_norm=float(grad.norm()),
                             closure_calls=calls, iterations=optimizer.state[c]["n_iter"])


def final_iou(c, image, gt):
    proto = image["proto"].cuda()
    z = (c @ proto.flatten(1)).reshape(1, *proto.shape[-2:])
    z = F.interpolate(z[None], (640, 640), mode="bilinear", align_corners=False)[0]
    box = image["boxes"][image["raw_id"]][None].cuda()
    binary = ops.crop_mask(z, box).gt(0).byte()
    original = ops.scale_masks(binary[None], image["original_shape"])[0, 0].cpu().numpy() > 0
    return float(np.logical_and(original, gt).sum() / max(np.logical_or(original, gt).sum(), 1))


def main(a):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bank = torch.load(a.bank / "val.pt", weights_only=False, map_location="cpu")
    source = COCO(str(a.annotations))
    rng = random.Random(20260924)
    bad = [i for i,r in enumerate(bank) if r["meta"]["native_iou"] < .75]
    good = [i for i,r in enumerate(bank) if r["meta"]["native_iou"] >= .75]
    chosen = rng.sample(bad, min(a.each_group, len(bad))) + rng.sample(good, min(a.each_group, len(good)))
    rows, values = [], {}
    start = time.monotonic()
    for n, idx in enumerate(chosen, 1):
        original = bank[idx]
        r = {**original, "p": original["p"].cuda(), "y": original["y"].cuda()}
        initial = original["c"].cuda()
        from_prediction, state_p = optimize(r, initial)
        from_zero, state_z = optimize(r, torch.zeros_like(initial))
        best = from_prediction if state_p["loss"] <= state_z["loss"] else from_zero
        image = torch.load(a.bank / "images" / f'{r["meta"]["image_id"]:012d}.pt', weights_only=False, map_location="cpu")
        image["raw_id"] = r["meta"]["raw_id"]
        gt = source.annToMask(source.anns[r["meta"]["annotation_id"]]).astype(bool)
        before_iou, after_iou = final_iou(initial, image, gt), final_iou(best, image, gt)
        rows.append({**r["meta"], "group": "native_failure" if idx in bad else "native_success",
                     "initial_loss": float(loss(initial, r)), "from_prediction":state_p,"from_zero":state_z,
                     "init_loss_difference":abs(state_p["loss"]-state_z["loss"]),
                     "final_iou_before":before_iou, "final_iou_after":after_iou,
                     "oracle_loss": min(state_p["loss"],state_z["loss"])})
        values[str(r["meta"]["annotation_id"])] = best.cpu()
        if n % 10 == 0 or n == len(chosen):
            state = dict(stage="individual", instances=n,total=len(chosen),elapsed=time.monotonic()-start)
            print(json.dumps(state),flush=True)
            (a.out/"PROGRESS.json").write_text(json.dumps(state))
    image_groups = defaultdict(list)
    for i,r in enumerate(bank):
        image_groups[r["meta"]["image_id"]].append(i)
    candidate_groups = [v for v in image_groups.values() if len(v)>=2]
    rng.shuffle(candidate_groups)
    controls = []
    for indices in candidate_groups[:20]:
        entries = [bank[i] for i in rng.sample(indices, 2)]
        entries = [{**r,"p":r["p"].cuda(),"y":r["y"].cuda()} for r in entries]
        initial = torch.stack([r["c"] for r in entries]).cuda()
        c = initial.clone().requires_grad_(True)
        joint_loss = sum(loss(c[k],r) for k,r in enumerate(entries))
        joint_gradient = torch.autograd.grad(joint_loss,c)[0]
        independent_gradient=[]
        independent_states=[]
        for k,r in enumerate(entries):
            ci=initial[k].clone().requires_grad_(True)
            independent_gradient.append(torch.autograd.grad(loss(ci,r),ci)[0])
            _, info=optimize(r,initial[k])
            independent_states.append(info)
        max_error=float((joint_gradient-torch.stack(independent_gradient)).abs().max())
        assert max_error < 1e-6
        optimizer=torch.optim.LBFGS([c],lr=1,max_iter=100,line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-7,tolerance_change=1e-9)
        def closure():
            optimizer.zero_grad()
            objective=sum(loss(c[k],r) for k,r in enumerate(entries))
            objective.backward()
            return objective
        optimizer.step(closure)
        controls.append(dict(image_id=entries[0]["meta"]["image_id"],gradient_max_error=max_error,
            independent_loss=sum(r["loss"] for r in independent_states),
            joint_loss=float(sum(loss(c[k],r) for k,r in enumerate(entries)).detach())))
    summary={"selection":dict(native_failures=len(bad),native_successes=len(good),sampled=len(rows)),"groups":{},
             "joint_controls":controls,"elapsed":time.monotonic()-start,
             "limits":"Balanced diagnostic subset; finite LBFGS solution, GT used in oracle; not AP or deployable prediction."}
    for group in ("native_failure","native_success"):
        rr=[r for r in rows if r["group"]==group]
        summary["groups"][group]=dict(n=len(rr),
            mean_iou_before=float(np.mean([r["final_iou_before"] for r in rr])),
            mean_iou_after=float(np.mean([r["final_iou_after"] for r in rr])),
            mask75_before=sum(r["final_iou_before"]>=.75 for r in rr),
            mask75_after=sum(r["final_iou_after"]>=.75 for r in rr),
            repairs=sum(r["final_iou_before"]<.75<=r["final_iou_after"] for r in rr),
            damages=sum(r["final_iou_after"]<.75<=r["final_iou_before"] for r in rr),
            mean_loss_gap=float(np.mean([r["initial_loss"]-r["oracle_loss"] for r in rr])))
    (a.out/"ROWS.json").write_text(json.dumps(rows,indent=2))
    torch.save(values,a.out/"oracle_coefficients.pt")
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2))
    (a.out/"COMPLETE.json").write_text(json.dumps({"instances":len(rows),"pairs":len(controls)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for key in ("bank","annotations","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--each-group",type=int,default=100)
    main(p.parse_args())
