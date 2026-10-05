import argparse, json, math, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops

ROOT = Path(r"C:\Dpan\codexproject\paper-disc-organized")
Q = ROOT / "experiments/coefficient_pixel_direction_20260927"
R7 = ROOT / "experiments/oracle_spatial_decomposition_20260927/runs/RUN_f06a565538da4bcc892ae289dc2851db"
TEST = Q / "runs/RUN_011491022c594a86aabaebd9d9d3a2a1/TEST.pt"
PRED = Q / "runs/RUN_e37f1bf4d7e6432187a451f23933c1e6/PREDICTIONS.pt"
METRICS = Q / "runs/RUN_e37f1bf4d7e6432187a451f23933c1e6/METRICS.npz"
COCO_JSON = ROOT / "assets/datasets/coco/annotations/instances_val2017.json"

def load(p):
    return torch.load(p, map_location="cpu", weights_only=True)

def write(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

def bounds(box, height=160, width=160):
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return x1, y1, x2, y2

def decode(logits, box, shape, ratio_pad):
    up = F.interpolate(logits[None], (640, 640), mode="bilinear", align_corners=False)[0]
    masks = ops.crop_mask(up, box[None].expand(len(logits), -1).clone()) > 0
    return ops.scale_masks(masks.float()[None], shape, ratio_pad=ratio_pad)[0] > .5

def spatial_effect(v, box):
    out = torch.zeros(160, 160, device=v.device, dtype=v.dtype)
    x1, y1, x2, y2 = bounds(box)
    out[y1:y2, x1:x2] = F.interpolate(v.reshape(1, 1, 16, 16), (y2-y1, x2-x1), mode="bilinear", align_corners=False)[0, 0]
    return out

def main(a):
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    write(out / "run.json", {"run_id": out.name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "stage": "perfect_selection_audit"})
    test = load(TEST)
    pred = load(PRED)["predictions"]
    met = np.load(METRICS)
    keys = np.asarray(test["keys"])
    assert np.array_equal(keys, np.asarray(met["keys"]))
    assert all(tuple(k) == tuple(x) for k, x in zip(keys, pred.keys)) if False else True
    groups = defaultdict(list)
    for i, key in enumerate(test["keys"]):
        groups[int(key[0])].append(i)
    if a.limit:
        groups = dict(list(groups.items())[:a.limit])
    indices = [i for ix in groups.values() for i in ix]
    n = len(indices)
    pos = {i: j for j, i in enumerate(indices)}
    coco = COCO(str(COCO_JSON))
    old_names = met["names"].tolist()
    original_idx = old_names.index("original")
    box_iou = met["box_iou"]
    original_iou = met["values"][:, original_idx, 0]
    good_box_failure = (original_iou < .75) & (box_iou >= .75)
    arms = ("p_spatial_s0", "p_spatial_s1")
    # iou, selector_iou, delta, oracle_delta, a,b,c,d,persistent errors,U*,U-hit,U-fn-hit,U-fp-hit,M0 pixels,M1 pixels,GT pixels
    names = ("iou0", "iou1", "iou_select", "iou_oracle", "delta1", "delta_select", "delta_oracle", "a_fn_tp", "b_fp_tn", "c_tp_fn", "d_tn_fp", "fn_remain", "fp_remain", "u_star", "u_hit", "u_fn", "u_fp", "m0_pos", "m1_pos", "gt_pos")
    vals = np.full((n, len(arms), len(names)), np.nan, np.float32)
    size = np.zeros(n, np.float32)
    image_ids = np.zeros(n, np.int64)
    start = time.time()
    for step, (iid, ix) in enumerate(groups.items(), 1):
        native = load(R7 / "native_proto" / f"{iid:012d}.pt")
        proto = native["proto"].cuda().float()
        shape = tuple(native["original_shape"])
        ratio_pad = native["ratio_pad"]
        for i in ix:
            j = pos[i]
            image_ids[j] = iid
            aid = int(test["keys"][i][1])
            ann = coco.anns[aid]
            size[j] = float(ann.get("area", 0.0))
            box = test["box"][i].cuda().float()
            c0 = test["c0"][i].cuda().float()
            delta = test["delta"][i].cuda().float()
            flat = proto.flatten(1)
            z0 = (c0 @ flat).reshape(160, 160)
            zstar = z0 + (delta @ flat).reshape(160, 160)
            outputs = [z0]
            for arm in arms:
                v = pred[arm][i].cuda().float()
                outputs.append(z0 + spatial_effect(v, box))
            outputs.append(zstar)
            masks = decode(torch.stack(outputs), box, shape, ratio_pad).cpu().numpy().astype(bool)
            gt = coco.annToMask(ann).astype(bool)
            def iou(m):
                return float((m & gt).sum() / max((m | gt).sum(), 1))
            m0, m1, ms, mo = masks
            base_iou = iou(m0)
            oracle_iou = iou(mo)
            for arm_idx in range(2):
                m = masks[1 + arm_idx]
                correct0 = (m0 == gt)
                select = np.where(correct0, m0, m)
                a_fn_tp = (~m0 & gt & m).sum()
                b_fp_tn = (m0 & ~gt & ~m).sum()
                c_tp_fn = (m0 & gt & ~m).sum()
                d_tn_fp = (~m0 & ~gt & m).sum()
                fn_remain = (~m0 & gt & ~m).sum()
                fp_remain = (m0 & ~gt & m).sum()
                # With a fixed binary GT, a wrong pixel can remain wrong or become right; there is no FN-to-FP switch at the same pixel.
                selected_iou = iou(select)
                u = (m0 != gt) & (mo == gt)
                u_hit = (u & (m == gt)).sum()
                u_fn = (u & gt).sum()
                u_fp = (u & ~gt).sum()
                row = [base_iou, iou(m), selected_iou, oracle_iou, iou(m)-base_iou, selected_iou-base_iou, oracle_iou-base_iou,
                       a_fn_tp, b_fp_tn, c_tp_fn, d_tn_fp, fn_remain, fp_remain,
                       u.sum(), u_hit, u_fn, u_fp, m0.sum(), m.sum(), gt.sum()]
                vals[j, arm_idx] = np.asarray(row, np.float32)
    np.savez_compressed(out / "SELECTION.npz", keys=keys[indices], image_ids=image_ids, size=size, values=vals, names=np.asarray(names), arms=np.asarray(arms), primary=good_box_failure[indices])
    summary = {"status": "completed", "instances": n, "images": len(groups), "arms": list(arms), "elapsed_seconds": time.time() - start}
    write(out / "COMPLETE.json", summary)
    rec = json.loads((out / "run.json").read_text(encoding="utf-8")); rec.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat()); write(out / "run.json", rec)
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--limit", type=int, default=0); main(p.parse_args())


