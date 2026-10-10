"""Stage 1: export per-arm native one-to-one coefficients for the dev panel.

Reuses the verified geometry of experiments/coefficient_predictability_20260924/scripts/native_bank.py
and the overlay insertion of experiments/acd_native_coefficient_20261006/scripts/train_acd.py.

Arms:
  native   - official FP32 coefficients (no overlay)
  baseline - ACD coefficient-only BCE 3-epoch EMA overlay (same-budget capacity control)
  acd      - ACD action-guided auxiliary 3-epoch EMA overlay (oracle/action target)

For every dev image: official LetterBox(640) forward, one-to-one decode positions, GT association
by the native_bank rule (Hungarian, same-class box IoU >= 0.5, repeated GT allowed - diagnostic
association, not COCO matching). Per candidate: identity tuple (branch, level, raw_id, grid_y,
grid_x, stride, input_shape), predicted box, GT-box ROI (prototype crop + GT pixels), native decode
IoU per arm, coverage, box-FPR, GT-box AUC, Mask75 flag.

Outputs (Run directory):
  ENVIRONMENT.json / DATA_RECEIPT.json / PROGRESS.json / COMPLETE.json / SUMMARY.json
  images/<iid>.pt          shared per-image object (proto, boxes, top-k, levels, rows with ROIs)
  arms/<arm>/<iid>.json    per-arm coefficients for row/top raw ids + per-row metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_vendor_parser = argparse.ArgumentParser(add_help=False)
_vendor_parser.add_argument("--vendor", type=Path)
_vendor_args, _ = _vendor_parser.parse_known_args()
if _vendor_args.vendor:
    _vendor = _vendor_args.vendor.resolve()
    if not (_vendor / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"invalid vendor root: {_vendor}")
    sys.path.insert(0, str(_vendor))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou

if ultralytics.__version__ != "8.4.100":
    raise RuntimeError(f"expected vendor ultralytics 8.4.100, got {ultralytics.__version__} from {ultralytics.__file__}")

EXPECTED_TRAINABLE = sorted(
    [f"model.23.{b}.{l}.{i}.{w}" for b in ("cv4", "one2one_cv4") for l in range(3)
     for i, w in ((0, "conv.weight"), (1, "conv.weight"), (2, "weight"), (2, "bias"))]
)
LEVEL_COUNTS = {0: 6400, 1: 1600, 2: 400}  # 80x80, 40x40, 20x20 at 640 input
LEVEL_STRIDES = {0: 8, 1: 16, 2: 32}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_overlay(path: Path) -> dict:
    state = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        meta = {k: v for k, v in state.items() if k != "state_dict"}
        state = state["state_dict"]
    else:
        meta = {}
    for key in state:
        if not (key.startswith("model.23.cv4.") or key.startswith("model.23.one2one_cv4.")):
            raise ValueError(f"overlay key outside coefficient branches: {key}")
    return {"state_dict": state, "meta": meta}


def apply_overlay(model, overlay_path: Path) -> dict:
    """Insert overlay; return audit info including the exact differing-key set."""
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    overlay = load_overlay(overlay_path)
    missing, unexpected = model.load_state_dict(overlay["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"overlay contains unknown model keys: {unexpected}")
    # A partial overlay is expected: it must map into the model and change exactly the
    # 24 trainable coefficient parameters; every other tensor is verified unchanged below.
    after = model.state_dict()
    changed = sorted(k for k, v in after.items() if k in before and not torch.equal(before[k], v.detach()))
    if changed != EXPECTED_TRAINABLE:
        raise RuntimeError(f"overlay changed unexpected keys: {sorted(set(changed) ^ set(EXPECTED_TRAINABLE))}")
    del before
    return {"changed_parameters": changed, "changed_parameter_count": len(changed),
            "overlay_kind": overlay["meta"].get("kind"),
            "base_weights_sha256": overlay["meta"].get("base_weights_sha256"),
            "audit_passed": overlay["meta"].get("audit_passed")}


def decode_metrics(c: torch.Tensor, proto: torch.Tensor, box: torch.Tensor,
                   truth: torch.Tensor, support_gt: torch.Tensor, y_gt: torch.Tensor) -> dict:
    """Predicted-box decode metrics plus GT-box ROI AUC. All on the 160 letterbox grid, FP32."""
    z = (c @ proto.flatten(1)).reshape(160, 160)
    cropped = ops.crop_mask(z[None].clone(), box[None] / 4)[0] > 0
    inter = int((cropped & truth).sum())
    union = int((cropped | truth).sum())
    gt_area = int(truth.sum())
    iou = inter / max(union, 1)
    coverage = inter / max(gt_area, 1)
    fpr = int((cropped & ~truth).sum()) / max(gt_area, 1)
    logits = (proto[:, support_gt].T.contiguous() @ c)
    pos, neg = logits[y_gt > .5], logits[y_gt <= .5]
    if len(pos) and len(neg):
        auc = float((pos[:, None] > neg[None, :]).float().mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).float().mean())
    else:
        auc = None
    return dict(iou=float(iou), coverage=float(coverage), fpr=fpr, auc=auc, mask75=int(iou >= .75))


def main(a):
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "images").mkdir(exist_ok=True)
    for arm in a.arms:
        (a.out / "arms" / arm).mkdir(parents=True, exist_ok=True)

    dev_ids = []
    for token in a.dev_list.read_text().split():
        text = token.strip()
        if not text:
            continue
        stem = Path(text).stem if not text.isdigit() else text
        dev_ids.append(int(stem))
    if a.limit:
        dev_ids = dev_ids[: a.limit]
    coco = COCO(str(a.data / "annotations" / "instances_train2017.json"))
    cat_ids = sorted(coco.cats)

    write_json(a.out / "DATA_RECEIPT.json", {
        "weights": {"path": str(a.weights), "sha256": file_hash(a.weights)},
        "dev_list": {"path": str(a.dev_list), "sha256": file_hash(a.dev_list), "count": len(dev_ids)},
        "annotations": {"path": str(a.data / "annotations" / "instances_train2017.json"),
                        "sha256": file_hash(a.data / "annotations" / "instances_train2017.json")},
        "overlays": {arm: {"path": str(p), "sha256": file_hash(p)} for arm, p in
                     (("baseline", a.overlay_baseline), ("acd", a.overlay_acd)) if p},
        "scope": "197-image historical dev panel (subset when --limit); reused research images, not a blind test",
    })
    write_json(a.out / "ENVIRONMENT.json", {
        "python": sys.executable, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "ultralytics_path": ultralytics.__file__, "cuda": torch.cuda.get_device_name(0),
        "tf32": False, "input_shape": [640, 640],
    })

    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    official_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    transform = LetterBox((640, 640), auto=False, stride=32)
    level_offsets = {l: sum(LEVEL_COUNTS[k] for k in range(l)) for l in LEVEL_COUNTS}
    start = time.monotonic()
    total_rows = 0

    for arm in a.arms:
        model.load_state_dict(official_state, strict=True)
        audit = {"arm": arm, "overlay": None}
        if arm == "baseline" and a.overlay_baseline:
            audit["overlay"] = apply_overlay(model, a.overlay_baseline)
        elif arm == "acd" and a.overlay_acd:
            audit["overlay"] = apply_overlay(model, a.overlay_acd)
        write_json(a.out / "arms" / arm / "ARM_AUDIT.json", audit)

        for position, iid in enumerate(dev_ids, 1):
            im = cv2.imread(str(a.data / "images" / "train2017" / coco.imgs[iid]["file_name"]))
            if im is None:
                raise FileNotFoundError(iid)
            shaped = transform(image=im)
            tensor = torch.from_numpy(np.ascontiguousarray(shaped[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
            with torch.no_grad():
                _, raw = model(tensor)
            pred = raw["one2one"]
            proto = pred["proto"][0]
            coeff = pred["mask_coefficient"][0].T
            boxes = head._get_decode_boxes(pred)[0].T
            scores = pred["scores"][0]
            top_score, top_cl, top_rid = head.get_topk_index(pred["scores"].permute(0, 2, 1).sigmoid(), 300)
            levels = torch.cat([torch.full((LEVEL_COUNTS[l],), l, dtype=torch.long, device="cuda")
                                for l in range(3)])
            grids = {l: (int(np.sqrt(LEVEL_COUNTS[l])), int(np.sqrt(LEVEL_COUNTS[l]))) for l in range(3)}

            anns = [r for r in coco.imgToAnns[iid] if not r.get("iscrowd", 0) and not r.get("ignore", 0)]
            oh, ow = im.shape[:2]
            gain = min(640 / oh, 640 / ow)
            nh, nw = round(oh * gain), round(ow * gain)
            top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
            rows, shared_rows = [], []
            if anns:
                gt_boxes = torch.tensor([r["bbox"] for r in anns], device="cuda", dtype=torch.float32)
                gt_boxes[:, 2:] += gt_boxes[:, :2]
                original_boxes = ops.scale_boxes((640, 640), boxes.clone(), (oh, ow))
                overlaps = box_iou(gt_boxes, original_boxes).cpu().numpy()
                benefit = (overlaps >= .5) * (1 + overlaps / (len(anns) + 1))
                gi, pi = linear_sum_assignment(-np.concatenate([benefit, np.zeros((len(anns), len(anns)))], axis=1))
                for g, raw_id in zip(gi, pi):
                    if raw_id >= len(coeff) or overlaps[g, raw_id] < .5:
                        continue
                    ann = anns[g]
                    gt = torch.from_numpy(coco.annToMask(ann)).cuda().float()[None, None]
                    padded = F.pad(F.interpolate(gt, (nh, nw), mode="nearest"),
                                   (left, 640 - nw - left, top, 640 - nh - top))
                    native_gt = F.interpolate(padded, (160, 160), mode="nearest")[0, 0]
                    box = gt_boxes[g] * gain + torch.tensor([left, top, left, top], device="cuda")
                    native_box = box / 4
                    xx = torch.arange(160, device="cuda")[None, :]
                    yy = torch.arange(160, device="cuda")[:, None]
                    support = ((xx >= native_box[0]) & (xx < native_box[2])
                               & (yy >= native_box[1]) & (yy < native_box[3]))
                    if not support.any():
                        continue
                    c = coeff[raw_id]
                    truth = native_gt > .5
                    metrics = decode_metrics(c, proto, boxes[raw_id], truth, support, native_gt[support])
                    level = int(levels[raw_id])
                    level_id = int(raw_id) - level_offsets[level]
                    shared_rows.append(dict(
                        annotation_id=ann["id"], area=ann["area"],
                        fill=ann["area"] / max(ann["bbox"][2] * ann["bbox"][3], 1e-8),
                        level=level, raw_id=int(raw_id), level_id=level_id,
                        grid_y=level_id // grids[level][0], grid_x=level_id % grids[level][0],
                        stride=LEVEL_STRIDES[level], branch="one2one",
                        box_iou=float(overlaps[g, raw_id]),
                        gt_class_score=float(scores[cat_ids.index(ann["category_id"]), raw_id].sigmoid()),
                        p=proto[:, support].T.contiguous().cpu(),
                        y=native_gt[support].cpu(), area_px=int(support.sum()),
                        truth_px=int(truth.sum()),
                        pred_box=boxes[raw_id].cpu(), gt_native_box=native_box.cpu(),
                        truth160=truth.cpu(),
                    ))
                    rows.append(dict(annotation_id=ann["id"], raw_id=int(raw_id), level=level, **metrics))

            shared_ids = sorted({r["raw_id"] for r in shared_rows} | {int(i) for i in top_rid.flatten().tolist()})
            if arm == "native":
                torch.save(dict(proto=proto.cpu(), boxes=boxes.cpu(), levels=levels.cpu(),
                                top_ids=top_rid.cpu(), top_scores=top_score.cpu(), top_classes=top_cl.cpu(),
                                original_shape=(oh, ow), input_shape=(640, 640), gain=gain,
                                left=left, top=top, rows=shared_rows),
                           a.out / "images" / f"{iid:012d}.pt")
            torch.save(dict(arm=arm, ids=shared_ids,
                            coeff=coeff[torch.tensor(shared_ids, device="cuda", dtype=torch.long)].cpu(),
                            rows=rows),
                       a.out / "arms" / arm / f"{iid:012d}.pt")
            total_rows += len(rows)
            if position % 20 == 0 or position == len(dev_ids):
                state = dict(arm=arm, images=position, total=len(dev_ids), rows=total_rows,
                             elapsed=round(time.monotonic() - start, 3))
                print(json.dumps(state), flush=True)
                write_json(a.out / "arms" / arm / "PROGRESS.json", state)

    summary = dict(arms=list(a.arms), dev_images=len(dev_ids), total_rows=total_rows)
    write_json(a.out / "SUMMARY.json", summary)
    write_json(a.out / "COMPLETE.json", {"status": "complete", "summary": summary})
    print("EXPORT_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("vendor", "weights", "data", "dev_list", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--overlay_baseline", type=Path)
    p.add_argument("--overlay_acd", type=Path)
    p.add_argument("--arms", nargs="+", default=["native", "baseline", "acd"])
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
