"""Cache a fixed raw GT mapping and native prototype pixels for coefficient probes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
import ultralytics


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "images").mkdir(exist_ok=True)
    split = json.loads(a.split.read_text())
    torch.manual_seed(0)
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    captured, inputs = {}, {}
    handles = []
    for level, branch in enumerate(head.one2one_cv4):
        handles.append(branch[-1].register_forward_pre_hook(
            lambda _, x, level=level: captured.__setitem__(level, x[0].detach())))
        handles.append(branch.register_forward_pre_hook(
            lambda _, x, level=level: inputs.__setitem__(level, x[0].detach())))
    transform = LetterBox((640, 640), auto=False, stride=32)
    coco = {s: COCO(str(a.data / "annotations" / f"instances_{s}2017.json")) for s in ("train", "val")}
    banks = {s: [] for s in ("fit", "dev", "val")}
    counts = Counter()
    start = time.monotonic()
    for group in ("fit", "dev", "val"):
        source_split = "val" if group == "val" else "train"
        source = coco[source_split]
        cat_ids = sorted(source.cats)
        cat_to_class = {c: j for j, c in enumerate(cat_ids)}
        for position, iid in enumerate(split[group], 1):
            im = cv2.imread(str(a.data / "images" / f"{source_split}2017" / source.imgs[iid]["file_name"]))
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
            h = torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            levels = torch.cat([torch.full((captured[l].shape[2] * captured[l].shape[3],), l,
                                          dtype=torch.long, device="cuda") for l in range(3)])
            raw_sizes = [captured[l].shape[-2:] for l in range(3)]
            recovered = torch.cat([head.one2one_cv4[l][-1](captured[l])[0].flatten(1).T for l in range(3)])
            assert float((recovered - coeff).abs().max()) < 1e-5
            assert proto.shape == (32, 160, 160) and len(coeff) == 8400
            phi = F.adaptive_avg_pool2d(proto, (4, 4)).flatten().cpu()
            anns = [r for r in source.imgToAnns[iid] if not r.get("iscrowd", 0) and not r.get("ignore", 0)]
            oh, ow = im.shape[:2]
            gain = min(640 / oh, 640 / ow)
            nh, nw = round(oh * gain), round(ow * gain)
            top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
            image_rows = []
            if anns:
                gt_boxes = torch.tensor([r["bbox"] for r in anns], device="cuda", dtype=torch.float32)
                gt_boxes[:, 2:] += gt_boxes[:, :2]
                original_boxes = ops.scale_boxes((640, 640), boxes.clone(), (oh, ow))
                overlaps = box_iou(gt_boxes, original_boxes).cpu().numpy()
                # One valid extra match outweighs the sum of all IoU tie-breakers.
                benefit = (overlaps >= .5) * (1 + overlaps / (len(anns) + 1))
                gi, pi = linear_sum_assignment(-np.concatenate([benefit, np.zeros((len(anns), len(anns)))], axis=1))
                for g, raw_id in zip(gi, pi):
                    if raw_id >= len(coeff) or overlaps[g, raw_id] < .5:
                        continue
                    ann = anns[g]
                    gt = torch.from_numpy(source.annToMask(ann)).cuda().float()[None, None]
                    padded = F.pad(F.interpolate(gt, (nh, nw), mode="nearest"), (left, 640 - nw - left, top, 640 - nh - top))
                    native_gt = F.interpolate(padded, (160, 160), mode="nearest")[0, 0]
                    box = gt_boxes[g] * gain + torch.tensor([left, top, left, top], device="cuda")
                    native_box = box / 4
                    xx = torch.arange(160, device="cuda")[None, :]
                    yy = torch.arange(160, device="cuda")[:, None]
                    support = (xx >= native_box[0]) & (xx < native_box[2]) & (yy >= native_box[1]) & (yy < native_box[3])
                    if not support.any():
                        counts["empty_native_gt_crop"] += 1
                        continue
                    loss_p = proto[:, support].T.contiguous()
                    y = native_gt[support]
                    c = coeff[raw_id]
                    z = (c @ proto.flatten(1)).reshape(160, 160)
                    cropped = ops.crop_mask(z[None].clone(), boxes[raw_id][None] / 4)[0] > 0
                    truth = native_gt > .5
                    native_iou = float((cropped & truth).sum() / (cropped | truth).sum().clamp_min(1))
                    level = int(levels[raw_id])
                    offset = sum(int(np.prod(shape)) for shape in raw_sizes[:level])
                    level_id = raw_id - offset
                    z_in = inputs[level][0].flatten(1)[:, level_id].cpu()
                    gt_area_native = float(((native_box[2:] - native_box[:2]).clamp_min(0)).prod())
                    meta = dict(image_id=iid, annotation_id=ann["id"], raw_id=int(raw_id), level=level,
                                box_iou=float(overlaps[g, raw_id]), native_iou=native_iou,
                                area=ann["area"], fill=ann["area"] / max(ann["bbox"][2] * ann["bbox"][3], 1e-8),
                                gt_class_score=float(pred["scores"][0, cat_to_class[ann["category_id"]], raw_id].sigmoid()),
                                native_gt_box_area=gt_area_native)
                    row = dict(meta=meta, h=h[raw_id].cpu(), c=c.cpu(), phi=phi,
                               p=loss_p.cpu(), y=y.cpu(), z_in=z_in,
                               factor=len(y) / max(gt_area_native, 1e-8))
                    banks[group].append(row)
                    image_rows.append(meta)
            counts[group + "_gt"] += len(anns)
            counts[group + "_mapped"] += len(image_rows)
            score, cl, rid = head.get_topk_index(pred["scores"].permute(0, 2, 1).sigmoid(), 300)
            torch.save(dict(proto=proto.cpu(), coeff=coeff.cpu(), boxes=boxes.cpu(), h=h.cpu(), levels=levels.cpu(),
                            class_logits=pred["scores"][0].cpu(), top_ids=rid.cpu(), top_scores=score.cpu(), top_classes=cl.cpu(),
                            original_shape=(oh, ow), input_shape=(640, 640), gain=gain, left=left, top=top,
                            rows=image_rows, source_split=source_split), a.out / "images" / f"{iid:012d}.pt")
            if position % 25 == 0 or position == len(split[group]):
                state = dict(group=group, images=position, total=len(split[group]), mapped=len(banks[group]), elapsed=time.monotonic()-start)
                print(json.dumps(state), flush=True)
                save_json(a.out / "PROGRESS.json", state)
        torch.save(banks[group], a.out / f"{group}.pt")
        banks[group] = []
    metadata = dict(counts=counts, software=dict(torch=torch.__version__, ultralytics=ultralytics.__version__),
                    checkpoint_train_args={k: wrapper.ckpt.get("train_args", {}).get(k) for k in ("mask_ratio", "overlap_mask", "imgsz", "optimizer", "epochs")},
                    input_shape=[640, 640], proto_shape=[32,160,160], tf32=False,
                    mapping="max cardinality then max IoU, raw score/class independent Box50 matching")
    save_json(a.out / "COMPLETE.json", metadata)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("data", "weights", "split", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
