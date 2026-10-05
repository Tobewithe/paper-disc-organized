"""Measure same-scale coefficient-head gradient conflict near different GT instances."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.loss import v8SegmentationLoss


def small_low_failure(row, ann):
    x, y, w, h = ann["bbox"]
    return (row["area_group"] == "small" and row["geometry_state"] == "box_good_mask_unavailable"
            and w * h > 0 and ann["area"] / (w * h) < .5)


def box_proximal(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    return min(ax + aw, bx + bw) + 5 > max(ax, bx) and min(ay + ah, by + bh) + 5 > max(ay, by)


def cosine(a, b):
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-12 else None


def mean(rows, key):
    vals = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows):
    return {"pairs": len(rows), "images": len({r["image_id"] for r in rows}),
            "endpoints": len({(r["image_id"], r["left_annotation_id"]) for r in rows}),
            "mean_weight_grad_cos": mean(rows, "weight_grad_cos"),
            "negative_weight_grad_fraction": float(np.mean([r["weight_grad_cos"] < 0 for r in rows])) if rows else None,
            "mean_bias_grad_cos": mean(rows, "bias_grad_cos"),
            "mean_feature_cos": mean(rows, "feature_cos"),
            "mean_coefficient_cos": mean(rows, "coefficient_cos"),
            "mean_left_to_right_leak_pixels": mean(rows, "left_to_right_leak_pixels"),
            "leak_positive_fraction": float(np.mean([r["left_to_right_leak_pixels"] > 0 for r in rows])) if rows else None}


def main(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    coco = COCO(str(args.annotations))
    by_image = defaultdict(list)
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("bbox_coco75") and row.get("bbox_raw_id") is not None:
                row["is_failure"] = small_low_failure(row, coco.anns[row["annotation_id"]])
                by_image[row["image_id"]].append(row)
    possible = []
    for image_id, rows in by_image.items():
        failure_rows = [r for r in rows if r["is_failure"]]
        if not failure_rows:
            continue
        for left in failure_rows:
            a = coco.anns[left["annotation_id"]]
            others = [coco.anns[r["annotation_id"]] for r in rows
                      if r["annotation_id"] != left["annotation_id"] and r["category_id"] == left["category_id"]]
            if any(box_proximal(a, b) for b in others) and any(not box_proximal(a, b) for b in others):
                possible.append(image_id)
                break
    rng = random.Random(20260923)
    image_ids = sorted(rng.sample(sorted(possible), min(args.images, len(possible))))
    model = YOLO(str(args.weights)).model.cuda().float().eval()
    head = model.model[-1]
    assert head.end2end
    captured = {}
    hooks = []
    for level, branch in enumerate(head.one2one_cv4):
        hooks.append(branch[-1].register_forward_pre_hook(
            lambda module, inputs, level=level: captured.__setitem__(level, inputs[0].detach())))
    transform = LetterBox((640, 640), auto=True, stride=32)
    pairs = []
    counts = Counter()
    for number, image_id in enumerate(image_ids, 1):
        rows = by_image[image_id]
        original = cv2.imread(str(args.images_dir / coco.imgs[image_id]["file_name"]))
        if original is None:
            raise FileNotFoundError(image_id)
        shaped = transform.apply_image({"img": original}, transform.get_params({"img": original}))["img"]
        inp = torch.from_numpy(np.ascontiguousarray(shaped[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
        with torch.no_grad():
            _, raw = model(inp)
        prediction = raw["one2one"]
        proto = prediction["proto"][0].detach()
        coefficient = prediction["mask_coefficient"][0].T.detach()
        features = torch.cat([captured[level][0].flatten(1).T for level in range(3)])
        levels = torch.cat([torch.full((captured[level].shape[2] * captured[level].shape[3],), level,
                                        device="cuda", dtype=torch.long) for level in range(3)])
        with np.load(args.archive / "images" / f"{image_id:012d}.npz") as archive:
            indices = torch.tensor([r["bbox_raw_id"] for r in rows], device="cuda")
            identity_error = float((coefficient[indices].cpu() -
                                    torch.as_tensor(archive["coefficients"][:, indices.cpu().numpy()].T)).abs().max())
            proto_error = float((proto.cpu() - torch.as_tensor(archive["proto"])).abs().max())
        assert identity_error < 1e-5 and proto_error < 1e-5, (image_id, identity_error, proto_error)
        counts["verified_images"] += 1
        raw_owners = Counter(r["bbox_raw_id"] for r in rows)
        unique = [r for r in rows if raw_owners[r["bbox_raw_id"]] == 1]
        counts["ambiguous_raw_excluded"] += len(rows) - len(unique)
        if len(unique) < 2:
            continue
        ih, iw = map(int, inp.shape[-2:])
        oh, ow = original.shape[:2]
        ph, pw = proto.shape[-2:]
        gain = min(ih / oh, iw / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top, left = round((ih - nh) / 2 - .1), round((iw - nw) / 2 - .1)
        targets = []
        boxes = []
        masks = {}
        for row in unique:
            ann = coco.anns[row["annotation_id"]]
            mask = coco.annToMask(ann).astype(bool)
            masks[row["annotation_id"]] = mask
            gt = torch.as_tensor(mask, device="cuda").float()[None, None]
            input_gt = F.pad(F.interpolate(gt, (nh, nw), mode="nearest"),
                             (left, iw - nw - left, top, ih - nh - top))
            targets.append(F.interpolate(input_gt, (ph, pw), mode="nearest")[0, 0])
            x, y, w, h = ann["bbox"]
            boxes.append([(x * gain + left) * pw / iw, (y * gain + top) * ph / ih,
                          ((x + w) * gain + left) * pw / iw, ((y + h) * gain + top) * ph / ih])
        target = torch.stack(targets)
        box = torch.tensor(boxes, device="cuda", dtype=torch.float32)
        area = ((box[:, 2] - box[:, 0]) / pw * (box[:, 3] - box[:, 1]) / ph).clamp_min(1e-8)
        raw_ids = torch.tensor([r["bbox_raw_id"] for r in unique], device="cuda")
        coeff = coefficient[raw_ids].clone().detach().requires_grad_(True)
        loss = v8SegmentationLoss.single_mask_loss(target, coeff, proto, box, area)
        grads = torch.autograd.grad(loss, coeff)[0].detach().cpu().numpy()
        hidden = features[raw_ids].detach().cpu().numpy()
        coeff_numpy = coeff.detach().cpu().numpy()
        level_numpy = levels[raw_ids].detach().cpu().numpy()
        normal = json.loads((args.archive / "images" / f"{image_id:012d}.json").read_text())
        predictions = {(p["raw_id"], p["category_id"]): p for p in normal["predictions"]}
        decoded = {}
        for row in unique:
            key = (row["bbox_raw_id"], row["category_id"])
            if key in predictions:
                decoded[row["annotation_id"]] = mask_utils.decode(predictions[key]["segmentation"]).astype(bool)
        dilated = {r["annotation_id"]: cv2.dilate(masks[r["annotation_id"]].astype(np.uint8),
                                                   np.ones((11, 11), np.uint8)).astype(bool) for r in unique}
        for i, arow in enumerate(unique):
            if not arow["is_failure"] or arow["annotation_id"] not in decoded:
                continue
            a = coco.anns[arow["annotation_id"]]
            ma = masks[arow["annotation_id"]]
            pred = decoded[arow["annotation_id"]]
            extra = pred & ~ma
            extra_count = int(extra.sum())
            for j, brow in enumerate(unique):
                if i == j or arow["category_id"] != brow["category_id"] or level_numpy[i] != level_numpy[j]:
                    continue
                b = coco.anns[brow["annotation_id"]]
                mb = masks[brow["annotation_id"]]
                adjacent = bool(np.any(dilated[arow["annotation_id"]] & mb))
                size_ratio = max(a["area"], b["area"]) / max(min(a["area"], b["area"]), 1e-8)
                gcos = cosine(grads[i], grads[j])
                hcos = cosine(hidden[i], hidden[j])
                leak = int(np.count_nonzero(extra & mb))
                pairs.append({"image_id": image_id, "left_annotation_id": arow["annotation_id"],
                              "right_annotation_id": brow["annotation_id"], "adjacent5": adjacent,
                              "area_ratio": size_ratio, "same_size2": size_ratio <= 2,
                              "level": int(level_numpy[i]), "feature_cos": hcos,
                              "coefficient_cos": cosine(coeff_numpy[i], coeff_numpy[j]),
                              "bias_grad_cos": gcos,
                              "weight_grad_cos": gcos * hcos if gcos is not None and hcos is not None else None,
                              "left_to_right_leak_pixels": leak,
                              "left_to_right_leak_fraction_extra": leak / max(extra_count, 1),
                              "left_mask_extra_pixels": extra_count,
                              "left_is_failure": True})
        if number % 25 == 0 or number == len(image_ids):
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"images": number, "total": len(image_ids),
                                                                  "pairs": len(pairs)}), encoding="utf-8")
            print(json.dumps({"images": number, "total": len(image_ids), "pairs": len(pairs)}), flush=True)
    for hook in hooks:
        hook.remove()
    groups = {"adjacent": [r for r in pairs if r["adjacent5"]],
              "nonadjacent": [r for r in pairs if not r["adjacent5"]],
              "adjacent_size2": [r for r in pairs if r["adjacent5"] and r["same_size2"]],
              "nonadjacent_size2": [r for r in pairs if not r["adjacent5"] and r["same_size2"]]}
    endpoint_groups = defaultdict(lambda: {"adjacent": [], "nonadjacent": []})
    for row in pairs:
        endpoint_groups[(row["image_id"], row["left_annotation_id"])][
            "adjacent" if row["adjacent5"] else "nonadjacent"].append(row)
    paired = []
    for (image_id, annotation_id), group in endpoint_groups.items():
        if not group["adjacent"] or not group["nonadjacent"]:
            continue
        paired.append({"image_id": image_id, "annotation_id": annotation_id,
                       "delta_weight_grad_cos": mean(group["adjacent"], "weight_grad_cos") -
                                                mean(group["nonadjacent"], "weight_grad_cos"),
                       "delta_bias_grad_cos": mean(group["adjacent"], "bias_grad_cos") -
                                              mean(group["nonadjacent"], "bias_grad_cos")})
    image_deltas = defaultdict(list)
    for row in paired:
        image_deltas[row["image_id"]].append(row["delta_weight_grad_cos"])
    per_image_delta = np.asarray([np.mean(v) for v in image_deltas.values()], dtype=float)
    ci = None
    if len(per_image_delta):
        boot_rng = np.random.default_rng(20260923)
        samples = boot_rng.choice(per_image_delta, (2000, len(per_image_delta)), replace=True).mean(1)
        ci = list(map(float, np.quantile(samples, [.025, .975])))
    summary = {"selected_images": len(image_ids), "candidate_images": len(possible),
               "selection_image_ids": image_ids, "counts": dict(counts),
               "groups": {k: summarize(v) for k, v in groups.items()},
               "paired_failure_endpoints": len(paired), "paired_images": len(per_image_delta),
               "paired_image_mean_adjacent_minus_nonadjacent_weight_grad_cos":
                   float(per_image_delta.mean()) if len(per_image_delta) else None,
               "paired_image_bootstrap95": ci,
               "limitations": "Current output, fixed proto and last coefficient readout only; paired image bootstrap does not account for model-training variability."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "PAIRS.jsonl").write_text("\n".join(json.dumps(r) for r in pairs) + "\n", encoding="utf-8")
    (args.out / "PAIRED_ENDPOINTS.jsonl").write_text("\n".join(json.dumps(r) for r in paired) + "\n", encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"images": len(image_ids), "pairs": len(pairs)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("joined", "archive", "annotations", "images_dir", "weights", "out"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    parser.add_argument("--images", type=int, default=240)
    main(parser.parse_args())
