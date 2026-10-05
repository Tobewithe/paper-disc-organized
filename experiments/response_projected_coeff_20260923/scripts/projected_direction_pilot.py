"""Project an existing spatial correction into each candidate's prototype span."""
import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import ops


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selected_images(joined, coco, limit):
    image_ids = set()
    with joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("geometry_state") != "box_good_mask_unavailable" or row["area_group"] != "small":
                continue
            a = coco.anns[row["annotation_id"]]
            if a["bbox"][2] * a["bbox"][3] > 0 and a["area"] / (a["bbox"][2] * a["bbox"][3]) < .5:
                image_ids.add(row["image_id"])
    rng = random.Random(20260923)
    return sorted(rng.sample(sorted(image_ids), min(limit, len(image_ids)))), len(image_ids)


def roi_correction(correction):
    t = (torch.arange(32, device=correction.device) + .5) / 32 * 2 - 1
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None].expand(len(correction), -1, -1, -1)
    return F.grid_sample(4 * correction.reshape(-1, 1, 4, 4), grid,
                         padding_mode="border", align_corners=True)[:, 0]


def coefficient_projection(proto_roi, original_roi, requested, guard_strength):
    p = proto_roi.flatten(2).transpose(1, 2).float()
    target = requested.flatten(1).float()
    gram = p.transpose(1, 2) @ p
    penalty = (original_roi.abs().flatten(1) > 1).float()
    protected = p.transpose(1, 2) @ (penalty[:, :, None] * p)
    ridge = .01 * gram.diagonal(dim1=1, dim2=2).sum(1) / p.shape[-1]
    eye = torch.eye(p.shape[-1], device=p.device)[None]
    matrix = gram + guard_strength * protected + ridge[:, None, None] * eye
    rhs = p.transpose(1, 2) @ target[:, :, None]
    result = torch.linalg.solve(matrix, rhs)[:, :, 0]
    assert bool(torch.isfinite(result).all())
    return result


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def matched_mask75(ev, coco, ids):
    threshold = int(np.argmin(np.abs(ev.params.iouThrs - .75)))
    good = set()
    for item in ev.evalImgs:
        if item is None or item["aRng"] != ev.params.areaRng[0]:
            continue
        good.update(int(g) for j, g in enumerate(item["gtIds"])
                    if not item["gtIgnore"][j] and item["gtMatches"][threshold, j] > 0)
    valid = {a["id"] for iid in ids for a in coco.imgToAnns[iid]
             if not a.get("iscrowd", 0) and not a.get("ignore", 0)}
    return good & valid


def evaluate(coco, image_ids, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(predictions), "segm")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}, matched_mask75(ev, coco, image_ids)


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.helper_dir))
    import component_seed_probe as helper
    import learn_refinement as original

    coco = COCO(str(args.annotations))
    image_ids, pool_size = selected_images(args.joined, coco, args.images)
    if args.trained_dir:
        arms = ("coeff", "coeff_guard", "local4", "project_0", "project_4")
        models = {}
        for arm in arms:
            mode = "coeff" if arm.startswith("coeff") else "local4"
            net = original.Refiner(mode).cuda().eval()
            net.load_state_dict(torch.load(args.trained_dir / f"{arm}_epoch8.pt", map_location="cuda",
                                           weights_only=False)["state_dict"])
            models[arm] = net
    else:
        local = original.Refiner("local4").cuda().eval()
        coefficient = original.Refiner("coeff").cuda().eval()
        local.load_state_dict(torch.load(args.local_checkpoint, map_location="cuda", weights_only=False)["state_dict"])
        coefficient.load_state_dict(torch.load(args.coeff_checkpoint, map_location="cuda", weights_only=False)["state_dict"])
    cats = sorted(coco.cats)
    variants = ["baseline_cached", "baseline"]
    if args.trained_dir:
        variants += [f"{arm}_{str(a).replace('.', '')}" for arm in arms for a in (.25, .5, 1.)]
    else:
        variants += ["spatial_05"]
        variants += [f"coeff_{str(a).replace('.', '')}" for a in (.25, .5, 1.)]
        variants += [f"project_{g}_{str(a).replace('.', '')}" for g in (0, 4) for a in (.25, .5, 1.)]
    predictions = {name: [] for name in variants}
    identity = {"images_checked": 0, "predictions_checked": 0, "missing_from_recomputed": 0,
                "extra_in_recomputed": 0, "changed_rle": 0, "max_changed_score": 0., "examples": []}
    for number, image_id in enumerate(image_ids, 1):
        with np.load(args.archive / "images" / f"{image_id:012d}.npz") as archive:
            proto = torch.as_tensor(archive["proto"], device="cuda")
            all_coeff = torch.as_tensor(archive["coefficients"].T, device="cuda")
            all_boxes = torch.as_tensor(archive["boxes_input"], device="cuda")
            shape = tuple(map(int, archive["input_shape"]))
            original_shape = tuple(map(int, archive["original_shape"]))
            top = archive["top_scores"] > .001
            indices = archive["top_ids"][top].astype(np.int64)
            scores = archive["top_scores"][top].astype(float)
            classes = archive["top_classes"][top].astype(int)
        scale = proto.square().mean((1, 2)).sqrt().clamp_min(.1)
        low_all = (all_coeff @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])
        masks_by_id = {name: {} for name in variants if name != "baseline_cached"}
        unique_ids = sorted(set(indices.tolist()))
        for first in range(0, len(unique_ids), 32):
            batch_ids = unique_ids[first:first + 32]
            ids = torch.as_tensor(batch_ids, device="cuda")
            boxes = all_boxes[ids]
            coeff = all_coeff[ids]
            x, base_roi, _ = original.features(proto, coeff, boxes, shape)
            with torch.no_grad():
                base_full = F.interpolate(low_all[ids][None], shape, mode="bilinear", align_corners=False)[0]
                logits = {"baseline": base_full}
                if args.trained_dir:
                    for arm, net in models.items():
                        output = net(x)
                        if arm.startswith("project_"):
                            strength = int(arm.split("_")[1])
                            direction = coefficient_projection(x[:, :32], base_roi, roi_correction(output), strength)
                            delta = F.interpolate(((direction / scale[None]) @ proto.flatten(1))
                                                  .reshape(-1, *proto.shape[-2:])[None], shape,
                                                  mode="bilinear", align_corners=False)[0]
                        for alpha in (.25, .5, 1.):
                            label = f"{arm}_{str(alpha).replace('.', '')}"
                            if arm.startswith("project_"):
                                logits[label] = base_full + alpha * delta
                            else:
                                mode = "coeff" if arm.startswith("coeff") else "local4"
                                logits[label] = helper.image_correct(base_full, proto, scale, boxes, shape,
                                                                     output * alpha, mode)
                else:
                    spatial_output = local(x)
                    coeff_output = coefficient(x)
                    proposal = roi_correction(spatial_output)
                    proto_roi = x[:, :32] * scale[None, :, None, None]
                    projected = {strength: coefficient_projection(proto_roi, base_roi, proposal, strength)
                                 for strength in (0, 4)}
                    delta_full = {strength: F.interpolate(
                        (direction @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])[None], shape,
                        mode="bilinear", align_corners=False)[0] for strength, direction in projected.items()}
                    logits["spatial_05"] = helper.image_correct(base_full, proto, scale, boxes, shape,
                                                                   spatial_output * .5, "local4")
                    for alpha in (.25, .5, 1.):
                        label = str(alpha).replace(".", "")
                        logits[f"coeff_{label}"] = helper.image_correct(base_full, proto, scale, boxes, shape,
                                                                         coeff_output * alpha, "coeff")
                        for strength in (0, 4):
                            logits[f"project_{strength}_{label}"] = base_full + alpha * delta_full[strength]
                for name, z in logits.items():
                    masks = ops.scale_masks(ops.crop_mask(z.clone(), boxes).gt(0).byte()[None], original_shape)[0]
                    for k, mask in enumerate(masks.cpu().numpy()):
                        if not mask.any():
                            continue
                        masks_by_id[name][batch_ids[k]] = encode(mask)
        variants_image = {name: [] for name in variants}
        for raw_id, score, category_index in zip(indices, scores, classes):
            for name in masks_by_id:
                rle = masks_by_id[name].get(int(raw_id))
                if rle is None:
                    continue
                variants_image[name].append({"image_id": image_id, "category_id": cats[category_index],
                                             "score": score, "raw_id": int(raw_id), "segmentation": rle})
        normal = json.loads((args.archive / "images" / f"{image_id:012d}.json").read_text())
        variants_image["baseline_cached"] = normal["predictions"]
        expected = {(p["raw_id"], p["category_id"]): p for p in normal["predictions"]}
        actual = {(p["raw_id"], p["category_id"]): p for p in variants_image["baseline"]}
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        changed = [key for key in expected.keys() & actual.keys()
                   if expected[key]["segmentation"] != actual[key]["segmentation"]]
        identity["images_checked"] += 1
        identity["predictions_checked"] += len(expected)
        identity["missing_from_recomputed"] += len(missing)
        identity["extra_in_recomputed"] += len(extra)
        identity["changed_rle"] += len(changed)
        for key in list(missing) + list(extra) + changed:
            source = expected.get(key, actual.get(key))
            identity["max_changed_score"] = max(identity["max_changed_score"], source["score"])
            if len(identity["examples"]) < 12:
                identity["examples"].append({"image_id": image_id, "raw_id": key[0],
                                             "category_id": key[1], "score": source["score"]})
        for name in variants:
            predictions[name].extend(variants_image[name])
        if number % 20 == 0 or number == len(image_ids):
            print(json.dumps({"images": number, "total": len(image_ids)}), flush=True)
            write(args.out / "PROGRESS.json", {"images": number, "total": len(image_ids)})
    results, good = {}, {}
    for name in variants:
        metrics, matched = evaluate(coco, image_ids, predictions[name])
        good[name] = matched
        results[name] = {"metrics": metrics, "matched75": len(matched), "predictions": len(predictions[name])}
    baseline = good["baseline"]
    for name in variants:
        results[name]["repairs75"] = len(good[name] - baseline)
        results[name]["damages75"] = len(baseline - good[name])
    summary = {"images": len(image_ids), "pool_images": pool_size, "image_ids": image_ids,
               "identity": identity, "results": results,
               "cached_vs_recomputed_matched75_difference": len(good["baseline_cached"] ^ good["baseline"]),
               "limitation": "Exploratory validation-GT-enriched cohort; projection strength and confidence rule were not trained or selected on an independent set."}
    write(args.out / "SUMMARY.json", summary)
    write(args.out / "MATCHED_GT75.json", {key: sorted(value) for key, value in good.items()})
    write(args.out / "COMPLETE.json", {"images": len(image_ids), "variants": len(variants)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("annotations", "joined", "archive", "helper_dir", "out"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    for key in ("local_checkpoint", "coeff_checkpoint", "trained_dir"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=Path)
    parser.add_argument("--images", type=int, default=200)
    args = parser.parse_args()
    assert args.trained_dir or (args.local_checkpoint and args.coeff_checkpoint)
    main(args)
