"""Compare mask capacity of both internal branches on fixed COCO objects."""
import argparse
from collections import Counter, defaultdict
import contextlib
import io
import json
import random
from pathlib import Path
import sys

import numpy as np


def main(args):
    sys.path.insert(0, str(args.root / "shared/vendor/ultralytics_8_4_100"))
    import cv2
    import torch
    import torch.nn.functional as F
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    from pycocotools.coco import COCO

    assert ultralytics.__version__ == "8.4.100" and torch.cuda.is_available()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotations))
    joined = defaultdict(dict)
    failure_images = set()
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            joined[row["image_id"]][row["annotation_id"]] = row
            if row["geometry_state"] == "box_good_mask_unavailable":
                failure_images.add(row["image_id"])
    assert sum(map(len, joined.values())) == 36335
    rng = random.Random(20260923)
    failure_sample = rng.sample(sorted(failure_images), args.failure_images)
    control_sample = rng.sample(sorted(set(coco.imgs) - failure_images), args.control_images)
    selected = sorted(failure_sample + control_sample)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SAMPLE.json").write_text(json.dumps({
        "failure_images": failure_sample, "control_images": control_sample,
        "selection_seed": 20260923,
    }), encoding="utf-8")

    model = YOLO(str(args.model)).model.cuda().float().eval()
    head = model.model[-1]
    assert head.end2end
    transform = LetterBox((640, 640), auto=True, stride=32)
    records = []
    shared_proto_differences = []
    raw_candidate_counts = Counter()
    with torch.inference_mode():
        for number, image_id in enumerate(selected, 1):
            image = cv2.imread(str(args.images / coco.imgs[image_id]["file_name"]))
            assert image is not None
            original_shape = image.shape[:2]
            params = transform.get_params({"img": image})
            resized = transform.apply_image({"img": image}, params)["img"]
            x = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
            shape = tuple(x.shape[2:])
            _, raw = model(x)
            assert set(raw) == {"one2one", "one2many"}
            shared_proto_differences.append(float((raw["one2one"]["proto"] - raw["one2many"]["proto"]).abs().max()))
            annotations = sorted((a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0) and not a.get("ignore", 0)), key=lambda a: a["id"])
            if not annotations:
                continue
            gt_masks = np.stack([coco.annToMask(a) for a in annotations])
            gt_flat = torch.from_numpy(gt_masks.reshape(len(annotations), -1)).cuda().float()
            gt_area = gt_flat.sum(1)
            gt_boxes = torch.tensor([a["bbox"] for a in annotations], device="cuda", dtype=torch.float32)
            gt_boxes[:, 2:] += gt_boxes[:, :2]
            branch_values = {}
            for branch in ("one2one", "one2many"):
                q = raw[branch]
                boxes = head._get_decode_boxes(q)[0].T
                original_boxes = ops.scale_boxes(shape, boxes.clone(), original_shape)
                box_quality = box_iou(original_boxes, gt_boxes).cpu().numpy()
                eligible = box_quality >= .75
                raw_candidate_counts[branch] += len(boxes)
                indices = np.flatnonzero(eligible.any(axis=1))
                quality = np.full((len(boxes), len(annotations)), -1.0, dtype=np.float32)
                # Use the same full coefficient/prototype multiplication as the source cache.
                proto = q["proto"][0]
                logits = (q["mask_coefficient"][0].T @ proto.float().flatten(1)).reshape(-1, *proto.shape[-2:])
                for start in range(0, len(indices), 24):
                    ids = indices[start:start + 24]
                    idx = torch.as_tensor(ids, device="cuda", dtype=torch.long)
                    mask_logits = F.interpolate(logits[idx][None], shape, mode="bilinear")[0]
                    binary = ops.crop_mask(mask_logits, boxes[idx]).gt_(0).byte()
                    masks = ops.scale_masks(binary[None], original_shape)[0].byte()
                    pixels = masks.flatten(1).float()
                    intersection = pixels @ gt_flat.T
                    union = pixels.sum(1)[:, None] + gt_area[None] - intersection
                    quality[ids] = (intersection / union.clamp(min=1)).cpu().numpy()
                values = []
                for gt_index, ann in enumerate(annotations):
                    candidate_ids = np.flatnonzero(eligible[:, gt_index])
                    if len(candidate_ids):
                        best_mask_iou = float(quality[candidate_ids, gt_index].max())
                        best_box_id = int(candidate_ids[np.argmax(box_quality[candidate_ids, gt_index])])
                        best_box_mask_iou = float(quality[best_box_id, gt_index])
                        best_box_iou = float(box_quality[best_box_id, gt_index])
                    else:
                        best_mask_iou = best_box_mask_iou = -1.0
                        best_box_iou = float(box_quality[:, gt_index].max())
                    values.append({
                        "box75_candidates": int(len(candidate_ids)),
                        "max_mask_iou_with_box75": best_mask_iou,
                        "best_box_mask_iou": best_box_mask_iou,
                        "max_box_iou": best_box_iou,
                    })
                branch_values[branch] = values
                del logits, quality
            for gt_index, ann in enumerate(annotations):
                prior = joined[image_id][ann["id"]]
                o2o = branch_values["one2one"][gt_index]
                if o2o["box75_candidates"]:
                    target = prior["best_mask_with_box75"]
                    assert target is not None
                    assert abs(o2o["max_mask_iou_with_box75"] - target["mask_iou"]) <= 2e-6, (image_id, ann["id"], o2o, target)
                records.append({
                    "image_id": image_id,
                    "annotation_id": ann["id"],
                    "area_group": prior["area_group"],
                    "original_state": prior["geometry_state"],
                    "o2o": o2o,
                    "o2m": branch_values["one2many"][gt_index],
                })
            if number % 25 == 0:
                (args.out / "PROGRESS.json").write_text(json.dumps({"images": number, "total": len(selected), "objects": len(records)}), encoding="utf-8")
            del raw
    assert max(shared_proto_differences) == 0, max(shared_proto_differences)
    (args.out / "OBJECTS.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    result = {}
    for group_name, subset in {
        "all": records,
        "o2o_box_good_mask_unavailable": [r for r in records if r["original_state"] == "box_good_mask_unavailable"],
        "o2o_joint_good": [r for r in records if r["original_state"] == "joint_good"],
    }.items():
        result[group_name] = {
            "n": len(subset),
            "o2o_box75": sum(r["o2o"]["box75_candidates"] > 0 for r in subset),
            "o2m_box75": sum(r["o2m"]["box75_candidates"] > 0 for r in subset),
            "o2o_joint75": sum(r["o2o"]["max_mask_iou_with_box75"] >= .75 for r in subset),
            "o2m_joint75": sum(r["o2m"]["max_mask_iou_with_box75"] >= .75 for r in subset),
            "o2m_best_box_mask75": sum(r["o2m"]["best_box_mask_iou"] >= .75 for r in subset),
            "o2m_recovered_from_o2o_failure": sum(r["o2o"]["max_mask_iou_with_box75"] < .75 and r["o2m"]["max_mask_iou_with_box75"] >= .75 for r in subset),
        }
    result["source"] = {"images": len(selected), "gt": len(records), "shared_proto_max_abs_diff": max(shared_proto_differences), "raw_candidate_counts": dict(raw_candidate_counts)}
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"images": len(selected), "gt": len(records)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("root", "model", "images", "annotations", "joined", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--failure-images", type=int, default=400)
    parser.add_argument("--control-images", type=int, default=100)
    main(parser.parse_args())
