"""Recompute frozen official features and GT-only diagnostic labels for 7I/7J."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics


def crop_pool(feature, box, size=8):
    channels, height, width = feature.shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return F.adaptive_avg_pool2d(feature[:, y1:y2, x1:x2], (size, size))


def basic_features(box, level, logits):
    b = box.float() / 640
    width = float((b[2] - b[0]).clamp_min(1e-6))
    height = float((b[3] - b[1]).clamp_min(1e-6))
    scores = logits.float().sigmoid()
    largest = torch.topk(scores, 2).values
    geo = torch.tensor([
        float(b[0]), float(b[1]), float(b[2]), float(b[3]),
        (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2,
        math.log(width), math.log(height), math.log(width * height), math.log(width / height),
        float(largest[0]), float(largest[0] - largest[1]),
        float(level == 0), float(level == 1), float(level == 2),
    ], dtype=torch.float32)
    return torch.cat((geo, logits.float().cpu()))


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.official_source / "scripts"))
    from official_pipeline import decode, load, target_rois

    target_rows = torch.load(a.targets / "DATA.pt", weights_only=True, map_location="cpu")
    controls = torch.load(a.controls / "DESCRIPTORS.pt", weights_only=True, map_location="cpu")
    assert len(target_rows) == len(controls["p8"])
    index = json.loads((a.bank / "INDEX.json").read_text())[a.split]
    grouped = defaultdict(list)
    for position, row in enumerate(target_rows):
        grouped[int(row["image_id"])].append(position)
    assert set(grouped) == {int(item["image_id"]) for item in index if item["n"]}
    domain = "val" if a.split == "val" else "train"
    coco = COCO(str(a.annotations))

    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    maps, detection_maps = {}, {}
    handles = []
    for level in range(3):
        handles.append(head.one2one_cv4[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: maps.__setitem__(level, x[0].detach())))
        handles.append(head.one2one_cv2[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: detection_maps.__setitem__(level, x[0].detach())))

    dataset = YOLODataset(img_path=str(a.bank / "official_data" / f"{a.split}.txt"),
        imgsz=640, batch_size=1, augment=False, hyp=deepcopy(model.args),
        rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    assert set(by_id) == {int(item["image_id"]) for item in index}
    result = [None] * len(target_rows)
    max_h_error, max_coeff_error, max_box_error, max_class_error, max_iou_error = 0., 0., 0., 0., 0.
    for step, item in enumerate(index, 1):
        iid = int(item["image_id"])
        if not item["n"]:
            continue
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        sample = dataset[by_id[iid]]
        tensor = sample["img"].cuda().float()[None] / 255
        with torch.no_grad():
            _, raw = model(tensor)
        pred = raw["one2one"]
        flat_h = torch.cat([maps[level][0].flatten(1).T for level in range(3)])
        flat_detection = torch.cat([detection_maps[level][0].flatten(1).T for level in range(3)])
        flat_coeff = pred["mask_coefficient"][0].T
        flat_raw_box = pred["boxes"][0].T
        flat_class = pred["scores"][0].T
        decoded_box = head._get_decode_boxes(pred)[0].T
        original_rois = target_rois(image)
        ordered_indices = grouped[iid]
        raw_ids = torch.tensor([target_rows[k]["raw_id"] for k in ordered_indices], dtype=torch.long)
        original_masks = decode(image, image["coeff"][raw_ids].cuda(), raw_ids)
        for mask, position in zip(original_masks, ordered_indices):
            row = target_rows[position]
            saved = image["rows"][row["row_index"]]
            assert saved["annotation_id"] == row["annotation_id"]
            raw_id = int(row["raw_id"])
            level = int(image["levels"][raw_id])
            max_h_error = max(max_h_error, float((flat_h[raw_id].cpu() - row["h"]).abs().max()))
            max_coeff_error = max(max_coeff_error, float((flat_coeff[raw_id].cpu() - row["c0"]).abs().max()))
            max_box_error = max(max_box_error, float((decoded_box[raw_id].cpu() - image["boxes"][raw_id]).abs().max()))
            max_class_error = max(max_class_error, float((flat_class[raw_id].cpu() - image["class_logits"][:, raw_id]).abs().max()))
            gt = coco.annToMask(coco.anns[int(row["annotation_id"])]).astype(bool)
            original_iou = float(np.logical_and(mask, gt).sum() / max(np.logical_or(mask, gt).sum(), 1))
            if row["initial_iou"] is not None:
                max_iou_error = max(max_iou_error, abs(original_iou - row["initial_iou"]))
            p, y, area = original_rois[row["row_index"]]
            c0 = row["c0"].cuda()
            c_star = c0 + row["delta"].cuda()
            with torch.no_grad():
                l0 = F.binary_cross_entropy_with_logits(p @ c0, y, reduction="sum") / area
                ls = F.binary_cross_entropy_with_logits(p @ c_star, y, reduction="sum") / area
            gap = float(l0 - ls)
            local = crop_pool(maps[level][0], decoded_box[raw_id], 8)
            result[position] = dict(image_id=iid, annotation_id=int(row["annotation_id"]),
                raw_id=raw_id, level=level, h=row["h"].float(),
                basic=basic_features(decoded_box[raw_id], level, flat_class[raw_id]),
                raw_detection=torch.cat((flat_detection[raw_id].cpu().float(),
                    flat_raw_box[raw_id].cpu().float())),
                local_hmap=local.cpu().float(),
                original_iou=original_iou, failure=int(original_iou < .75),
                oracle_gap=gap, loss_original=float(l0), loss_oracle=float(ls),
                box_iou=float(row["box_iou"]))
        if step % 50 == 0 or step == len(index):
            print(json.dumps(dict(split=a.split, images=step, total=len(index),
                records=sum(x is not None for x in result))), flush=True)
    assert all(x is not None for x in result)
    assert max_h_error < 1e-4 and max_coeff_error < 1e-4 and max_box_error < 1e-3 and max_class_error < 1e-4
    assert max_iou_error < 1e-8
    assert min(x["oracle_gap"] for x in result) >= -1e-5
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rows=result, wrong_instance=controls["wrong_instance"],
        wrong_image=controls["wrong_image"]), a.out / "FEATURES.pt")
    summary = dict(split=a.split, images=len(index), effective_images=len(grouped),
        instances=len(result), failures=sum(x["failure"] for x in result),
        max_h_error=max_h_error, max_coeff_error=max_coeff_error,
        max_box_error=max_box_error, max_class_error=max_class_error,
        max_initial_iou_error=max_iou_error,
        gap_min=min(x["oracle_gap"] for x in result),
        gap_median=float(np.median([x["oracle_gap"] for x in result])),
        gap_max=max(x["oracle_gap"] for x in result))
    (a.out / "COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("official_source", "weights", "bank", "targets", "controls", "annotations", "out"):
        ap.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--split", choices=("fit", "dev", "val"), required=True)
    main(ap.parse_args())
