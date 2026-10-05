"""Recheck frozen official 10k cache; build finite targets and GT-free candidate descriptors."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
import ultralytics


def crop_pool(feature, box, size=8):
    _, height, width = feature.shape
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
    geometry = torch.tensor([
        float(b[0]), float(b[1]), float(b[2]), float(b[3]),
        (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2,
        math.log(width), math.log(height), math.log(width * height), math.log(width / height),
        float(largest[0]), float(largest[0] - largest[1]),
        float(level == 0), float(level == 1), float(level == 2)], dtype=torch.float32)
    return torch.cat((geometry, logits.detach().cpu().float()))


def decompressed(cache):
    array = np.frombuffer(zlib.decompress(cache["mask_zlib"]), dtype="<u2").copy()
    mask = torch.from_numpy(array.reshape(cache["mask_shape"]).astype(np.int64))
    return dict(cache, masks=mask, proto=cache["proto"].float())


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.official_source / "scripts"))
    sys.path.insert(0, str(a.oracle_source / "scripts"))
    from official_pipeline import decode, load, target_rois
    from finite_oracle import solve

    manifest = json.loads(a.manifest.read_text())
    bank_index = json.loads((a.bank / "INDEX.json").read_text())["fit"]
    assert [int(x["image_id"]) for x in bank_index] == manifest["train_images"]
    if a.count:
        chosen = bank_index[a.start:a.start + a.count]
    else:
        chosen = bank_index[a.start:]
    assert chosen
    order = [int(x["image_id"]) for x in chosen]
    coco = COCO(str(a.annotations))
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    maps, detmaps = {}, {}
    handles = []
    for level in range(3):
        handles.append(head.one2one_cv4[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: maps.__setitem__(level, x[0].detach())))
        handles.append(head.one2one_cv2[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: detmaps.__setitem__(level, x[0].detach())))
    dataset = YOLODataset(img_path=str(a.bank / "official_data" / "fit.txt"),
        imgsz=640, batch_size=1, augment=False, hyp=deepcopy(model.args),
        rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    assert set(by_id) == set(manifest["train_images"])
    a.out.mkdir(parents=True, exist_ok=True)
    shard_dir = a.out / "shards"
    shard_dir.mkdir()
    block, progress = [], []
    max_h_error = max_c_error = max_p_quantization_error = 0.
    counts = Counter()
    for step, item in enumerate(chosen, 1):
        iid = int(item["image_id"])
        if not item["n"]:
            progress.append(dict(image_id=iid, instances=0))
        else:
            cache = decompressed(load(a.bank / "images" / f"{iid:012d}.pt"))
            sample = dataset[by_id[iid]]
            assert torch.equal(sample["masks"][0].long(), cache["masks"]), iid
            with torch.no_grad():
                _, raw = model(sample["img"].cuda().float()[None] / 255)
            pred = raw["one2one"]
            h_all = torch.cat([maps[l][0].flatten(1).T for l in range(3)])
            det_all = torch.cat([detmaps[l][0].flatten(1).T for l in range(3)])
            coeff_all = pred["mask_coefficient"][0].T
            raw_box_all = pred["boxes"][0].T
            class_all = pred["scores"][0].T
            box_all = head._get_decode_boxes(pred)[0].T
            selected_raw = [int(r["original_raw_id"]) for r in cache["rows"]]
            assert len(selected_raw) == item["n"] == len(cache["rows"])
            max_h_error = max(max_h_error, float((h_all[selected_raw].cpu() - cache["h"]).abs().max()))
            max_c_error = max(max_c_error, float((coeff_all[selected_raw].cpu() - cache["coeff"]).abs().max()))
            prototype = pred["proto"][0].detach()
            max_p_quantization_error = max(max_p_quantization_error,
                float((prototype.cpu() - cache["proto"]).abs().max()))
            image = dict(cache, proto=prototype.cpu(), boxes=box_all[selected_raw].cpu(),
                         original_shape=sample["ori_shape"], ratio_pad=sample["ratio_pad"])
            rois = target_rois(image)
            masks = decode(image, cache["coeff"].cuda(), torch.arange(len(selected_raw)))
            up = F.interpolate(prototype[None], (640, 640), mode="bilinear", align_corners=False)[0]
            for j, (saved, roi, mask) in enumerate(zip(cache["rows"], rois, masks)):
                raw_id = selected_raw[j]
                box = box_all[raw_id]
                logits = class_all[raw_id]
                c0 = cache["coeff"][j].cuda()
                delta, state = solve(c0, roi, a.penalty, a.iterations)
                if not torch.isfinite(delta).all() or state["stationary_norm"] > 1e-3:
                    delta, state = solve(c0, roi, a.penalty, 240)
                if not torch.isfinite(delta).all() or state["stationary_norm"] > 1e-3:
                    raise RuntimeError(f"Oracle did not converge: {iid}/{saved['annotation_id']} {state}")
                support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
                pixels = up[:, support].T.contiguous()
                gram = pixels.T @ pixels / max(len(pixels), 1)
                local = crop_pool(maps[int(saved["level"])][0], box)
                pooled = F.adaptive_avg_pool2d(local, (2, 2)).flatten()
                std = local.flatten(1).std(1)
                ann = coco.anns[int(saved["annotation_id"])]
                gt = coco.annToMask(ann).astype(bool)
                original_iou = float(np.logical_and(mask, gt).sum() / max(np.logical_or(mask, gt).sum(), 1))
                original_box = ops.scale_boxes((640, 640), box[None].clone(),
                    sample["ori_shape"], ratio_pad=sample["ratio_pad"])[0]
                gt_box = torch.tensor(ann["bbox"], dtype=torch.float32, device="cuda")
                gt_box[2:] += gt_box[:2]
                block.append(dict(image_id=iid, annotation_id=int(saved["annotation_id"]),
                    original_raw_id=raw_id, level=int(saved["level"]),
                    h=h_all[raw_id].detach().cpu().float().clone(),
                    c0=c0.detach().cpu().float().clone(), delta=delta.cpu().float().clone(),
                    gram=gram.detach().cpu().float().clone(),
                    basic=basic_features(box, int(saved["level"]), logits).clone(),
                    raw_detection=torch.cat((det_all[raw_id].cpu().float(),
                        raw_box_all[raw_id].cpu().float())).clone(),
                    local_summary=torch.cat((pooled.cpu().float(), std.cpu().float())).clone(),
                    predicted_box=box.detach().cpu().float().clone(),
                    original_iou=original_iou,
                    box_iou=float(box_iou(gt_box[None], original_box[None])[0, 0]),
                    stationary_norm=state["stationary_norm"]))
                counts["instances"] += 1
            counts["effective_images"] += 1
            progress.append(dict(image_id=iid, instances=len(cache["rows"])))
        if step % a.shard_images == 0 or step == len(chosen):
            suffix = f"{a.start + step - len(progress):05d}_{a.start + step:05d}"
            torch.save(block, shard_dir / f"PART_{suffix}.pt")
            (shard_dir / f"PART_{suffix}.json").write_text(json.dumps(dict(images=progress,
                records=len(block), penalty=a.penalty)), encoding="utf-8")
            block, progress = [], []
            state = dict(images=step, total=len(chosen), instances=counts["instances"],
                max_h_error=max_h_error, max_c_error=max_c_error,
                max_p_quantization_error=max_p_quantization_error)
            (a.out / "PROGRESS.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(json.dumps(state), flush=True)
    assert max_h_error < 1e-4 and max_c_error < 1e-4
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=len(chosen),
        effective_images=counts["effective_images"], instances=counts["instances"],
        shards=len(list(shard_dir.glob("PART_*.pt"))), penalty=a.penalty,
        max_h_error=max_h_error, max_c_error=max_c_error,
        max_p_quantization_error=max_p_quantization_error), indent=2), encoding="utf-8")
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("manifest", "bank", "annotations", "weights", "official_source", "oracle_source", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--penalty", type=float, default=.003)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--shard-images", type=int, default=100)
    main(p.parse_args())
