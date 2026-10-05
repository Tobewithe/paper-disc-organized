"""Score an image-held-out correction fold on original-size COCO masks."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
from pycocotools.coco import COCO
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import ops
from types import SimpleNamespace
import ultralytics

from train_oof_correction import HOnlySpatialNet, load_shards


def mask_iou(predicted, target):
    return float(np.logical_and(predicted, target).sum() /
                 max(np.logical_or(predicted, target).sum(), 1))


def roc_auc_score(y, score):
    positives = int(y.sum())
    negatives = len(y) - positives
    return (rankdata(score, method="average")[y.astype(bool)].sum() -
            positives * (positives + 1) / 2) / (positives * negatives)


def proto_support(box):
    """At least one native prototype pixel for boxes smaller than its stride."""
    coords = [float(q) / 4 for q in box]
    x1 = max(0, min(159, math.floor(coords[0])))
    y1 = max(0, min(159, math.floor(coords[1])))
    x2 = max(x1 + 1, min(160, math.ceil(coords[2])))
    y2 = max(y1 + 1, min(160, math.ceil(coords[3])))
    support = torch.zeros((160, 160), dtype=torch.bool, device="cuda")
    support[y1:y2, x1:x2] = True
    return support


def scalar_features(c0, delta, z0, z1):
    pixels = z0.numel()
    dz = z1 - z0
    return torch.tensor([
        float(delta.norm()),
        float(F.cosine_similarity(c0[None], delta[None])[0]),
        float(dz.norm() / pixels ** .5), float(dz.abs().mean()),
        *[float(q) for q in torch.quantile(z0, torch.tensor([.1, .5, .9], device=z0.device))],
        *[float(q) for q in torch.quantile(z1, torch.tensor([.1, .5, .9], device=z1.device))],
        float((z0 > 0).float().mean()), float((z1 > 0).float().mean()),
        float((z1 > 0).float().mean() - (z0 > 0).float().mean()),
    ], dtype=torch.float32)


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.official_source / "scripts"))
    from official_pipeline import decode, load

    manifest = json.loads(a.manifest.read_text())
    fold_ids = set(int(i) for i in manifest["folds"][a.fold])
    train_ids = set(int(i) for i in manifest["train_images"])
    assert len(fold_ids) == 2000 and fold_ids <= train_ids
    checkpoint = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
    assert checkpoint["fold"] == a.fold and checkpoint["held_instances"] > 0
    assert fold_ids.isdisjoint(set(map(int, checkpoint["fit_image_ids"])))
    model_corr = HOnlySpatialNet().cuda().eval()
    model_corr.load_state_dict(checkpoint["state_dict"])
    mean, std = checkpoint["mean"].cuda(), checkpoint["std"].cuda()

    rows = load_shards(a.targets)
    grouped = defaultdict(list)
    for row in rows:
        if int(row["image_id"]) in fold_ids:
            grouped[int(row["image_id"])].append(row)
    assert sum(map(len, grouped.values())) == checkpoint["held_instances"]
    index = json.loads((a.bank / "INDEX.json").read_text())["fit"]
    chosen = [item for item in index if int(item["image_id"]) in fold_ids]
    assert len(chosen) == 2000
    if a.smoke:
        chosen = [item for item in chosen if int(item["image_id"]) in grouped]
    if a.count:
        chosen = chosen[a.start:a.start + a.count]
    else:
        chosen = chosen[a.start:]
    assert chosen

    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end
    dataset = YOLODataset(img_path=str(a.bank / "official_data" / "fit.txt"),
        imgsz=640, batch_size=1, augment=False, hyp=deepcopy(model.args),
        rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    coco = COCO(str(a.annotations))
    a.out.mkdir(parents=True, exist_ok=True)
    shard_dir = a.out / "shards"
    shard_dir.mkdir()
    block = []
    progress = []
    biggest_baseline_discrepancy = 0.
    for step, item in enumerate(chosen, 1):
        iid = int(item["image_id"])
        image_rows = grouped.get(iid, [])
        assert len(image_rows) == int(item["n"])
        if image_rows:
            sample = dataset[by_id[iid]]
            cache = load(a.bank / "images" / f"{iid:012d}.pt")
            owner_mask = torch.from_numpy(np.frombuffer(
                __import__("zlib").decompress(cache["mask_zlib"]), dtype="<u2").copy()
                .reshape(cache["mask_shape"]).astype(np.int64))
            assert torch.equal(sample["masks"][0].long(), owner_mask)
            with torch.no_grad():
                _, raw = model(sample["img"].cuda().float()[None] / 255)
                proto = raw["one2one"]["proto"][0].detach()
                coeff = raw["one2one"]["mask_coefficient"][0].T
                boxes = head._get_decode_boxes(raw["one2one"])[0].T
                raw_ids = [int(r["original_raw_id"]) for r in cache["rows"]]
                by_key = {(int(r["annotation_id"]), int(r["original_raw_id"])): j
                          for j, r in enumerate(cache["rows"])}
                order = [by_key[(int(r["annotation_id"]), int(r["original_raw_id"]))]
                         for r in image_rows]
                assert len(set(order)) == len(image_rows)
                assert max(float((coeff[raw_ids[j]].cpu() - cache["coeff"][j]).abs().max())
                           for j in order) < 1e-4
                assert max(float((boxes[raw_ids[j]].cpu() - image_rows[k]["predicted_box"]).abs().max())
                           for k, j in enumerate(order)) < 1e-4
                h = torch.stack([r["h"] for r in image_rows]).cuda()
                unit, logradius = model_corr((h - mean) / std)
                delta = unit * logradius.expm1()[:, None]
                c0 = torch.stack([r["c0"] for r in image_rows]).cuda()
                used_boxes = torch.stack([r["predicted_box"] for r in image_rows])
                decoding = dict(proto=proto.cpu(), boxes=used_boxes,
                    original_shape=sample["ori_shape"], ratio_pad=sample["ratio_pad"])
                ids = torch.arange(len(image_rows))
                original_masks = decode(decoding, c0, ids)
                corrected_masks = decode(decoding, c0 + delta, ids)
                owner160 = F.interpolate(owner_mask[None, None].float().cuda(),
                    (160, 160), mode="nearest")[0, 0].long()
            for k, row in enumerate(image_rows):
                ann = coco.anns[int(row["annotation_id"])]
                gt = coco.annToMask(ann).astype(bool)
                iou_before = mask_iou(original_masks[k], gt)
                iou_after = mask_iou(corrected_masks[k], gt)
                biggest_baseline_discrepancy = max(biggest_baseline_discrepancy,
                    abs(iou_before - float(row["original_iou"])))
                support = proto_support(used_boxes[k])
                pixels = proto[:, support].T.contiguous()
                z0 = pixels @ c0[k]
                z1 = pixels @ (c0[k] + delta[k])
                y = owner160[support] == int(cache["owners"][order[k]]) + 1
                if y.any() and (~y).any():
                    y_np = y.cpu().numpy().astype(np.uint8)
                    auc_before = float(roc_auc_score(y_np, z0.cpu().numpy()))
                    auc_after = float(roc_auc_score(y_np, z1.cpu().numpy()))
                else:
                    auc_before = auc_after = float("nan")
                block.append(dict(image_id=iid, annotation_id=int(row["annotation_id"]),
                    original_raw_id=int(row["original_raw_id"]), fold=a.fold,
                    checkpoint=str(a.checkpoint),
                    basic=row["basic"], h=row["h"],
                    raw_detection=row["raw_detection"],
                    local_summary=row["local_summary"],
                    correction_stats=scalar_features(c0[k], delta[k], z0, z1),
                    correction=delta[k].cpu().float(),
                    box_iou=float(row["box_iou"]),
                    original_iou=iou_before, corrected_iou=iou_after,
                    benefit_iou=iou_after - iou_before,
                    auc_before=auc_before, auc_after=auc_after,
                    benefit_auc=auc_after - auc_before,
                    repair=iou_after - iou_before > .01,
                    damage=iou_after - iou_before < -.01,
                    mask75_repair=iou_before < .75 <= iou_after,
                    mask75_damage=iou_before >= .75 > iou_after))
        progress.append(dict(image_id=iid, instances=len(image_rows)))
        if step % a.shard_images == 0 or step == len(chosen):
            suffix = f"{a.start + step - len(progress):05d}_{a.start + step:05d}"
            torch.save(block, shard_dir / f"PART_{suffix}.pt")
            (shard_dir / f"PART_{suffix}.json").write_text(json.dumps(dict(images=progress,
                records=len(block), fold=a.fold)), encoding="utf-8")
            block, progress = [], []
            status = dict(fold=a.fold, images=step, total=len(chosen),
                max_baseline_discrepancy=biggest_baseline_discrepancy)
            (a.out / "PROGRESS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status), flush=True)
    assert biggest_baseline_discrepancy < 1e-5
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(fold=a.fold,
        images=len(chosen), instances=sum(int(item["n"]) for item in chosen),
        max_baseline_discrepancy=biggest_baseline_discrepancy,
        checkpoint=str(a.checkpoint)), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "targets", "bank", "annotations", "weights",
                "official_source", "checkpoint", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--shard-images", type=int, default=100)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
