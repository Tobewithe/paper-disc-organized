"""Cache official one-to-one positives for a nested full-COCO size curve."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
from pycocotools.coco import COCO
import torch
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as augment_module


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import prepare_dataset, roi_losses, setup, target_rois, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    images = a.out / "images"
    images.mkdir(exist_ok=True)
    dataset_root = a.out / "data"
    (dataset_root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val2017").mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations").mkdir(exist_ok=True)
    for domain in ("train", "val"):
        dest = dataset_root / "annotations" / f"instances_{domain}2017.json"
        if not dest.exists():
            dest.symlink_to(a.source / "data" / "annotations" / dest.name)
    with (dataset_root / "annotations" / "instances_train2017.json").open() as f:
        coco_data = json.load(f)
    existing = json.loads((a.source / "SPLIT.json").read_text())
    excluded = set(existing["dev"])
    available = sorted(im["id"] for im in coco_data["images"] if im["id"] not in excluded)
    order = random.Random(a.seed).sample(available, a.images)
    write(a.out / "ORDER.json", dict(seed=a.seed, images=order, excluded_dev=existing["dev"],
         population=len(available), source="COCO train2017 instances JSON; uniform sample without replacement"))
    filename = {im["id"]: im["file_name"] for im in coco_data["images"]}
    with zipfile.ZipFile(a.zip_file) as archive:
        members = set(archive.namelist())
        for step, iid in enumerate(order, 1):
            name = filename[iid]
            member = f"train2017/{name}"
            assert member in members, member
            dest = dataset_root / "images" / "train2017" / name
            if not dest.exists():
                with archive.open(member) as src, dest.open("wb") as output:
                    while chunk := src.read(1024 * 1024):
                        output.write(chunk)
            if step % 500 == 0:
                print(json.dumps(dict(stage="extract", images=step, total=len(order))), flush=True)
    del coco_data

    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = model.init_criterion().one2one
    head = model.model[-1]
    dataset_args = SimpleNamespace(out=a.out, data=dataset_root)
    data, _, identity = prepare_dataset(dataset_args, dict(fit=order, dev=[], val=[]), wrapper.names)
    captured = {}
    hooks = [branch[-1].register_forward_pre_hook(
        lambda _, x, level=l: captured.__setitem__(level, x[0].detach()))
        for l, branch in enumerate(head.one2one_cv4)]
    rasterizer = augment_module.polygons2masks_overlap
    label_order = {}

    def record_order(*args, **kwargs):
        mask, indices = rasterizer(*args, **kwargs)
        label_order["indices"] = indices.copy()
        return mask, indices

    augment_module.polygons2masks_overlap = record_order
    dataset = YOLODataset(img_path=str(data / "fit.txt"), imgsz=640, batch_size=1,
        augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    assert set(by_id) == set(order)
    index, counts, audit = [], Counter(), []
    try:
        for step, iid in enumerate(order, 1):
            raw_label = dataset.labels[by_id[iid]]
            pre_ids = []
            for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
                key = np.r_[cls, polygon.flatten()].astype(np.float32)
                matches = [aid for arr, aid in identity[iid] if np.array_equal(key, arr)]
                assert len(matches) == 1, (iid, matches)
                pre_ids.append(matches[0])
            label_order.clear()
            sample = dataset[by_id[iid]]
            ann_ids = np.asarray(pre_ids)[label_order["indices"]].tolist() if pre_ids else []
            assert len(ann_ids) == len(sample["cls"])
            batch = YOLODataset.collate_fn([sample])
            batch = {k: value.cuda() if isinstance(value, torch.Tensor) else value
                     for k, value in batch.items()}
            with torch.no_grad():
                _, raw = model(batch["img"].float() / 255)
                assigned, _, _ = criterion.get_assigned_targets_and_loss(raw["one2one"], batch)
            fg, owner, gt_boxes = (item[0] for item in assigned[:3])
            positive = torch.where(fg)[0]
            owners = owner[positive]
            assert len(set(owners.tolist())) == len(owners)
            count = len(positive)
            counts["images"] += 1
            counts["positive"] += count
            index.append(dict(image_id=iid, n=count))
            if count:
                pred = raw["one2one"]
                proto = pred["proto"][0].detach()
                coeff = pred["mask_coefficient"][0].T[positive].detach()
                h_all = torch.cat([captured[level][0].flatten(1).T for level in range(3)])
                h = h_all[positive].detach()
                level_all = torch.cat([torch.full(
                    (captured[level].shape[-2] * captured[level].shape[-1],), level,
                    dtype=torch.long, device="cuda") for level in range(3)])
                rows = [dict(image_id=iid, annotation_id=ann_ids[g], raw_id=j,
                         original_raw_id=rid, level=int(level_all[rid]))
                        for j, (rid, g) in enumerate(zip(positive.tolist(), owners.tolist()))]
                mask = sample["masks"][0].cpu().numpy().astype("<u2", copy=False)
                image = dict(proto=proto.cpu().half(), coeff=coeff.cpu().float(),
                    h=h.cpu().float(), levels=level_all[positive].cpu(),
                    mask_zlib=zlib.compress(mask.tobytes(), level=1), mask_shape=mask.shape,
                    owners=owners.cpu(), target_boxes=gt_boxes[positive].cpu(),
                    rows=rows, segmentation_gain=float(model.args.box))
                if len(audit) < 8:
                    original = dict(image, proto=proto.cpu().float(), masks=torch.from_numpy(mask.astype(np.int64)))
                    compact = dict(image, proto=image["proto"].float(), masks=original["masks"])
                    with torch.no_grad():
                        loss32 = float(roi_losses(coeff[None], target_rois(original))[0])
                        loss16 = float(roi_losses(coeff[None], target_rois(compact))[0])
                    audit.append(dict(image_id=iid, loss_fp32=loss32, loss_proto_fp16=loss16,
                                      absolute_error=abs(loss32-loss16), positive=count))
                torch.save(image, images / f"{iid:012d}.pt")
            if step % 100 == 0 or step == len(order):
                write(a.out / "INDEX.json", dict(fit=index))
                write(a.out / "PROGRESS.json", dict(stage="cache", images=step, total=len(order),
                                                    positive=counts["positive"]))
                print(json.dumps(dict(stage="cache", images=step, total=len(order))), flush=True)
    finally:
        augment_module.polygons2masks_overlap = rasterizer
        for hook in hooks:
            hook.remove()
    write(a.out / "QUANTIZATION_AUDIT.json", audit)
    write(a.out / "COMPLETE.json", dict(counts=counts, mask_ratio=model.args.mask_ratio,
        overlap_mask=model.args.overlap_mask, branch="one2one", images=len(order),
        annotation_source="official COCO convert_coco, YOLODataset/Format, one2one assigner",
        prototype_cache="float16; loss computed float32"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "weights", "zip_file", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260924)
    main(parser.parse_args())
