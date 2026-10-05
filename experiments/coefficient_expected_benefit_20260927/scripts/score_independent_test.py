"""Score frozen full-fit correction on sealed COCO val images after gate lock."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import ultralytics.data.augment as augment_module
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import ops
import ultralytics

from build_oracle_features import basic_features, crop_pool
from score_oof_benefit import mask_iou, scalar_features, proto_support
from train_oof_correction import HOnlySpatialNet


def auc(y, score):
    positives = int(y.sum())
    negatives = len(y) - positives
    if min(positives, negatives) == 0:
        return float("nan")
    return float((rankdata(score, method="average")[y.astype(bool)].sum() -
                  positives * (positives + 1) / 2) / (positives * negatives))


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    assert (a.gate / "COMPLETE.json").is_file(), "Gate selection must be frozen"
    gate = json.loads((a.gate / "COMPLETE.json").read_text())
    assert gate["selected"] and gate["test_labels_read"] is False
    manifest = json.loads(a.manifest.read_text())
    test_ids = set(map(int, manifest["independent_test_images"]))
    assert len(test_ids) == 2000
    assert json.loads((a.test_labels / "COMPLETE.json").read_text())["images"] == 2000
    identities = json.loads((a.test_labels / "IDENTITIES.json").read_text())
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.official_source / "scripts"))
    from official_pipeline import decode

    cp = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
    assert cp["fold"] == 5 and cp["fit_images"] > 9000
    assert test_ids.isdisjoint(set(map(int, cp["fit_image_ids"])))
    correction = HOnlySpatialNet().cuda().eval()
    correction.load_state_dict(cp["state_dict"])
    mean, std = cp["mean"].cuda(), cp["std"].cuda()
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    criterion = model.init_criterion().one2one
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    maps, detmaps, handles = {}, {}, []
    for level in range(3):
        handles.append(head.one2one_cv4[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: maps.__setitem__(level, x[0].detach())))
        handles.append(head.one2one_cv2[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: detmaps.__setitem__(level, x[0].detach())))
    order = {}
    original_rasterizer = augment_module.polygons2masks_overlap
    def record_order(*args, **kwargs):
        masks, indices = original_rasterizer(*args, **kwargs)
        order["indices"] = indices.copy()
        return masks, indices
    augment_module.polygons2masks_overlap = record_order
    data = a.test_labels / "official_data"
    ds = YOLODataset(img_path=str(data / "test.txt"), imgsz=640, batch_size=1,
        augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(ds.im_files)}
    assert set(by_id) == test_ids
    coco = COCO(str(a.test_labels / "conversion_input" / "instances_val2017.json"))
    selected = manifest["independent_test_images"]
    if a.count:
        selected = selected[a.start:a.start + a.count]
    else:
        selected = selected[a.start:]
    assert selected
    a.out.mkdir(parents=True, exist_ok=True)
    shard_dir = a.out / "shards"
    shard_dir.mkdir()
    block, progress = [], []
    counts = Counter()
    try:
        for step, iid in enumerate(selected, 1):
            iid = int(iid)
            j = by_id[iid]
            label = ds.labels[j]
            source_path = data / "labels" / "val2017" / f"{iid:012d}.txt"
            lines = source_path.read_text().strip().splitlines() if source_path.exists() else []
            source_keys = [(np.asarray(line.split(), dtype=np.float32), aid)
                           for line, aid in zip(lines, identities[str(iid)])]
            pre_ids = []
            for cls, polygon in zip(label["cls"].flatten(), label["segments"]):
                key = np.r_[cls, polygon.flatten()].astype(np.float32)
                matches = [aid for arr, aid in source_keys if np.array_equal(arr, key)]
                assert len(matches) == 1, (iid, matches)
                pre_ids.append(matches[0])
            order.clear()
            sample = ds[j]
            ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
            assert len(ann_ids) == len(sample["cls"])
            batch = YOLODataset.collate_fn([sample])
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                _, raw = model(batch["img"].float() / 255)
                assignment, _, _ = criterion.get_assigned_targets_and_loss(raw["one2one"], batch)
                fg, owners, _ = (t[0] for t in assignment[:3])
                indices = torch.where(fg)[0]
                owner_ids = owners[indices]
                assert len(set(owner_ids.tolist())) == len(owner_ids)
                pred = raw["one2one"]
                proto = pred["proto"][0].detach()
                coeff = pred["mask_coefficient"][0].T
                boxes = head._get_decode_boxes(pred)[0].T
                logits = pred["scores"][0].T
                h_all = torch.cat([maps[l][0].flatten(1).T for l in range(3)])
                det_all = torch.cat([detmaps[l][0].flatten(1).T for l in range(3)])
                raw_boxes = pred["boxes"][0].T
                levels = torch.cat([torch.full((maps[l].shape[-2] * maps[l].shape[-1],),
                    l, dtype=torch.long, device="cuda") for l in range(3)])
                if len(indices):
                    h = h_all[indices]
                    unit, logradius = correction((h - mean) / std)
                    delta = unit * logradius.expm1()[:, None]
                    c0 = coeff[indices]
                    chosen_boxes = boxes[indices]
                    decoded = dict(proto=proto.cpu(), boxes=chosen_boxes.cpu(),
                        original_shape=sample["ori_shape"], ratio_pad=sample["ratio_pad"])
                    candidate_ids = torch.arange(len(indices))
                    original_masks = decode(decoded, c0, candidate_ids)
                    corrected_masks = decode(decoded, c0 + delta, candidate_ids)
                    owner160 = F.interpolate(sample["masks"][0][None, None].float().cuda(),
                        (160, 160), mode="nearest")[0, 0].long()
                else:
                    delta = torch.empty((0, 32), device="cuda")
            for k, (raw_id, owner) in enumerate(zip(indices.tolist(), owner_ids.tolist())):
                aid = int(ann_ids[owner])
                gt = coco.annToMask(coco.anns[aid]).astype(bool)
                before = mask_iou(original_masks[k], gt)
                after = mask_iou(corrected_masks[k], gt)
                box = chosen_boxes[k]
                support = proto_support(box)
                pixels = proto[:, support].T.contiguous()
                z0 = pixels @ c0[k]
                z1 = pixels @ (c0[k] + delta[k])
                y = (owner160[support] == owner + 1).cpu().numpy().astype(np.uint8)
                auc0 = auc(y, z0.cpu().numpy())
                auc1 = auc(y, z1.cpu().numpy())
                local = crop_pool(maps[int(levels[raw_id])][0], box)
                local_summary = torch.cat((F.adaptive_avg_pool2d(local, (2,2)).flatten(),
                                           local.flatten(1).std(1)))
                block.append(dict(image_id=iid, annotation_id=aid,
                    original_raw_id=raw_id, fold=5, checkpoint=str(a.checkpoint),
                    basic=basic_features(box, int(levels[raw_id]), logits[raw_id]),
                    h=h_all[raw_id].cpu().float().clone(),
                    raw_detection=torch.cat((det_all[raw_id], raw_boxes[raw_id])).cpu().float().clone(),
                    local_summary=local_summary.cpu().float().clone(),
                    correction_stats=scalar_features(c0[k], delta[k], z0, z1),
                    correction=delta[k].cpu().float().clone(),
                    original_iou=before, corrected_iou=after, benefit_iou=after-before,
                    auc_before=auc0, auc_after=auc1, benefit_auc=auc1-auc0,
                    repair=after-before>.01, damage=after-before<-.01,
                    mask75_repair=before<.75<=after, mask75_damage=before>=.75>after))
                counts["instances"] += 1
            counts["images"] += 1
            progress.append(dict(image_id=iid, instances=len(indices)))
            if step % a.shard_images == 0 or step == len(selected):
                suffix = f"{a.start + step - len(progress):05d}_{a.start + step:05d}"
                torch.save(block, shard_dir / f"PART_{suffix}.pt")
                (shard_dir / f"PART_{suffix}.json").write_text(json.dumps(dict(images=progress,
                    records=len(block))), encoding="utf-8")
                block, progress = [], []
                status = dict(images=step, total=len(selected), instances=counts["instances"])
                (a.out / "PROGRESS.json").write_text(json.dumps(status), encoding="utf-8")
                print(json.dumps(status), flush=True)
    finally:
        augment_module.polygons2masks_overlap = original_rasterizer
        for handle in handles:
            handle.remove()
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=counts["images"],
        instances=counts["instances"], final_checkpoint=str(a.checkpoint),
        gate_frozen=str(a.gate / "COMPLETE.json")), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "test_labels", "gate", "checkpoint", "weights",
                "official_source", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--shard-images", type=int, default=100)
    main(p.parse_args())
