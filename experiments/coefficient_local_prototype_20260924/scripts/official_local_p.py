"""Matched local-prototype controls on official one-to-one mask supervision."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F

from local_p_controls import ARMS, MatchedHead


def roi_features(image, indices):
    boxes = image["boxes"][[image["rows"][k]["raw_id"] for k in indices]].cuda()
    prototype = image["proto"].cuda()
    grid_values = (torch.arange(32, device="cuda") + .5) / 32
    yy, xx = torch.meshgrid(grid_values, grid_values, indexing="ij")
    gx = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
    gy = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
    grid = torch.stack([2 * gx / 640 - 1, 2 * gy / 640 - 1], dim=-1)
    result = F.grid_sample(prototype[None].expand(len(indices), -1, -1, -1),
                           grid, align_corners=False)
    return F.adaptive_avg_pool2d(result, (4, 4)).flatten(1).cpu()


def prepare(a, index, load):
    groups, stats = {}, {}
    for split in ("fit", "dev", "val"):
        by_image = {}
        by_level = defaultdict(list)
        for entry in index[split]:
            iid = entry["image_id"]
            if entry["n"] < 2:
                continue
            image = load(a.bank / "images" / f"{iid:012d}.pt")
            level_rows = defaultdict(list)
            for k, row in enumerate(image["rows"]):
                level_rows[row["level"]].append(k)
            eligible = sorted(k for values in level_rows.values() if len(values) >= 2 for k in values)
            if not eligible:
                continue
            same = {}
            for values in level_rows.values():
                if len(values) >= 2:
                    same.update({k: values[(j + 1) % len(values)] for j, k in enumerate(values)})
            all_features = roi_features(image, list(range(len(image["rows"]))))
            by_image[iid] = dict(indices=eligible, same=same, phi=all_features)
            for level in set(image["rows"][k]["level"] for k in eligible):
                by_level[level].append(iid)
        # Donor indices are image-separated and level-matched.
        for level, image_ids in by_level.items():
            assert len(image_ids) >= 2, (split, level)
            image_ids.sort()
            random.Random(20260924 + level).shuffle(image_ids)
            for j, iid in enumerate(image_ids):
                donor_iid = image_ids[(j + 1) % len(image_ids)]
                by_image[iid].setdefault("donors", {})[level] = donor_iid
        groups[split] = by_image
        stats[split] = dict(images=len(by_image), instances=sum(len(item["indices"]) for item in by_image.values()),
                            per_level={str(level): len(ids) for level, ids in by_level.items()})
        print(json.dumps(dict(stage="features", split=split, **stats[split])), flush=True)
    return groups, stats


def inputs(iid, image, meta, group, stats):
    chosen = meta["indices"]
    records = [image["rows"][k] for k in chosen]
    raw = torch.tensor([row["raw_id"] for row in records])
    h = (image["h"][raw].cuda() - stats["h_mean"]) / stats["h_std"]
    c = image["coeff"][raw].cuda()
    levels = image["levels"][raw].cuda()
    phi_true = meta["phi"][chosen].cuda()
    phi_same = meta["phi"][[meta["same"][k] for k in chosen]].cuda()
    phi_other = []
    for row in records:
        donor = group[meta["donors"][row["level"]]]
        options = [k for k in donor["indices"] if donor["levels"][k] == row["level"]]
        phi_other.append(donor["phi"][options[len(phi_other) % len(options)]])
    phi_other = torch.stack(phi_other).cuda()
    values = {"h_only": torch.zeros_like(phi_true),
        "true_local": (phi_true - stats["p_mean"]) / stats["p_std"],
        "wrong_instance": (phi_same - stats["p_mean"]) / stats["p_std"],
        "wrong_image": (phi_other - stats["p_mean"]) / stats["p_std"]}
    return records, h, c, levels, values


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import decode, load, roi_losses, setup, target_rois, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.bank / "INDEX.json").read_text())
    groups, scope = prepare(a, index, load)
    write(a.out / "SCOPE.json", scope)
    for split, group in groups.items():
        for iid, meta in group.items():
            image = load(a.bank / "images" / f"{iid:012d}.pt")
            meta["levels"] = [row["level"] for row in image["rows"]]
    hs, ps, cs = [], [], []
    for iid, meta in groups["fit"].items():
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        raw = [image["rows"][k]["raw_id"] for k in meta["indices"]]
        hs.append(image["h"][raw]); cs.append(image["coeff"][raw]); ps.append(meta["phi"][meta["indices"]])
    h, p, c = torch.cat(hs), torch.cat(ps), torch.cat(cs)
    stats = dict(h_mean=h.mean(0).cuda(), h_std=h.std(0).clamp_min(.01).cuda(),
                 p_mean=p.mean(0).cuda(), p_std=p.std(0).clamp_min(.01).cuda(),
                 c_scale=c.std(0).clamp_min(.1).cuda())
    models, optimizers = {}, {}
    for arm in ARMS:
        torch.manual_seed(0)
        models[arm] = MatchedHead(stats["c_scale"]).cuda()
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=.0001)
    parameter_counts = {arm: sum(v.numel() for v in model.parameters()) for arm, model in models.items()}
    assert len(set(parameter_counts.values())) == 1

    def evaluate(split, include_masks=False):
        sums = defaultdict(float)
        rows = []
        coco = COCO(str(a.source / "data/annotations/instances_val2017.json")) if include_masks else None
        with torch.no_grad():
            for iid, meta in groups[split].items():
                image = load(a.bank / "images" / f"{iid:012d}.pt")
                records, hh, cc, levels, phi = inputs(iid, image, meta, groups[split], stats)
                rois = target_rois(image, meta["indices"])
                coefficients = {"original": cc}
                coefficients.update({arm: cc + models[arm](hh, phi[arm], levels) for arm in ARMS})
                losses = {arm: roi_losses(values[None], rois)[0] for arm, values in coefficients.items()}
                for arm, value in losses.items():
                    sums[arm] += float(value)
                if include_masks:
                    raw_ids = torch.tensor([record["raw_id"] for record in records])
                    masks = {arm: decode(image, values, raw_ids) for arm, values in coefficients.items()}
                    for k, record in enumerate(records):
                        gt = coco.annToMask(coco.anns[record["annotation_id"]]).astype(bool)
                        ious = {arm: float((results[k] & gt).sum() / max((results[k] | gt).sum(), 1))
                                for arm, results in masks.items()}
                        rows.append(dict(image_id=iid, annotation_id=record["annotation_id"], iou=ious))
        count = scope[split]["instances"]
        return {arm: total / count for arm, total in sums.items()}, rows

    baseline_dev, _ = evaluate("dev")
    best = {arm: (float("inf"), 0) for arm in ARMS}
    history = []
    for epoch in range(1, a.epochs + 1):
        ordered = list(groups["fit"])
        random.Random(20260924 + epoch).shuffle(ordered)
        train_sums = defaultdict(float)
        for start in range(0, len(ordered), 8):
            batch = ordered[start:start + 8]
            n = sum(len(groups["fit"][iid]["indices"]) for iid in batch)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for iid in batch:
                image = load(a.bank / "images" / f"{iid:012d}.pt")
                meta = groups["fit"][iid]
                _, hh, cc, levels, phi = inputs(iid, image, meta, groups["fit"], stats)
                rois = target_rois(image, meta["indices"])
                losses = []
                for arm in ARMS:
                    candidate = cc + models[arm](hh, phi[arm], levels)
                    value = roi_losses(candidate[None], rois)[0] * image["segmentation_gain"]
                    losses.append(value)
                    train_sums[arm] += float(value.detach())
                (torch.stack(losses).sum() / n).backward()
            for arm in ARMS:
                torch.nn.utils.clip_grad_norm_(models[arm].parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        dev, _ = evaluate("dev")
        row = dict(epoch=epoch, train={arm: train_sums[arm] / scope["fit"]["instances"] for arm in ARMS},
                   dev=dev, original_dev=baseline_dev["original"])
        history.append(row)
        write(a.out / "HISTORY.json", history)
        print(json.dumps(row), flush=True)
        for arm, model in models.items():
            if dev[arm] < best[arm][0]:
                best[arm] = (dev[arm], epoch)
                torch.save(dict(epoch=epoch, state_dict=model.state_dict()), a.out / f"{arm}_best.pt")
        if epoch >= a.min_epochs and all(epoch - best[arm][1] >= a.patience for arm in ARMS):
            break
    for arm, model in models.items():
        model.load_state_dict(load(a.out / f"{arm}_best.pt")["state_dict"])
        model.eval()
    val, rows = evaluate("val", include_masks=True)
    write(a.out / "VAL_ROWS.json", rows)
    summary = dict(scope=scope, parameter_counts=parameter_counts,
        dev_original=baseline_dev["original"], val_bce=val,
        best={arm: dict(epoch=best[arm][1], dev=best[arm][0],
                        gated_epoch=best[arm][1] if best[arm][0] < baseline_dev["original"] else 0)
              for arm in ARMS},
        val_iou={arm: float(np.mean([r["iou"][arm] for r in rows])) for arm in ("original", *ARMS)},
        scope_note="Official one2one GT identity, 640 overlap mask supervision; fixed original predicted boxes")
    write(a.out / "SUMMARY.json", summary)
    write(a.out / "COMPLETE.json", dict(epochs=len(history), val_instances=len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    main(parser.parse_args())
