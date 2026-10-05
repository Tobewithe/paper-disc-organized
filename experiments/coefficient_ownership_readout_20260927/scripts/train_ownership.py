"""Capacity-matched dense ownership probes on frozen neck ROI."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


ARMS = ("base_h", "true_roi", "wrong_instance", "wrong_image")


class OwnershipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_proj = nn.Linear(64, 16)
        self.roi_proj = nn.Conv2d(512, 32, 1)
        self.body = nn.Sequential(nn.Conv2d(49, 64, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, h, base, roi):
        n = len(h)
        a = self.h_proj(h)[:, :, None, None].expand(n, 16, 8, 8)
        b = self.roi_proj(roi)
        x = torch.cat((a, base, b), dim=1)
        return base + self.body(x)


def controls(records):
    groups, levels = defaultdict(list), defaultdict(list)
    keys = []
    for i, r in enumerate(records):
        log_sqrt_area = float(r["basic"][8]) / 2
        area = 0 if log_sqrt_area < math.log(32 / 640) else (
            1 if log_sqrt_area < math.log(96 / 640) else 2)
        key = (int(r["level"]), area)
        keys.append(key)
        groups[key].append(i)
        levels[key[0]].append(i)
    wrong_instance, wrong_image, fallback = [], [], 0
    for i, r in enumerate(records):
        iid, key = int(r["image_id"]), keys[i]
        remote = [j for j in groups[key] if int(records[j]["image_id"]) != iid]
        if not remote:
            remote = [j for j in levels[key[0]] if int(records[j]["image_id"]) != iid]
        assert remote
        donor = remote[i % len(remote)]
        same = [j for j in groups[key] if int(records[j]["image_id"]) == iid and j != i]
        if same:
            wrong_instance.append(same[i % len(same)])
        else:
            wrong_instance.append(donor)
            fallback += 1
        wrong_image.append(donor)
    return torch.tensor(wrong_instance, dtype=torch.long), torch.tensor(
        wrong_image, dtype=torch.long), fallback


def prepare(data_path, neck_path, metadata_path, fit_stats=None):
    d = torch.load(data_path, map_location="cpu", weights_only=True)
    neck = torch.load(neck_path, map_location="cpu", weights_only=True)
    records = torch.load(metadata_path, map_location="cpu", weights_only=True)["rows"]
    assert len(d["keys"]) == len(neck["keys"]) == len(records)
    assert d["keys"] == [tuple(map(int, key)) for key in neck["keys"]]
    assert d["keys"] == [(int(r["image_id"]), int(r["annotation_id"]), int(r["raw_id"]))
                         for r in records]
    wrong_inst, wrong_img, fallback = controls(records)
    h, roi = d["h"].float(), neck["roi"].float()
    if fit_stats is None:
        fit_stats = {"h_mean": h.mean(0), "h_std": h.std(0).clamp_min(.01),
                     "roi_mean": roi.mean((0, 2, 3), keepdim=True),
                     "roi_std": roi.std((0, 2, 3), keepdim=True).clamp_min(.01)}
    h = ((h - fit_stats["h_mean"]) / fit_stats["h_std"]).cuda()
    roi = ((roi - fit_stats["roi_mean"]) / fit_stats["roi_std"]).cuda()
    return {"keys": d["keys"], "h": h, "roi": roi, "base": d["base"].cuda(),
            "label": d["label"].cuda(), "box_iou": d["box_iou"],
            "wrong_inst": wrong_inst.cuda(), "wrong_img": wrong_img.cuda(),
            "fallback": fallback}, fit_stats


def select_roi(data, arm, indices):
    if arm == "base_h":
        return torch.zeros_like(data["roi"][indices])
    if arm == "true_roi":
        return data["roi"][indices]
    if arm == "wrong_instance":
        return data["roi"][data["wrong_inst"][indices]]
    if arm == "wrong_image":
        return data["roi"][data["wrong_img"][indices]]
    raise ValueError(arm)


@torch.no_grad()
def predict(model, data, arm, batch=256):
    model.eval()
    result = []
    for lo in range(0, len(data["h"]), batch):
        idx = torch.arange(lo, min(lo + batch, len(data["h"])), device="cuda")
        result.append(model(data["h"][idx], data["base"][idx],
                            select_roi(data, arm, idx)).cpu())
    return torch.cat(result)


@torch.no_grad()
def dev_bce(model, data, arm):
    logits = predict(model, data, arm).cuda()
    return float(F.binary_cross_entropy_with_logits(logits, data["label"]))


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    fit, stats = prepare(args.data / "FIT.pt", args.fit_neck / "NECK.pt",
                         args.fit_metadata / "METADATA.pt")
    dev, _ = prepare(args.data / "DEV.pt", args.dev_neck / "NECK.pt",
                     args.dev_metadata / "METADATA.pt", stats)
    val, _ = prepare(args.data / "VAL.pt", args.val_neck / "NECK.pt",
                     args.val_metadata / "METADATA.pt", stats)
    assert len(set(key[0] for key in fit["keys"]) & set(key[0] for key in dev["keys"])) == 0
    assert len(set(key[0] for key in fit["keys"]) & set(key[0] for key in val["keys"])) == 0
    assert len(set(key[0] for key in dev["keys"]) & set(key[0] for key in val["keys"])) == 0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out / "NORMALIZATION.pt")
    initial = deepcopy(OwnershipNet().state_dict())
    models, optimizers, info = {}, {}, {}
    for arm in ARMS:
        model = OwnershipNet().cuda()
        model.load_state_dict(initial)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        info[arm] = {"best_dev": dev_bce(model, dev, arm), "epoch": 0,
                     "stale": 0, "history": [], "parameters": sum(
                         p.numel() for p in model.parameters())}
        torch.save({"model": model.state_dict(), "epoch": 0}, args.out / f"{arm}_best.pt")
    active = set(ARMS)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(fit["h"]), generator=torch.Generator().manual_seed(
            args.seed + epoch)).cuda()
        for lo in range(0, len(order), args.batch):
            idx = order[lo:lo + args.batch]
            for arm in ARMS:
                if arm not in active:
                    continue
                model = models[arm]
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                output = model(fit["h"][idx], fit["base"][idx], select_roi(fit, arm, idx))
                loss = F.binary_cross_entropy_with_logits(output, fit["label"][idx])
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        for arm in ARMS:
            if arm not in active:
                continue
            fit_loss, dev_loss = dev_bce(models[arm], fit, arm), dev_bce(models[arm], dev, arm)
            info[arm]["history"].append({"epoch": epoch, "fit_bce": fit_loss,
                                          "dev_bce": dev_loss})
            torch.save({"model": models[arm].state_dict(), "epoch": epoch,
                        "fit_bce": fit_loss, "dev_bce": dev_loss},
                       args.out / f"{arm}_epoch{epoch:02d}.pt")
            if dev_loss < info[arm]["best_dev"] - 1e-5:
                info[arm].update(best_dev=dev_loss, epoch=epoch, stale=0)
                torch.save({"model": models[arm].state_dict(), "epoch": epoch,
                            "dev_bce": dev_loss}, args.out / f"{arm}_best.pt")
            else:
                info[arm]["stale"] += 1
                if info[arm]["stale"] >= args.patience:
                    active.remove(arm)
        (args.out / "TRAINING.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "active": sorted(active), "best": {
            arm: {"epoch": info[arm]["epoch"], "dev_bce": info[arm]["best_dev"]}
            for arm in ARMS}}), flush=True)
        if not active:
            break
    result = {}
    for arm in ARMS:
        checkpoint = torch.load(args.out / f"{arm}_best.pt", map_location="cpu",
                                weights_only=True)
        models[arm].load_state_dict(checkpoint["model"])
        result[arm] = predict(models[arm], val, arm)
    torch.save({"keys": val["keys"], "logits": result, "box_iou": val["box_iou"]},
               args.out / "VAL_PREDICTIONS.pt")
    complete = {"fit_instances": len(fit["h"]), "dev_instances": len(dev["h"]),
                "val_instances": len(val["h"]), "seed": args.seed,
                "selected": {arm: info[arm]["epoch"] for arm in ARMS},
                "parameters": info["base_h"]["parameters"],
                "wrong_instance_fallback": {split: data["fallback"] for split, data in
                    (("fit", fit), ("dev", dev), ("val", val))},
                "val_keys_match": result["base_h"].shape[0] == len(val["keys"])}
    (args.out / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("data", "fit_neck", "dev_neck", "val_neck", "fit_metadata",
                "dev_metadata", "val_metadata", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=.001)
    main(parser.parse_args())
