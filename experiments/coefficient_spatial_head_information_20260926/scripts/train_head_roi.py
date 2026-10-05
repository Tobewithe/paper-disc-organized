"""Capacity-matched current versus mismatched spatial-head ROI predictors."""
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


ARMS = ("h_only", "true_local", "wrong_instance", "wrong_image")


class SpatialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                     nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.predictor = nn.Sequential(nn.Linear(64 + 32 * 4 * 4, 192), nn.SiLU(),
                                       nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 33))

    def forward(self, h, p):
        result = self.predictor(torch.cat((h, self.encoder(p)), dim=1))
        return F.normalize(result[:, :32], dim=1), F.softplus(result[:, 32])


def matrix(rows, key):
    return torch.stack([row[key] for row in rows]).float()


def controls(records):
    groups = defaultdict(list)
    levels = defaultdict(list)
    key_list = []
    for i, r in enumerate(records):
        log_sqrt_area = float(r["basic"][8]) / 2
        area_bin = 0 if log_sqrt_area < math.log(32 / 640) else (1 if log_sqrt_area < math.log(96 / 640) else 2)
        key = (int(r["level"]), area_bin)
        key_list.append(key)
        groups[key].append(i)
        levels[key[0]].append(i)
    wrong_instance, wrong_image, fallback = [], [], 0
    for i, r in enumerate(records):
        iid = r["image_id"]
        key = key_list[i]
        remote = [j for j in groups[key] if records[j]["image_id"] != iid]
        if not remote:
            remote = [j for j in levels[key[0]] if records[j]["image_id"] != iid]
        assert remote
        donor = remote[i % len(remote)]
        same = [j for j in groups[key] if records[j]["image_id"] == iid and j != i]
        if same:
            wrong_instance.append(same[i % len(same)])
        else:
            wrong_instance.append(donor)
            fallback += 1
        wrong_image.append(donor)
    return torch.tensor(wrong_instance), torch.tensor(wrong_image), fallback


def prepare(rows, features, normalization=None):
    records = features["rows"]
    assert len(records) == len(rows)
    assert all((r["image_id"], r["annotation_id"], r["raw_id"]) ==
               (f["image_id"], f["annotation_id"], f["raw_id"])
               for r, f in zip(rows, records))
    h = matrix(rows, "h")
    local = matrix(records, "local_hmap")
    assert local.shape == (len(rows), 64, 8, 8)
    wrong_instance, wrong_image, fallback = controls(records)
    if normalization is None:
        normalization = dict(h_mean=h.mean(0), h_std=h.std(0).clamp_min(.01),
                             f_mean=local.mean((0, 2, 3), keepdim=True),
                             f_std=local.std((0, 2, 3), keepdim=True).clamp_min(.01))
    h = ((h - normalization["h_mean"]) / normalization["h_std"]).cuda()
    def norm(q):
        return ((q - normalization["f_mean"]) / normalization["f_std"]).cuda()
    inputs = dict(h_only=torch.zeros_like(local, device="cuda"), true_local=norm(local),
                  wrong_instance=norm(local[wrong_instance]), wrong_image=norm(local[wrong_image]))
    target = matrix(rows, "delta").cuda()
    radius = target.norm(dim=1)
    return dict(h=h, inputs=inputs, direction=F.normalize(target, dim=1),
                radius=radius.log1p(), gram=matrix(rows, "gram").reshape(-1, 32, 32).cuda(),
                control_fallback=fallback), normalization


def components(unit, logradius, data, indices):
    target = data["direction"][indices]
    gram = data["gram"][indices]
    g_target = torch.bmm(gram, target.unsqueeze(-1)).squeeze(-1)
    g_unit = torch.bmm(gram, unit.unsqueeze(-1)).squeeze(-1)
    effect = ((unit * g_target).sum(1) /
              ((target * g_target).sum(1).clamp_min(1e-12) *
               (unit * g_unit).sum(1).clamp_min(1e-12)).sqrt()).clamp(-1, 1)
    coefficient = (unit * target).sum(1).clamp(-1, 1)
    radius_abs = (logradius - data["radius"][indices]).abs()
    loss = 1 - effect + .25 * (1 - coefficient) + .25 * F.smooth_l1_loss(
        logradius, data["radius"][indices], reduction="none")
    return dict(loss=loss, effect_cos=effect, coefficient_cos=coefficient,
                radius_log_abs=radius_abs)


@torch.no_grad()
def evaluate(model, data, arm):
    model.eval()
    chunks = []
    for lo in range(0, len(data["h"]), 256):
        idx = torch.arange(lo, min(lo + 256, len(data["h"])), device="cuda")
        unit, radius = model(data["h"][idx], data["inputs"][arm][idx])
        chunks.append(components(unit, radius, data, idx))
    return {key: float(torch.cat([part[key] for part in chunks]).mean()) for key in chunks[0]}


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    fit_rows = torch.load(args.fit_targets / "DATA.pt", weights_only=True, map_location="cpu")
    dev_rows = torch.load(args.dev_targets / "DATA.pt", weights_only=True, map_location="cpu")
    assert {r["image_id"] for r in fit_rows}.isdisjoint({r["image_id"] for r in dev_rows})
    fit_features = torch.load(args.fit_features / "FEATURES.pt", weights_only=True, map_location="cpu")
    dev_features = torch.load(args.dev_features / "FEATURES.pt", weights_only=True, map_location="cpu")
    fit, normalization = prepare(fit_rows, fit_features)
    dev, _ = prepare(dev_rows, dev_features, normalization)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(normalization, args.out / "NORMALIZATION.pt")
    initial = deepcopy(SpatialNet().state_dict())
    models, optimizers, records = {}, {}, {}
    for arm in ARMS:
        model = SpatialNet().cuda()
        model.load_state_dict(initial)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        records[arm] = dict(best_dev=float("inf"), selected=0, stale=0, history=[],
                            parameters=sum(p.numel() for p in model.parameters()))
        score = evaluate(model, dev, arm)
        records[arm]["best_dev"] = score["loss"]
        records[arm]["history"].append(dict(epoch=0, fit=evaluate(model, fit, arm), dev=score))
        torch.save(dict(state_dict=model.state_dict(), epoch=0, arm=arm, dev=score),
                   args.out / f"{arm}_best.pt")
    active = set(ARMS)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(fit_rows), generator=torch.Generator().manual_seed(args.seed + epoch)).cuda()
        for lo in range(0, len(order), args.batch):
            indices = order[lo:lo + args.batch]
            for arm in ARMS:
                if arm not in active:
                    continue
                model = models[arm]
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                unit, radius = model(fit["h"][indices], fit["inputs"][arm][indices])
                loss = components(unit, radius, fit, indices)["loss"].mean()
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        for arm in ARMS:
            if arm not in active:
                continue
            fit_score = evaluate(models[arm], fit, arm)
            dev_score = evaluate(models[arm], dev, arm)
            records[arm]["history"].append(dict(epoch=epoch, fit=fit_score, dev=dev_score))
            if dev_score["loss"] < records[arm]["best_dev"] - 1e-5:
                records[arm].update(best_dev=dev_score["loss"], selected=epoch, stale=0)
                torch.save(dict(state_dict=models[arm].state_dict(), epoch=epoch, arm=arm,
                                dev=dev_score), args.out / f"{arm}_best.pt")
            else:
                records[arm]["stale"] += 1
                if records[arm]["stale"] >= args.patience:
                    active.remove(arm)
        (args.out / "TRAINING.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(dict(epoch=epoch, active=sorted(active),
                              best={arm: dict(epoch=records[arm]["selected"],
                                             dev_loss=records[arm]["best_dev"]) for arm in ARMS})), flush=True)
        if not active:
            break
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(fit_instances=len(fit_rows),
        dev_instances=len(dev_rows), selected={arm: records[arm]["selected"] for arm in ARMS},
        seed=args.seed, parameters=records["h_only"]["parameters"],
        wrong_instance_fallback_fit=fit["control_fallback"],
        wrong_instance_fallback_dev=dev["control_fallback"]), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("fit_targets", "dev_targets", "fit_features", "dev_features", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    main(parser.parse_args())
