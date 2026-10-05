"""Equal-architecture, split-isolated predictor comparison for 7E."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


ARMS = ("h_only", "true_local", "wrong_instance", "wrong_image")
LOCAL_KEYS = dict(true_local="p_true", wrong_instance="p_wrong_instance", wrong_image="p_wrong_image")


class DirectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(576, 192), nn.SiLU(),
                                    nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 33))

    def forward(self, x):
        output = self.layers(x)
        direction = F.normalize(output[:, :32], dim=1)
        log_radius = F.softplus(output[:, 32])
        return direction, log_radius


def matrix(rows, key):
    return torch.stack([row[key] for row in rows]).float()


def prepare(rows, normalization=None):
    h = matrix(rows, "h")
    p = matrix(rows, "p_true")
    if normalization is None:
        normalization = dict(h_mean=h.mean(0), h_std=h.std(0).clamp_min(.01),
                             p_mean=p.mean(0), p_std=p.std(0).clamp_min(.01))
    h = (h - normalization["h_mean"]) / normalization["h_std"]
    features = {}
    for arm in ARMS:
        local = torch.zeros_like(p) if arm == "h_only" else matrix(rows, LOCAL_KEYS[arm])
        if arm != "h_only":
            local = (local - normalization["p_mean"]) / normalization["p_std"]
        features[arm] = torch.cat((h, local), dim=1).cuda()
    target = matrix(rows, "delta").cuda()
    radius = target.norm(dim=1)
    values = dict(features=features, direction=F.normalize(target, dim=1),
                  log_radius=radius.log1p(), gram=matrix(rows, "gram").reshape(-1, 32, 32).cuda())
    return values, normalization


def components(direction, log_radius, data, indices):
    target = data["direction"][indices]
    gram = data["gram"][indices]
    gt_effect = torch.bmm(gram, target.unsqueeze(-1)).squeeze(-1)
    pred_effect = torch.bmm(gram, direction.unsqueeze(-1)).squeeze(-1)
    numerator = (direction * gt_effect).sum(1)
    denominator = ((target * gt_effect).sum(1).clamp_min(1e-12) *
                   (direction * pred_effect).sum(1).clamp_min(1e-12)).sqrt()
    effect_cos = (numerator / denominator).clamp(-1, 1)
    coefficient_cos = (direction * target).sum(1).clamp(-1, 1)
    magnitude_error = F.smooth_l1_loss(log_radius, data["log_radius"][indices], reduction="none")
    loss = (1 - effect_cos) + 0.25 * (1 - coefficient_cos) + 0.25 * magnitude_error
    return dict(loss=loss, effect_cos=effect_cos, coefficient_cos=coefficient_cos,
                radius_log_abs=(log_radius - data["log_radius"][indices]).abs())


@torch.no_grad()
def evaluate(model, data, arm):
    model.eval()
    indices = torch.arange(len(data["direction"]), device="cuda")
    direction, log_radius = model(data["features"][arm])
    values = components(direction, log_radius, data, indices)
    return {key: float(value.mean()) for key, value in values.items()}


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    fit_rows = torch.load(args.fit / "DATA.pt", weights_only=True, map_location="cpu")
    dev_rows = torch.load(args.dev / "DATA.pt", weights_only=True, map_location="cpu")
    assert fit_rows and dev_rows
    assert {r["image_id"] for r in fit_rows}.isdisjoint({r["image_id"] for r in dev_rows})
    assert all(r["initial_iou"] is None for r in fit_rows + dev_rows)
    fit, normalization = prepare(fit_rows)
    dev, _ = prepare(dev_rows, normalization)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(normalization, args.out / "NORMALIZATION.pt")
    initial = deepcopy(DirectionNet().state_dict())
    models, optimizers, records = {}, {}, {}
    for arm in ARMS:
        model = DirectionNet().cuda()
        model.load_state_dict(initial)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        records[arm] = dict(best_dev=float("inf"), chosen_epoch=None, stale=0, history=[],
                            parameters=sum(p.numel() for p in model.parameters()))
    active = set(ARMS)
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(fit_rows), generator=generator).cuda()
        for offset in range(0, len(order), args.batch):
            indices = order[offset:offset + args.batch]
            for arm in ARMS:
                if arm not in active:
                    continue
                model = models[arm]
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                direction, log_radius = model(fit["features"][arm][indices])
                loss = components(direction, log_radius, fit, indices)["loss"].mean()
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        for arm in ARMS:
            if arm not in active:
                continue
            train_score = evaluate(models[arm], fit, arm)
            dev_score = evaluate(models[arm], dev, arm)
            records[arm]["history"].append(dict(epoch=epoch, train=train_score, dev=dev_score))
            if dev_score["loss"] < records[arm]["best_dev"] - 1e-5:
                records[arm].update(best_dev=dev_score["loss"], chosen_epoch=epoch, stale=0)
                torch.save(dict(state_dict=models[arm].state_dict(), epoch=epoch, arm=arm,
                                dev=dev_score), args.out / f"{arm}_best.pt")
            else:
                records[arm]["stale"] += 1
                if records[arm]["stale"] >= args.patience:
                    active.remove(arm)
        (args.out / "TRAINING.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(dict(epoch=epoch, active=sorted(active),
                              dev={arm: records[arm]["history"][-1]["dev"] for arm in ARMS})), flush=True)
        if not active:
            break
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(fit_instances=len(fit_rows),
                    dev_instances=len(dev_rows), selected={arm: records[arm]["chosen_epoch"] for arm in ARMS},
                    parameters=records["h_only"]["parameters"], seed=args.seed), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
