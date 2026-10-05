"""Train the 7J-N h-only correction head with image-disjoint OOF folds."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F


class HOnlySpatialNet(nn.Module):
    """Exact 7J-N h-only architecture, evaluating its constant ROI just once."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(512, 32, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                     nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.predictor = nn.Sequential(nn.Linear(64 + 32 * 4 * 4, 192), nn.SiLU(),
                                       nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 33))
        self.register_buffer("zero_roi", torch.zeros(1, 512, 8, 8), persistent=False)

    def forward(self, h):
        constant = self.encoder(self.zero_roi).expand(len(h), -1)
        result = self.predictor(torch.cat((h, constant), dim=1))
        return F.normalize(result[:, :32], dim=1), F.softplus(result[:, 32])


def matrix(rows, key):
    return torch.stack([row[key] for row in rows]).float()


def components(unit, logradius, direction, radius, gram):
    g_target = torch.bmm(gram, direction.unsqueeze(-1)).squeeze(-1)
    g_unit = torch.bmm(gram, unit.unsqueeze(-1)).squeeze(-1)
    effect = ((unit * g_target).sum(1) /
              ((direction * g_target).sum(1).clamp_min(1e-12) *
               (unit * g_unit).sum(1).clamp_min(1e-12)).sqrt()).clamp(-1, 1)
    coefficient = (unit * direction).sum(1).clamp(-1, 1)
    loss = 1 - effect + .25 * (1 - coefficient) + .25 * F.smooth_l1_loss(
        logradius, radius, reduction="none")
    return loss, effect, coefficient


def load_shards(directory):
    rows = []
    paths = sorted((directory / "shards").glob("PART_*.pt"))
    assert paths, directory
    for path in paths:
        rows.extend(torch.load(path, weights_only=True, map_location="cpu"))
    return rows


def pack(rows, mean, std, device):
    delta = matrix(rows, "delta")
    return dict(h=((matrix(rows, "h") - mean) / std).to(device),
                direction=F.normalize(delta, dim=1).to(device),
                radius=delta.norm(dim=1).log1p().to(device),
                gram=matrix(rows, "gram").to(device))


@torch.no_grad()
def evaluate(model, data, batch=512):
    model.eval()
    aggregate = torch.zeros(4, dtype=torch.float64)
    for lo in range(0, len(data["h"]), batch):
        hi = min(lo + batch, len(data["h"]))
        unit, radius = model(data["h"][lo:hi])
        loss, effect, coefficient = components(unit, radius, data["direction"][lo:hi],
            data["radius"][lo:hi], data["gram"][lo:hi])
        aggregate += torch.tensor([loss.sum(), effect.sum(), coefficient.sum(), len(loss)],
                                  dtype=torch.float64)
    return dict(loss=float(aggregate[0] / aggregate[3]),
                effect_cos=float(aggregate[1] / aggregate[3]),
                coefficient_cos=float(aggregate[2] / aggregate[3]))


def main(a):
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    manifest = json.loads(a.manifest.read_text())
    image_ids = [int(x) for x in manifest["train_images"]]
    fold_by_id = {int(iid): k for k, fold in enumerate(manifest["folds"]) for iid in fold}
    assert len(image_ids) == 10000 and len(fold_by_id) == 10000
    assert set(fold_by_id) == set(image_ids)
    assert a.fold in (0, 1, 2, 3, 4, 5)
    all_rows = load_shards(a.targets)
    assert all(int(r["image_id"]) in fold_by_id for r in all_rows)
    fit_rows = [r for r in all_rows if a.fold == 5 or fold_by_id[int(r["image_id"])] != a.fold]
    held_rows = [] if a.fold == 5 else [r for r in all_rows if fold_by_id[int(r["image_id"])] == a.fold]
    assert {r["image_id"] for r in fit_rows}.isdisjoint({r["image_id"] for r in held_rows})
    assert fit_rows and (held_rows or a.fold == 5)
    dev_rows = torch.load(a.dev_targets, weights_only=True, map_location="cpu")
    assert {r["image_id"] for r in dev_rows}.isdisjoint(image_ids)
    h = matrix(fit_rows, "h")
    mean = h.mean(0)
    std = h.std(0).clamp_min(.01)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = pack(fit_rows, mean, std, device)
    dev = pack(dev_rows, mean, std, device)
    model = HOnlySpatialNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    a.out.mkdir(parents=True, exist_ok=True)
    fit_ids = sorted({int(r["image_id"]) for r in fit_rows})
    checkpoint_meta = dict(fold=a.fold, fit_images=len(fit_ids), fit_instances=len(fit_rows),
        held_images=len({r["image_id"] for r in held_rows}), held_instances=len(held_rows),
        fit_image_ids=fit_ids,
        fit_id_sha256=hashlib.sha256(json.dumps(fit_ids).encode()).hexdigest(), seed=a.seed,
        parameters=sum(p.numel() for p in model.parameters()))
    best = float("inf")
    selected_epoch = stale = 0
    history = []
    for epoch in range(a.epochs + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(fit_rows),
                                   generator=torch.Generator().manual_seed(a.seed + epoch)).to(device)
            for lo in range(0, len(order), a.batch):
                idx = order[lo:lo + a.batch]
                optimizer.zero_grad(set_to_none=True)
                unit, radius = model(fit["h"][idx])
                loss = components(unit, radius, fit["direction"][idx], fit["radius"][idx],
                                  fit["gram"][idx])[0].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite train loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
        dev_score = evaluate(model, dev)
        fit_score = evaluate(model, fit)
        history.append(dict(epoch=epoch, fit=fit_score, dev=dev_score))
        if dev_score["loss"] < best - 1e-5:
            best, selected_epoch, stale = dev_score["loss"], epoch, 0
            torch.save(dict(state_dict=deepcopy(model.state_dict()), mean=mean, std=std,
                selected_epoch=epoch, dev=dev_score, **checkpoint_meta), a.out / "BEST.pt")
        else:
            stale += 1
        (a.out / "TRAINING.json").write_text(json.dumps(dict(history=history,
            selected_epoch=selected_epoch, best_dev_loss=best, **checkpoint_meta), indent=2),
            encoding="utf-8")
        print(json.dumps(dict(fold=a.fold, epoch=epoch, fit=fit_score, dev=dev_score,
                              selected_epoch=selected_epoch)), flush=True)
        if stale >= a.patience:
            break
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(selected_epoch=selected_epoch,
        best_dev_loss=best, **checkpoint_meta), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "targets", "dev_targets", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=.001)
    main(p.parse_args())
