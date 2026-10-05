"""Matched h-only, true-local-P, same-image and other-image controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ARMS = ("h_only", "true_local", "wrong_instance", "wrong_image")


class MatchedHead(nn.Module):
    def __init__(self, c_scale):
        super().__init__()
        self.register_buffer("c_scale", c_scale)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(576, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 32)) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, h, phi, levels):
        result = torch.zeros(len(h), 32, device=h.device)
        for level, head in enumerate(self.heads):
            index = torch.where(levels == level)[0]
            if len(index):
                result[index] = head(torch.cat([h[index], phi[index]], dim=1))
        return result * self.c_scale


def eligible_controls(rows):
    groups = defaultdict(list)
    for k, row in enumerate(rows):
        meta = row["meta"]
        groups[(meta["image_id"], meta["level"])].append(k)
    eligible = sorted(k for group in groups.values() if len(group) > 1 for k in group)
    same = {}
    for group in groups.values():
        if len(group) > 1:
            same.update({k: group[(j+1) % len(group)] for j, k in enumerate(group)})
    per_level = defaultdict(list)
    for key, group in groups.items():
        if len(group) > 1:
            per_level[key[1]].append((key[0], group))
    other = {}
    for level, image_groups in per_level.items():
        image_groups.sort()
        random.Random(20260924 + level).shuffle(image_groups)
        assert len(image_groups) > 1
        for j, (_, recipients) in enumerate(image_groups):
            donors = image_groups[(j + 1) % len(image_groups)][1]
            other.update({k: donors[i % len(donors)] for i, k in enumerate(recipients)})
    return [{**rows[k], "phi_same": rows[same[k]]["phi"],
             "phi_other": rows[other[k]]["phi"]} for k in eligible]


def collate(rows):
    from feature_probes import collate as base_collate
    batch = base_collate(rows)
    batch["phi_same"] = torch.stack([row["phi_same"] for row in rows])
    batch["phi_other"] = torch.stack([row["phi_other"] for row in rows])
    batch["image_ids"] = torch.tensor([row["meta"]["image_id"] for row in rows])
    batch["annotation_ids"] = torch.tensor([row["meta"]["annotation_id"] for row in rows])
    return batch


def evaluate(models, rows, normalization, batch_size):
    h_mean, h_std, p_mean, p_std = normalization
    loader = DataLoader(rows, batch_size=batch_size, collate_fn=collate)
    losses = {arm: [] for arm in ("original", *ARMS)}
    records = []
    with torch.no_grad():
        for batch in loader:
            ids = batch.pop("image_ids")
            anns = batch.pop("annotation_ids")
            value = {key: tensor.cuda() for key, tensor in batch.items()}
            h = (value["h"] - h_mean) / h_std
            phi = {
                "h_only": torch.zeros_like(value["phi"]),
                "true_local": (value["phi"] - p_mean) / p_std,
                "wrong_instance": (value["phi_same"] - p_mean) / p_std,
                "wrong_image": (value["phi_other"] - p_mean) / p_std,
            }
            def loss(coeff):
                z = torch.bmm(value["p"], coeff[:, :, None])[:, :, 0]
                return (F.binary_cross_entropy_with_logits(z, value["y"], reduction="none")
                        * value["weight"]).sum(1)
            now = {"original": loss(value["c"])}
            for arm, net in models.items():
                now[arm] = loss(value["c"] + net(h, phi[arm], value["levels"]))
            for arm, tensor in now.items():
                losses[arm].extend(tensor.cpu().tolist())
            records.extend(zip(ids.tolist(), anns.tolist()))
    return losses, records


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import setup, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    train = eligible_controls(torch.load(a.bank / "fit.pt", weights_only=False))
    dev = eligible_controls(torch.load(a.bank / "dev.pt", weights_only=False))
    val = eligible_controls(torch.load(a.bank / "val.pt", weights_only=False))
    assert all(len(items) for items in (train, dev, val))
    h = torch.stack([row["h"] for row in train])
    p = torch.stack([row["phi"] for row in train])
    c = torch.stack([row["c"] for row in train])
    h_mean, h_std = h.mean(0).cuda(), h.std(0).clamp_min(.01).cuda()
    p_mean, p_std = p.mean(0).cuda(), p.std(0).clamp_min(.01).cuda()
    normalization = (h_mean, h_std, p_mean, p_std)
    c_scale = c.std(0).clamp_min(.1).cuda()
    models = {}
    for arm in ARMS:
        torch.manual_seed(0)
        models[arm] = MatchedHead(c_scale).cuda()
    optimizer = torch.optim.AdamW([param for net in models.values()
        for param in net.parameters()], lr=.001, weight_decay=.0001)
    parameters = {arm: sum(param.numel() for param in net.parameters()) for arm, net in models.items()}
    assert len(set(parameters.values())) == 1
    initial, _ = evaluate({}, dev, normalization, a.batch_size)
    original_dev = float(np.mean(initial["original"]))
    best = {arm: (float("inf"), 0) for arm in ARMS}
    history = []
    for epoch in range(1, a.epochs + 1):
        for net in models.values():
            net.train()
        loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(20260924 + epoch), collate_fn=collate)
        sums = {arm: 0.0 for arm in ARMS}
        count = 0
        for batch in loader:
            batch.pop("image_ids")
            batch.pop("annotation_ids")
            value = {key: tensor.cuda() for key, tensor in batch.items()}
            normalized_h = (value["h"] - h_mean) / h_std
            phi = {
                "h_only": torch.zeros_like(value["phi"]),
                "true_local": (value["phi"] - p_mean) / p_std,
                "wrong_instance": (value["phi_same"] - p_mean) / p_std,
                "wrong_image": (value["phi_other"] - p_mean) / p_std,
            }
            optimizer.zero_grad(set_to_none=True)
            objectives = []
            for arm, net in models.items():
                coefficients = value["c"] + net(normalized_h, phi[arm], value["levels"])
                z = torch.bmm(value["p"], coefficients[:, :, None])[:, :, 0]
                loss = (F.binary_cross_entropy_with_logits(z, value["y"], reduction="none")
                        * value["weight"]).sum(1).mean()
                objectives.append(loss)
                sums[arm] += float(loss.detach()) * len(value["c"])
            torch.stack(objectives).sum().backward()
            for net in models.values():
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            count += len(value["c"])
        for net in models.values():
            net.eval()
        dev_losses, _ = evaluate(models, dev, normalization, a.batch_size)
        row = dict(epoch=epoch, train={arm: sums[arm] / count for arm in ARMS},
                   dev={arm: float(np.mean(dev_losses[arm])) for arm in ARMS},
                   original_dev=original_dev)
        history.append(row)
        write(a.out / "HISTORY.json", history)
        print(json.dumps(row), flush=True)
        for arm, net in models.items():
            if row["dev"][arm] < best[arm][0]:
                best[arm] = (row["dev"][arm], epoch)
                torch.save(dict(state_dict=net.state_dict(), epoch=epoch), a.out / f"{arm}_best.pt")
        if epoch >= a.min_epochs and all(epoch - best[arm][1] >= a.patience for arm in ARMS):
            break
    for arm, net in models.items():
        net.load_state_dict(torch.load(a.out / f"{arm}_best.pt", weights_only=False)["state_dict"])
        net.eval()
    dev_losses, dev_records = evaluate(models, dev, normalization, a.batch_size)
    val_losses, val_records = evaluate(models, val, normalization, a.batch_size)
    write(a.out / "VAL_ROWS.json", [dict(image_id=iid, annotation_id=ann,
        losses={arm: val_losses[arm][k] for arm in val_losses})
        for k, (iid, ann) in enumerate(val_records)])
    summary = dict(counts=dict(train=len(train), dev=len(dev), val=len(val)),
        parameter_count=parameters, baseline_dev=original_dev,
        best_trained={arm: dict(epoch=best[arm][1], dev=best[arm][0],
            val=float(np.mean(val_losses[arm])),
            gated_epoch=best[arm][1] if best[arm][0] < original_dev else 0) for arm in ARMS},
        val_original=float(np.mean(val_losses["original"])),
        scope="Frozen official one2one matched-head diagnostic on multi-positive same-level images")
    write(a.out / "SUMMARY.json", summary)
    write(a.out / "COMPLETE.json", dict(epochs=len(history), counts=summary["counts"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
