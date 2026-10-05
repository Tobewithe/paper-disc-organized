"""Fixed-update h-only target-direction curve, with dev-only checkpointing."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SIZES = (800, 1600, 3200, 6400, 10000)


class DirectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, 192), nn.SiLU(),
                                    nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 33))

    def forward(self, h):
        output = self.layers(h)
        return F.normalize(output[:, :32], dim=1), F.softplus(output[:, 32])


def matrix(rows, key):
    return torch.stack([row[key] for row in rows]).float()


@torch.no_grad()
def eval_dev(model, h, direction, radius, gram):
    model.eval()
    result = []
    for lo in range(0, len(h), 256):
        unit, lograd = model(h[lo:lo + 256])
        target = direction[lo:lo + 256]
        g = gram[lo:lo + 256]
        gtarget = torch.bmm(g, target.unsqueeze(-1)).squeeze(-1)
        gunit = torch.bmm(g, unit.unsqueeze(-1)).squeeze(-1)
        effect = (unit * gtarget).sum(1) / (((target * gtarget).sum(1).clamp_min(1e-12) *
                                                  (unit * gunit).sum(1).clamp_min(1e-12)).sqrt())
        result.append((effect.clamp(-1, 1).cpu(), (unit * target).sum(1).cpu(),
                       (lograd - radius[lo:lo + 256]).abs().cpu()))
    values = [torch.cat([item[k] for item in result]) for k in range(3)]
    return dict(effect_cos=float(values[0].mean()), coefficient_cos=float(values[1].mean()),
                radius_log_abs=float(values[2].mean()))


@torch.no_grad()
def eval_fit(model, h, direction):
    model.eval()
    total = 0.0
    for lo in range(0, len(h), 1024):
        unit, _ = model(h[lo:lo + 1024])
        total += float((unit * direction[lo:lo + 1024]).sum())
    return total / len(h)


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    index = json.loads((args.targets / "INDEX.json").read_text())
    assert index["complete"] and index["images_processed"] == 10000
    chunks = index["chunks"]
    assert [item["through_images"] for item in chunks] == list(range(200, 10001, 200))
    fit_rows = [row for item in chunks for row in
                torch.load(args.targets / item["file"], weights_only=True, map_location="cpu")]
    assert len(fit_rows) == index["instances"]
    fit_h, fit_delta = matrix(fit_rows, "h"), matrix(fit_rows, "delta")
    assert torch.isfinite(fit_h).all() and torch.isfinite(fit_delta).all()
    fit_unit = F.normalize(fit_delta, dim=1)
    fit_radius = fit_delta.norm(dim=1).log1p()
    del fit_rows

    dev_rows = torch.load(args.dev / "DATA.pt", weights_only=True, map_location="cpu")
    assert all(r["initial_iou"] is None for r in dev_rows)
    dev_h = matrix(dev_rows, "h")
    dev_delta = matrix(dev_rows, "delta")
    dev_unit = F.normalize(dev_delta, dim=1).cuda()
    dev_radius = dev_delta.norm(dim=1).log1p().cuda()
    dev_gram = matrix(dev_rows, "gram").reshape(-1, 32, 32).cuda()
    del dev_rows
    assert len(dev_h) == 1402
    first = deepcopy(DirectionNet().state_dict())
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for images in SIZES:
        n = next(item["cumulative_instances"] for item in chunks if item["through_images"] == images)
        mean = fit_h[:n].mean(0)
        std = fit_h[:n].std(0).clamp_min(.01)
        h = ((fit_h[:n] - mean) / std).cuda()
        target = fit_unit[:n].cuda()
        radii = fit_radius[:n].cuda()
        d_h = ((dev_h - mean) / std).cuda()
        model = DirectionNet().cuda()
        model.load_state_dict(first)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best, best_step, stale = -float("inf"), 0, 0
        record = dict(images=images, instances=n, history=[], normalization=dict(mean=mean.tolist(), std=std.tolist()))
        def checkpoint(step, dev):
            nonlocal best, best_step, stale
            if dev["effect_cos"] > best + 1e-5:
                best, best_step, stale = dev["effect_cos"], step, 0
                torch.save(dict(state_dict=model.state_dict(), h_mean=mean, h_std=std,
                                images=images, instances=n, step=step, dev=dev),
                           args.out / f"n{images}_best.pt")
            else:
                stale += 1

        dev_score = eval_dev(model, d_h, dev_unit, dev_radius, dev_gram)
        checkpoint(0, dev_score)
        record["history"].append(dict(step=0, fit_coefficient_cos=eval_fit(model, h, target), dev=dev_score))
        generator = torch.Generator().manual_seed(args.seed + images)
        order, cursor = torch.empty(0, dtype=torch.long), 0
        for step in range(1, args.max_updates + 1):
            if cursor >= len(order):
                order = torch.randperm(n, generator=generator)
                cursor = 0
            indices = order[cursor:cursor + args.batch].cuda()
            cursor += len(indices)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            direction, logradius = model(h[indices])
            loss = (1 - (direction * target[indices]).sum(1) +
                    .25 * F.smooth_l1_loss(logradius, radii[indices], reduction="none")).mean()
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.max_updates:
                dev_score = eval_dev(model, d_h, dev_unit, dev_radius, dev_gram)
                checkpoint(step, dev_score)
                record["history"].append(dict(step=step,
                                              fit_coefficient_cos=eval_fit(model, h, target), dev=dev_score))
                print(json.dumps(dict(images=images, step=step, fit_coefficient_cos=record["history"][-1]["fit_coefficient_cos"],
                                      dev=dev_score, best_step=best_step, best_effect_cos=best)), flush=True)
                (args.out / "PROGRESS.json").write_text(json.dumps(dict(current_images=images, step=step,
                                                                    completed=list(results))), encoding="utf-8")
                if stale >= args.patience:
                    break
        record.update(best_step=best_step, best_dev_effect_cos=best, final_step=step)
        results[str(images)] = record
        (args.out / "TRAINING.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        del h, target, radii, d_h, model, optimizer
        torch.cuda.empty_cache()
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(sizes=SIZES, selected={k: v["best_step"] for k, v in results.items()},
                                                      seed=args.seed, max_updates=args.max_updates,
                                                      fit_instances=len(fit_h), dev_instances=len(dev_h),
                                                      selection="maximum dev functional cosine"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    main(parser.parse_args())
