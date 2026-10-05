"""Train matched refinement heads on one COCO train2017 prediction bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from projected_direction_pilot import coefficient_projection, roi_correction


def corrected(base, x, output, arm, original, helper):
    if arm.startswith("project_"):
        strength = int(arm.split("_")[1])
        p = x[:, :32]
        direction = coefficient_projection(p, base, roi_correction(output), strength)
        return base + (p * direction[:, :, None, None]).sum(1)
    mode = "coeff" if arm.startswith("coeff") else "local4"
    return helper.roi_correct(base, x, output, mode)


def loss_fn(logits, target):
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = 1 - (2 * (prob * target).sum((1, 2)) + 1) / (prob.sum((1, 2)) + target.sum((1, 2)) + 1)
    return bce + .5 * dice.mean()


def hard_iou(logits, target):
    prediction = logits > 0
    gt = target >= .5
    return (prediction & gt).sum((1, 2)).float() / (prediction | gt).sum((1, 2)).clamp_min(1)


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.helper_dir))
    import component_seed_probe as helper
    import learn_refinement as original

    bank = torch.load(args.bank, map_location="cpu", weights_only=False, mmap=True)
    image_ids = sorted({r["image_id"] for r in bank["records"]})
    random.Random(20260923).shuffle(image_ids)
    fit_ids = set(image_ids[:min(800, len(image_ids) - 1)])
    dev_ids = set(image_ids) - fit_ids
    assert len(fit_ids) >= 400 and len(dev_ids) >= 50
    fit = [i for i, r in enumerate(bank["records"]) if r["image_id"] in fit_ids]
    dev = [i for i, r in enumerate(bank["records"]) if r["image_id"] in dev_ids]
    fitset = TensorDataset(*(bank[k][fit] for k in ("x", "base", "target")))
    devset = TensorDataset(*(bank[k][dev] for k in ("x", "base", "target")))
    (args.out / "SPLIT.json").write_text(json.dumps({"fit_images": sorted(fit_ids), "dev_images": sorted(dev_ids),
        "fit_instances": len(fit), "dev_instances": len(dev)}, indent=2), encoding="utf-8")
    arms = ("coeff", "coeff_guard", "local4", "project_0", "project_4")
    all_results = {}
    for arm in arms:
        torch.manual_seed(0)
        mode = "coeff" if arm.startswith("coeff") else "local4"
        net = original.Refiner(mode).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(fitset, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(0), num_workers=0)
        losses = []
        for epoch in range(1, 9):
            net.train()
            epoch_losses = []
            for x, base, target in loader:
                x, base, target = x.cuda().float(), base.cuda().float(), target.cuda().float()
                z = corrected(base, x, net(x), arm, original, helper)
                objective = loss_fn(z, target)
                if arm == "coeff_guard":
                    objective = objective + helper.correctness_guard(z, base, target).mean()
                assert bool(torch.isfinite(objective))
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
                epoch_losses.append(float(objective.detach()))
            losses.append(float(np.mean(epoch_losses)))
            torch.save({"state_dict": net.state_dict(), "arm": arm, "epoch": epoch,
                        "fit_instances": len(fit)}, args.out / f"{arm}_epoch{epoch}.pt")
            print(json.dumps({"arm": arm, "epoch": epoch, "loss": losses[-1]}), flush=True)
        net.eval()
        dev_stats = {str(a): {"n": 0, "sum_iou": 0., "repairs": 0, "damages": 0}
                     for a in (.25, .5, 1.)}
        with torch.no_grad():
            for x, base, target in DataLoader(devset, batch_size=128, num_workers=0):
                x, base, target = x.cuda().float(), base.cuda().float(), target.cuda().float()
                output = net(x)
                before = hard_iou(base, target)
                for alpha in (.25, .5, 1.):
                    after = hard_iou(corrected(base, x, output * alpha, arm, original, helper), target)
                    s = dev_stats[str(alpha)]
                    s["n"] += len(x)
                    s["sum_iou"] += float(after.sum())
                    s["repairs"] += int(((before < .75) & (after >= .75)).sum())
                    s["damages"] += int(((before >= .75) & (after < .75)).sum())
        for s in dev_stats.values():
            s["mean_iou"] = s.pop("sum_iou") / s["n"]
        all_results[arm] = {"losses": losses, "dev": dev_stats, "checkpoint": f"{arm}_epoch8.pt"}
    (args.out / "TRAINING.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"arms": list(arms), "fit_instances": len(fit),
        "dev_instances": len(dev)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
