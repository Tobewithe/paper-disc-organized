"""GT-dependent, strongly convex coefficient oracle for the fixed prototype bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from ultralytics.utils import ops


LAMBDAS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def solve(c0, roi, penalty, iterations, start=None, weights=None):
    p, y, area = roi
    p, y = p.double(), y.double()
    c0 = c0.double()
    area = area.double()
    if weights is None:
        weights = torch.ones_like(y)
    else:
        weights = weights.double()
    delta = (torch.zeros_like(c0) if start is None else start.double().clone()).requires_grad_(True)
    optimizer = torch.optim.LBFGS([delta], lr=1, max_iter=iterations,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                  tolerance_change=1e-12)

    def objective():
        logits = p @ (c0 + delta)
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights).sum() / area
        return bce + penalty * delta.square().sum() / 2

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    value = objective()
    gradient = torch.autograd.grad(value, delta)[0]
    return delta.detach().float(), dict(objective=float(value.detach()),
                                        stationary_norm=float(gradient.norm()),
                                        iterations=int(optimizer.state[delta]["n_iter"]))


def subsample_roi(roi, seed):
    p, y, area = roi
    chosen, weights = [], []
    for label in (0, 1):
        indices = (y == label).nonzero(as_tuple=True)[0]
        if not len(indices):
            continue
        keep = max(1, round(0.8 * len(indices)))
        generator = torch.Generator(device=p.device).manual_seed(seed + label)
        sample = indices[torch.randperm(len(indices), generator=generator, device=p.device)[:keep]]
        chosen.append(sample)
        weights.append(torch.full((keep,), len(indices) / keep, device=p.device))
    indices = torch.cat(chosen)
    return (p[indices], y[indices], area), torch.cat(weights)


def item_key(row):
    return (row["image_id"], row["annotation_id"])


def main(args):
    sys.path.insert(0, str(args.source / "scripts"))
    sys.path.insert(0, str(args.metrics_source / "scripts"))
    from official_pipeline import load, setup, target_rois, write
    from gradient_geometry import cosine, metrics

    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    prior_rows = json.loads((args.oracle / "ROWS.json").read_text())
    if args.limit:
        prior_rows = prior_rows[:args.limit]
    old_coefficients = load(args.oracle / "oracle_coefficients.pt")
    grouped = defaultdict(list)
    for row in prior_rows:
        grouped[row["image_id"]].append(row)
    failures = [row for row in prior_rows if row["initial_iou"] < .75]
    rng = random.Random(20260925)
    stability_ids = set()
    if not args.limit:
        for good_box in (False, True):
            available = [row["annotation_id"] for row in failures if (row["box_iou"] >= .75) == good_box]
            stability_ids.update(rng.sample(available, min(20, len(available))))
    rows, checks, targets = [], [], {}
    for image_number, (image_id, selected) in enumerate(grouped.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        rois = target_rois(image, [r["row_index"] for r in selected])
        prototype = F.interpolate(image["proto"].cuda()[None], (640, 640),
                                  mode="bilinear", align_corners=False)[0]
        owners = image["masks"].cuda()
        for row, roi in zip(selected, rois):
            c0 = image["coeff"][row["raw_id"]].cuda()
            old = old_coefficients[str(row["annotation_id"])].cuda() - c0
            instance_targets = {}
            box = image["boxes"][row["raw_id"]].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            predicted = (prototype[:, support].T.contiguous(),
                         (owners[support] == image["owners"][row["row_index"]].item() + 1).float())
            original = metrics(c0, roi, predicted)
            regularized = {}
            for penalty in LAMBDAS:
                delta, state = solve(c0, roi, penalty, args.iterations)
                regularized[penalty] = delta
                instance_targets[str(penalty)] = delta.cpu()
                current = metrics(c0 + delta, roi, predicted)
                rows.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                                 original_mask_iou=row["initial_iou"], box_iou=row["box_iou"],
                                 penalty=penalty, delta_norm=float(delta.norm()),
                                 old_oracle_norm=float(old.norm()),
                                 old_coefficient_cos=cosine(delta, old),
                                 old_response_cos=cosine(roi[0] @ delta, roi[0] @ old),
                                 **state, **{f"initial_{k}": v for k, v in original.items()},
                                 **{f"new_{k}": v for k, v in current.items()}))
            targets[str(row["annotation_id"])] = instance_targets
            if row["annotation_id"] in stability_ids:
                penalty = 0.03
                full = regularized[penalty]
                generator = torch.Generator(device=c0.device).manual_seed(20260925 + row["annotation_id"])
                alternate_start = torch.randn(c0.shape, generator=generator, device=c0.device) * 0.25
                alternate, alternate_state = solve(c0, roi, penalty, args.iterations, start=alternate_start)
                samples = []
                for seed in (100003 + row["annotation_id"], 200003 + row["annotation_id"]):
                    sampled_roi, weights = subsample_roi(roi, seed)
                    candidate, sampled_state = solve(c0, sampled_roi, penalty, args.iterations, weights=weights)
                    samples.append((candidate, sampled_state))
                full_obj = next(r["objective"] for r in rows[::-1]
                                if r["annotation_id"] == row["annotation_id"] and r["penalty"] == penalty)
                checks.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                                   full_norm=float(full.norm()),
                                   alternate_relative_distance=float((alternate - full).norm() / full.norm().clamp_min(1e-9)),
                                   alternate_objective_gap=alternate_state["objective"] - full_obj,
                                   alternate_stationary_norm=alternate_state["stationary_norm"],
                                   sample_direction_cos=[cosine(candidate, full) for candidate, _ in samples],
                                   sample_pair_cos=cosine(samples[0][0], samples[1][0]),
                                   sample_relative_distance=[float((candidate - full).norm() / full.norm().clamp_min(1e-9))
                                                             for candidate, _ in samples],
                                   sample_stationary_norm=[state["stationary_norm"] for _, state in samples]))
        if image_number % 10 == 0 or image_number == len(grouped):
            progress = dict(images=image_number, total_images=len(grouped), instances=len({item_key(r) for r in rows}))
            print(json.dumps(progress), flush=True)
            write(args.out / "PROGRESS.json", progress)
            write(args.out / "ROWS.json", rows)
            write(args.out / "STABILITY.json", checks)
            torch.save(targets, args.out / "TARGETS.pt")
    write(args.out / "COMPLETE.json", dict(status="completed", instances=len(prior_rows),
                                           penalties=list(LAMBDAS), rows=len(rows), stability=len(checks)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metrics-source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=120)
    main(parser.parse_args())
