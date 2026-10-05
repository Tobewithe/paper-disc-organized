"""Experiment 7C on the existing official one-to-one oracle sample."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if float(denom) > 1e-12 else None


def auc(logits, labels):
    scores = logits.detach().cpu().numpy().astype(np.float64)
    truth = labels.detach().cpu().numpy().astype(np.int64)
    npos = int(truth.sum())
    nneg = len(truth) - npos
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind="stable")
    ranked, y = scores[order], truth[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    ends = np.r_[starts[1:], len(ranked)]
    counts = np.add.reduceat(y, starts)
    rank_sum = (counts * (starts + ends + 1) / 2).sum()
    return float((rank_sum - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(c, roi, predicted):
    p, y, area = roi
    z = p @ c
    pos = F.softplus(-z[y.bool()]).sum() / area
    neg = F.softplus(z[~y.bool()]).sum() / area
    p_eval, y_eval = predicted
    z_eval = p_eval @ c
    label = y_eval.bool()
    hard = z_eval > 0
    tp = int((hard & label).sum())
    fp = int((hard & ~label).sum())
    fn = int((~hard & label).sum())
    nneg = int((~label).sum())
    npos = int(label.sum())
    return dict(loss=float(pos + neg), positive_loss=float(pos), negative_loss=float(neg),
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                coverage=tp / npos if npos else None,
                fpr=fp / nneg if nneg else None, auc=auc(z_eval, y_eval))


def gradient_record(c0, cstar, roi, row, validate):
    p, y, area = roi
    z = p @ c0
    residual = z.sigmoid() - y
    positive = p[y.bool()].T @ residual[y.bool()] / area
    negative = p[~y.bool()].T @ residual[~y.bool()] / area
    total = positive + negative
    direction = cstar - c0
    initial = F.binary_cross_entropy_with_logits(z, y, reduction="sum") / area
    final = F.binary_cross_entropy_with_logits(p @ cstar, y, reduction="sum") / area
    if validate:
        probe = c0.detach().clone().requires_grad_(True)
        loss = F.binary_cross_entropy_with_logits(p @ probe, y, reduction="sum") / area
        reference = torch.autograd.grad(loss, probe)[0]
        assert float((reference - total).abs().max()) < 2e-6
    descent_dot = float(total @ direction)
    loss_delta = float(final - initial)
    # Convexity gives L(c*) >= L(c0) + grad L(c0) . (c* - c0).
    assert descent_dot <= loss_delta + 1e-3, (row["annotation_id"], descent_dot, loss_delta)
    return dict(image_id=row["image_id"], annotation_id=row["annotation_id"],
                initial_iou=row["initial_iou"], box_iou=row["box_iou"],
                oracle_loss=float(final), initial_loss=float(initial),
                positive_norm=float(positive.norm()), negative_norm=float(negative.norm()),
                total_norm=float(total.norm()), oracle_distance=float(direction.norm()),
                pos_neg_cos=cosine(positive, negative),
                cancellation=float(total.norm() / (positive.norm() + negative.norm()).clamp_min(1e-12)),
                total_descent_oracle_cos=cosine(-total, direction),
                positive_descent_oracle_cos=cosine(-positive, direction),
                negative_descent_oracle_cos=cosine(-negative, direction),
                positive_directional_derivative=float(positive @ direction),
                negative_directional_derivative=float(negative @ direction),
                total_directional_derivative=descent_dot,
                simultaneous_oracle_descent=bool(float(positive @ direction) < 0 and float(negative @ direction) < 0))


def optimize_path(c0, roi, predicted, iterations):
    c = c0.detach().clone().requires_grad_(True)
    p, y, area = roi
    optimizer = torch.optim.LBFGS([c], lr=1, max_iter=1, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)
    result = [dict(step=0, distance=0.0, **metrics(c.detach(), roi, predicted))]

    def closure():
        optimizer.zero_grad()
        value = F.binary_cross_entropy_with_logits(p @ c, y, reduction="sum") / area
        value.backward()
        return value

    for step in range(1, iterations + 1):
        before = c.detach().clone()
        optimizer.step(closure)
        result.append(dict(step=step, distance=float((c.detach() - c0).norm()),
                           **metrics(c.detach(), roi, predicted)))
        if float((c.detach() - before).norm()) < 1e-8:
            break
    return result


def summarize(rows):
    def group_metrics(items):
        keys = ("pos_neg_cos", "cancellation", "total_descent_oracle_cos",
                "positive_descent_oracle_cos", "negative_descent_oracle_cos")
        result = {key: float(np.mean([r[key] for r in items if r[key] is not None]))
                  if any(r[key] is not None for r in items) else None for key in keys}
        defined = [r["pos_neg_cos"] for r in items if r["pos_neg_cos"] is not None]
        result.update(instances=len(items), images=len({r["image_id"] for r in items}),
                      gradient_cosine_defined=len(defined),
                      antagonistic_fraction=float(np.mean([value < 0 for value in defined])) if defined else None,
                      simultaneous_oracle_descent_fraction=float(np.mean([r["simultaneous_oracle_descent"] for r in items])))
        return result

    groups = dict(all=rows, original_failure=[r for r in rows if r["initial_iou"] < .75],
                  original_success=[r for r in rows if r["initial_iou"] >= .75],
                  failure_good_box=[r for r in rows if r["initial_iou"] < .75 and r["box_iou"] >= .75])
    rng = random.Random(20260925)
    output = {}
    for name, items in groups.items():
        by_image = defaultdict(list)
        for row in items:
            by_image[row["image_id"]].append(row)
        images = list(by_image)
        point = group_metrics(items)
        boot = defaultdict(list)
        for _ in range(2000):
            draw = [row for iid in rng.choices(images, k=len(images)) for row in by_image[iid]]
            for key, value in group_metrics(draw).items():
                if key not in ("instances", "images", "gradient_cosine_defined") and value is not None:
                    boot[key].append(value)
        output[name] = dict(point=point, image_cluster_ci={key: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                                                               for key, v in boot.items()})
    return output


def main(args):
    sys.path.insert(0, str(args.source / "scripts"))
    from official_pipeline import load, setup, target_rois, write
    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        gradients = json.loads((args.out / "GRADIENTS.json").read_text())
        summary = summarize(gradients)
        summary["sampling"] = dict(instances=len(gradients),
                                   trajectory_instances=len({r["annotation_id"] for r in json.loads((args.out / "TRAJECTORIES.json").read_text())}),
                                   interpolation_failures=len({r["annotation_id"] for r in json.loads((args.out / "INTERPOLATION.json").read_text())}),
                                   lbfgs_iterations=args.iterations)
        write(args.out / "SUMMARY.json", summary)
        write(args.out / "COMPLETE.json", dict(status="completed", instances=len(gradients),
                                               trajectory_instances=summary["sampling"]["trajectory_instances"]))
        print(json.dumps({key: value["point"] for key, value in summary.items() if "point" in value}, indent=2), flush=True)
        return
    oracle_rows = json.loads((args.oracle / "ROWS.json").read_text())
    coefficients = load(args.oracle / "oracle_coefficients.pt")
    by_image = defaultdict(list)
    for row in oracle_rows:
        by_image[row["image_id"]].append(row)
    failures = [r for r in oracle_rows if r["initial_iou"] < .75]
    rng = random.Random(20260925)
    trajectory_ids = set()
    for good_box in (False, True):
        candidates = [r["annotation_id"] for r in failures if (r["box_iou"] >= .75) == good_box]
        trajectory_ids.update(rng.sample(candidates, min(20, len(candidates))))
    gradients, interpolation, trajectories = [], [], []
    alphas = (0, .001, .005, .01, .02, .05, .1, .2, .4, .6, .8, 1)
    for image_number, (image_id, selected) in enumerate(by_image.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        row_indices = [row["row_index"] for row in selected]
        rois = target_rois(image, row_indices)
        prototype = F.interpolate(image["proto"].cuda()[None], (640, 640),
                                  mode="bilinear", align_corners=False)[0]
        owner_map = image["masks"].cuda()
        for row, roi in zip(selected, rois):
            raw_id = row["raw_id"]
            c0 = image["coeff"][raw_id].cuda()
            cstar = coefficients[str(row["annotation_id"])].cuda()
            gradient = gradient_record(c0, cstar, roi, row, len(gradients) < 3)
            gradients.append(gradient)
            box = image["boxes"][raw_id].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            p_eval = prototype[:, support].T.contiguous()
            y_eval = (owner_map[support] == image["owners"][row["row_index"]].item() + 1).float()
            predicted = (p_eval, y_eval)
            if row["initial_iou"] < .75:
                for alpha in alphas:
                    c = c0 + alpha * (cstar - c0)
                    interpolation.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                                              alpha=alpha, **metrics(c, roi, predicted)))
            if row["annotation_id"] in trajectory_ids:
                for value in optimize_path(c0, roi, predicted, args.iterations):
                    trajectories.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                                             box_iou=row["box_iou"], **value))
        if image_number % 10 == 0 or image_number == len(by_image):
            print(json.dumps(dict(images=image_number, total_images=len(by_image),
                                  gradients=len(gradients), trajectories=len(trajectory_ids))), flush=True)
            write(args.out / "PROGRESS.json", dict(images=image_number, instances=len(gradients)))
    write(args.out / "GRADIENTS.json", gradients)
    write(args.out / "INTERPOLATION.json", interpolation)
    write(args.out / "TRAJECTORIES.json", trajectories)
    summary = summarize(gradients)
    summary["sampling"] = dict(instances=len(gradients), trajectory_instances=len(trajectory_ids),
                               interpolation_failures=len(failures), lbfgs_iterations=args.iterations)
    write(args.out / "SUMMARY.json", summary)
    write(args.out / "COMPLETE.json", dict(status="completed", instances=len(gradients),
                                           trajectory_instances=len(trajectory_ids)))
    print(json.dumps({key: value["point"] for key, value in summary.items() if "point" in value}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--summarize-only", action="store_true")
    main(parser.parse_args())
