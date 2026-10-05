"""Experiment 7A: decompose fixed-prototype oracle changes in pixel-logit space."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def fit_calibration(z, y, mode, steps=80):
    """Fit per-instance bias or affine logit calibration only for diagnosis."""
    dtype = z.dtype
    if mode == "bias":
        raw = torch.zeros(1, device=z.device, dtype=dtype, requires_grad=True)
    else:
        raw = torch.tensor([0.0, 0.0], device=z.device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        if mode == "bias":
            logits = z + raw[0]
        else:
            alpha = F.softplus(raw[0])
            logits = alpha * z + raw[1]
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        if mode == "bias":
            logits = z + raw[0]
            params = {"bias": float(raw[0])}
        else:
            alpha = F.softplus(raw[0])
            logits = alpha * z + raw[1]
            params = {"alpha": float(alpha), "bias": float(raw[1])}
        loss = float(F.binary_cross_entropy_with_logits(logits, y))
    return logits, params, loss


def iou_from_logits(logits, truth):
    pred = logits > 0
    inter = (pred & truth).sum().item()
    union = (pred | truth).sum().item()
    return float(inter / union) if union else 0.0


def best_threshold_iou(logits, truth):
    """GT-guided one-scalar IoU upper bound, with ties kept at one threshold."""
    values = logits.detach().cpu().numpy().astype(np.float64)
    labels = truth.detach().cpu().numpy().astype(bool)
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    positives = labels[order].astype(np.int64)
    cutoff = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    tp = np.cumsum(positives)[cutoff]
    predicted = cutoff + 1
    union = labels.sum() + predicted - tp
    scores = np.divide(tp, union, out=np.zeros_like(tp, dtype=np.float64), where=union > 0)
    winning = int(np.argmax(scores))
    return float(scores[winning]), float(ranked[cutoff[winning]])


def roc_auc(logits, truth):
    """Tie-aware pairwise ranking of GT pixels against non-GT pixels."""
    values = logits.detach().cpu().numpy().astype(np.float64)
    labels = truth.detach().cpu().numpy().astype(np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    ranked_labels = labels[order]
    boundaries = np.r_[0, np.flatnonzero(ranked[:-1] != ranked[1:]) + 1, len(ranked)]
    positive_rank_sum = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        positive_rank_sum += int(ranked_labels[left:right].sum()) * ((left + 1 + right) / 2)
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def region_stats(delta, region, prefix):
    values = delta[region]
    if values.numel() == 0:
        return {f"{prefix}_pixels": 0}
    return {
        f"{prefix}_pixels": int(values.numel()),
        f"{prefix}_delta_mean": float(values.mean()),
        f"{prefix}_delta_abs_mean": float(values.abs().mean()),
        f"{prefix}_delta_pos_fraction": float((values > 0).float().mean()),
        f"{prefix}_delta_neg_fraction": float((values < 0).float().mean()),
        f"{prefix}_abs_energy": float(values.abs().sum()),
    }


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, setup, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    oracle_rows = json.loads((a.oracle / "ROWS.json").read_text())
    coefficients = load(a.oracle / "oracle_coefficients.pt")
    grouped = defaultdict(list)
    for row in oracle_rows:
        grouped[row["image_id"]].append(row)

    records = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for step, (iid, subset) in enumerate(grouped.items(), 1):
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        prototype = F.interpolate(image["proto"].cuda()[None], (640, 640),
                                   mode="bilinear", align_corners=False)[0]
        label = image["masks"].numpy()
        rows = {row["annotation_id"]: (k, row) for k, row in enumerate(image["rows"])}
        for selected in subset:
            k, row = rows[selected["annotation_id"]]
            raw_id = row["raw_id"]
            c0 = image["coeff"][raw_id].cuda()
            cstar = coefficients[str(row["annotation_id"])].cuda()
            z0_full = torch.einsum("c,chw->hw", c0, prototype)
            zs_full = torch.einsum("c,chw->hw", cstar, prototype)
            box = image["boxes"][raw_id].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            truth = torch.from_numpy((label == int(image["owners"][k]) + 1)).to(device="cuda")
            z0 = z0_full[support]
            zs = zs_full[support]
            y = truth[support].float()
            raw_delta = zs - z0
            # Coefficient BCE can drive separable pixels to arbitrary logit magnitudes.
            # Bound the descriptive logit contrast; keep threshold/IoU tests on raw logits.
            delta = zs.clamp(-10, 10) - z0.clamp(-10, 10)
            probability_delta = zs.sigmoid() - z0.sigmoid()

            truth_np = truth.cpu().numpy().astype(np.uint8)
            boundary_np = cv2.dilate(truth_np, kernel).astype(bool) & ~cv2.erode(truth_np, kernel).astype(bool)
            target = truth & support
            boundary = torch.from_numpy(boundary_np).to(device="cuda") & target
            interior = target & ~boundary
            outside = support & ~truth
            neighbor = outside & torch.from_numpy(label != 0).to(device="cuda")
            background = outside & ~torch.from_numpy(label != 0).to(device="cuda")
            bx = box.detach().cpu().numpy()
            yy, xx = np.mgrid[:640, :640]
            edge_np = support.cpu().numpy() & ((xx < bx[0] + 8) | (xx >= bx[2] - 8) |
                                                (yy < bx[1] + 8) | (yy >= bx[3] - 8))
            edge = torch.from_numpy(edge_np).to(device="cuda")

            with torch.no_grad():
                # How much of the continuous oracle change is a global shift or affine transform?
                dmean = delta.mean()
                centered_z = z0 - z0.mean()
                centered_d = delta - dmean
                denom = (centered_z * centered_z).sum()
                slope = (centered_z * centered_d).sum() / denom if denom > 1e-12 else torch.zeros_like(denom)
                intercept = dmean - slope * z0.mean()
                delta_bias = torch.full_like(delta, dmean)
                delta_affine = slope * z0 + intercept
                total_energy = delta.abs().sum().clamp_min(1e-12)
                affine_r2 = 1.0 - ((delta - delta_affine) ** 2).sum() / ((delta - dmean) ** 2).sum().clamp_min(1e-12)
                delta_energy = (delta * delta).sum().clamp_min(1e-12)
                constant_energy_share = (delta_bias * delta_bias).sum() / delta_energy
                affine_energy_share = 1.0 - ((delta - delta_affine) ** 2).sum() / delta_energy

            bias_logits, bias_params, bias_loss = fit_calibration(z0, y, "bias")
            affine_logits, affine_params, affine_loss = fit_calibration(z0, y, "affine")
            threshold_iou, threshold_cut = best_threshold_iou(z0, y.bool())
            row_out = dict(
                image_id=iid, annotation_id=row["annotation_id"], box_iou=row["box_iou"],
                initial_iou=row["initial_iou"], oracle_iou=selected["final_iou_after"],
                support_pixels=int(support.sum()), target_pixels=int(target.sum()),
                box_width=float(bx[2] - bx[0]), box_height=float(bx[3] - bx[1]),
                box_edge_pixel_fraction=float(edge.sum() / support.sum()),
                original_logit_abs_p95=float(torch.quantile(z0.abs(), .95)),
                oracle_logit_abs_p95=float(torch.quantile(zs.abs(), .95)),
                oracle_raw_delta_abs_p95=float(torch.quantile(raw_delta.abs(), .95)),
                oracle_logit_saturation_fraction=float((zs.abs() > 10).float().mean()),
                oracle_clipped_delta_abs_mean=float(delta.abs().mean()),
                oracle_clipped_delta_abs_energy=float(delta.abs().sum()),
                bias_delta_mean=float(dmean), bias_delta_slope=float(slope), bias_delta_intercept=float(intercept),
                constant_delta_energy_share=float(constant_energy_share),
                affine_delta_energy_share=float(affine_energy_share), affine_centered_r2=float(affine_r2),
                initial_grid_iou=iou_from_logits(z0, y.bool()),
                oracle_grid_iou=iou_from_logits(zs, y.bool()),
                initial_roc_auc=roc_auc(z0, y.bool()),
                oracle_roc_auc=roc_auc(zs, y.bool()),
                mean_shift_iou=iou_from_logits(z0 + dmean, y.bool()),
                fitted_bias_iou=iou_from_logits(bias_logits, y.bool()),
                fitted_affine_iou=iou_from_logits(affine_logits, y.bool()),
                best_threshold_iou=threshold_iou, best_threshold_cut=threshold_cut,
                fitted_bias_loss=bias_loss, fitted_affine_loss=affine_loss,
                fitted_bias=float(bias_params["bias"]), fitted_affine_alpha=float(affine_params["alpha"]),
                fitted_affine_bias=float(affine_params["bias"]),
            )
            for name, mask in (("target_boundary", boundary), ("target_interior", interior),
                               ("outside", outside), ("neighbor", neighbor),
                               ("background", background), ("box_edge", edge)):
                stats = region_stats(delta, mask[support], name)
                row_out.update(stats)
                prob = probability_delta[mask[support]]
                if prob.numel():
                    row_out[f"{name}_probability_delta_mean"] = float(prob.mean())
                    row_out[f"{name}_probability_delta_abs_mean"] = float(prob.abs().mean())
                if stats.get(f"{name}_pixels", 0):
                    row_out[f"{name}_abs_energy_share"] = stats[f"{name}_abs_energy"] / float(total_energy)
            records.append(row_out)
        if step % 25 == 0 or step == len(grouped):
            state = dict(images=step, total=len(grouped), instances=len(records))
            print(json.dumps(state), flush=True)
            write(a.out / "PROGRESS.json", state)

    def mean(rows, key):
        values = [r[key] for r in rows if key in r and r[key] is not None]
        return float(np.mean(values)) if values else None

    def summarize(rows):
        useful = [r for r in rows if r["oracle_iou"] > r["initial_iou"]]
        return dict(
            instances=len(rows), improved=len(useful),
            mean_initial_iou=mean(rows, "initial_iou"), mean_oracle_iou=mean(rows, "oracle_iou"),
            mean_initial_grid_iou=mean(rows, "initial_grid_iou"),
            mean_oracle_grid_iou=mean(rows, "oracle_grid_iou"),
            mean_initial_roc_auc=mean(rows, "initial_roc_auc"),
            mean_oracle_roc_auc=mean(rows, "oracle_roc_auc"),
            mean_fitted_bias_iou=mean(rows, "fitted_bias_iou"), mean_fitted_affine_iou=mean(rows, "fitted_affine_iou"),
            mean_best_threshold_iou=mean(rows, "best_threshold_iou"),
            mean_constant_delta_energy_share=mean(rows, "constant_delta_energy_share"),
            mean_affine_delta_energy_share=mean(rows, "affine_delta_energy_share"),
            mean_affine_centered_r2=mean(rows, "affine_centered_r2"),
            mean_oracle_clipped_delta_abs=mean(rows, "oracle_clipped_delta_abs_mean"),
            mean_oracle_logit_abs_p95=mean(rows, "oracle_logit_abs_p95"),
            mean_oracle_logit_saturation_fraction=mean(rows, "oracle_logit_saturation_fraction"),
            mean_target_boundary_energy_share=mean(rows, "target_boundary_abs_energy_share"),
            mean_target_interior_energy_share=mean(rows, "target_interior_abs_energy_share"),
            mean_outside_energy_share=mean(rows, "outside_abs_energy_share"),
            mean_neighbor_energy_share=mean(rows, "neighbor_abs_energy_share"),
            mean_background_energy_share=mean(rows, "background_abs_energy_share"),
            mean_box_edge_energy_share=mean(rows, "box_edge_abs_energy_share"),
            mean_box_edge_pixel_fraction=mean(rows, "box_edge_pixel_fraction"),
            mean_target_interior_probability_delta=mean(rows, "target_interior_probability_delta_mean"),
            mean_target_boundary_probability_delta=mean(rows, "target_boundary_probability_delta_mean"),
            mean_neighbor_probability_delta=mean(rows, "neighbor_probability_delta_mean"),
            mean_background_probability_delta=mean(rows, "background_probability_delta_mean"),
        )

    groups = dict(
        all=records,
        failed=[r for r in records if r["initial_iou"] < .75],
        success=[r for r in records if r["initial_iou"] >= .75],
        repaired=[r for r in records if r["initial_iou"] < .75 <= r["oracle_iou"]],
        good_box_failed=[r for r in records if r["box_iou"] >= .75 and r["initial_iou"] < .75],
    )
    write(a.out / "ROWS.json", records)
    write(a.out / "SUMMARY.json", {name: summarize(rows) for name, rows in groups.items()})
    write(a.out / "COMPLETE.json", dict(instances=len(records), images=len(grouped),
                                        experiment="7A_pixel_logit_decomposition",
                                        descriptive_logit_clip=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "oracle", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
