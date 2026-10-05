"""Experiment 7B: matched coefficient-head training with four pixel losses."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops


ARMS = {
    "official_bce": {"kind": "uniform"},
    "negative_weighted": {"kind": "fixed", "negative": 1.5},
    "positive_weighted": {"kind": "fixed", "positive": 1.5},
    "hard_negative": {"kind": "hard", "negative": 1.5, "gamma": 2.0},
}


def weighted_roi_loss(coefficients, rois, arm):
    result = coefficients.sum((1, 2)) * 0
    spec = ARMS[arm]
    for k, (p, y, area) in enumerate(rois):
        z = coefficients[:, k] @ p.T
        bce = F.binary_cross_entropy_with_logits(z, y.expand_as(z), reduction="none")
        if spec["kind"] == "uniform":
            weight = torch.ones_like(bce)
        elif spec["kind"] == "fixed":
            weight = torch.where(y[None] > 0.5,
                                 torch.full_like(bce, spec.get("positive", 1.0)),
                                 torch.full_like(bce, spec.get("negative", 1.0)))
        else:
            confidence = z.detach().sigmoid().pow(spec["gamma"])
            weight = torch.where(y[None] > 0.5, torch.ones_like(bce),
                                 1.0 + spec["negative"] * confidence)
        weight = weight / weight.mean().clamp_min(1e-8)
        result = result + (bce * weight).sum(1) / area
    return result


def auc(logits, truth):
    values = logits.detach().cpu().numpy().astype(np.float64)
    labels = truth.detach().cpu().numpy().astype(np.int64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(values, kind="stable")
    ranked, y = values[order], labels[order]
    boundaries = np.r_[0, np.flatnonzero(ranked[:-1] != ranked[1:]) + 1, len(ranked)]
    positive_counts = np.add.reduceat(y, boundaries[:-1])
    mean_ranks = (boundaries[:-1] + 1 + boundaries[1:]) / 2
    rank_sum = float(np.dot(positive_counts, mean_ranks))
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import decode, load, roi_losses, setup, target_rois, write
    from feature_probes import Probe
    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.bank / "INDEX.json").read_text())
    fit_rows = {r["image_id"]: r for r in index["fit"] if r["n"] > 0}
    dev_ids = [r["image_id"] for r in index["dev"] if r["n"] > 0]
    val_ids = [r["image_id"] for r in index["val"] if r["n"] > 0]
    selected = sorted(fit_rows)
    write(a.out / "SELECTION.json", dict(fit=selected, dev=dev_ids, val=val_ids,
                                           source="Official one2one frozen cache"))

    fit_images = {}
    hs, cs = [], []
    for iid in selected:
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        fit_images[iid] = image
        ids = torch.tensor([r["raw_id"] for r in image["rows"]])
        if len(ids):
            hs.append(image["h"][ids]); cs.append(image["coeff"][ids])
    h = torch.cat(hs); c = torch.cat(cs)
    nullp = torch.zeros(512)
    initial_model = Probe("mlp_large", h.mean(0), h.std(0).clamp_min(.01), nullp,
                          torch.ones_like(nullp), c.std(0).clamp_min(.1)).cuda()
    models = {arm: deepcopy(initial_model) for arm in ARMS}
    optimizers = {arm: torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=.0001)
                  for arm in ARMS}

    def image_loss(image, arm):
        ids = torch.tensor([r["raw_id"] for r in image["rows"]])
        if not len(ids):
            return None, 0
        features = image["h"][ids].cuda()
        coeff = image["coeff"][ids].cuda()
        levels = image["levels"][ids].cuda()
        predicted = coeff + models[arm](features, torch.zeros(len(ids), 512, device="cuda"), levels)
        return weighted_roi_loss(predicted[None], target_rois(image), arm)[0], len(ids)

    def mean_loss(ids, arm, training=False, epoch_number=0):
        total, count = 0.0, 0
        sequence = list(ids)
        if training:
            random.Random(a.seed + epoch_number).shuffle(sequence)
            for start in range(0, len(sequence), 8):
                batch = sequence[start:start + 8]
                n_batch = sum(fit_rows[i]["n"] for i in batch)
                optimizers[arm].zero_grad(set_to_none=True)
                for iid in batch:
                    image = fit_images[iid]
                    value, n = image_loss(image, arm)
                    (value * image["segmentation_gain"] * gradient_scale[arm] / n_batch).backward()
                    total += float(value.detach()); count += n
                torch.nn.utils.clip_grad_norm_(models[arm].parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        else:
            with torch.no_grad():
                for iid in sequence:
                    image = fit_images[iid] if iid in fit_images else load(a.bank / "images" / f"{iid:012d}.pt")
                    value, n = image_loss(image, arm)
                    if n:
                        total += float(value); count += n
        return total / count if count else None

    def official_dev_loss(arm):
        total, count = 0.0, 0
        with torch.no_grad():
            for iid in dev_ids:
                image = load(a.bank / "images" / f"{iid:012d}.pt")
                ids = torch.tensor([r["raw_id"] for r in image["rows"]])
                if not len(ids): continue
                candidate = image["coeff"][ids].cuda() + models[arm](
                    image["h"][ids].cuda(), torch.zeros(len(ids), 512, device="cuda"), image["levels"][ids].cuda())
                total += float(roi_losses(candidate[None], target_rois(image))[0]); count += len(ids)
        return total / count

    # Match the initial parameter-gradient scale on the same first eight images.
    reference_batch = selected[:8]
    initial_gradient_norm = {}
    for arm in ARMS:
        models[arm].zero_grad(set_to_none=True)
        n_batch = sum(fit_rows[i]["n"] for i in reference_batch)
        for iid in reference_batch:
            image = fit_images[iid]
            value, _ = image_loss(image, arm)
            (value * image["segmentation_gain"] / n_batch).backward()
        initial_gradient_norm[arm] = float(torch.sqrt(sum(
            parameter.grad.detach().square().sum() for parameter in models[arm].parameters()
            if parameter.grad is not None)))
        models[arm].zero_grad(set_to_none=True)
    baseline_norm = initial_gradient_norm["official_bce"]
    gradient_scale = {arm: baseline_norm / norm for arm, norm in initial_gradient_norm.items()}
    write(a.out / "GRADIENT_SCALE.json", dict(reference_images=reference_batch,
        initial_gradient_norm=initial_gradient_norm, multiplier=gradient_scale))

    dev_original = None
    with torch.no_grad():
        total, count = 0.0, 0
        for iid in dev_ids:
            image = load(a.bank / "images" / f"{iid:012d}.pt")
            ids = torch.tensor([r["raw_id"] for r in image["rows"]])
            if not len(ids): continue
            total += float(roi_losses(image["coeff"][ids].cuda()[None], target_rois(image))[0]); count += len(ids)
        dev_original = total / count

    def dev_auc():
        scores = {arm: [] for arm in ARMS}
        with torch.no_grad():
            for iid in dev_ids:
                image = load(a.bank / "images" / f"{iid:012d}.pt")
                rows = image["rows"]
                if not rows: continue
                ids = torch.tensor([r["raw_id"] for r in rows])
                proto = F.interpolate(image["proto"].cuda()[None], (640, 640),
                                      mode="bilinear", align_corners=False)[0]
                base = image["coeff"][ids].cuda()
                features = image["h"][ids].cuda()
                levels = image["levels"][ids].cuda()
                supports = ops.crop_mask(torch.ones(len(ids), 640, 640, device="cuda"),
                                          image["boxes"][ids].cuda()).bool()
                labels = image["masks"].cuda()
                for arm in ARMS:
                    candidate = base + models[arm](features, torch.zeros(len(ids), 512, device="cuda"), levels)
                    logits = (candidate @ proto.flatten(1)).reshape(-1, 640, 640)
                    for j, row_item in enumerate(rows):
                        truth = labels == image["owners"][j].item() + 1
                        score = auc(logits[j][supports[j]], truth[supports[j]])
                        if score is not None: scores[arm].append(score)
        return {arm: float(np.mean(values)) for arm, values in scores.items()}

    original_dev_auc = None
    # At initialization every arm has an exact zero coefficient residual.
    for arm in ARMS: models[arm].eval()
    original_dev_auc = dev_auc()["official_bce"]

    history = []
    best = {arm: dev_original for arm in ARMS}
    chosen = {arm: 0 for arm in ARMS}
    stale = {arm: 0 for arm in ARMS}
    for epoch in range(1, a.epochs + 1):
        row = {"epoch": epoch, "arms": {}}
        for arm in ARMS:
            models[arm].train()
            train = mean_loss(selected, arm, training=True, epoch_number=epoch)
            models[arm].eval()
            dev_weighted = mean_loss(dev_ids, arm)
            dev_official = official_dev_loss(arm)
            row["arms"][arm] = dict(train_loss=train, dev_weighted_loss=dev_weighted,
                                     dev_official_bce=dev_official, dev_original_official_bce=dev_original)
        auc_values = dev_auc()
        for arm in ARMS:
            row["arms"][arm]["dev_auc"] = auc_values[arm]
            dev_official = row["arms"][arm]["dev_official_bce"]
            if dev_official < best[arm]:
                best[arm] = dev_official; chosen[arm] = epoch; stale[arm] = 0
                torch.save({"state_dict": models[arm].state_dict(), "epoch": epoch,
                            "dev_official_bce": dev_official, "dev_auc": auc_values[arm]},
                           a.out / f"{arm}_best.pt")
            else:
                stale[arm] += 1
            torch.save({"state_dict": models[arm].state_dict(), "epoch": epoch,
                        "dev_official_bce": row["arms"][arm]["dev_official_bce"]},
                       a.out / f"{arm}_last.pt")
        history.append(row); write(a.out / "HISTORY.json", history); write(a.out / "PROGRESS.json", row)
        print(json.dumps(row), flush=True)
        if all(stale[arm] >= a.patience and epoch >= a.min_epochs for arm in ARMS): break

    coco = COCO(str(a.source / "data/annotations/instances_val2017.json"))
    oracle_rows = json.loads((a.oracle / "ROWS.json").read_text())
    oracle_coeff = load(a.oracle / "oracle_coefficients.pt")
    oracle_lookup = {(r["image_id"], r["annotation_id"]): r for r in oracle_rows}
    def evaluate():
        val_rows = {arm: [] for arm in ("original", *ARMS)}
        alignment = {arm: [] for arm in ARMS}
        with torch.no_grad():
            for iid in val_ids:
                image = load(a.bank / "images" / f"{iid:012d}.pt")
                rows = image["rows"]
                if not rows: continue
                ids = torch.tensor([r["raw_id"] for r in rows])
                features = image["h"][ids].cuda(); base = image["coeff"][ids].cuda(); levels = image["levels"][ids].cuda()
                truth_masks = [torch.from_numpy(coco.annToMask(coco.anns[r["annotation_id"]]).astype(bool)) for r in rows]
                proto = F.interpolate(image["proto"].cuda()[None], (640, 640), mode="bilinear", align_corners=False)[0]
                supports = ops.crop_mask(torch.ones(len(ids), 640, 640, device="cuda"),
                                          image["boxes"][ids].cuda()).bool()
                labels = image["masks"].cuda()
                for arm in ("original", *ARMS):
                    candidate = base if arm == "original" else base + models[arm](
                        features, torch.zeros(len(ids), 512, device="cuda"), levels)
                    masks = decode(image, candidate, ids)
                    for j, (row_item, mask, truth_np) in enumerate(zip(rows, masks, truth_masks)):
                        truth = truth_np.cuda(); pred = torch.from_numpy(mask).cuda()
                        iou = float((pred & truth).sum() / (pred | truth).sum().clamp_min(1))
                        support = supports[j]
                        label = labels == image["owners"][j].item() + 1
                        logits = torch.einsum("c,chw->hw", candidate[j], proto)
                        auc_value = auc(logits[support], label[support])
                        false_positive = ((logits[support] > 0) & ~label[support]).sum()
                        negatives = (~label[support]).sum().clamp_min(1)
                        coverage = float(((logits[support] > 0) & label[support]).sum() /
                                         label[support].sum().clamp_min(1))
                        val_rows[arm].append(dict(image_id=iid, annotation_id=row_item["annotation_id"],
                            iou=iou, auc=auc_value, fp_inside_box=int(false_positive),
                            fp_rate=float(false_positive / negatives), target_coverage=coverage,
                            baseline_iou=row_item["initial_iou"], box_iou=row_item["box_iou"]))
                        oracle = oracle_lookup.get((iid, row_item["annotation_id"]))
                        if oracle is not None and arm != "original":
                            base_z = torch.einsum("c,chw->hw", base[j], proto)[support]
                            oracle_z = torch.einsum("c,chw->hw", oracle_coeff[str(row_item["annotation_id"])].cuda(), proto)[support]
                            model_z = logits[support]
                            dm = model_z.clamp(-10, 10) - base_z.clamp(-10, 10)
                            do = oracle_z.clamp(-10, 10) - base_z.clamp(-10, 10)
                            alignment[arm].append(dict(image_id=iid, annotation_id=row_item["annotation_id"],
                                failed=oracle["initial_iou"] < .75,
                                cosine=float((dm @ do) / (dm.norm() * do.norm()).clamp_min(1e-8))))
        return val_rows, alignment

    def summarize_rows(rows):
        return dict(n=len(rows), images=len({r["image_id"] for r in rows}),
                    mean_iou=float(np.mean([r["iou"] for r in rows])),
                    mask75=int(sum(r["iou"] >= .75 for r in rows)),
                    mean_auc=float(np.mean([r["auc"] for r in rows if r["auc"] is not None])),
                    mean_fp_inside_box=float(np.mean([r["fp_inside_box"] for r in rows])),
                    mean_fp_rate=float(np.mean([r["fp_rate"] for r in rows])),
                    mean_target_coverage=float(np.mean([r["target_coverage"] for r in rows])))

    def summarize_alignment(rows):
        result = {}
        for name, subset in (("all", rows), ("failed", [r for r in rows if r["failed"]])):
            result[name] = dict(n=len(subset), mean_cosine=float(np.mean([r["cosine"] for r in subset])) if subset else None)
        return result

    last_rows, last_alignment = evaluate()
    for arm in ARMS:
        if chosen[arm]:
            models[arm].load_state_dict(load(a.out / f"{arm}_best.pt")["state_dict"])
        else:
            models[arm].load_state_dict(initial_model.state_dict())
        models[arm].eval()
    selected_rows, selected_alignment = evaluate()
    output = dict(arms=list(ARMS), chosen_epoch=chosen, best_dev_official_bce=best,
                  original_dev_auc=original_dev_auc,
                  selected_beats_original_dev={arm: best[arm] < dev_original for arm in ARMS},
                  initial_gradient_norm=initial_gradient_norm, gradient_scale=gradient_scale,
                  last_val={arm: summarize_rows(rows) for arm, rows in last_rows.items()},
                  selected_val={arm: summarize_rows(rows) for arm, rows in selected_rows.items()},
                  last_oracle_alignment={arm: summarize_alignment(rows) for arm, rows in last_alignment.items()},
                  selected_oracle_alignment={arm: summarize_alignment(rows) for arm, rows in selected_alignment.items()},
                  scope="Frozen official one2one candidates; single seed; not COCO AP")
    write(a.out / "LAST_VAL_ROWS.json", last_rows); write(a.out / "SELECTED_VAL_ROWS.json", selected_rows)
    write(a.out / "LAST_ORACLE_ALIGNMENT.json", last_alignment)
    write(a.out / "SELECTED_ORACLE_ALIGNMENT.json", selected_alignment)
    write(a.out / "SUMMARY.json", output)
    write(a.out / "COMPLETE.json", dict(epochs_run=len(history), arms=list(ARMS), chosen_epoch=chosen))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("source", "bank", "oracle", "out"): p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=10); p.add_argument("--seed", type=int, default=20260924)
    main(p.parse_args())
