"""Matched official-mask-loss probes for h64 and upstream PCA64 features."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
from pycocotools.coco import COCO
import torch


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import decode, load, roi_losses, setup, target_rois, write
    from feature_probes import Probe

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.bank / "INDEX.json").read_text())
    def item(iid):
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        feature = load(a.features / "features" / f"{iid:012d}.pt")
        raw_ids = torch.tensor([row["raw_id"] for row in image["rows"]])
        assert torch.equal(raw_ids, feature["raw_ids"])
        return image, feature, raw_ids
    per_level = defaultdict(list)
    hs, cs = [], []
    for entry in index["fit"]:
        if not entry["n"]:
            continue
        image, feature, raw_ids = item(entry["image_id"])
        hs.append(image["h"][raw_ids]); cs.append(image["coeff"][raw_ids])
        for tensor, level in zip(feature["features"], feature["levels"].tolist()):
            per_level[level].append(tensor)
    pca = {}
    projected_train = []
    for level in range(3):
        x = torch.stack(per_level[level]).double()
        mean = x.mean(0)
        centered = x - mean
        cov = centered.T @ centered / max(len(x) - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        components = eigenvectors[:, -64:].flip(1)
        pca[level] = dict(mean=mean.float(), components=components.float(),
                          retained_variance=float(eigenvalues[-64:].sum() / eigenvalues.clamp_min(0).sum()))
        projected_train.append((centered @ components).float())
    h = torch.cat(hs)
    c = torch.cat(cs)
    z = torch.cat(projected_train)
    stats = {"h": (h.mean(0), h.std(0).clamp_min(.01)),
             "upstream": (z.mean(0), z.std(0).clamp_min(.01))}
    write(a.out / "PCA.json", {str(level): dict(samples=len(per_level[level]),
        original_dim=len(pca[level]["mean"]), retained_variance=pca[level]["retained_variance"])
        for level in range(3)})
    torch.save(pca, a.out / "PCA.pt")
    models, optimizers = {}, {}
    for arm in ("h", "upstream"):
        torch.manual_seed(0)
        models[arm] = Probe("mlp_large", stats[arm][0], stats[arm][1],
            torch.zeros(512), torch.ones(512), c.std(0).clamp_min(.1)).cuda()
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=.0001)
    parameter_counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    assert len(set(parameter_counts.values())) == 1

    def projected(feature):
        rows = []
        for x, level in zip(feature["features"], feature["levels"].tolist()):
            rows.append((x - pca[level]["mean"]) @ pca[level]["components"])
        return torch.stack(rows)

    def coefficients(image, feature, raw_ids, arm):
        current = image["coeff"][raw_ids].cuda()
        x = image["h"][raw_ids] if arm == "h" else projected(feature)
        return current + models[arm](x.cuda(), torch.zeros(len(raw_ids), 512, device="cuda"),
                                      image["levels"][raw_ids].cuda())

    def evaluate(split, final=False):
        total = defaultdict(float)
        n = 0
        rows = []
        coco = COCO(str(a.source / "data/annotations/instances_val2017.json")) if final else None
        with torch.no_grad():
            for entry in index[split]:
                if not entry["n"]:
                    continue
                image, feature, raw_ids = item(entry["image_id"])
                rois = target_rois(image)
                values = {"original": image["coeff"][raw_ids].cuda()}
                values.update({arm: coefficients(image, feature, raw_ids, arm) for arm in models})
                for arm, coeff in values.items():
                    total[arm] += float(roi_losses(coeff[None], rois)[0])
                n += len(raw_ids)
                if final:
                    predictions = {arm: decode(image, coeff, raw_ids) for arm, coeff in values.items()}
                    for k, record in enumerate(image["rows"]):
                        gt = coco.annToMask(coco.anns[record["annotation_id"]]).astype(bool)
                        ious = {arm: float((masks[k] & gt).sum() / max((masks[k] | gt).sum(), 1))
                                for arm, masks in predictions.items()}
                        rows.append(dict(image_id=record["image_id"], annotation_id=record["annotation_id"], iou=ious))
        return {arm: loss / n for arm, loss in total.items()}, rows

    baseline, _ = evaluate("dev")
    best = {arm: (float("inf"), 0) for arm in models}
    history = []
    for epoch in range(1, a.epochs + 1):
        ordered = [entry for entry in index["fit"] if entry["n"]]
        random.Random(20260924 + epoch).shuffle(ordered)
        train_total = defaultdict(float)
        for start in range(0, len(ordered), 8):
            chunk = ordered[start:start + 8]
            n = sum(entry["n"] for entry in chunk)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for entry in chunk:
                image, feature, raw_ids = item(entry["image_id"])
                rois = target_rois(image)
                losses = []
                for arm in models:
                    value = roi_losses(coefficients(image, feature, raw_ids, arm)[None], rois)[0]
                    losses.append(value)
                    train_total[arm] += float(value.detach())
                (torch.stack(losses).sum() * image["segmentation_gain"] / n).backward()
            for arm, model in models.items():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizers[arm].step()
        dev, _ = evaluate("dev")
        record = dict(epoch=epoch,
            train={arm: train_total[arm] / sum(entry["n"] for entry in index["fit"]) for arm in models},
            dev=dev, original_dev=baseline["original"])
        history.append(record)
        write(a.out / "HISTORY.json", history)
        print(json.dumps(record), flush=True)
        for arm, model in models.items():
            if dev[arm] < best[arm][0]:
                best[arm] = (dev[arm], epoch)
                torch.save(dict(epoch=epoch, state_dict=model.state_dict()), a.out / f"{arm}_best.pt")
        if epoch >= a.min_epochs and all(epoch - best[arm][1] >= a.patience for arm in models):
            break
    for arm, model in models.items():
        model.load_state_dict(load(a.out / f"{arm}_best.pt")["state_dict"])
        model.eval()
    val, val_rows = evaluate("val", final=True)
    write(a.out / "VAL_ROWS.json", val_rows)
    write(a.out / "SUMMARY.json", dict(parameter_counts=parameter_counts,
        dev_original=baseline["original"], val_bce=val,
        best={arm: dict(epoch=best[arm][1], dev=best[arm][0],
            gated_epoch=best[arm][1] if best[arm][0] < baseline["original"] else 0) for arm in models},
        val_iou={arm: float(np.mean([row["iou"][arm] for row in val_rows])) for arm in ("original", *models)},
        val_mask75={arm: int(sum(row["iou"][arm] >= .75 for row in val_rows)) for arm in ("original", *models)},
        scope="Frozen official one2one positives; same MLP, 800fit/200dev/200val, PCA learned on fit only"))
    write(a.out / "COMPLETE.json", dict(epochs=len(history), val_instances=len(val_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "features", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    main(parser.parse_args())
