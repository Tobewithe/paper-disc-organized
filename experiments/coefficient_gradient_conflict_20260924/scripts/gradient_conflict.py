"""Compare per-instance gradients of shared h-only coefficient-head weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, roi_losses, setup, target_rois, write
    from feature_probes import Probe

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    oracle = json.loads((a.oracle / "ROWS.json").read_text())
    model_state = load(a.checkpoint)["state_dict"]
    net = Probe("mlp_large", model_state["h_mean"], model_state["h_std"],
        model_state["p_mean"], model_state["p_std"], model_state["c_scale"]).cuda()
    net.load_state_dict(model_state)
    net.eval()
    groups = defaultdict(list)
    for row in oracle:
        groups[row["image_id"]].append(row)
    entries = []
    for iid, records in groups.items():
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        lookup = {row["annotation_id"]: k for k, row in enumerate(image["rows"])}
        for record in records:
            k = lookup[record["annotation_id"]]
            row = image["rows"][k]
            raw_id = row["raw_id"]
            level = row["level"]
            h = image["h"][raw_id:raw_id + 1].cuda()
            c = image["coeff"][raw_id:raw_id + 1].cuda()
            predicted = c + net(h, torch.zeros(1, 512, device="cuda"),
                                image["levels"][raw_id:raw_id + 1].cuda())
            loss = roi_losses(predicted[None], target_rois(image, [k]))[0]
            params = tuple(net.heads[level].parameters())
            grads = torch.autograd.grad(loss, params)
            vector = torch.cat([grad.flatten() for grad in grads]).detach().cpu().numpy()
            norm = np.linalg.norm(vector)
            assert norm > 0 and np.isfinite(norm)
            entries.append(dict(image_id=iid, annotation_id=row["annotation_id"],
                level=level, initial_iou=record["initial_iou"],
                oracle_gap=record["initial_loss"] - record["oracle_loss"],
                current_loss=float(loss.detach()),
                gradient=vector / norm, gradient_norm=float(norm)))
    results = []
    for level in range(3):
        subset = [entry for entry in entries if entry["level"] == level]
        if len(subset) < 3:
            continue
        matrix = np.stack([entry["gradient"] for entry in subset])
        cos = matrix @ matrix.T
        for i, entry in enumerate(subset):
            other = np.array([j for j, x in enumerate(subset) if j != i and
                              x["image_id"] != entry["image_id"]])
            same = np.array([j for j, x in enumerate(subset) if j != i and
                             x["image_id"] == entry["image_id"]])
            assert len(other)
            results.append(dict(image_id=entry["image_id"], annotation_id=entry["annotation_id"],
                level=level, initial_iou=entry["initial_iou"], oracle_gap=entry["oracle_gap"],
                current_loss=entry["current_loss"], gradient_norm=entry["gradient_norm"],
                cross_image_negative_fraction=float(np.mean(cos[i, other] < 0)),
                cross_image_mean_cos=float(np.mean(cos[i, other])),
                same_image_pairs=len(same),
                same_image_negative_fraction=float(np.mean(cos[i, same] < 0)) if len(same) else None,
                same_image_mean_cos=float(np.mean(cos[i, same])) if len(same) else None))
    def summarize(rows):
        if len(rows) < 3:
            return dict(n=len(rows))
        gaps = np.array([row["oracle_gap"] for row in rows])
        conflict = np.array([row["cross_image_negative_fraction"] for row in rows])
        same = [row for row in rows if row["same_image_pairs"]]
        rho, p = spearmanr(gaps, conflict)
        return dict(n=len(rows), cross_image_negative_fraction=float(conflict.mean()),
            cross_image_mean_cos=float(np.mean([row["cross_image_mean_cos"] for row in rows])),
            same_image_n=len(same),
            same_image_negative_fraction=float(np.mean([row["same_image_negative_fraction"] for row in same])) if same else None,
            gap_vs_conflict_spearman=float(rho), gap_vs_conflict_p=float(p))
    summary = dict(all=summarize(results),
        failed=summarize([r for r in results if r["initial_iou"] < .75]),
        success=summarize([r for r in results if r["initial_iou"] >= .75]),
        per_level={str(level): summarize([r for r in results if r["level"] == level]) for level in range(3)},
        checkpoint=str(a.checkpoint))
    write(a.out / "ROWS.json", results)
    write(a.out / "SUMMARY.json", summary)
    write(a.out / "COMPLETE.json", dict(instances=len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "oracle", "checkpoint", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
