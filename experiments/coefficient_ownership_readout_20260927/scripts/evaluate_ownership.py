"""Evaluate dense ownership readout on existing 7J-N validation candidates."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def cluster_ci(ids, values, mask, reps=2000, seed=20260927):
    images, inv = np.unique(ids[mask], return_inverse=True)
    sums = np.bincount(inv, weights=values[mask], minlength=len(images))
    counts = np.bincount(inv, minlength=len(images))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(images), size=(reps, len(images)), dtype=np.int32)
    boot = sums[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(values[mask].mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist()}


def metrics(logits, label):
    n = len(label)
    bce = F.binary_cross_entropy_with_logits(
        logits, label.expand_as(logits), reduction="none").flatten(1).mean(1).numpy()
    pred = logits.sigmoid().numpy()
    truth = label.numpy()
    binary_pred = pred >= .5
    binary_true = truth >= .5
    intersection = (binary_pred & binary_true).reshape(n, -1).sum(1)
    union = (binary_pred | binary_true).reshape(n, -1).sum(1)
    iou = np.where(union > 0, intersection / np.maximum(union, 1), np.nan)
    auc = np.full(n, np.nan)
    for i in range(n):
        y = binary_true[i].ravel()
        if y.any() and not y.all():
            auc[i] = roc_auc_score(y, pred[i].ravel())
    return {"bce": bce, "iou": iou, "auc": auc}


def main(args):
    val = torch.load(args.val_data, map_location="cpu", weights_only=True)
    pred = torch.load(args.predictions, map_location="cpu", weights_only=True)
    assert val["keys"] == [tuple(map(int, key)) for key in pred["keys"]]
    assert len(val["keys"]) == 1346
    prior = json.loads(args.prior_rows.read_text(encoding="utf-8"))
    prior_index = {(int(r["image_id"]), int(r["annotation_id"])): r for r in prior}
    assert len(prior_index) == len(prior)
    original_iou = np.array([prior_index[(iid, aid)]["original_image_iou"]
                            for iid, aid, raw in val["keys"]])
    assert np.max(np.abs(original_iou - val["full_iou"].numpy())) < 1e-5
    box_iou = val["box_iou"].numpy()
    image_ids = np.array([key[0] for key in val["keys"]])
    assert np.max(np.abs(box_iou - np.array([
        prior_index[(iid, aid)]["box_iou"] for iid, aid, raw in val["keys"]]))) < 1e-5
    label = val["label"]
    outputs = {"original": metrics(val["base"], label)}
    for arm, logits in pred["logits"].items():
        outputs[arm] = metrics(logits, label)
    groups = {"all": np.ones(len(image_ids), bool),
              "failure": original_iou < .75,
              "success": original_iou >= .75,
              "failure_good_box": (original_iou < .75) & (box_iou >= .75),
              "success_good_box": (original_iou >= .75) & (box_iou >= .75),
              "severe_failure": original_iou < .5}
    summary = {"audit": {"instances": len(image_ids),
               "images": len(set(image_ids)),
               "source": "frozen YOLO26m-seg one-to-one GT-conditioned candidates",
               "mask_grid": "predicted-box 8x8 soft ownership",
               "val_previously_reviewed": True}, "groups": {}}
    for group, mask in groups.items():
        data = {"instances": int(mask.sum()), "images": int(len(set(image_ids[mask]))),
                "arms": {}, "paired": {}}
        for arm, measures in outputs.items():
            data["arms"][arm] = {metric: float(np.nanmean(values[mask]))
                                  for metric, values in measures.items()}
        for other in ("base_h", "wrong_instance", "wrong_image", "original"):
            data["paired"]["true_minus_" + other] = {}
            for metric in ("bce", "iou", "auc"):
                diff = outputs["true_roi"][metric] - outputs[other][metric]
                valid = mask & np.isfinite(diff)
                data["paired"]["true_minus_" + other][metric] = cluster_ci(
                    image_ids, diff, valid)
        summary["groups"][group] = data
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(args.out / "ROWS.jsonl.gz", "wt", encoding="utf-8") as stream:
        for i, (iid, aid, raw) in enumerate(val["keys"]):
            row = {"image_id": iid, "annotation_id": aid, "raw_id": raw,
                   "box_iou": float(box_iou[i]), "original_full_iou": float(original_iou[i]),
                   "metrics": {arm: {m: float(v[i]) for m, v in x.items()}
                               for arm, x in outputs.items()}}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": len(image_ids),
        "images": len(set(image_ids)), "arms": list(outputs)}, indent=2), encoding="utf-8")
    print(json.dumps({"complete": len(image_ids),
                      "failure_good_box": summary["groups"]["failure_good_box"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prior-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
