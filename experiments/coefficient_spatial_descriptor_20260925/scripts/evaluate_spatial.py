"""Held-out 7F-B direction and original-image mask evaluation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

from train_spatial import ARMS, SpatialNet, components, prepare


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main(args):
    sys.path.insert(0, str(args.official_source / "scripts"))
    sys.path.insert(0, str(args.metrics_source / "scripts"))
    from official_pipeline import decode, load, setup
    from gradient_geometry import auc

    def grid_stats(coefficient, pixels, labels):
        logits = pixels @ coefficient
        positive = labels.bool()
        hard = logits > 0
        tp = int((hard & positive).sum())
        fp = int((hard & ~positive).sum())
        fn = int((~hard & positive).sum())
        npos = int(positive.sum())
        nneg = int((~positive).sum())
        return dict(iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                    coverage=tp / npos if npos else None,
                    fpr=fp / nneg if nneg else None,
                    auc=auc(logits, labels))

    setup()
    rows = torch.load(args.val_targets / "DATA.pt", weights_only=True, map_location="cpu")
    descriptors = torch.load(args.val_descriptors / "DESCRIPTORS.pt", weights_only=True, map_location="cpu")
    normalization = torch.load(args.training / "NORMALIZATION.pt", weights_only=True, map_location="cpu")
    val, _ = prepare(rows, descriptors, normalization)
    chosen = json.loads((args.training / "COMPLETE.json").read_text())
    assert chosen["fit_instances"] == 6058 and chosen["dev_instances"] == 1402
    predictions = {}
    for arm in ARMS:
        checkpoint = torch.load(args.training / f"{arm}_best.pt", weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == chosen["selected"][arm]
        model = SpatialNet().cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            unit, logradius = model(val["h"], val["inputs"][arm])
            scores = components(unit, logradius, val,
                                torch.arange(len(rows), device="cuda"))
            predictions[arm] = dict(correction=(unit * logradius.expm1()[:, None]).cpu(),
                                    scores={key: value.cpu().tolist() for key, value in scores.items()})
    coco = COCO(str(args.annotations))
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["image_id"]].append((index, row))
    output = []
    args.out.mkdir(parents=True, exist_ok=True)
    for image_number, (image_id, group) in enumerate(grouped.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        up = F.interpolate(image["proto"].cuda()[None], (640, 640),
                           mode="bilinear", align_corners=False)[0]
        owner = image["masks"].cuda()
        coefficients, raw_ids = [], []
        for index, row in group:
            c0 = image["coeff"][row["raw_id"]].cuda()
            for arm in ARMS:
                coefficients.append(c0 + predictions[arm]["correction"][index].cuda())
                raw_ids.append(row["raw_id"])
        coefficients = torch.stack(coefficients)
        masks = decode(image, coefficients, torch.tensor(raw_ids))
        for j, (index, row) in enumerate(group):
            c0 = image["coeff"][row["raw_id"]].cuda()
            box = image["boxes"][row["raw_id"]].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            pixels = up[:, support].T.contiguous()
            labels = (owner[support] == image["owners"][row["row_index"]].item() + 1).float()
            gt = coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
            before = grid_stats(c0, pixels, labels)
            record = dict(image_id=image_id, annotation_id=row["annotation_id"],
                          box_iou=row["box_iou"], original_image_iou=row["initial_iou"],
                          original_grid={k: before[k] for k in ("iou", "coverage", "fpr", "auc")},
                          arms={})
            for k, arm in enumerate(ARMS):
                coefficient = coefficients[j * len(ARMS) + k]
                after = grid_stats(coefficient, pixels, labels)
                mask = masks[j * len(ARMS) + k]
                iou = float(np.logical_and(mask, gt).sum() / max(np.logical_or(mask, gt).sum(), 1))
                score = predictions[arm]["scores"]
                record["arms"][arm] = dict(image_iou=iou,
                    grid={key: after[key] for key in ("iou", "coverage", "fpr", "auc")},
                    effect_cos=score["effect_cos"][index],
                    coefficient_cos=score["coefficient_cos"][index],
                    radius_log_abs=score["radius_log_abs"][index])
            output.append(record)
        if image_number % 25 == 0 or image_number == len(grouped):
            state = dict(images=image_number, total_images=len(grouped), instances=len(output))
            write(args.out / "PROGRESS.json", state)
            write(args.out / "ROWS.json", output)
            print(json.dumps(state), flush=True)
    write(args.out / "COMPLETE.json", dict(images=len(grouped), instances=len(output), arms=ARMS,
                                          training_run=args.training.name, val_targets=args.val_targets.name,
                                          val_descriptors=args.val_descriptors.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("official_source", "metrics_source", "val_targets", "val_descriptors", "training", "bank", "annotations", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(parser.parse_args())
