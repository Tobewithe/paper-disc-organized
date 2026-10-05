"""Evaluate frozen 7F-A checkpoints on untouched 7E val images."""
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

from train_size_curve import DirectionNet, SIZES, matrix


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def functional_cos(direction, target, gram):
    gt_effect = torch.bmm(gram, target.unsqueeze(-1)).squeeze(-1)
    pred_effect = torch.bmm(gram, direction.unsqueeze(-1)).squeeze(-1)
    return ((direction * gt_effect).sum(1) /
            ((target * gt_effect).sum(1).clamp_min(1e-12) *
             (direction * pred_effect).sum(1).clamp_min(1e-12)).sqrt()).clamp(-1, 1)


def main(args):
    sys.path.insert(0, str(args.official_source / "scripts"))
    sys.path.insert(0, str(args.metrics_source / "scripts"))
    from official_pipeline import decode, load, setup
    from gradient_geometry import auc

    setup()
    rows = torch.load(args.val / "DATA.pt", weights_only=True, map_location="cpu")
    reference = {row["annotation_id"]: row for row in
                 json.loads((args.reference_eval / "ROWS.json").read_text())}
    assert len(rows) == len(reference) == 1346
    training = json.loads((args.training / "COMPLETE.json").read_text())
    h = matrix(rows, "h").cuda()
    delta = matrix(rows, "delta").cuda()
    target = F.normalize(delta, dim=1)
    target_radius = delta.norm(dim=1).log1p()
    gram = matrix(rows, "gram").reshape(-1, 32, 32).cuda()
    predicted = {}
    for images in SIZES:
        checkpoint = torch.load(args.training / f"n{images}_best.pt", weights_only=True, map_location="cpu")
        assert checkpoint["step"] == training["selected"][str(images)]
        model = DirectionNet().cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        normalized = (h - checkpoint["h_mean"].cuda()) / checkpoint["h_std"].cuda()
        with torch.no_grad():
            unit, lograd = model(normalized)
        predicted[images] = dict(correction=(unit * lograd.expm1()[:, None]).cpu(),
                                 effect_cos=functional_cos(unit, target, gram).cpu().tolist(),
                                 coefficient_cos=(unit * target).sum(1).cpu().tolist(),
                                 radius_log_abs=(lograd - target_radius).abs().cpu().tolist())
    coco = COCO(str(args.annotations))
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["image_id"]].append((index, row))
    output = []
    args.out.mkdir(parents=True, exist_ok=True)
    for image_number, (image_id, group) in enumerate(grouped.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        up = F.interpolate(image["proto"].cuda()[None], (640, 640), mode="bilinear", align_corners=False)[0]
        owner = image["masks"].cuda()
        coeffs, raw_ids = [], []
        for index, row in group:
            c0 = image["coeff"][row["raw_id"]].cuda()
            for images in SIZES:
                coeffs.append(c0 + predicted[images]["correction"][index].cuda())
                raw_ids.append(row["raw_id"])
        coeffs = torch.stack(coeffs)
        masks = decode(image, coeffs, torch.tensor(raw_ids))
        assert len(masks) == len(group) * len(SIZES)
        for position, (index, row) in enumerate(group):
            original = reference[row["annotation_id"]]
            assert original["image_id"] == image_id and original["raw_id"] == row["raw_id"]
            assert abs(original["original_image_iou"] - row["initial_iou"]) < 1e-6
            box = image["boxes"][row["raw_id"]].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            pixels = up[:, support].T.contiguous()
            label = (owner[support] == image["owners"][row["row_index"]].item() + 1).float()
            local_coeffs = coeffs[position * len(SIZES):(position + 1) * len(SIZES)]
            logits = pixels @ local_coeffs.T
            gt = coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
            record = dict(image_id=image_id, annotation_id=row["annotation_id"],
                          box_iou=row["box_iou"], original_image_iou=row["initial_iou"],
                          original_auc=original["original_grid"]["auc"], sizes={})
            for j, images in enumerate(SIZES):
                mask = masks[position * len(SIZES) + j]
                iou = float(np.logical_and(mask, gt).sum() / max(np.logical_or(mask, gt).sum(), 1))
                record["sizes"][str(images)] = dict(image_iou=iou,
                    auc=auc(logits[:, j], label),
                    effect_cos=predicted[images]["effect_cos"][index],
                    coefficient_cos=predicted[images]["coefficient_cos"][index],
                    radius_log_abs=predicted[images]["radius_log_abs"][index])
            output.append(record)
        if image_number % 25 == 0 or image_number == len(grouped):
            state = dict(images=image_number, total_images=len(grouped), instances=len(output))
            write(args.out / "PROGRESS.json", state)
            write(args.out / "ROWS.json", output)
            print(json.dumps(state), flush=True)
    write(args.out / "COMPLETE.json", dict(images=len(grouped), instances=len(output), sizes=SIZES,
                                          training_run=args.training.name, validation_target_run=args.val.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("official_source", "metrics_source", "val", "reference_eval", "training", "bank", "annotations", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(parser.parse_args())
