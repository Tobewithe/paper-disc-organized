"""Held-out 7E evaluation after all arm checkpoints are fixed by dev."""
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

from train_predictor import ARMS, DirectionNet, components, prepare


def main(args):
    sys.path.insert(0, str(args.source / "scripts"))
    sys.path.insert(0, str(args.metrics_source / "scripts"))
    from official_pipeline import decode, load, setup, target_rois, write
    from gradient_geometry import metrics

    setup()
    rows = torch.load(args.val / "DATA.pt", weights_only=True, map_location="cpu")
    normalization = torch.load(args.training / "NORMALIZATION.pt", weights_only=True, map_location="cpu")
    val, _ = prepare(rows, normalization)
    summary = json.loads((args.training / "COMPLETE.json").read_text())
    assert summary["fit_instances"] and summary["dev_instances"]
    predictions, directions = {}, {}
    target_direction = val["direction"]
    target_radius = val["log_radius"].expm1()
    for arm in ARMS:
        checkpoint = torch.load(args.training / f"{arm}_best.pt", weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == summary["selected"][arm]
        model = DirectionNet().cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            unit, log_radius = model(val["features"][arm])
            score = components(unit, log_radius, val, torch.arange(len(rows), device="cuda"))
            radius = log_radius.expm1()
            predictions[arm] = dict(full=(unit * radius[:, None]).cpu(),
                                    direction_with_teacher_radius=(unit * target_radius[:, None]).cpu(),
                                    teacher_direction_with_radius=(target_direction * radius[:, None]).cpu(),
                                    scores={key: value.cpu().tolist() for key, value in score.items()},
                                    predicted_radius=radius.cpu().tolist())
            directions[arm] = dict(mean_coefficient_cos=float(score["coefficient_cos"].mean()),
                                   mean_effect_cos=float(score["effect_cos"].mean()),
                                   mean_radius_log_abs=float(score["radius_log_abs"].mean()))
    coco = COCO(str(args.annotations))
    by_image = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["image_id"]].append((index, row))
    output = []
    for image_number, (image_id, group) in enumerate(by_image.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        prototype = F.interpolate(image["proto"].cuda()[None], (640, 640),
                                  mode="bilinear", align_corners=False)[0]
        owner_map = image["masks"].cuda()
        rois = target_rois(image, [row["row_index"] for _, row in group])
        raw_ids = torch.tensor([row["raw_id"] for _, row in group])
        gt_masks = [coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool) for _, row in group]
        originals = [image["coeff"][row["raw_id"]].cuda() for _, row in group]
        correction_masks = {}
        for arm in ARMS:
            corrected = torch.stack([original + predictions[arm]["full"][index].cuda()
                                     for (index, _), original in zip(group, originals)])
            correction_masks[arm] = decode(image, corrected, raw_ids)
        for k, ((index, row), roi, gt, original) in enumerate(zip(group, rois, gt_masks, originals)):
            box = image["boxes"][row["raw_id"]].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            predicted = (prototype[:, support].T.contiguous(),
                         (owner_map[support] == image["owners"][row["row_index"]].item() + 1).float())
            original_grid = metrics(original, roi, predicted)
            assert row["initial_iou"] is not None
            record = dict(image_id=image_id, annotation_id=row["annotation_id"],
                          raw_id=row["raw_id"], box_iou=row["box_iou"],
                          original_image_iou=row["initial_iou"], original_grid=original_grid, arms={})
            for arm in ARMS:
                delta = predictions[arm]["full"][index].cuda()
                after_grid = metrics(original + delta, roi, predicted)
                mask = correction_masks[arm][k]
                after_image_iou = float((mask & gt).sum() / max((mask | gt).sum(), 1))
                hybrid = {}
                for name in ("direction_with_teacher_radius", "teacher_direction_with_radius"):
                    c = original + predictions[arm][name][index].cuda()
                    hybrid[name] = metrics(c, roi, predicted)
                record["arms"][arm] = dict(coefficient_cos=predictions[arm]["scores"]["coefficient_cos"][index],
                                           effect_cos=predictions[arm]["scores"]["effect_cos"][index],
                                           radius_log_abs=predictions[arm]["scores"]["radius_log_abs"][index],
                                           predicted_radius=predictions[arm]["predicted_radius"][index],
                                           teacher_radius=float(target_radius[index]),
                                           image_iou=after_image_iou, grid=after_grid, hybrids=hybrid)
            output.append(record)
        if image_number % 25 == 0 or image_number == len(by_image):
            state = dict(images=image_number, total_images=len(by_image), instances=len(output))
            print(json.dumps(state), flush=True)
            write(args.out / "PROGRESS.json", state)
            write(args.out / "ROWS.json", output)
    write(args.out / "DIRECTION_SUMMARY.json", directions)
    write(args.out / "COMPLETE.json", dict(status="completed", images=len(by_image), instances=len(output),
                                           arms=list(ARMS), training_run=args.training.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metrics-source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
