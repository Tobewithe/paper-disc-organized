"""7E.1 GT-assisted direction/utility curve on frozen official candidates."""
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


RHO = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
ARMS = ("h_only", "true_local")


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def g_dot(a, b, gram):
    return torch.dot(a, gram @ b)


def g_norm(a, gram):
    return g_dot(a, a, gram).clamp_min(1e-20).sqrt()


def construct(oracle, predicted, gram, rho):
    """Construct an exact functional-cosine direction, then fix L2 radius."""
    oracle, predicted, gram = oracle.double(), predicted.double(), gram.double()
    gram = (gram + gram.T) / 2 + torch.eye(len(oracle), device=gram.device, dtype=gram.dtype) * 1e-8
    e1 = oracle / g_norm(oracle, gram)
    residual = predicted - g_dot(predicted, e1, gram) * e1
    if float(g_norm(residual, gram)) < 1e-8:
        eye = torch.eye(len(oracle), device=oracle.device, dtype=oracle.dtype)
        candidates = eye - (eye @ (gram @ e1))[:, None] * e1[None]
        strengths = torch.einsum("ni,ij,nj->n", candidates, gram, candidates)
        residual = candidates[int(strengths.argmax())]
    e2 = residual / g_norm(residual, gram)
    direction = rho * e1 + (1.0 - rho * rho) ** 0.5 * e2
    achieved = float(g_dot(direction, e1, gram) / (g_norm(direction, gram) * g_norm(e1, gram)))
    assert abs(achieved - rho) < 2e-4, (rho, achieved)
    return (direction / direction.norm() * oracle.norm()).float(), achieved


def grid_metrics(logits, label, auc):
    positive = label.bool()
    hard = logits > 0
    tp = int((hard & positive).sum())
    fp = int((hard & ~positive).sum())
    fn = int((~hard & positive).sum())
    npos, nneg = int(positive.sum()), int((~positive).sum())
    return dict(iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                coverage=tp / npos if npos else None,
                fpr=fp / nneg if nneg else None,
                auc=auc(logits, label))


def main(args):
    sys.path.insert(0, str(args.predictor_source / "scripts"))
    sys.path.insert(0, str(args.official_source / "scripts"))
    sys.path.insert(0, str(args.metrics_source / "scripts"))
    from train_predictor import DirectionNet, prepare
    from official_pipeline import decode, load, setup
    from gradient_geometry import auc

    setup()
    rows = torch.load(args.targets / "DATA.pt", weights_only=True, map_location="cpu")
    normalization = torch.load(args.training / "NORMALIZATION.pt", weights_only=True, map_location="cpu")
    prepared, _ = prepare(rows, normalization)
    frozen = json.loads((args.training / "COMPLETE.json").read_text())
    predictions = {}
    for arm in ARMS:
        checkpoint = torch.load(args.training / f"{arm}_best.pt", weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == frozen["selected"][arm]
        model = DirectionNet().cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            direction, log_radius = model(prepared["features"][arm])
        predictions[arm] = (direction.cpu(), log_radius.expm1().cpu())
    coco = COCO(str(args.annotations))
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["image_id"]].append((i, row))
    if args.max_images:
        grouped = dict(list(grouped.items())[:args.max_images])
    result, max_cos_error, max_original_error = [], 0.0, 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    for image_number, (image_id, group) in enumerate(grouped.items(), 1):
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        up = F.interpolate(image["proto"].cuda()[None], (640, 640), mode="bilinear", align_corners=False)[0]
        owner = image["masks"].cuda()
        specs, by_candidate = [], []
        for index, row in group:
            c0 = image["coeff"][row["raw_id"]].cuda()
            target = row["delta"].cuda()
            gram = row["gram"].cuda()
            record = dict(image_id=image_id, annotation_id=row["annotation_id"], raw_id=row["raw_id"],
                          box_iou=row["box_iou"], target_radius=float(target.norm()), conditions={})
            conditions = [("baseline", c0, None, None)]
            for arm in ARMS:
                direction = predictions[arm][0][index].cuda()
                radius = predictions[arm][1][index].cuda()
                predicted_cos = float(g_dot(direction, target, gram) /
                                      (g_norm(direction, gram) * g_norm(target, gram)))
                conditions.append((f"{arm}:raw", c0 + direction * radius, predicted_cos, float(radius)))
                conditions.append((f"{arm}:oracle_radius", c0 + direction * target.norm(), predicted_cos, float(target.norm())))
                for rho in RHO:
                    delta, achieved = construct(target, direction, gram, rho)
                    max_cos_error = max(max_cos_error, abs(achieved - rho))
                    conditions.append((f"{arm}:rho_{rho:g}", c0 + delta, achieved, float(target.norm())))
            for condition, coeff, cosine, radius in conditions:
                specs.append((index, row, condition, coeff, cosine, radius))
            by_candidate.append(record)
        coefficients = torch.stack([s[3] for s in specs])
        raw_ids = torch.tensor([s[1]["raw_id"] for s in specs])
        masks = decode(image, coefficients, raw_ids)
        assert len(masks) == len(specs)
        gt_masks = {row["annotation_id"]: coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
                    for _, row in group}
        by_index = {index: rec for (index, _), rec in zip(group, by_candidate)}
        for index, row in group:
            indices = [k for k, spec in enumerate(specs) if spec[0] == index]
            box = image["boxes"][row["raw_id"]].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            p_eval = up[:, support].T.contiguous()
            y_eval = (owner[support] == image["owners"][row["row_index"]].item() + 1).float()
            logits = p_eval @ coefficients[indices].T
            gt = gt_masks[row["annotation_id"]]
            rec = by_index[index]
            for local, k in enumerate(indices):
                _, _, name, _, cosine, radius = specs[k]
                mask = masks[k]
                image_iou = float(np.logical_and(mask, gt).sum() / max(np.logical_or(mask, gt).sum(), 1))
                rec["conditions"][name] = dict(image_iou=image_iou, effect_cos=cosine,
                                                radius=radius, grid=grid_metrics(logits[:, local], y_eval, auc))
            original_error = abs(rec["conditions"]["baseline"]["image_iou"] - row["initial_iou"])
            max_original_error = max(max_original_error, original_error)
            assert original_error < 1e-6, (image_id, row["annotation_id"], original_error)
            result.append(rec)
        if image_number % 25 == 0 or image_number == len(grouped):
            state = dict(images=image_number, total_images=len(grouped), instances=len(result))
            write(args.out / "PROGRESS.json", state)
            write(args.out / "ROWS.json", result)
            print(json.dumps(state), flush=True)
    write(args.out / "COMPLETE.json", dict(images=len(grouped), instances=len(result),
                                          target_cosines=RHO, arms=ARMS,
                                          maximal_cosine_error=max_cos_error,
                                          maximal_baseline_iou_error=max_original_error,
                                          diagnostic_image_limit=args.max_images,
                                          training_run=args.training.name,
                                          target_run=args.targets.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-source", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--metrics-source", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    main(parser.parse_args())
