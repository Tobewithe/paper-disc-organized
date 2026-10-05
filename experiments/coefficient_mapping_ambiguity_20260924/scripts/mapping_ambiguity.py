"""Ask whether nearby coefficient-head features require aligned mask corrections."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, setup, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    oracle = json.loads((a.oracle / "ROWS.json").read_text())
    optimum = load(a.oracle / "oracle_coefficients.pt")
    coco = COCO(str(a.source / "data/annotations/instances_val2017.json"))
    feature_rows = []
    by_image = {}
    for record in oracle:
        iid = record["image_id"]
        if iid not in by_image:
            by_image[iid] = load(a.bank / "images" / f"{iid:012d}.pt")
        image = by_image[iid]
        match = next(r for r in image["rows"] if r["annotation_id"] == record["annotation_id"])
        raw_id = match["raw_id"]
        coefficient = image["coeff"][raw_id].cuda()
        delta = optimum[str(record["annotation_id"])].cuda() - coefficient
        proto = image["proto"].cuda()
        box = image["boxes"][raw_id].cuda()
        t = (torch.arange(16, device="cuda") + .5) / 16
        yy, xx = torch.meshgrid(t, t, indexing="ij")
        gx = box[0] + xx * (box[2] - box[0])
        gy = box[1] + yy * (box[3] - box[1])
        grid = torch.stack([2 * gx / 640 - 1, 2 * gy / 640 - 1], dim=-1)
        local = F.grid_sample(proto[None], grid[None], align_corners=False)[0]
        correction = (delta[:, None, None] * local).sum(0).flatten().cpu().numpy()
        feature_rows.append(dict(image_id=iid, annotation_id=record["annotation_id"],
            level=record["level"], area=record["area"],
            category_id=coco.anns[record["annotation_id"]]["category_id"],
            oracle_gap=record["initial_loss"] - record["oracle_loss"],
            h=image["h"][raw_id].numpy(), delta_c=delta.cpu().numpy(), correction=correction,
            initial_iou=record["initial_iou"]))
    h = np.stack([row["h"] for row in feature_rows])
    # Train-derived scale, fixed before looking at these validation oracle corrections.
    fit_index = json.loads((a.bank / "INDEX.json").read_text())["fit"]
    train_h = []
    for entry in fit_index:
        if not entry["n"]:
            continue
        image = load(a.bank / "images" / f'{entry["image_id"]:012d}.pt')
        train_h.append(image["h"][[r["raw_id"] for r in image["rows"]]].numpy())
    train_h = np.concatenate(train_h)
    h_std = np.maximum(train_h.std(0), .01)
    normalized = h / h_std
    correction = np.stack([row["correction"] for row in feature_rows])
    correction /= np.maximum(np.linalg.norm(correction, axis=1, keepdims=True), 1e-8)
    delta = np.stack([row["delta_c"] for row in feature_rows])
    delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(20260924)
    pairs, excluded = [], Counter()
    for i, row in enumerate(feature_rows):
        matched = [j for j, other in enumerate(feature_rows) if j != i and
            other["image_id"] != row["image_id"] and other["level"] == row["level"] and
            .5 <= other["area"] / row["area"] <= 2]
        if len(matched) < 3:
            excluded["insufficient_level_area_candidates"] += 1
            continue
        distances = np.linalg.norm(normalized[matched] - normalized[i], axis=1)
        near = matched[int(np.argmin(distances))]
        randoms = rng.choice(matched, size=100, replace=True)
        same_category = [j for j in matched if feature_rows[j]["category_id"] == row["category_id"]]
        same_near = min(same_category, key=lambda j: np.linalg.norm(normalized[j] - normalized[i])) if len(same_category) >= 3 else None
        pairs.append(dict(image_id=row["image_id"], annotation_id=row["annotation_id"],
            level=row["level"], area=row["area"], category_id=row["category_id"],
            initial_iou=row["initial_iou"], oracle_gap=row["oracle_gap"],
            matched_pool=len(matched), same_category_pool=len(same_category),
            h_distance_nearest=float(np.linalg.norm(normalized[near] - normalized[i])),
            h_distance_random=float(np.mean(np.linalg.norm(normalized[randoms] - normalized[i], axis=1))),
            correction_cos_nearest=float(correction[i] @ correction[near]),
            correction_cos_random=float(np.mean(correction[randoms] @ correction[i])),
            coefficient_cos_nearest=float(delta[i] @ delta[near]),
            coefficient_cos_random=float(np.mean(delta[randoms] @ delta[i])),
            same_category_correction_cos_nearest=float(correction[i] @ correction[same_near]) if same_near is not None else None,
            same_category_correction_cos_random=float(np.mean(correction[rng.choice(same_category,100)] @ correction[i])) if same_near is not None else None))
    def summarize(rows):
        if not rows:
            return dict(n=0)
        return dict(n=len(rows), h_distance_nearest=float(np.mean([r["h_distance_nearest"] for r in rows])),
            h_distance_random=float(np.mean([r["h_distance_random"] for r in rows])),
            correction_cos_nearest=float(np.mean([r["correction_cos_nearest"] for r in rows])),
            correction_cos_random=float(np.mean([r["correction_cos_random"] for r in rows])),
            correction_nearest_minus_random=float(np.mean([r["correction_cos_nearest"] - r["correction_cos_random"] for r in rows])),
            coefficient_nearest_minus_random=float(np.mean([r["coefficient_cos_nearest"] - r["coefficient_cos_random"] for r in rows])),
            opposite_correction_nearest=int(sum(r["correction_cos_nearest"] < 0 for r in rows)))
    groups = dict(all=pairs,
        initial_failure=[r for r in pairs if r["initial_iou"] < .75],
        high_oracle_gap=[r for r in pairs if r["oracle_gap"] >= .1],
        same_category=[r for r in pairs if r["same_category_correction_cos_nearest"] is not None])
    summary = {name: summarize(rows) for name, rows in groups.items()}
    if groups["same_category"]:
        rows = groups["same_category"]
        summary["same_category"]["nearest_minus_random_within_category"] = float(np.mean([
            r["same_category_correction_cos_nearest"] - r["same_category_correction_cos_random"] for r in rows]))
    summary["excluded"] = dict(excluded)
    write(a.out / "ROWS.json", pairs)
    write(a.out / "SUMMARY.json", summary)
    write(a.out / "COMPLETE.json", dict(oracle_instances=len(oracle), analyzed=len(pairs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "oracle", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
