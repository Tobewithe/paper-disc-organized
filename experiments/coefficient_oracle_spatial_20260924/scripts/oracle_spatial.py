"""Localize which pixels the fixed-prototype coefficient oracle changes."""
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
            original = image["coeff"][raw_id].cuda()
            oracle = coefficients[str(row["annotation_id"])].cuda()
            initial_logits = torch.einsum("c,chw->hw", original, prototype).cpu().numpy()
            oracle_logits = torch.einsum("c,chw->hw", oracle, prototype).cpu().numpy()
            predicted_box = image["boxes"][raw_id].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"),
                                    predicted_box[None])[0].bool().cpu().numpy()
            truth = label == int(image["owners"][k]) + 1
            before = (initial_logits > 0) & support
            after = (oracle_logits > 0) & support
            rescued = ~before & after & truth
            removed_fp = before & ~after & ~truth
            deleted_tp = before & ~after & truth
            introduced_fp = ~before & after & ~truth
            boundary = cv2.dilate(truth.astype(np.uint8), kernel).astype(bool) & ~cv2.erode(
                truth.astype(np.uint8), kernel).astype(bool)
            outside = support & ~truth
            neighbor = outside & (label != 0)
            background = outside & (label == 0)
            supervised = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"),
                image["target_boxes"][k:k + 1].cuda())[0].bool().cpu().numpy()
            initial_fp = before & ~truth
            box = predicted_box.cpu().numpy()
            yy, xx = np.mgrid[:640, :640]
            edge = support & ((xx < box[0] + 8) | (xx >= box[2] - 8) |
                              (yy < box[1] + 8) | (yy >= box[3] - 8))
            measures = dict(rescued_target=int(rescued.sum()), removed_false_positive=int(removed_fp.sum()),
                deleted_true_positive=int(deleted_tp.sum()), introduced_false_positive=int(introduced_fp.sum()),
                removed_neighbor=int((removed_fp & neighbor).sum()),
                removed_background=int((removed_fp & background).sum()),
                initial_fp_supervised=int((initial_fp & supervised).sum()),
                initial_fp_unsupervised=int((initial_fp & ~supervised).sum()),
                removed_fp_supervised=int((removed_fp & supervised).sum()),
                removed_fp_unsupervised=int((removed_fp & ~supervised).sum()),
                introduced_fp_unsupervised=int((introduced_fp & ~supervised).sum()),
                boundary_rescue=int((rescued & boundary).sum()),
                boundary_fp_removal=int((removed_fp & boundary).sum()),
                interior_rescue=int((rescued & ~boundary).sum()),
                outer_fp_removal=int((removed_fp & ~boundary).sum()),
                box_edge_fp_removal=int((removed_fp & edge).sum()),
                target_pixels_in_box=int((truth & support).sum()),
                outside_pixels_in_box=int(outside.sum()))
            records.append(dict(image_id=iid, annotation_id=row["annotation_id"],
                initial_iou=row["initial_iou"], oracle_iou=selected["final_iou_after"],
                box_iou=row["box_iou"], **measures))
        if step % 25 == 0 or step == len(grouped):
            state = dict(images=step, total=len(grouped), instances=len(records))
            print(json.dumps(state), flush=True)
            write(a.out / "PROGRESS.json", state)
    def summarize(rows):
        totals = {key: sum(row[key] for row in rows) for key in measures}
        useful = totals["rescued_target"] + totals["removed_false_positive"]
        harms = totals["deleted_true_positive"] + totals["introduced_false_positive"]
        return dict(instances=len(rows), totals=totals, useful_changed_pixels=useful,
            harmful_changed_pixels=harms,
            target_rescue_share=totals["rescued_target"] / useful if useful else None,
            fp_removal_share=totals["removed_false_positive"] / useful if useful else None,
            initial_fp_unsupervised_share=totals["initial_fp_unsupervised"] /
                (totals["initial_fp_supervised"] + totals["initial_fp_unsupervised"])
                if totals["initial_fp_supervised"] + totals["initial_fp_unsupervised"] else None,
            removed_fp_unsupervised_share=totals["removed_fp_unsupervised"] /
                totals["removed_false_positive"] if totals["removed_false_positive"] else None,
            neighbor_share_of_fp_removal=totals["removed_neighbor"] / totals["removed_false_positive"]
                if totals["removed_false_positive"] else None)
    groups = dict(all=records,
        initial_failed=[r for r in records if r["initial_iou"] < .75],
        initial_success=[r for r in records if r["initial_iou"] >= .75],
        repaired=[r for r in records if r["initial_iou"] < .75 <= r["oracle_iou"]],
        good_box_failed_mask=[r for r in records if r["box_iou"] >= .75 and r["initial_iou"] < .75],
        bad_box_failed_mask=[r for r in records if r["box_iou"] < .75 and r["initial_iou"] < .75])
    write(a.out / "ROWS.json", records)
    write(a.out / "SUMMARY.json", {name: summarize(rows) for name, rows in groups.items()})
    write(a.out / "COMPLETE.json", dict(instances=len(records), images=len(grouped)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "oracle", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
