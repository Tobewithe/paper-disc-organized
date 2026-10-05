"""Build split-isolated regularized targets and GT-free predictor descriptors."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def local_descriptor(proto, box):
    channels, height, width = proto.shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    assert channels == 32
    return F.adaptive_avg_pool2d(proto[:, y1:y2, x1:x2], (4, 4)).flatten()


def assign_controls(records):
    by_image = defaultdict(list)
    for i, row in enumerate(records):
        by_image[row["image_id"]].append(i)
    images = sorted(by_image)
    assert len(images) > 1
    donors = {iid: images[(k + 1) % len(images)] for k, iid in enumerate(images)}
    fallback = 0
    for iid in images:
        own = by_image[iid]
        other = by_image[donors[iid]]
        for j, index in enumerate(own):
            donor_image_index = other[j % len(other)]
            records[index]["p_wrong_image"] = records[donor_image_index]["p_true"].clone()
            if len(own) > 1:
                records[index]["p_wrong_instance"] = records[own[(j + 1) % len(own)]]["p_true"].clone()
            else:
                records[index]["p_wrong_instance"] = records[index]["p_wrong_image"].clone()
                fallback += 1
    return fallback


def main(args):
    sys.path.insert(0, str(args.source / "scripts"))
    sys.path.insert(0, str(args.oracle_source / "scripts"))
    from official_pipeline import load, setup, target_rois, write
    from finite_oracle import solve

    setup()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.bank / "INDEX.json").read_text())[args.split]
    if args.limit:
        index = index[:args.limit]
    rows = []
    for step, item in enumerate(index, 1):
        image_id = item["image_id"]
        image = load(args.bank / "images" / f"{image_id:012d}.pt")
        selected = image["rows"]
        if not selected:
            continue
        rois = target_rois(image)
        up = F.interpolate(image["proto"].cuda()[None], (640, 640),
                           mode="bilinear", align_corners=False)[0]
        native = image["proto"].cuda()
        for j, (row, roi) in enumerate(zip(selected, rois)):
            raw_id = row["raw_id"]
            c0 = image["coeff"][raw_id].cuda()
            delta, status = solve(c0, roi, args.penalty, args.iterations)
            if not torch.isfinite(delta).all() or status["stationary_norm"] > 1e-3:
                raise RuntimeError(f"Unconverged target: {image_id}/{row['annotation_id']}: {status}")
            box = image["boxes"][raw_id].cuda()
            support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
            p = up[:, support].T.contiguous()
            gram = p.T @ p / max(len(p), 1)
            rows.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                             raw_id=raw_id, row_index=j, box_iou=row["box_iou"],
                             initial_iou=row.get("initial_iou"),
                             h=image["h"][raw_id].cpu().float(),
                             p_true=local_descriptor(native, box).cpu().float(),
                             gram=gram.cpu().float(),
                             delta=delta.cpu().float(),
                             c0=c0.cpu().float(),
                             stationary_norm=status["stationary_norm"]))
        if step % 100 == 0 or step == len(index):
            write(args.out / "PROGRESS.json", dict(split=args.split, images=step,
                                                    total_images=len(index), instances=len(rows)))
            print(json.dumps(dict(split=args.split, images=step, instances=len(rows))), flush=True)
    fallback = assign_controls(rows)
    torch.save(rows, args.out / "DATA.pt")
    write(args.out / "COMPLETE.json", dict(split=args.split, images=len(index), instances=len(rows),
                                           penalty=args.penalty, singleton_fallback=fallback,
                                           max_stationary_norm=max(r["stationary_norm"] for r in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "dev", "val"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--penalty", type=float, default=0.003)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
