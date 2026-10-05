"""GT-free predicted-box local-prototype descriptors matched to 7E row order."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def crop_pool(proto, box, size):
    channels, height, width = proto.shape
    assert channels == 32
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return F.adaptive_avg_pool2d(proto[:, y1:y2, x1:x2], (size, size))


def main(args):
    torch.set_num_threads(6)
    rows = torch.load(args.targets / "DATA.pt", weights_only=True, map_location="cpu")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["image_id"]].append(index)
    p8 = [None] * len(rows)
    p4_difference = 0.0
    for step, (image_id, indices) in enumerate(grouped.items(), 1):
        image = torch.load(args.bank / "images" / f"{image_id:012d}.pt", weights_only=False, map_location="cpu")
        proto = image["proto"].float()
        for index in indices:
            row = rows[index]
            assert image["rows"][row["row_index"]]["annotation_id"] == row["annotation_id"]
            box = image["boxes"][row["raw_id"]]
            p8[index] = crop_pool(proto, box, 8).clone()
            if index < 20:
                difference = (crop_pool(proto, box, 4).flatten() - row["p_true"]).abs().max()
                p4_difference = max(p4_difference, float(difference))
        if step % 100 == 0 or step == len(grouped):
            print(json.dumps(dict(split=args.split, images=step, total_images=len(grouped),
                                  instances=sum(item is not None for item in p8))), flush=True)
    assert all(item is not None for item in p8)
    assert p4_difference < 1e-5, p4_difference
    images = sorted(grouped)
    donors = {iid: images[(k + 1) % len(images)] for k, iid in enumerate(images)}
    wrong_instance = torch.empty(len(rows), dtype=torch.long)
    wrong_image = torch.empty(len(rows), dtype=torch.long)
    fallback = 0
    for iid in images:
        own, other = grouped[iid], grouped[donors[iid]]
        for k, index in enumerate(own):
            wrong_image[index] = other[k % len(other)]
            if len(own) > 1:
                wrong_instance[index] = own[(k + 1) % len(own)]
            else:
                wrong_instance[index] = wrong_image[index]
                fallback += 1
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(p8=torch.stack(p8), wrong_instance=wrong_instance,
                    wrong_image=wrong_image), args.out / "DESCRIPTORS.pt")
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(split=args.split, images=len(grouped),
        instances=len(rows), singleton_fallback=fallback,
        first_20_p4_max_difference=p4_difference,
        source_targets=args.targets.name), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("fit", "dev", "val"), required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
