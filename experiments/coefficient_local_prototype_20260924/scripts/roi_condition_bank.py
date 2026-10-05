"""Sample predicted-box-local prototype maps for fixed official positives."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main(a):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    (a.out / "images").symlink_to((a.source / "images").resolve(), target_is_directory=True)
    counts = {}
    for split in ("fit", "dev", "val"):
        rows = torch.load(a.source / f"{split}.pt", map_location="cpu", weights_only=False)
        by_image = defaultdict(list)
        for k, row in enumerate(rows):
            by_image[row["meta"]["image_id"]].append(k)
        for iid, indices in by_image.items():
            image = torch.load(a.source / "images" / f"{iid:012d}.pt",
                               map_location="cpu", weights_only=False)
            raw_ids = [rows[k]["meta"]["raw_id"] for k in indices]
            boxes = image["boxes"][raw_ids].cuda()
            proto = image["proto"].cuda()
            t = (torch.arange(32, device="cuda") + .5) / 32
            yy, xx = torch.meshgrid(t, t, indexing="ij")
            gx = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
            gy = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
            grid = torch.stack([2 * gx / 640 - 1, 2 * gy / 640 - 1], dim=-1)
            features = F.grid_sample(proto[None].expand(len(indices), -1, -1, -1),
                                     grid, align_corners=False)
            features = F.adaptive_avg_pool2d(features, (4, 4)).flatten(1).cpu()
            for k, index in enumerate(indices):
                rows[index]["phi"] = features[k].clone()
        torch.save(rows, a.out / f"{split}.pt")
        counts[split] = len(rows)
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(counts=counts,
        source=str(a.source),
        feature="32x32 predicted-box ROI on actual prototype, pooled to 4x4; no GT input")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
