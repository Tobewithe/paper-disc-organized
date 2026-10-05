"""Reconstruct official coefficient-loss moments for the complete 7L OOF bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics.utils import ops

from analyze_conditional_direction import load_bank


def main(args):
    torch.set_num_threads(6)
    sys.path.insert(0, str(args.official_source / "scripts"))
    from official_pipeline import target_rois

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = load_bank(args.targets / "shards", args.annotations, manifest)
    n = len(bank["image_id"])
    by_image = defaultdict(list)
    for i, iid in enumerate(bank["image_id"]):
        by_image[int(iid)].append(i)
    s = np.empty((n, 32), dtype=np.float32)
    s_predbox = np.empty_like(s)
    t = np.empty_like(s)
    g = np.empty_like(s)
    stationarity = np.empty(n, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    for count, (iid, indices) in enumerate(sorted(by_image.items()), 1):
        cache = torch.load(args.cache / f"{iid:012d}.pt", map_location="cpu", weights_only=True)
        encoded = np.frombuffer(zlib.decompress(cache["mask_zlib"]), dtype="<u2").copy()
        mask = torch.from_numpy(encoded.reshape(cache["mask_shape"]).astype(np.int64))
        image = dict(cache, masks=mask, proto=cache["proto"].float())
        positions = {int(row["annotation_id"]): j for j, row in enumerate(cache["rows"])}
        assert len(positions) == len(cache["rows"])
        assert all(int(bank["annotation_id"][i]) in positions for i in indices)
        positions_selected = [positions[int(bank["annotation_id"][i])] for i in indices]
        rois = target_rois(image, positions_selected)
        assert len(rois) == len(indices)
        with torch.no_grad():
            up = F.interpolate(image["proto"].cuda()[None], (640, 640),
                               mode="bilinear", align_corners=False)[0]
            for i, pos, (pixels, y, area) in zip(indices, positions_selected, rois):
                c0 = cache["coeff"][pos].cuda().float()
                delta = torch.from_numpy(bank["delta"][i]).cuda()
                box = torch.from_numpy(bank["predicted_box"][i]).cuda().float()
                pred_support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"),
                                             box[None])[0].bool()
                pred_pixels = up[:, pred_support].T.contiguous()
                pred_area = ((box[2:] - box[:2]) / 640).prod() * (640 * 640)
                assert float(pred_area) > 0
                s_predbox[i] = (pred_pixels.T @ (pred_pixels @ c0).sigmoid()
                                / pred_area).cpu().numpy()
                p0 = (pixels @ c0).sigmoid()
                si = pixels.T @ p0 / area
                ti = pixels.T @ y / area
                gi = si - ti
                final = pixels.T @ ((pixels @ (c0 + delta)).sigmoid() - y) / area
                final = final + .003 * delta
                s[i], t[i], g[i] = (x.cpu().numpy() for x in (si, ti, gi))
                stationarity[i] = float(final.norm())
                seen[i] = True
        del cache, mask, image, rois
        if count % 100 == 0:
            print(json.dumps({"images": count, "total": len(by_image),
                              "instances": int(seen.sum())}), flush=True)
    assert seen.all() and np.isfinite(s).all() and np.isfinite(s_predbox).all() and np.isfinite(t).all()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "MOMENTS.npz", s=s, s_predbox=s_predbox, t=t, g=g, stationarity=stationarity,
             image_id=bank["image_id"], annotation_id=bank["annotation_id"],
             fold=bank["fold"], original_iou=bank["original_iou"],
             box_iou=bank["box_iou"], h=bank["h"],
             raw_detection=bank["raw_detection"],
             local_summary=bank["local_summary"],
             gram_diag=np.diagonal(bank["gram"], axis1=1, axis2=2).copy())
    audit = {"instances": n, "images": len(by_image),
             "unique_keys": len(set(zip(bank["image_id"], bank["annotation_id"]))),
             "fold_counts": np.bincount(bank["fold"], minlength=5).tolist(),
             "stationarity_median": float(np.median(stationarity)),
             "stationarity_p95": float(np.quantile(stationarity, .95)),
             "stationarity_max": float(np.max(stationarity)),
             "cached_prototype_dtype": "float16_to_float32"}
    (args.out / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("manifest", "targets", "annotations", "cache", "official_source", "out"):
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    main(parser.parse_args())
