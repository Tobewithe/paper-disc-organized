"""Finite per-instance coefficient oracle for the new 16-image train anchor."""
import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, optimize, roi_losses, setup, target_rois, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    order = json.loads((a.bank / "ORDER.json").read_text())["images"][:16]
    rows = []
    for step, iid in enumerate(order, 1):
        path = a.bank / "images" / f"{iid:012d}.pt"
        if not path.exists():
            continue
        image = load(path)
        shape = image.pop("mask_shape")
        mask = np.frombuffer(zlib.decompress(image.pop("mask_zlib")), dtype="<u2").copy()
        image["masks"] = torch.from_numpy(mask.reshape(shape).astype(np.int64))
        image["proto"] = image["proto"].float()
        rois = target_rois(image)
        for k, (record, roi) in enumerate(zip(image["rows"], rois)):
            initial = image["coeff"][k].cuda()
            initial_loss = float(roi_losses(initial[None, None], [roi])[0])
            _, from_initial = optimize(initial, roi, a.iterations)
            _, from_zero = optimize(torch.zeros_like(initial), roi, a.iterations)
            rows.append(dict(image_id=iid, annotation_id=record["annotation_id"],
                initial_loss=initial_loss,
                oracle_loss=min(initial_loss, from_initial["loss"], from_zero["loss"]),
                from_initial=from_initial, from_zero=from_zero))
        print(json.dumps(dict(stage="anchor_oracle", images=step, total=16,
                              instances=len(rows))), flush=True)
        write(a.out / "ROWS.json", rows)
    write(a.out / "SUMMARY.json", dict(images=16, instances=len(rows),
        original_mean=float(np.mean([r["initial_loss"] for r in rows])),
        oracle_mean=float(np.mean([r["oracle_loss"] for r in rows]))))
    write(a.out / "COMPLETE.json", dict(images=16, instances=len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    main(parser.parse_args())
