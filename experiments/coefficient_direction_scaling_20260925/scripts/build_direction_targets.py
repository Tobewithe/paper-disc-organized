"""Build compact λ=.003 GT-conditioned targets from the existing 10k bank."""
import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def ids(path, split):
    return {row["image_id"] for row in json.loads((path / "INDEX.json").read_text())[split]}


def main(args):
    sys.path.insert(0, str(args.official_source / "scripts"))
    sys.path.insert(0, str(args.oracle_source / "scripts"))
    from official_pipeline import load, setup, target_rois
    from finite_oracle import solve

    setup()
    order = json.loads((args.bank / "ORDER.json").read_text())["images"]
    assert len(order) == 10000 and len(set(order)) == 10000
    counts = {row["image_id"]: row["n"] for row in json.loads((args.bank / "INDEX.json").read_text())["fit"]}
    assert len(counts) == 10000 and set(counts) == set(order)
    assert not set(order) & ids(args.fixed_bank, "dev")
    assert not set(order) & ids(args.fixed_bank, "val")
    if args.limit_images:
        order = order[:args.limit_images]
    args.out.mkdir(parents=True, exist_ok=True)
    chunks, values, count, worst = [], [], 0, 0.0
    for step, image_id in enumerate(order, 1):
        if counts[image_id]:
            image = load(args.bank / "images" / f"{image_id:012d}.pt")
            assert len(image["rows"]) == counts[image_id]
            shape = image.pop("mask_shape")
            mask = np.frombuffer(zlib.decompress(image.pop("mask_zlib")), dtype="<u2").copy()
            image["masks"] = torch.from_numpy(mask.reshape(shape).astype(np.int64))
            image["proto"] = image["proto"].float()
            rois = target_rois(image)
            assert len(rois) == len(image["rows"])
            for row, roi in zip(image["rows"], rois):
                raw_id = row["raw_id"]
                c0 = image["coeff"][raw_id].cuda()
                delta, state = solve(c0, roi, args.penalty, args.iterations)
                if not torch.isfinite(delta).all() or state["stationary_norm"] > 1e-3:
                    raise RuntimeError(f"Target did not converge: {image_id}/{row['annotation_id']} {state}")
                values.append(dict(image_id=image_id, annotation_id=row["annotation_id"],
                                   h=image["h"][raw_id].clone().float(),
                                   delta=delta.cpu().clone().float()))
                worst = max(worst, state["stationary_norm"])
                count += 1
        if step % args.chunk_images == 0 or step == len(order):
            name = f"chunk_{len(chunks):04d}.pt"
            torch.save(values, args.out / name)
            chunks.append(dict(file=name, through_images=step, instances=len(values), cumulative_instances=count))
            values = []
            write(args.out / "INDEX.json", dict(chunks=chunks, penalty=args.penalty,
                                                images_processed=step, instances=count,
                                                maximal_stationarity=worst,
                                                complete=step == len(order)))
            print(json.dumps(dict(images=step, total_images=len(order), instances=count,
                                  maximal_stationarity=worst)), flush=True)
    write(args.out / "COMPLETE.json", dict(images=len(order), instances=count,
                                           chunks=len(chunks), penalty=args.penalty,
                                           maximal_stationarity=worst,
                                           reference_order=args.bank.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("official_source", "oracle_source", "bank", "fixed_bank", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--penalty", type=float, default=.003)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--chunk-images", type=int, default=200)
    parser.add_argument("--limit-images", type=int, default=0)
    main(parser.parse_args())
