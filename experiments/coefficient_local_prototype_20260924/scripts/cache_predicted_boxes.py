"""Replay frozen YOLO to recover the predicted boxes of cached positives."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, setup, write

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "boxes").mkdir(exist_ok=True)
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    head = model.model[-1]
    data = a.bank / "official_data"
    dataset = YOLODataset(img_path=str(data / "fit.txt"), imgsz=640, batch_size=1,
        augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    order = json.loads((a.bank / "ORDER.json").read_text())["images"]
    assert set(order) == set(by_id)
    max_coefficient_error = 0.0
    nonempty = 0
    with torch.no_grad():
        for step, iid in enumerate(order, 1):
            cached_path = a.bank / "images" / f"{iid:012d}.pt"
            if not cached_path.exists():
                continue
            cached = load(cached_path)
            sample = dataset[by_id[iid]]
            batch = YOLODataset.collate_fn([sample])
            x = batch["img"].cuda().float() / 255
            _, raw = model(x)
            pred = raw["one2one"]
            original_indices = torch.tensor([row["original_raw_id"] for row in cached["rows"]])
            coefficients = pred["mask_coefficient"][0].T[original_indices]
            error = float((coefficients.cpu() - cached["coeff"]).abs().max())
            max_coefficient_error = max(max_coefficient_error, error)
            assert error < 1e-4, (iid, error)
            boxes = head._get_decode_boxes(pred)[0].T[original_indices].cpu()
            assert torch.isfinite(boxes).all() and boxes.shape == (len(cached["rows"]), 4)
            torch.save(boxes, a.out / "boxes" / f"{iid:012d}.pt")
            nonempty += 1
            if step % 200 == 0 or step == len(order):
                state = dict(images=step, total=len(order), nonempty=nonempty,
                             max_coefficient_error=max_coefficient_error)
                write(a.out / "PROGRESS.json", state)
                print(json.dumps(state), flush=True)
    write(a.out / "COMPLETE.json", dict(images=len(order), nonempty=nonempty,
        max_coefficient_error=max_coefficient_error, crop="original predicted one2one box"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "weights", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
