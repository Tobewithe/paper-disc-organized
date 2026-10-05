"""Capture the input of the frozen one-to-one coefficient branch at official positives."""
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
    (a.out / "features").mkdir(exist_ok=True)
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    head = model.model[-1]
    inputs = {}
    hooks = [branch.register_forward_pre_hook(
        lambda _, x, level=level: inputs.__setitem__(level, x[0].detach()))
        for level, branch in enumerate(head.one2one_cv4)]
    official_data = a.bank / "official_data"
    index = json.loads((a.bank / "INDEX.json").read_text())
    counts = {}
    max_error = 0.0
    try:
        for split in ("fit", "dev", "val"):
            dataset = YOLODataset(img_path=str(official_data / f"{split}.txt"),
                imgsz=640, batch_size=1, augment=False, hyp=deepcopy(model.args),
                rect=False, cache=False, stride=32,
                data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
            by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
            assert set(by_id) == {entry["image_id"] for entry in index[split]}
            positives = 0
            for step, entry in enumerate(index[split], 1):
                iid = entry["image_id"]
                if not entry["n"]:
                    continue
                cached = load(a.bank / "images" / f"{iid:012d}.pt")
                sample = dataset[by_id[iid]]
                batch = YOLODataset.collate_fn([sample])
                with torch.no_grad():
                    _, raw = model(batch["img"].cuda().float() / 255)
                positions = torch.tensor([row["raw_id"] for row in cached["rows"]])
                current = raw["one2one"]["mask_coefficient"][0].T[positions]
                error = float((current.cpu() - cached["coeff"][positions]).abs().max())
                max_error = max(max_error, error)
                assert error < 1e-4, (iid, error)
                level = cached["levels"][positions]
                maps = [inputs[lev][0].flatten(1).T for lev in range(3)]
                offsets = [0, len(maps[0]), len(maps[0]) + len(maps[1])]
                assert sum(len(value) for value in maps) == 8400
                features = [maps[lev][pos - offsets[lev]].cpu().float()
                            for pos, lev in zip(positions.tolist(), level.tolist())]
                assert all(len(feature) == (256 if lev == 0 else 512)
                           for feature, lev in zip(features, level.tolist()))
                torch.save(dict(features=features, levels=level, raw_ids=positions),
                    a.out / "features" / f"{iid:012d}.pt")
                positives += len(positions)
                if step % 100 == 0 or step == len(index[split]):
                    state = dict(split=split, images=step, total=len(index[split]),
                        positives=positives, max_coefficient_error=max_error)
                    write(a.out / "PROGRESS.json", state)
                    print(json.dumps(state), flush=True)
            counts[split] = positives
    finally:
        for hook in hooks:
            hook.remove()
    write(a.out / "COMPLETE.json", dict(counts=counts, max_coefficient_error=max_error,
        source="pre-hook on original one2one_cv4 branch; official assigned raw positions"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "weights", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
