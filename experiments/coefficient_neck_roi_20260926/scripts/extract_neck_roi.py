"""Frozen neck ROI before the coefficient branch's 256/512-to-64 compression."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics


def crop_pool(feature, box):
    channels, height, width = feature.shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return F.adaptive_avg_pool2d(feature[:, y1:y2, x1:x2], (8, 8))


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.official_source / "scripts"))
    from official_pipeline import load

    rows = torch.load(a.targets / "DATA.pt", weights_only=True, map_location="cpu")
    index = json.loads((a.bank / "INDEX.json").read_text())[a.split]
    grouped = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[int(row["image_id"])].append(position)
    assert set(grouped) == {int(item["image_id"]) for item in index if item["n"]}
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nm == 32
    neck, hmap = {}, {}
    handles = []
    for level in range(3):
        handles.append(head.one2one_cv4[level].register_forward_pre_hook(
            lambda _, x, level=level: neck.__setitem__(level, x[0].detach())))
        handles.append(head.one2one_cv4[level][-1].register_forward_pre_hook(
            lambda _, x, level=level: hmap.__setitem__(level, x[0].detach())))
    dataset = YOLODataset(img_path=str(a.bank / "official_data" / f"{a.split}.txt"),
        imgsz=640, batch_size=1, augment=False, hyp=deepcopy(model.args),
        rect=False, cache=False, stride=32,
        data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
    by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
    assert set(by_id) == {int(item["image_id"]) for item in index}
    output = [None] * len(rows)
    max_h_error, max_coeff_error, max_box_error = 0., 0., 0.
    for step, item in enumerate(index, 1):
        iid = int(item["image_id"])
        if not item["n"]:
            continue
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        sample = dataset[by_id[iid]]
        with torch.no_grad():
            _, raw = model(sample["img"].cuda().float()[None] / 255)
        pred = raw["one2one"]
        flat_h = torch.cat([hmap[level][0].flatten(1).T for level in range(3)])
        flat_coeff = pred["mask_coefficient"][0].T
        decoded_box = head._get_decode_boxes(pred)[0].T
        assert [neck[level].shape[1] for level in range(3)] == [256, 512, 512]
        for position in grouped[iid]:
            row = rows[position]
            saved = image["rows"][row["row_index"]]
            assert saved["annotation_id"] == row["annotation_id"]
            raw_id = int(row["raw_id"])
            level = int(image["levels"][raw_id])
            max_h_error = max(max_h_error, float((flat_h[raw_id].cpu() - row["h"]).abs().max()))
            max_coeff_error = max(max_coeff_error, float((flat_coeff[raw_id].cpu() - row["c0"]).abs().max()))
            max_box_error = max(max_box_error, float((decoded_box[raw_id].cpu() - image["boxes"][raw_id]).abs().max()))
            local = crop_pool(neck[level][0], decoded_box[raw_id]).cpu().half()
            padded = torch.zeros((512, 8, 8), dtype=torch.float16)
            padded[:local.shape[0]] = local
            output[position] = padded
        if step % 50 == 0 or step == len(index):
            print(json.dumps(dict(split=a.split, images=step, total=len(index),
                records=sum(q is not None for q in output))), flush=True)
    assert all(q is not None for q in output)
    assert max_h_error < 1e-4 and max_coeff_error < 1e-4 and max_box_error < 1e-3
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(roi=torch.stack(output), keys=[
        (int(r["image_id"]), int(r["annotation_id"]), int(r["raw_id"])) for r in rows]),
        a.out / "NECK.pt")
    summary = dict(split=a.split, images=len(index), effective_images=len(grouped),
        instances=len(output), roi_shape=[512, 8, 8], dtype="float16",
        max_h_error=max_h_error, max_coeff_error=max_coeff_error,
        max_box_error=max_box_error)
    (a.out / "COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("official_source", "weights", "bank", "targets", "out"):
        ap.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--split", choices=("fit", "dev", "val"), required=True)
    main(ap.parse_args())
