"""S057: native YOLO26 one-to-one path on the same complete COCO val images."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "local_readout_runtime_20260912"
sys.path.insert(0, str(RUNTIME / "vendor"))
os.environ["YOLO_CONFIG_DIR"] = str(RUNTIME / "settings")
if os.name == "nt":
    os.environ["PATH"] = str(Path(sys.prefix) / "Library/bin") + os.pathsep + os.environ.get("PATH", "")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import argparse
import contextlib
import csv
import io
import json
import shutil
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER, ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mask_error_ap_probe import (ANNOTATION, CENSUS, SOURCE, dump, encode,
                                 evaluate, paired_stats, read_gz, save_csv, save_gz, sha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    (out / "images").mkdir()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    images = Path(source_protocol["image_root"])
    weight = ROOT / "weights/yolo26m-seg.pt"
    with CENSUS.open(encoding="utf-8-sig") as handle:
        metadata = {int(r["annotation_id"]): r for r in csv.DictReader(handle)}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANNOTATION))
    ids = sorted(gt.imgs)
    assert len(ids) == 5000
    assert sha(weight) == "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
    model = YOLO(str(weight))
    loaded_end2end = bool(model.model.model[-1].end2end)
    protocol = dict(experiment="S057_NATIVE_ONE2ONE_FULL_VAL", training=False, image_ids=ids,
                    image_root=str(images), weight_sha256=sha(weight), script_sha256=sha(__file__),
                    helper_sha256=sha(ROOT / "mask_error_ap_probe.py"), annotation_sha256=sha(ANNOTATION),
                    torch=torch.__version__, ultralytics=ultralytics.__version__,
                    ultralytics_path=ultralytics.__file__, loaded_head_end2end=loaded_end2end,
                    inference=dict(end2end=True, imgsz=640, rect=False, conf=.001, iou=.7,
                                   max_det=300, half=False, retina_masks=False, device=0, tf32=False),
                    comparison="Same weights/runtime/images/decoder and confidence as S036; use one-to-one head/native top-k instead of one-to-many head/class NMS. Entire prediction sets may differ; this is an inference-path comparison, not isolated NMS causality.",
                    decode="Official SegmentationPredictor process_mask, binary scale_masks to original image and >.5, same export as S036.",
                    metrics="Official full COCO Mask AP, Box AP and all-GT R75/P90 by unchanged GT-only E(4) groups. Exploratory val previously used, no tuning.")
    dump(out / "protocol.json", protocol)
    shutil.copy2(__file__, out / Path(__file__).name)
    LOGGER.setLevel(40)
    categories = sorted(gt.cats)
    start = time.monotonic()
    predictions, boxes, witnesses = [], [], []
    receipts = json.loads((SOURCE / "image_receipts.json").read_text())
    with torch.inference_mode():
        for number, iid in enumerate(ids, 1):
            path = images / gt.imgs[iid]["file_name"]
            assert sha(path) == receipts[str(iid)]["jpeg_sha256"]
            result = model.predict(str(path), end2end=True, imgsz=640, rect=False, conf=.001,
                                   iou=.7, max_det=300, half=False, retina_masks=False,
                                   device=0, verbose=False)[0]
            head = model.predictor.model.model.model[-1]
            assert bool(head.end2end) and bool(model.predictor.model.end2end), "Native branch not active"
            det = result.boxes.data.cpu().numpy()
            assert len(det) <= 300
            mask = (ops.scale_masks(result.masks.data[:, None], result.orig_shape)[:, 0] > .5).cpu().numpy() if len(det) else []
            assert len(mask) == len(det)
            pp, bb = [], []
            for j, detection in enumerate(det):
                x1, y1, x2, y2, score, category = map(float, detection[:6])
                common = dict(image_id=iid, category_id=categories[int(category)], score=score)
                pp.append(dict(**common, segmentation=encode(mask[j])))
                bb.append(dict(**common, bbox=[x1, y1, x2 - x1, y2 - y1]))
            save_gz(out / "images" / f"{iid}.json.gz", dict(predictions=pp, boxes=bb))
            predictions.extend(pp)
            boxes.extend(bb)
            witnesses.append(dict(image_id=iid, end2end=True, predictions=len(pp), jpeg_hash_exact=True))
            if number % 100 == 0:
                value = dict(stage="INFERENCE", images=number, total=5000, predictions=len(predictions),
                             seconds=round(time.monotonic() - start, 2), pid=os.getpid())
                dump(out / "progress.json", value)
                print(json.dumps(value), flush=True)
    del model
    torch.cuda.empty_cache()
    dump(out / "WITNESS.json", witnesses)
    summary, records = evaluate(gt, predictions, ids, metadata, "native_one2one")
    save_csv(out / "native_one2one_gt.csv", records)
    dump(out / "native_one2one.json", dict(summary=summary))
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(boxes), "bbox")
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    dump(out / "box_ap.json", dict(box_ap=float(ev.stats[0]), box_ap50=float(ev.stats[1]),
                                  box_ap75=float(ev.stats[2])))
    prior = ROOT / "diagnostics/mask_error_ap_20260913"
    with (prior / "original_gt.csv").open() as handle:
        original = list(csv.DictReader(handle))
    for row in original:
        for key in ("annotation_id", "image_id"):
            row[key] = int(row[key])
        row["hit75"] = row["hit75"] == "True"
    dump(out / "paired_stats.json", paired_stats(original, records))
    save_csv(out / "comparison.csv", [json.loads((prior / "original.json").read_text())["summary"], summary])
    dump(out / "COMPLETE.json", dict(status="COMPLETE", images=5000, gt=len(records),
                                      seconds=round(time.monotonic() - start, 3), predictions=len(predictions),
                                      comparison_sha256=sha(out / "comparison.csv")))
    dump(out / "progress.json", dict(stage="COMPLETE", images=5000, pid=os.getpid()))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
