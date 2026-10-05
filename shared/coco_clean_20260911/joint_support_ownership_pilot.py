"""Train/evaluate a prediction-only pair ownership gate on one support domain.

The gate is trained on pixels from the intersection of two predicted same-class
boxes. COCO masks provide ownership labels only during fitting. At transfer,
the identical predicted-box intersection is used, but no GT enters the gate.
This is an exploratory pilot intended to test whether the earlier ownership
probe was disabled by a train/inference support mismatch.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import ops

from antisymmetric_ownership import PairGate, pair_features
from native_label_pipeline_probe import format_native, mask_of


ROOT = Path(__file__).resolve().parent
DEVICE = "cuda"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    wh = (torch.minimum(a[2:], b[2:]) - torch.maximum(a[:2], b[:2])).clamp_min(0)
    inter = float(wh.prod())
    aa = float((a[2:] - a[:2]).clamp_min(0).prod())
    ab = float((b[2:] - b[:2]).clamp_min(0).prod())
    return inter / max(aa + ab - inter, 1e-9)


def candidate_x(item, k: int) -> torch.Tensor:
    h, w = [float(v) for v in item["input_shape"]]
    bb = torch.as_tensor(item["boxes"][k], device=DEVICE, dtype=torch.float32)
    feat = torch.as_tensor(item["h"][k], device=DEVICE, dtype=torch.float32)
    level = F.one_hot(torch.tensor(int(item["level"][k]), device=DEVICE), 3).float()
    norm = bb / bb.new_tensor([w, h, w, h])
    wh = (norm[2:] - norm[:2]).clamp_min(1e-5)
    x = torch.cat([feat, level, norm, wh])
    # PairGate's documented geometry slice is normalized xyxy + normalized wh.
    x[67:73] = torch.cat([norm, wh])
    return x


def common_positions(box_i, box_j, shape):
    h, w = [int(v) for v in shape]
    x0 = max(0, int(np.ceil(max(float(box_i[0]), float(box_j[0])))))
    y0 = max(0, int(np.ceil(max(float(box_i[1]), float(box_j[1])))))
    x1 = min(w, int(np.floor(min(float(box_i[2]), float(box_j[2])))))
    y1 = min(h, int(np.floor(min(float(box_i[3]), float(box_j[3])))))
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, np.int64)
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    return (yy.reshape(-1) * w + xx.reshape(-1)).astype(np.int64)


def balanced_positions(pool, yi, yj, per_pair, seed):
    """Sample equal ownership states when available, with deterministic RNG."""
    yi = yi.astype(np.uint8)
    yj = yj.astype(np.uint8)
    state = yi * 2 + yj
    rng = np.random.default_rng(seed)
    groups = []
    quota = max(1, per_pair // 4)
    for value in range(4):
        q = np.flatnonzero(state == value)
        if len(q):
            take = min(quota, len(q))
            groups.append(rng.choice(q, size=take, replace=len(q) < take))
    if not groups:
        return np.empty(0, np.int64)
    selected = np.concatenate(groups)
    if len(selected) < per_pair:
        selected = np.concatenate([selected, rng.choice(len(pool), size=per_pair-len(selected), replace=True)])
    rng.shuffle(selected)
    return pool[selected[:per_pair]]


def build_pair_arrays(cache: Path, ann: COCO, ds, ids, per_pair=256):
    features, reverse, zi_all, zj_all, yi_all, yj_all = [], [], [], [], [], []
    pair_rows = []
    for number, iid in enumerate(ids, 1):
        npz_path = cache / "images" / f"{iid}.npz"
        meta_path = cache / "images" / f"{iid}.json"
        with np.load(npz_path) as z:
            item = {k: z[k] for k in z.files}
        meta = json.loads(meta_path.read_text())
        native_ids = list(map(int, meta["native_ids"]))
        ds_idx = {int(Path(x["im_file"]).stem): k for k, x in enumerate(ds.labels)}[int(iid)]
        lab = ds.labels[ds_idx]
        raw = ds.get_image_and_label(ds_idx)
        for transform in ds.transforms.transforms[:-1]:
            raw = transform(raw)
        formatted = format_native(raw, native_ids, 1, True)
        aids = list(map(int, item["annotation_ids"]))
        predidx = list(map(int, item["prediction_indices"]))
        aid_to_pred = dict(zip(aids, predidx))
        boxes = item["boxes"]
        det = item["detections"]
        input_shape = tuple(int(v) for v in item["input_shape"])
        p = torch.as_tensor(item["proto"], device=DEVICE, dtype=torch.float32)
        p640 = F.interpolate(p[None], input_shape, mode="bilinear", align_corners=False)[0]
        masks = {aid: mask_of(formatted, aid).astype(np.uint8).reshape(-1) for aid in native_ids}
        same = 0
        for ai in range(len(aids)):
            for aj in range(ai + 1, len(aids)):
                aid_i, aid_j = aids[ai], aids[aj]
                i, j = predidx[ai], predidx[aj]
                if int(det[i, 5]) != int(det[j, 5]):
                    continue
                if box_iou(torch.as_tensor(boxes[i]), torch.as_tensor(boxes[j])) <= 0.05:
                    continue
                pool = common_positions(boxes[i], boxes[j], input_shape)
                if not len(pool):
                    continue
                yi_pool, yj_pool = masks[aid_i][pool], masks[aid_j][pool]
                pos = balanced_positions(pool, yi_pool, yj_pool, per_pair, 7001 + int(iid) * 31 + ai * 7 + aj)
                if not len(pos):
                    continue
                pos_t = torch.as_tensor(pos, device=DEVICE, dtype=torch.long)
                zi = (p640.flatten(1)[:, pos_t].T @ torch.as_tensor(item["coeff"][i], device=DEVICE)).flatten()
                zj = (p640.flatten(1)[:, pos_t].T @ torch.as_tensor(item["coeff"][j], device=DEVICE)).flatten()
                xi, xj = candidate_x(item, i), candidate_x(item, j)
                fi = pair_features(zi, zj, xi, xj, pos_t.float(), input_shape)
                fr = pair_features(zj, zi, xj, xi, pos_t.float(), input_shape)
                features.append(fi.detach().cpu().numpy().astype(np.float32))
                reverse.append(fr.detach().cpu().numpy().astype(np.float32))
                zi_all.append(zi.detach().cpu().numpy().astype(np.float32))
                zj_all.append(zj.detach().cpu().numpy().astype(np.float32))
                yi_all.append(masks[aid_i][pos].astype(np.float32))
                yj_all.append(masks[aid_j][pos].astype(np.float32))
                pair_rows.append(dict(image_id=int(iid), aid_i=int(aid_i), aid_j=int(aid_j), pixels=int(len(pos)),
                                      state_counts=np.bincount(yi_pool * 2 + yj_pool, minlength=4).tolist()))
                same += 1
        if number % 25 == 0 or number == len(ids):
            print(json.dumps(dict(stage="pairs", images=number, total=len(ids), pairs=len(pair_rows))), flush=True)
    if not features:
        raise RuntimeError("No same-class overlapping candidate pairs were found")
    return dict(features=np.concatenate(features), reverse=np.concatenate(reverse),
                zi=np.concatenate(zi_all), zj=np.concatenate(zj_all),
                yi=np.concatenate(yi_all), yj=np.concatenate(yj_all), rows=pair_rows)


def train_gate(arrays, out: Path, seeds=(0, 1, 2), epochs=15, lr=1e-4):
    x = torch.as_tensor(arrays["features"], device=DEVICE)
    xr = torch.as_tensor(arrays["reverse"], device=DEVICE)
    zi = torch.as_tensor(arrays["zi"], device=DEVICE)
    zj = torch.as_tensor(arrays["zj"], device=DEVICE)
    yi = torch.as_tensor(arrays["yi"], device=DEVICE)
    yj = torch.as_tensor(arrays["yj"], device=DEVICE)
    n = len(x)
    perm = np.random.default_rng(20260913).permutation(n)
    cut = max(1, int(n * 0.8))
    fit = torch.as_tensor(perm[:cut], device=DEVICE)
    hold = torch.as_tensor(perm[cut:], device=DEVICE)
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    stats = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = PairGate().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(fit))
            total = 0.0
            for start in range(0, len(order), 8192):
                batch = fit[torch.as_tensor(order[start:start + 8192], device=DEVICE)]
                d = 0.5 * (model(x[batch]) - model(xr[batch]))
                loss = (F.binary_cross_entropy_with_logits(zi[batch] + d, yi[batch]) +
                        F.binary_cross_entropy_with_logits(zj[batch] - d, yj[batch])) * 0.5
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                opt.step()
                total += float(loss.detach()) * len(batch)
            with torch.no_grad():
                d = 0.5 * (model(x[hold]) - model(xr[hold]))
                before = 0.5 * (F.binary_cross_entropy_with_logits(zi[hold], yi[hold]) + F.binary_cross_entropy_with_logits(zj[hold], yj[hold]))
                after = 0.5 * (F.binary_cross_entropy_with_logits(zi[hold] + d, yi[hold]) + F.binary_cross_entropy_with_logits(zj[hold] - d, yj[hold]))
                active = int(((zi[hold] > 0) & (zj[hold] > 0) & (d.abs() > 1e-4)).sum())
            rec = dict(seed=seed, epoch=epoch, train_loss=total / len(fit), hold_before=float(before), hold_after=float(after),
                       hold_delta=float(after - before), hold_active_pixels=active)
            stats.append(rec)
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch}, out / "checkpoints" / f"pair_gate_s{seed}_epoch{epoch:02d}.pt")
    write_json(out / "TRAINING.json", dict(status="COMPLETE", rows=int(len(arrays["rows"])), pixels=int(n), fit=int(len(fit)), holdout=int(len(hold)), stats=stats))


def rle(mask):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


@torch.inference_mode()
def decode(item, model):
    p = torch.as_tensor(item["proto"], device=DEVICE, dtype=torch.float32)
    c = torch.as_tensor(item["coeff"], device=DEVICE, dtype=torch.float32)
    b = torch.as_tensor(item["boxes"], device=DEVICE, dtype=torch.float32)
    det = torch.as_tensor(item["detections"], device=DEVICE, dtype=torch.float32)
    h, w = [int(v) for v in item["input_shape"]]
    full = (c @ p.flatten(1)).reshape(1, -1, *p.shape[-2:])
    full = F.interpolate(full, (h, w), mode="bilinear", align_corners=False)[0]
    correction = torch.zeros_like(full)
    updates = 0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if int(det[i, 5]) != int(det[j, 5]) or box_iou(b[i], b[j]) <= 0.05:
                continue
            pos = common_positions(b[i].cpu().numpy(), b[j].cpu().numpy(), (h, w))
            if not len(pos):
                continue
            ids = torch.as_tensor(pos, device=DEVICE, dtype=torch.long)
            zi, zj = full[i].flatten()[ids], full[j].flatten()[ids]
            xi, xj = candidate_x(item, i), candidate_x(item, j)
            fi = pair_features(zi, zj, xi, xj, ids.float(), (h, w))
            fr = pair_features(zj, zi, xj, xi, ids.float(), (h, w))
            d = 0.5 * (model(fi) - model(fr))
            # Prediction-only conservative gate. The training objective can
            # produce a large ownership correction, but deployment clips it
            # to a small logit residual and only acts on ambiguous positives.
            # This preserves the stock decoder when the pair is already
            # decisive and prevents dense pair accumulation from exploding.
            active = (zi > 0) & (zj > 0) & ((zi - zj).abs() < 1.0)
            # Smoke-test setting: retain the learned bounded correction (the
            # PairGate output itself is bounded at +/-2) so a later full run
            # can measure whether a moderate logit shift actually changes
            # masks. The stock crop below remains mandatory.
            d = d
            d = torch.where(active, d, torch.zeros_like(d))
            correction[i].flatten()[ids] += d
            correction[j].flatten()[ids] -= d
            updates += int(active.sum())
    corrected = (full + correction > 0).to(torch.uint8)
    # Keep the official decoder contract: pair correction happens before the
    # stock predicted-box crop, exactly as in ops.process_mask.
    corrected = ops.crop_mask(corrected, b)
    return corrected, updates, float(correction.abs().sum())


def evaluate(cache: Path, ann: COCO, training: Path, ids, out: Path):
    models = {"initial": None}
    for seed in (0, 1, 2):
        candidates = sorted((training / "checkpoints").glob(f"pair_gate_s{seed}_epoch*.pt"))
        if not candidates:
            raise RuntimeError(f"missing checkpoint for seed {seed}")
        path = candidates[-1]
        model = PairGate().to(DEVICE).eval()
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model"])
        models[f"gate_s{seed}"] = model
    categories = sorted(ann.cats)
    preds = {name: [] for name in models}
    diagnostics = {name: dict(updates=0, correction_abs=0.0) for name in models}
    for iid in ids:
        with np.load(cache / "images" / f"{iid}.npz") as z:
            item = {k: z[k] for k in z.files}
        for name, model in models.items():
            if model is None:
                p = torch.as_tensor(item["proto"], device=DEVICE)
                c = torch.as_tensor(item["coeff"], device=DEVICE)
                b = torch.as_tensor(item["boxes"], device=DEVICE)
                masks = ops.process_mask(p, c, b, tuple(int(v) for v in item["input_shape"]), upsample=True)
            else:
                masks, count, corr = decode(item, model)
                diagnostics[name]["updates"] += count
                diagnostics[name]["correction_abs"] += corr
            restored = ops.scale_masks(masks[:, None], tuple(int(v) for v in item["shape"]))[:, 0].cpu().numpy()
            for j, mask in enumerate(restored):
                if not bool(mask.any()):
                    continue
                preds[name].append(dict(image_id=int(iid), category_id=categories[int(item["detections"][j, 5])],
                                        score=float(item["detections"][j, 4]), segmentation=rle(mask)))
    rows = []
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions").mkdir()
    for name, pp in preds.items():
        with gzip.open(out / "predictions" / f"{name}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(pp, fh, separators=(",", ":"))
        with contextlib.redirect_stdout(io.StringIO()):
            dt = ann.loadRes(pp)
            ev = COCOeval(ann, dt, "segm")
            ev.params.imgIds = list(map(int, ids))
            ev.evaluate(); ev.accumulate(); ev.summarize()
        rows.append(dict(arm=name, mask_ap=float(ev.stats[0]), mask_ap50=float(ev.stats[1]), mask_ap75=float(ev.stats[2]),
                         predictions=len(pp), **diagnostics[name]))
    write_json(out / "SUMMARY.json", dict(status="COMPLETE", images=len(ids), rows=rows,
                                          support="predicted same-class box intersection", GT_used_in_inference=False))
    print(json.dumps(rows, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-limit", type=int, default=120)
    ap.add_argument("--transfer-limit", type=int, default=60)
    ap.add_argument("--pixels-per-pair", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cache = args.cache.resolve(); out = args.out.resolve()
    selection = json.loads((cache / "selection.json").read_text())
    fit_ids = list(map(int, selection["fit"][:args.fit_limit]))
    transfer_ids = list(map(int, selection["transfer"][:args.transfer_limit]))
    ann_path = ROOT / "data" / "annotations" / "instances_train2017.json"
    if not ann_path.exists():
        ann_path = ROOT.parent.parent / "datasets" / "coco" / "annotations" / "instances_train2017.json"
    with contextlib.redirect_stdout(io.StringIO()):
        ann = COCO(str(ann_path))
    image_dir = cache / "converted" / "images" / "probe"
    cfg = get_cfg(overrides=dict(task="segment", imgsz=640, mask_ratio=1, overlap_mask=True, rect=False, cache=False, workers=0, fraction=1.0))
    data = dict(names={k: ann.cats[c]["name"] for k, c in enumerate(sorted(ann.cats))}, nc=80, channels=3)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = build_yolo_dataset(cfg, str(image_dir), 1, data, mode="val", rect=False)
    arrays = build_pair_arrays(cache, ann, ds, fit_ids, args.pixels_per_pair)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "pair_arrays.npz", **{k: arrays[k] for k in ["features", "reverse", "zi", "zj", "yi", "yj"]})
    write_json(args.out / "pair_rows.json", arrays["rows"])
    train_gate(arrays, args.out / "training", epochs=args.epochs)
    evaluate(cache, ann, args.out / "training", transfer_ids, args.out / "evaluation")
    write_json(args.out / "PROTOCOL.json", dict(status="COMPLETE", cache=str(cache), fit_images=fit_ids, transfer_images=transfer_ids,
        pixels_per_pair=args.pixels_per_pair, epochs=args.epochs, seeds=[0, 1, 2],
        train_support="same-class candidate predicted-box intersection", inference_support="same; prediction-only",
        labels="COCO masks used only in fitting; no GT in evaluate/decode", balanced_states="yi,yj in {00,01,10,11}"))


if __name__ == "__main__":
    main()
