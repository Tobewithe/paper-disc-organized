"""Seed-0 full COCO val2017 confirmation for Ada+Center.

The backbone is run once per image with the vendor runtime used by the
existing full-val audit.  Candidate boxes, classes and scores are shared by
all arms; only the mask readout is changed.  No GT enters inference.
"""
import os, sys
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "4")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "local_readout_runtime_20260912" / "vendor"
sys.path.insert(0, str(VENDOR)); sys.path.insert(0, str(ROOT))
import contextlib, csv, gzip, hashlib, io, json, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from ultralytics import YOLO
from ultralytics.utils import ops, nms
from structure_candidate_trace import TraceCapture
from rich_pixel_readout import GlobalHead, instance_features
from ada_calib_readout import AdaCalibHead
from contrastive_spatial_loss import apply_centerness_prior
from eval_readout_input_pilot import ici

ANN = ROOT.parent.parent / "datasets/coco/annotations/instances_val2017.json"
IMAGES = ROOT / "local_readout_runtime_20260912/data/images/val2017"
WEIGHT = ROOT / "weights/yolo26m-seg.pt"
NORMALIZER = ROOT / "diagnostics/shared_label_controls_20260912/normalizer.pt"
S032 = ROOT / "diagnostics/shared_label_controls_20260912/raw_coco_s0/checkpoints/epoch015.pt"
ADA = ROOT / "runs/innovation_reaudit_20260913/corrected_training/ada_s0/epoch015.pt"
OUT = ROOT / "diagnostics/final_candidate_fullval_s0_20260914"
ARMS = ["s032_s0", "ada_s0", "center_s0", "ada_center_s0"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_gz(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(path)


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)


def groups(v):
    return "low" if v <= 1e-10 else "middle" if v <= .5 + 1e-10 else "high"


def load_heads(device):
    s = GlobalHead().to(device)
    s.load_state_dict(torch.load(S032, map_location=device, weights_only=False)["model"])
    s.eval().requires_grad_(False)
    a = AdaCalibHead().to(device)
    st = torch.load(ADA, map_location=device, weights_only=False)["model"]
    st = {k[5:] if k.startswith("base.") else k: v for k, v in st.items()}
    a.load_state_dict(st); a.eval().requires_grad_(False)
    return s, a


def decode(z, boxes, input_shape, shape, threshold):
    up = F.interpolate(z[:, None], input_shape, mode="bilinear", align_corners=False)[:, 0]
    bb = ops.crop_mask(up, boxes)
    return (ops.scale_masks(bb[:, None], shape)[:, 0] > threshold).cpu().numpy()


def evaluate(gt, ids, meta, predictions, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, "segm"); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    t = int(np.flatnonzero(np.isclose(ev.params.iouThrs, .75))[0])
    records, scores, tp, fp = [], [], [], []
    for item in ev.evalImgs:
        if item is None or item["aRng"] != [0, 1e10] or item["maxDet"] != 100: continue
        score_map = dict(zip(item["dtIds"], item["dtScores"]))
        valid = ~item["dtIgnore"][t].astype(bool)
        hit = item["dtMatches"][t] > 0
        scores.extend(np.asarray(item["dtScores"])[valid].tolist())
        tp.extend(hit[valid].tolist()); fp.extend((~hit[valid]).tolist())
        for j, aid in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]: continue
            aid = int(aid); ann = gt.anns[aid]; matched = int(item["gtMatches"][t, j])
            v = meta[aid]
            records.append(dict(arm=arm, image_id=item["image_id"], annotation_id=aid,
                category_id=ann["category_id"], area=ann["area"], ici=v,
                group=groups(v), hit75=bool(matched),
                matched_score75=float(score_map[matched]) if matched else None))
    scores = np.asarray(scores); order = np.argsort(-scores, kind="stable")
    ct = np.cumsum(np.asarray(tp, dtype=np.int64)[order]); cf = np.cumsum(np.asarray(fp, dtype=np.int64)[order])
    endpoints = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    precision = ct[endpoints] / np.maximum(ct[endpoints] + cf[endpoints], 1)
    q = endpoints[(precision >= .90) & (ct[endpoints] > 0)]
    threshold = float(scores[q[-1]]) if len(q) else None
    for r in records: r["hit90"] = bool(threshold is not None and r["hit75"] and r["matched_score75"] >= threshold)
    row = dict(arm=arm, mask_ap=float(ev.stats[0]), mask_ap50=float(ev.stats[1]),
        mask_ap75=float(ev.stats[2]), predictions=len(predictions), gt=len(records),
        p90_threshold=threshold)
    for g in ["all", "low", "middle", "high", "nonhigh"]:
        rr = [r for r in records if g == "all" or (r["group"] != "high" if g == "nonhigh" else r["group"] == g)]
        row[f"n_{g}"] = len(rr)
        row[f"r75_{g}"] = float(np.mean([r["hit75"] for r in rr])) if rr else None
        row[f"r90_{g}"] = float(np.mean([r["hit90"] for r in rr])) if rr else None
    row["gap"] = row["r75_nonhigh"] - row["r75_high"]
    return row, records


def main():
    OUT.mkdir(parents=True, exist_ok=True); shard = OUT / "images"; shard.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with contextlib.redirect_stdout(io.StringIO()): gt = COCO(str(ANN))
    ids = sorted(gt.imgs)
    if len(ids) != 5000: raise RuntimeError(f"Expected 5000 val images, got {len(ids)}")
    normalizer = torch.load(NORMALIZER, map_location=device, weights_only=True)
    s032, ada = load_heads(device)
    model = YOLO(str(WEIGHT)); model.model.eval().requires_grad_(False)
    head = model.model.model[-1]
    if head.end2end: head.end2end = False
    maps, hooks = {}, []
    for level, branch in enumerate(head.cv4):
        def hook(module, args, output, level=level): maps[level] = args[0].detach().clone()
        hooks.append(branch[-1].register_forward_hook(hook))
    receipts_path = OUT / "receipts.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8")) if receipts_path.exists() else {}
    # A previous attempt stopped on the float empty-index corner case.  The
    # per-image shards are valid through the last completed receipt; remove
    # only the stale failure marker so the run resumes safely.
    failed_marker = OUT / "FAILED.json"
    if failed_marker.exists(): failed_marker.unlink()
    categories = sorted(gt.cats)
    start = time.monotonic()
    for num, iid in enumerate(ids, 1):
        path = shard / f"{iid}.json.gz"
        if str(iid) in receipts:
            continue
        maps.clear(); image_path = IMAGES / gt.imgs[iid]["file_name"]
        with torch.inference_mode():
            model.predict(str(image_path), predictor=TraceCapture, imgsz=640, rect=False,
                conf=.001, iou=.7, max_det=300, half=False, retina_masks=False, device=0, verbose=False)
            pred = model.predictor; raw, cap, pre = pred.dense, pred.capture, pred.before_empty
            allnms, idx = nms.non_max_suppression(raw.clone(), conf_thres=.001, iou_thres=.7,
                nc=80, max_det=raw.shape[-1], return_idxs=True, end2end=False)
            if not torch.equal(allnms[0][:300], pre): raise RuntimeError(f"NMS replay mismatch {iid}")
            pre_mask = ops.process_mask(cap["proto"], pre[:, 6:], pre[:, :4], (640, 640), upsample=True)
            # Ultralytics returns a float tensor for the empty-index branch
            # on some images; normalize before using it as a tensor index.
            keep_all = idx[0].flatten()[:300].to(dtype=torch.long)
            keep = keep_all[pre_mask.flatten(1).any(1).bool()]
            if not torch.equal(raw[0, 84:, keep].T, cap["coeff"]): raise RuntimeError(f"Coeff replay mismatch {iid}")
            h_all = torch.cat([maps[k][0].flatten(1).T for k in range(3)], dim=0)
            level_all = torch.cat([torch.full((maps[k].shape[-2] * maps[k].shape[-1],), k, device=device, dtype=torch.long) for k in range(3)])
            h = h_all[keep]; level = level_all[keep]
            x = (instance_features(h, level, cap["boxes"], cap["input_shape"]) - normalizer["mean"]) / normalizer["std"]
            c, p, boxes = cap["coeff"], cap["proto"].float(), cap["boxes"]
            z_s = torch.einsum("bc,chw->bhw", c + s032(x), p)
            z_a = ada.forward_inference(x, c, p)[0]
            z_c = apply_centerness_prior(z_s, boxes * (160.0 / cap["input_shape"][0]), (160, 160), margin_scale=.5)
            z_ac = apply_centerness_prior(z_a, boxes * (160.0 / cap["input_shape"][0]), (160, 160), margin_scale=.5)
            # Match the established innovation evaluator: S032 uses the
            # stock sigmoid/process-mask equivalent; adaptive arms threshold
            # the interpolated logit at zero.
            base = ops.process_mask(p, c + s032(x), boxes, cap["input_shape"], upsample=True)
            binaries = {
                "s032_s0": (ops.scale_masks(base[:, None], cap["shape"])[:, 0] > .5).cpu().numpy(),
                "ada_s0": decode(z_a, boxes, cap["input_shape"], cap["shape"], 0.0),
                "center_s0": decode(z_c, boxes, cap["input_shape"], cap["shape"], 0.0),
                "ada_center_s0": decode(z_ac, boxes, cap["input_shape"], cap["shape"], 0.0),
            }
            det = cap["detections"].cpu().numpy(); payload = {arm: [] for arm in ARMS}
            for arm in ARMS:
                nonempty = binaries[arm].any(axis=(1, 2)) if binaries[arm].ndim == 3 else np.zeros(0, dtype=bool)
                for j in np.flatnonzero(nonempty):
                    rle = mu.encode(np.asfortranarray(binaries[arm][j].astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
                    payload[arm].append(dict(image_id=iid, category_id=categories[int(det[j, 5])], score=float(det[j, 4]), segmentation=rle))
            payload["original_boxes"] = [dict(image_id=iid, category_id=categories[int(q[5])], score=float(q[4]), bbox=[float(q[0]), float(q[1]), float(q[2]-q[0]), float(q[3]-q[1])]) for q in det]
            payload["witness"] = dict(image_id=iid, slots=len(c), source_indices=keep.cpu().tolist(), counts={a: len(payload[a]) for a in ARMS}, jpeg_sha256=sha(image_path), decoder="S032 process_mask; adaptive logits interpolate/crop/>0")
        save_gz(path, payload); receipts[str(iid)] = dict(sha256=sha(path), slots=len(c), counts=payload["witness"]["counts"]); dump(receipts_path, receipts)
        if num % 50 == 0 or num == len(ids):
            prog = dict(stage="INFERENCE", completed=num, total=len(ids), seconds=round(time.monotonic()-start, 1)); dump(OUT / "progress.json", prog); print(json.dumps(prog), flush=True)
    for h in hooks: h.remove()
    meta = {ann["id"]: ici(ann, [q for q in gt.imgToAnns[ann["image_id"]] if not q.get("iscrowd", 0)]) for ann in gt.anns.values() if not ann.get("iscrowd", 0)}
    summary, all_records = [], []
    for arm in ARMS:
        predictions = []; 
        for iid in ids: predictions.extend(load_gz(shard / f"{iid}.json.gz")[arm])
        row, records = evaluate(gt, ids, meta, predictions, arm); summary.append(row); all_records.extend(records)
        (OUT / f"{arm}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{arm}] AP={row['mask_ap']*100:.3f}% AP75={row['mask_ap75']*100:.3f}% highR75={row['r75_high']*100:.3f}% lowR75={row['r75_low']*100:.3f}% gap={row['gap']*100:.3f} pts", flush=True)
    fields = list(all_records[0]);
    with (OUT / "gt_records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_records)
    (OUT / "summary.csv").write_text(pd.DataFrame(summary).to_csv(index=False), encoding="utf-8") if False else None
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    dump(OUT / "COMPLETE.json", dict(status="COMPLETE", images=5000, ordinary_gt=len(meta), arms=ARMS, seconds=time.monotonic()-start, source_hashes={"weight": sha(WEIGHT), "annotation": sha(ANN), "s032": sha(S032), "ada": sha(ADA), "normalizer": sha(NORMALIZER)}, receipts_sha256=sha(receipts_path)))


if __name__ == "__main__":
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True); dump(OUT / "FAILED.json", {"error": repr(e), "traceback": traceback.format_exc()}); raise
