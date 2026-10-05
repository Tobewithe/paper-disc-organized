"""COCO pilot: norm-preserving coefficient interventions in a frozen decoder.

GT-only pair/image selection is persisted before inference. Ordinary COCO bbox
matching determines ownership. Small symmetric rotations have matched random
controls; cropped GT pixel errors and raw representation overlap are distinct.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

from coco_instance_metrics import ROOT, digest, write_csv, write_json
from coco_spatial_diagnosis import box_assignment

WEIGHTS = {
    "baseline": ROOT / "gemini/remote_runs/baseline_stage2/weights/best.pt",
    "ccl01": ROOT / "gemini/models/yolo26_coco_ccl_w0.1_stage2_best.pt",
}
GT_PATH = ROOT / "gemini/data/coco_dense/coco_dense_val_gt.json"
PREVIOUS = ROOT / "gemini/results/coco_instance_eval/20260911_034617"
ANGLES = (2., 5., 10.)
SEED = 20260911


def pair_rotations(coeff, degrees, seed):
    """Rotate each coefficient by equal angle in its pair plane or random tangent."""
    c = np.asarray(coeff, dtype=np.float64)
    norms = np.linalg.norm(c, axis=1)
    if np.min(norms) < 1e-8:
        raise ValueError("zero_coefficient_norm")
    u, v = c / norms[:, None]
    rho = float(np.clip(u @ v, -1., 1.))
    phi = math.acos(rho)
    effective = min(math.radians(degrees), phi/2 - 1e-6, (math.pi-phi)/2 - 1e-6)
    if effective < 1e-6:
        raise ValueError("collinear_coefficients")
    e1 = u
    e2 = (v - rho*u) / math.sin(phi)
    cs, sn = math.cos(effective), math.sin(effective)
    push = np.stack([cs*e1-sn*e2, math.cos(phi+effective)*e1+math.sin(phi+effective)*e2])
    pull = np.stack([cs*e1+sn*e2, math.cos(phi-effective)*e1+math.sin(phi-effective)*e2])
    rng = np.random.RandomState(seed)
    units = c / norms[:, None]
    tangent = rng.normal(size=c.shape)
    tangent -= (tangent*units).sum(axis=1)[:, None] * units
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    variants = {"push": push*norms[:, None], "pull": pull*norms[:, None],
                "random_plus": (cs*units+sn*tangent)*norms[:, None],
                "random_minus": (cs*units-sn*tangent)*norms[:, None]}
    for x in variants.values():
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), norms, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose((x*c).sum(axis=1) / norms**2, cs, atol=1e-10, rtol=1e-10)
    assert np.dot(push[0], push[1]) < rho and np.dot(pull[0], pull[1]) > rho
    return variants, math.degrees(effective), rho


class CapturingPredictor(SegmentationPredictor):
    def construct_result(self, pred, img, orig_img, img_path, proto):
        raw = pred.detach().clone()
        output = super().construct_result(pred, img, orig_img, img_path, proto)
        if len(raw):
            replay = ops.process_mask(proto, raw[:, 6:], raw[:, :4], img.shape[2:], upsample=True)
            keep = replay.amax((-2, -1)) > 0
            assert output.masks is not None
            assert torch.equal(replay[keep], output.masks.data), "Capture changed predictor masks"
        else:
            keep = torch.zeros(0, dtype=torch.bool, device=raw.device)
        self.capture = {"proto": proto.detach().clone(), "coeff": raw[keep, 6:], "boxes_pad": raw[keep, :4],
                        "boxes_original": output.boxes.data.detach().clone(), "input_shape": tuple(img.shape[2:]),
                        "original_shape": tuple(orig_img.shape[:2])}
        return output


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def freeze_selection(out, n_images):
    path = out / "gt_selection.json"
    if path.exists():
        selection = json.loads(path.read_text(encoding="utf-8"))
        assert selection["requested_images"] == n_images
        return selection
    density_path = ROOT / "refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv"
    density = {int(r["annotation_id"]): float(r["ici_same"]) for r in read_csv(density_path)}
    qpath = ROOT / "gemini/data/coco_dense/coco_dense_quartiles.csv"
    quartiles = {int(r["image_id"]): r["quartile"] for r in read_csv(qpath)}
    grouped = {}
    for r in read_csv(PREVIOUS / "gt_pairs.csv"):
        a, b, iid = int(r["annotation_a"]), int(r["annotation_b"]), int(r["image_id"])
        if r["same_category"] == "True" and max(density[a], density[b]) > .5:
            grouped.setdefault(iid, []).append({"image_id": iid, "annotation_a": a, "annotation_b": b,
                                                "ici_a": density[a], "ici_b": density[b], "gt_box_iou": float(r["box_iou"]), "quartile": quartiles[iid]})
    # A stable hash orders images/pairs without using outcomes or model tensors.
    def rank(value):
        return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()
    selected = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        quota = n_images//4 + (int(q[1])-1 < n_images % 4)
        ids = sorted([i for i in grouped if quartiles[i] == q], key=rank)
        assert len(ids) >= quota
        for iid in ids[:quota]:
            pair = min(grouped[iid], key=lambda r: rank(f"{iid}:{r['annotation_a']}:{r['annotation_b']}"))
            selected.append(pair)
    selection = {"created": datetime.now().isoformat(), "requested_images": n_images, "seed": SEED,
                 "rule": "Equal image-Q quotas; stable hash sample among GT same-category BoxIoU>.05 pairs with >=one instance ICI>.5; exactly one GT pair/image; no outcome selection",
                 "gt_sha256": digest(GT_PATH), "density_sha256": digest(density_path), "pairs": sorted(selected, key=lambda r: r["image_id"])}
    write_json(path, selection)
    return selection


def gpu_pixel_metrics(masks, own, union, same_union, valid):
    """N binary masks for the same target, exact raster pixel decomposition."""
    own = own & valid
    ga = own.sum()
    if ga == 0:
        raise ValueError("fully_ignored_target")
    m = masks & valid
    tp = (m & own).sum((1, 2)).double()
    near = (m & union & ~own).sum((1, 2)).double()
    same = (m & same_union & ~own).sum((1, 2)).double()
    background = (m & ~union).sum((1, 2)).double()
    pa = m.sum((1, 2)).double()
    assert torch.equal(tp+near+background, pa)
    return {"coverage": (tp/ga).cpu().numpy(), "neighbor": (near/ga).cpu().numpy(), "same_neighbor": (same/ga).cpu().numpy(),
            "background": (background/ga).cpu().numpy(), "mask_iou": (tp/(ga+near+background)).cpu().numpy()}


def tensor_digest(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def diagnose_pair(capture, assignments, gt, pair, model_name, out):
    ids = [pair["annotation_a"], pair["annotation_b"]]
    pred_idx = [assignments.get(a, -1) for a in ids]
    if min(pred_idx) < 0:
        return [], {"status": "unmatched", "matched_a": pred_idx[0] >= 0, "matched_b": pred_idx[1] >= 0}
    assert pred_idx[0] != pred_idx[1]
    p = capture["proto"]
    coeff = capture["coeff"][pred_idx].float()
    boxes = capture["boxes_pad"][pred_idx].float()
    input_shape, shape = capture["input_shape"], capture["original_shape"]
    coeff_np = coeff.cpu().numpy()
    variants = [("zero", 0., 0., coeff_np)]
    for angle in ANGLES:
        try:
            changes, effective, rho = pair_rotations(coeff_np, angle, (SEED + pair["image_id"]) % (2**32-1))
        except ValueError as e:
            return [], {"status": str(e)}
        variants += [(name, angle, effective, value) for name, value in changes.items()]
    stacked = torch.as_tensor(np.concatenate([v[3] for v in variants]), dtype=torch.float32, device=coeff.device)
    tiled_boxes = boxes.repeat(len(variants), 1)
    before = (tensor_digest(p), tensor_digest(boxes), tensor_digest(capture["boxes_original"]))
    masks = ops.process_mask(p, stacked, tiled_boxes, input_shape, upsample=True)
    masks = ops.scale_masks(masks[:, None], shape).squeeze(1) > .5
    expected = ops.process_mask(p, capture["coeff"], capture["boxes_pad"], input_shape, upsample=True)[pred_idx]
    expected = ops.scale_masks(expected[:, None], shape).squeeze(1) > .5
    zero_difference = int((masks[:2] != expected).sum().item())
    zero_fraction = zero_difference / expected.numel()
    assert zero_fraction <= 1e-5, f"Zero perturbation replay exceeds fixed tolerance: {zero_fraction}"
    masks = masks.reshape(len(variants), 2, *shape)
    anns = gt.imgToAnns[pair["image_id"]]
    raster = {a["id"]: torch.as_tensor(gt.annToMask(a).astype(bool), device=coeff.device) for a in anns}
    ordinary = [a for a in anns if not a.get("iscrowd", 0)]
    union = torch.stack([raster[a["id"]] for a in ordinary]).any(0)
    crowds = [raster[a["id"]] for a in anns if a.get("iscrowd", 0)]
    valid = ~torch.stack(crowds).any(0) if crowds else torch.ones(shape, dtype=torch.bool, device=coeff.device)
    metrics = []
    for j, aid in enumerate(ids):
        same = torch.stack([raster[a["id"]] for a in ordinary if a["category_id"] == gt.anns[aid]["category_id"]]).any(0)
        try:
            metrics.append(gpu_pixel_metrics(masks[:, j], raster[aid], union, same, valid))
        except ValueError as e:
            return [], {"status": str(e)}
    logits = (stacked @ p.float().flatten(1)).reshape(len(variants), 2, -1)
    raw = logits > 0
    intersection = (raw[:, 0] & raw[:, 1]).sum(1)
    raw_union = (raw[:, 0] | raw[:, 1]).sum(1)
    raw_iou = (intersection / raw_union.clamp(min=1)).cpu().numpy()
    positive = logits.clamp(min=0)
    q = (positive[:, 0]*positive[:, 1]).mean(1).cpu().numpy()
    zcos = torch.nn.functional.cosine_similarity(logits[:, 0], logits[:, 1], dim=1).cpu().numpy()
    original_logits = logits[0].clone()
    logit_delta = ((logits-original_logits).square().mean((1, 2)).sqrt() / original_logits.square().mean().sqrt().clamp(min=1e-12)).cpu().numpy()
    gram = p.float().flatten(1) @ p.float().flatten(1).T
    eig = torch.linalg.eigvalsh(gram.double()).cpu().numpy()
    rows = []
    for k, (name, requested, effective, v) in enumerate(variants):
        cosine = float(np.dot(v[0], v[1]) / (np.linalg.norm(v[0])*np.linalg.norm(v[1])))
        common = {"model": model_name, "image_id": pair["image_id"], "annotation_a": ids[0], "annotation_b": ids[1],
                  "mode": name, "degrees_requested": requested, "degrees_effective": effective, "coefficient_cosine": cosine,
                  "logit_cosine": float(zcos[k]), "positive_product": float(q[k]), "raw_pair_iou": float(raw_iou[k]),
                  "relative_logit_rms_change": float(logit_delta[k])}
        for j, aid in enumerate(ids):
            row = {**common, "target_annotation": aid, "target_ici": pair["ici_a"] if j == 0 else pair["ici_b"], "target_position": j,
                   "detection_index": pred_idx[j], "fixed_score": float(capture["boxes_original"][pred_idx[j], 4].item())}
            row.update({field: float(values[k]) for field, values in metrics[j].items()})
            row.update({"delta_"+field: float(values[k]-values[0]) for field, values in metrics[j].items()})
            rows.append(row)
    after = (tensor_digest(p), tensor_digest(boxes), tensor_digest(capture["boxes_original"]))
    assert before == after, "Prototype, boxes or scores changed"
    artifact = out / "tensors" / f"{model_name}_{pair['image_id']}.npz"
    artifact.parent.mkdir(exist_ok=True)
    np.savez_compressed(artifact, proto=p.float().cpu().numpy(), coeff=coeff_np, boxes_pad=boxes.cpu().numpy(),
                        input_shape=np.array(input_shape), original_shape=np.array(shape), scores=capture["boxes_original"][pred_idx, 4].cpu().numpy(),
                        annotation_ids=np.array(ids), variants=stacked.cpu().numpy())
    return rows, {"status": "ok", "zero_replay_pixel_difference": zero_difference, "zero_replay_fraction": zero_fraction,
                  "prototype_unchanged": True, "boxes_scores_unchanged": True, "gram_eigenvalues": eig.tolist(),
                  "coefficient_norms": np.linalg.norm(coeff_np, axis=1).tolist(), "tensor_file": str(artifact), "tensor_sha256": digest(artifact)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection = freeze_selection(args.out, args.images)
    pairs = selection["pairs"][:args.limit] if args.limit else selection["pairs"]
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    witness = torch.randn(16, 16, device="cuda")
    assert torch.isfinite(witness @ witness.T).all()
    versions = {n: importlib.metadata.version(n) for n in ("torch", "ultralytics", "numpy", "pycocotools")}
    metadata = {"started": datetime.now().isoformat(), "scope": "COCO frozen-decoder pilot, new rect=False batch1 predictions; not standard AP or a training effect estimate",
                "script_sha256": digest(__file__), "ops_sha256": digest(ops.__file__), "gt_sha256": digest(GT_PATH), "selection_sha256": digest(args.out / "gt_selection.json"),
                "weights": {n: digest(p) for n, p in WEIGHTS.items()}, "versions": versions, "device": torch.cuda.get_device_name(),
                "inference": {"imgsz": 640, "conf": .001, "iou": .7, "max_det": 300, "batch": 1, "rect": False, "retina_masks": False, "half": False},
                "angle_degrees_per_vector": ANGLES, "zero_replay_tolerance_fraction": 1e-5, "requested_images": args.images, "current_image_limit": args.limit}
    write_json(args.out / ("pilot_run.json" if args.limit else "run.json"), metadata)
    gt = COCO(str(GT_PATH))
    catids = sorted(gt.cats)
    rows, statuses = [], []
    for name, weights in WEIGHTS.items():
        model = YOLO(str(weights))
        for k, pair in enumerate(pairs, 1):
            result_path = args.out / f"pair_{name}_{pair['image_id']}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                assert saved["script_sha256"] == metadata["script_sha256"]
                assert saved["weights_sha256"] == metadata["weights"][name]
                rows.extend(saved["rows"])
                statuses.append(saved["status"])
                continue
            im = gt.imgs[pair["image_id"]]
            path = ROOT / "gemini/data/coco_dense/images/val2017" / im["file_name"]
            with torch.inference_mode():
                result = model.predict(str(path), predictor=CapturingPredictor, imgsz=640, conf=.001, iou=.7, max_det=300,
                                       device=0, rect=False, retina_masks=False, half=False, verbose=False)[0]
                capture = model.predictor.capture
                predictions = []
                for b in result.boxes.data.cpu().numpy():
                    predictions.append({"image_id": im["id"], "category_id": catids[int(b[5])], "score": float(b[4]),
                                        "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])]})
                with contextlib.redirect_stdout(io.StringIO()):
                    local_gt = COCO()
                    local_gt.dataset = {"info": {}, "images": [copy.deepcopy(im)], "categories": copy.deepcopy(gt.dataset["categories"]),
                                        "annotations": copy.deepcopy(gt.imgToAnns[im["id"]])}
                    local_gt.createIndex()
                    aids = [a["id"] for a in local_gt.imgToAnns[im["id"]] if not a.get("iscrowd", 0)]
                    assigned = box_assignment(local_gt, predictions, aids)
                mapping = {aid: int(did)-1 for aid, did in zip(aids, assigned) if did > 0}
                new_rows, status = diagnose_pair(capture, mapping, gt, pair, name, args.out)
            status.update({"model": name, **pair, "detections": len(predictions)})
            rows.extend(new_rows)
            statuses.append(status)
            write_json(result_path, {"rows": new_rows, "status": status, "script_sha256": metadata["script_sha256"], "weights_sha256": metadata["weights"][name]})
            if k % 10 == 0 or k == len(pairs):
                print(f"{name}: {k}/{len(pairs)} images; last={status['status']}", flush=True)
        del model
        torch.cuda.empty_cache()
    write_csv(args.out / ("pilot_interventions.csv" if args.limit else "interventions.csv"), rows)
    write_json(args.out / ("pilot_statuses.json" if args.limit else "statuses.json"), statuses)
    metadata["completed"] = datetime.now().isoformat()
    metadata["valid_pairs_by_model"] = {n: sum(s["model"] == n and s["status"] == "ok" for s in statuses) for n in WEIGHTS}
    write_json(args.out / ("pilot_run.json" if args.limit else "run.json"), metadata)
    print(json.dumps({"completed": metadata["completed"], "valid_pairs": metadata["valid_pairs_by_model"], "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
