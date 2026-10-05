"""COCO frozen-readout crop-support mechanism probe.

For cached official candidates, replay the raw prototype response while changing
only the final crop scale. Fit a threshold for a no-GT border-evidence heuristic
on the disjoint fit images, then evaluate it on transfer images. This is a
diagnostic, not an AP result or a trained method.
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "local_readout_runtime_20260912" / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from ultralytics.utils import ops

SCALES = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def box_iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    return inter / max(1e-12, aw * ah + bw * bh - inter)


def expand_box(box, scale):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def decode_masks(proto, coeff, boxes, shape, scales, device):
    """Return original-resolution cropped binary masks, [N,S,H,W]."""
    n = coeff.shape[0]
    logits = (coeff @ proto.flatten(1)).reshape(n, *proto.shape[-2:])
    logits = F.interpolate(logits[:, None], (640, 640), mode="bilinear", align_corners=False)[:, 0]
    raw = logits.gt(0).byte()
    out = []
    for scale in scales:
        eb = torch.as_tensor([expand_box(b.tolist(), float(scale)) for b in boxes], device=device)
        cropped = ops.crop_mask(raw.clone(), eb)
        out.append((ops.scale_masks(cropped[:, None], shape)[:, 0] > 0.5).cpu().numpy())
    return np.stack(out, axis=1), logits, raw


def metrics(mask, gt, union, same):
    inter = np.logical_and(mask, gt).sum()
    union_i = np.logical_or(mask, gt).sum()
    denom = max(1, mask.sum())
    return float(inter / max(1, union_i)), float(inter / max(1, gt.sum())), float(np.logical_and(mask, same & ~gt).sum() / denom), float(np.logical_and(mask, ~union).sum() / denom)


def bootstrap(rows, field, group=None, n=2000):
    by_img = {}
    for r in rows:
        if group and r["group"] != group:
            continue
        by_img.setdefault(int(r["image_id"]), []).append(float(r[field]))
    ids = np.asarray(sorted(by_img), dtype=np.int64)
    vals = np.asarray([np.mean(by_img[i]) for i in ids], dtype=np.float64)
    rng = np.random.default_rng(20260913)
    draws = rng.integers(0, len(ids), size=(n, len(ids)))
    means = vals[draws].mean(1)
    return dict(mean=float(vals.mean()), ci_low=float(np.quantile(means, .025)), ci_high=float(np.quantile(means, .975)), images=int(len(ids)), targets=int(sum(map(len, by_img.values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "diagnostics/readout_input_scale1200_20260912/cache")
    ap.add_argument("--out", type=Path, default=ROOT / "diagnostics/crop_support_probe_20260913")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_path = ROOT / "local_readout_runtime_20260912/data/annotations/instances_train2017.json"
    coco = COCO(str(ann_path))
    sidecars = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((args.cache / "images").glob("*.json"))]
    sidecars = [x for x in sidecars if x["split"] in ("fit", "transfer")]
    rows = []; start = time.monotonic(); n_images = 0
    for meta in sidecars:
        iid = int(meta["image_id"]); split = meta["split"]
        npz = np.load(args.cache / "images" / f"{iid}.npz")
        aids = np.asarray(npz["annotation_ids"]).astype(int); pidx = np.asarray(npz["prediction_indices"]).astype(int)
        if not len(aids):
            continue
        proto = torch.as_tensor(npz["proto"], device=device, dtype=torch.float32)
        coeff = torch.as_tensor(npz["coeff"][pidx], device=device, dtype=torch.float32)
        boxes = torch.as_tensor(npz["boxes"][pidx], device=device, dtype=torch.float32)
        shape = tuple(int(x) for x in npz["shape"])
        decoded, logits, raw = decode_masks(proto, coeff, boxes, shape, SCALES, device)
        anns = [coco.anns[int(a)] for a in aids]
        all_anns = [a for a in coco.imgToAnns[iid] if not a.get("iscrowd", 0)]
        gt_masks = {int(a["id"]): coco.annToMask(a).astype(bool) for a in all_anns}
        union = np.logical_or.reduce(list(gt_masks.values())) if gt_masks else np.zeros(shape, dtype=bool)
        for j, ann in enumerate(anns):
            aid = int(ann["id"]); gt = gt_masks[aid]; cls = ann["category_id"]
            same_ids = [int(a["id"]) for a in all_anns if a["category_id"] == cls]
            same = np.logical_or.reduce([gt_masks[x] for x in same_ids]) if same_ids else gt
            ici = sum(box_iou(ann["bbox"], a["bbox"]) * a["bbox"][2] * a["bbox"][3] / max(1e-9, ann["bbox"][2] * ann["bbox"][3]) for a in all_anns if int(a["id"]) != aid and a["category_id"] == cls)
            group = "high" if ici > .5 else "low"
            vals = [metrics(decoded[j, s], gt, union, same) for s in range(len(SCALES))]
            box = npz["boxes"][pidx[j]].astype(np.float32)
            eb = np.asarray(expand_box(box, 1.2), dtype=np.float32)
            # Border evidence is measured before cropping, in the same 640 grid.
            b = torch.as_tensor(box, device=device); e = torch.as_tensor(eb, device=device)
            inside = ops.crop_mask(torch.ones((1, 640, 640), device=device, dtype=torch.uint8), b[None])[0].bool().cpu().numpy()
            ring = (ops.crop_mask(torch.ones((1, 640, 640), device=device, dtype=torch.uint8), e[None])[0].bool().cpu().numpy() & ~inside)
            z = logits[j].detach().cpu().numpy(); rb = raw[j].detach().cpu().numpy().astype(bool)
            ring_frac = float(rb[ring].mean()) if ring.any() else 0.0
            inside_frac = float(rb[inside].mean()) if inside.any() else 0.0
            ratio = ring_frac / max(inside_frac, 1e-6)
            rec = dict(image_id=iid, split=split, annotation_id=aid, group=group, ici=float(ici),
                       ring_positive_fraction=ring_frac, inside_positive_fraction=inside_frac, ring_inside_ratio=ratio)
            for k, scale in enumerate(SCALES):
                rec[f"iou_{scale:.1f}"] = vals[k][0]; rec[f"coverage_{scale:.1f}"] = vals[k][1]
                rec[f"neighbor_{scale:.1f}"] = vals[k][2]; rec[f"background_{scale:.1f}"] = vals[k][3]
            best = int(np.argmax([x[0] for x in vals])); rec["oracle_scale"] = float(SCALES[best]); rec["oracle_delta"] = float(vals[best][0] - vals[2][0]); rec["delta_1.2"] = float(vals[4][0] - vals[2][0])
            rows.append(rec)
        n_images += 1
        if n_images % 100 == 0:
            print(json.dumps({"images": n_images, "targets": len(rows), "seconds": round(time.monotonic() - start, 1)}), flush=True)
    write_csv(args.out / "per_target.csv", rows)

    # Select a threshold and a fixed expansion on fit only, then freeze it for transfer.
    fit = [r for r in rows if r["split"] == "fit"]; transfer = [r for r in rows if r["split"] == "transfer"]
    candidates = np.unique(np.quantile([r["ring_positive_fraction"] for r in fit], np.linspace(.05, .95, 91)))
    choices = [0.9, 1.1, 1.2, 1.3, 1.4]
    best = None
    for threshold in candidates:
        for scale in choices:
            field = f"iou_{scale:.1f}"
            pred = [r[field] if r["ring_positive_fraction"] >= threshold else r["iou_1.0"] for r in fit]
            score = float(np.mean(pred))
            if best is None or score > best["fit_mean"]:
                best = {"threshold": float(threshold), "scale": float(scale), "fit_mean": score}
    def summarize(part, name):
        out = {"split": name, "targets": len(part)}
        for group in ("all", "high", "low"):
            q = part if group == "all" else [r for r in part if r["group"] == group]
            for field in ("iou_1.0", "iou_1.2", "oracle_delta", "delta_1.2"):
                out[f"{group}_{field}"] = bootstrap(q, field) if field in q[0] else None
            if q:
                scale = best["scale"]; th = best["threshold"]; pred = [r[f"iou_{scale:.1f}"] if r["ring_positive_fraction"] >= th else r["iou_1.0"] for r in q]
                out[f"{group}_heuristic_iou"] = float(np.mean(pred)); out[f"{group}_heuristic_delta"] = float(np.mean(pred) - np.mean([r["iou_1.0"] for r in q]))
        return out
    summary = {"status": "COMPLETE", "device": str(device), "scales": SCALES.tolist(), "fit": summarize(fit, "fit"), "transfer": summarize(transfer, "transfer"), "heuristic_selected_on": "fit only", "heuristic": best, "source_cache_sha256": sha256(args.cache / "COMPLETE.json"), "script_sha256": sha256(__file__), "limitations": "Frozen matched candidate diagnostic; GT used for evaluation and fit-only threshold selection. No end-to-end AP or trained method."}
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Crop support mechanism probe (S077)", "", "Frozen COCO train2017 readout cache; fit threshold is selected only on fit images and evaluated on disjoint transfer images.", "", f"- Device: `{device}`; targets: fit {len(fit)}, transfer {len(transfer)}", f"- Heuristic: choose crop scale {best['scale']:.1f} when raw 20% border positive fraction >= {best['threshold']:.6f}, otherwise 1.0.", "", "## Transfer means"]
    for g in ("all", "high", "low"):
        t = summary["transfer"]; md.append(f"- {g}: IoU@1.0 {t[g+'_iou_1.0']['mean']:.4f}; fixed 1.2 {t[g+'_iou_1.2']['mean']:.4f}; oracle gain {t[g+'_oracle_delta']['mean']:.4f} [{t[g+'_oracle_delta']['ci_low']:.4f}, {t[g+'_oracle_delta']['ci_high']:.4f}]; heuristic gain {t[g+'_heuristic_delta']:.4f}.")
    md += ["", "Interpretation is restricted to crop sensitivity and an inference-visible border statistic; oracle gain is not a deployable method result."]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
