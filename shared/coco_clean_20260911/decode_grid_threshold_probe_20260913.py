"""S083: frozen mask-decoder grid and threshold probe.

The official candidate, box, coefficient, and prototype tensors are frozen.
Only the logit sampling grid and threshold are changed. The input640/t0 arm
must reproduce the S078 official decoder exactly before other arms are read.
This is a mechanism diagnostic, not an AP result or a deployable method.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
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

CACHE = ROOT / "diagnostics/readout_input_scale1200_20260912/cache"
S078 = ROOT / "diagnostics/crop_support_failure_cross_20260913/per_target.csv"
OUT = ROOT / "diagnostics/decode_grid_threshold_probe_20260913"
ANNOTATION = ROOT / "local_readout_runtime_20260912/data/annotations/instances_train2017.json"

# factor is relative to the model input grid in the frozen cache.
ARMS = (
    ("native160_t0", 0.25, 0.0),
    ("input320_t0", 0.5, 0.0),
    ("input640_t-0.5", 1.0, -0.5),
    ("input640_t-0.25", 1.0, -0.25),
    ("input640_t0", 1.0, 0.0),
    ("input640_t0.25", 1.0, 0.25),
    ("input640_t0.5", 1.0, 0.5),
    ("input1280_t0", 2.0, 0.0),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_bootstrap(rows, field, n=1000):
    if not rows:
        return {"mean": None, "ci_low": None, "ci_high": None, "images": 0, "targets": 0}
    by_image = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(float(row[field]))
    image_ids = np.asarray(sorted(by_image), dtype=np.int64)
    image_values = np.asarray([np.mean(by_image[i]) for i in image_ids], dtype=np.float64)
    rng = np.random.default_rng(20260913)
    draw = rng.integers(0, len(image_values), size=(n, len(image_values)))
    means = image_values[draw].mean(axis=1)
    return {
        "mean": float(image_values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "images": int(len(image_ids)),
        "targets": int(len(rows)),
    }


def decode_grid(logits, boxes, shape, input_shape, factor, threshold):
    """Decode binary masks at a chosen post-logit grid."""
    ih, iw = (int(input_shape[0]), int(input_shape[1]))
    gh = max(1, int(round(ih * factor)))
    gw = max(1, int(round(iw * factor)))
    grid_logits = F.interpolate(logits[:, None], (gh, gw), mode="bilinear", align_corners=False)[:, 0]
    binary = grid_logits.gt(float(threshold)).to(torch.uint8)
    scale = torch.tensor([gw / iw, gh / ih, gw / iw, gh / ih], device=boxes.device)
    grid_boxes = boxes * scale
    cropped = ops.crop_mask(binary, grid_boxes)
    restored = ops.scale_masks(cropped[:, None], shape)[:, 0] > 0.5
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--input", type=Path, default=S078)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # S078 supplies retrospective state labels; it does not enter decoding.
    s078 = read_csv(args.input)
    by_image = defaultdict(list)
    for row in s078:
        by_image[int(row["image_id"])].append(row)
    with io.StringIO() as sink:
        with __import__("contextlib").redirect_stdout(sink):
            coco = COCO(str(ANNOTATION))

    rows = []
    replay_xor = 0
    replay_iou_abs_diff = []
    processed_images = 0
    skipped_targets = 0
    start = time.monotonic()
    for image_id in sorted(by_image):
        npz_path = args.cache / "images" / f"{image_id}.npz"
        if not npz_path.exists():
            skipped_targets += len(by_image[image_id])
            continue
        data = np.load(npz_path, allow_pickle=False)
        ann_ids = np.asarray(data["annotation_ids"], dtype=np.int64)
        pred_ids = np.asarray(data["prediction_indices"], dtype=np.int64)
        ann_to_index = {int(a): i for i, a in enumerate(ann_ids)}
        target_rows = [r for r in by_image[image_id] if int(r["annotation_id"]) in ann_to_index]
        if not target_rows:
            continue
        target_indices = np.asarray([ann_to_index[int(r["annotation_id"])] for r in target_rows], dtype=np.int64)
        pred_indices = pred_ids[target_indices]
        proto = torch.as_tensor(data["proto"], dtype=torch.float32, device=device)
        coeff = torch.as_tensor(data["coeff"][pred_indices], dtype=torch.float32, device=device)
        boxes = torch.as_tensor(data["boxes"][pred_indices], dtype=torch.float32, device=device)
        shape = tuple(int(x) for x in data["shape"])
        input_shape = tuple(int(x) for x in data["input_shape"])
        logits = (coeff @ proto.flatten(1)).reshape(len(target_rows), *proto.shape[1:])

        anns = [coco.anns[int(r["annotation_id"])] for r in target_rows]
        all_anns = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0)]
        gt_masks = {int(a["id"]): coco.annToMask(a).astype(bool) for a in all_anns}
        union = np.logical_or.reduce(list(gt_masks.values())) if gt_masks else np.zeros(shape, dtype=bool)
        same_by_cat = {}
        for a in all_anns:
            same_by_cat.setdefault(int(a["category_id"]), []).append(int(a["id"]))

        decoded = {}
        for name, factor, threshold in ARMS:
            decoded[name] = decode_grid(logits, boxes, shape, input_shape, factor, threshold).cpu().numpy()
        official = decoded["input640_t0"]
        # Compare the official arm against the exact S078 formulas and labels.
        for j, target in enumerate(target_rows):
            aid = int(target["annotation_id"])
            gt = gt_masks[aid]
            ann = coco.anns[aid]
            same_ids = same_by_cat.get(int(ann["category_id"]), [])
            same = np.logical_or.reduce([gt_masks[x] for x in same_ids]) if same_ids else gt
            for name, _, _ in ARMS:
                mask = decoded[name][j]
                inter = np.logical_and(mask, gt).sum()
                union_i = np.logical_or(mask, gt).sum()
                coverage = inter / max(1, gt.sum())
                neighbor = np.logical_and(mask, same & ~gt).sum() / max(1, mask.sum())
                background = np.logical_and(mask, ~union).sum() / max(1, mask.sum())
                rec = dict(
                    image_id=image_id,
                    split=target["split"],
                    annotation_id=aid,
                    group=target["group"],
                    state=target["state"],
                    box_iou=target["box_iou"],
                    gt_box_support=target["gt_box_support"],
                    mask_good=target["mask_good"],
                    arm=name,
                    iou=float(inter / max(1, union_i)),
                    coverage=float(coverage),
                    neighbor_fraction=float(neighbor),
                    background_fraction=float(background),
                )
                rows.append(rec)
                if name == "input640_t0":
                    reference = float(target["iou_1.0"])
                    replay_iou_abs_diff.append(abs(rec["iou"] - reference))
                    # This validates the decoded mask itself against S078 IoU.
                    if abs(rec["iou"] - reference) > 1e-7:
                        raise RuntimeError(f"official IoU replay mismatch image={image_id} ann={aid}: {rec['iou']} vs {reference}")
        processed_images += 1
        if processed_images % 100 == 0:
            print(json.dumps({"images": processed_images, "targets": len(rows), "seconds": round(time.monotonic() - start, 1)}), flush=True)

    # Long form output and compact arm summary.
    write_csv(args.out / "per_target_arm.csv", rows)
    summary = {
        "status": "COMPLETE",
        "experiment": "S083",
        "device": str(device),
        "processed_images": processed_images,
        "target_rows": len(s078),
        "decoded_rows": len(rows),
        "skipped_targets": skipped_targets,
        "arms": [{"name": a, "factor": f, "threshold": t} for a, f, t in ARMS],
        "official_replay_max_iou_abs_diff": float(max(replay_iou_abs_diff or [0.0])),
        "official_replay_mean_iou_abs_diff": float(np.mean(replay_iou_abs_diff or [0.0])),
        "limitations": "Frozen paired-candidate decoder replay; S078 GT-derived state is used only for retrospective grouping and evaluation. No training, no end-to-end AP, and post-interpolation grids do not add feature information.",
    }
    summary_rows = []
    for arm, _, _ in ARMS:
        part = [r for r in rows if r["arm"] == arm]
        for stratum, selector in (
            ("all", lambda r: True),
            ("high", lambda r: r["group"] == "high"),
            ("low", lambda r: r["group"] == "low"),
            ("support_low_mask_bad", lambda r: r["state"] == "box_good_support_low_mask_bad"),
            ("support_sufficient_mask_bad", lambda r: r["state"] == "box_good_support_sufficient_mask_bad"),
            ("mask_good", lambda r: r["state"] == "box_good_mask_good"),
        ):
            q = [r for r in part if selector(r)]
            if not q:
                continue
            for metric in ("iou", "coverage", "neighbor_fraction", "background_fraction"):
                v = image_bootstrap(q, metric)
                summary_rows.append(dict(arm=arm, stratum=stratum, metric=metric, n_targets=v["targets"], n_images=v["images"], mean=v["mean"], ci_low=v["ci_low"], ci_high=v["ci_high"]))
    write_csv(args.out / "SUMMARY.csv", summary_rows)
    summary["summary_rows"] = len(summary_rows)
    summary["cache_sha256"] = sha256(args.cache / "COMPLETE.json")
    summary["input_sha256"] = sha256(args.input)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S083 frozen decoder grid and threshold probe",
        "",
        "The official candidate, box, coefficient, and prototype tensors are frozen. Only post-logit sampling grid and threshold change.",
        "",
        f"- Device: `{device}`; images: {processed_images}; decoded targets: {len(rows) // len(ARMS)}; skipped: {skipped_targets}",
        f"- Official `input640_t0` replay max absolute IoU difference vs S078: `{summary['official_replay_max_iou_abs_diff']:.3e}`",
        "- `native160` and post-interpolation 320/1280 arms are decoder diagnostics; they do not create new feature information.",
        "",
        "See `SUMMARY.csv` for image-bootstrap means and 95% intervals, and `per_target_arm.csv` for target-level values.",
        "",
        "Interpretation guardrail: a recoverable gap under a decoder arm localizes loss to sampling/threshold/cropping, but does not establish a trainable method or explain standard AP by itself.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
