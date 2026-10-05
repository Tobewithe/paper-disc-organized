"""Cross-tabulate S077 crop sensitivity with paired COCO failure states.

This is a retrospective diagnostic on the frozen train2017 readout cache. The
candidate is the one-to-one bbox-matched candidate already stored by S077; no
new model forward, training, or candidate selection is performed here.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


ROOT = Path(__file__).resolve().parent
S077 = ROOT / "diagnostics" / "crop_support_probe_20260913"
CACHE = ROOT / "diagnostics" / "readout_input_scale1200_20260912" / "cache"
ANN = ROOT / "local_readout_runtime_20260912" / "data" / "annotations" / "instances_train2017.json"
OUT = ROOT / "diagnostics" / "crop_support_failure_cross_20260913"
MASK_THR = 0.75
BOX_THR = 0.50
SUPPORT_THR = 0.95


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, aa + bb - inter)


def gt_box(ann):
    x, y, w, h = [float(v) for v in ann["bbox"]]
    return (x, y, x + w, y + h)


def box_mask(box, shape):
    h, w = [int(v) for v in shape]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(w, int(math.floor(x1))))
    y1 = max(0, min(h, int(math.floor(y1))))
    x2 = max(0, min(w, int(math.ceil(x2))))
    y2 = max(0, min(h, int(math.ceil(y2))))
    out = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = True
    return out


def bootstrap_mean(rows, field, n=2000):
    if not rows:
        return None
    by_image = defaultdict(list)
    for r in rows:
        by_image[int(r["image_id"])].append(float(r[field]))
    image_values = np.asarray([np.mean(v) for v in by_image.values()], dtype=np.float64)
    rng = np.random.default_rng(20260913)
    draws = rng.integers(0, len(image_values), size=(n, len(image_values)))
    means = image_values[draws].mean(1)
    return {
        "mean": float(image_values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "images": int(len(image_values)),
        "targets": int(len(rows)),
    }


def state(box_iou, mask_iou, support):
    box_good = box_iou >= BOX_THR
    mask_good = mask_iou >= MASK_THR
    if mask_good:
        return "box_good_mask_good" if box_good else "box_bad_mask_good"
    if not box_good:
        return "box_bad_mask_bad"
    return "box_good_support_sufficient_mask_bad" if support >= SUPPORT_THR else "box_good_support_low_mask_bad"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = read_csv(S077 / "per_target.csv")
    coco = COCO(str(ANN))
    meta = {}
    for p in (CACHE / "images").glob("*.npz"):
        z = np.load(p, mmap_mode="r")
        aids = [int(x) for x in z["annotation_ids"]]
        pidx = [int(x) for x in z["prediction_indices"]]
        meta[int(p.stem)] = (p, dict(zip(aids, pidx)))

    rows = []
    skipped = []
    for r in base:
        iid = int(r["image_id"])
        aid = int(r["annotation_id"])
        if iid not in meta or aid not in meta[iid][1]:
            skipped.append({"image_id": iid, "annotation_id": aid, "reason": "cache_target_missing"})
            continue
        npz_path, mapping = meta[iid]
        z = np.load(npz_path)
        pidx = mapping[aid]
        det = np.asarray(z["detections"][pidx], dtype=np.float64)
        ann = coco.anns[aid]
        gt = coco.annToMask(ann).astype(bool)
        candidate = det[:4]
        b_iou = xyxy_iou(candidate, gt_box(ann))
        support = float((gt & box_mask(candidate, gt.shape)).sum() / max(1, gt.sum()))
        m_iou = float(r["iou_1.0"])
        rec = dict(r)
        rec.update(
            candidate_score=float(det[4]),
            candidate_class_index=int(det[5]),
            box_iou=float(b_iou),
            gt_box_support=float(support),
            box_good=int(b_iou >= BOX_THR),
            mask_good=int(m_iou >= MASK_THR),
            state=state(b_iou, m_iou, support),
        )
        rows.append(rec)

    write_csv(OUT / "per_target.csv", rows)
    write_csv(OUT / "skipped.csv", skipped)

    fields = [
        "box_iou",
        "gt_box_support",
        "iou_1.0",
        "coverage_1.0",
        "oracle_delta",
        "delta_1.2",
        "ring_positive_fraction",
        "neighbor_1.0",
        "background_1.0",
    ]
    summary = {
        "status": "COMPLETE",
        "source": "S077 frozen train2017 readout cache and per-target crop replay",
        "thresholds": {"box_iou_good": BOX_THR, "mask_iou_good": MASK_THR, "gt_box_support_sufficient": SUPPORT_THR},
        "targets": len(rows),
        "images": len({int(r["image_id"]) for r in rows}),
        "skipped": len(skipped),
        "states": {},
        "by_group": {},
        "limitations": [
            "State is for the stored paired bbox candidate, not an official end-to-end COCO assignment.",
            "GT is used only for retrospective state labels and evaluation; no GT-dependent inference rule is tested.",
            "gt_box_support is an independent geometric support measure from the predicted candidate box.",
        ],
    }
    state_names = [
        "box_bad_mask_bad",
        "box_bad_mask_good",
        "box_good_support_low_mask_bad",
        "box_good_support_sufficient_mask_bad",
        "box_good_mask_good",
    ]
    for name in state_names:
        q = [r for r in rows if r["state"] == name]
        summary["states"][name] = {"count": len(q), "fraction": len(q) / max(1, len(rows))}
        for f in fields:
            summary["states"][name][f] = bootstrap_mean(q, f)
    for group in ["all", "high", "low"]:
        gq = rows if group == "all" else [r for r in rows if r["group"] == group]
        summary["by_group"][group] = {"count": len(gq), "states": {}}
        for name in state_names:
            q = [r for r in gq if r["state"] == name]
            summary["by_group"][group]["states"][name] = {
                "count": len(q),
                "fraction_within_group": len(q) / max(1, len(gq)),
                "oracle_delta": bootstrap_mean(q, "oracle_delta"),
                "delta_1.2": bootstrap_mean(q, "delta_1.2"),
                "iou_1.0": bootstrap_mean(q, "iou_1.0"),
                "box_iou": bootstrap_mean(q, "box_iou"),
                "gt_box_support": bootstrap_mean(q, "gt_box_support"),
            }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Crop sensitivity crossed with paired COCO failure states (S078)",
        "",
        "This retrospective analysis uses the S077 frozen train2017 readout cache. The stored bbox-matched candidate is retained; no new forward, training, or inference rule is introduced.",
        "",
        f"- Targets: {len(rows)} across {len({int(r['image_id']) for r in rows})} images; skipped: {len(skipped)}.",
        f"- State thresholds: candidate Box IoU >= {BOX_THR:.2f}; mask IoU@native crop >= {MASK_THR:.2f}; GT pixels inside candidate box >= {SUPPORT_THR:.2f}.",
        "- `gt_box_support` is computed from the candidate box and GT mask, independently of predicted mask positives.",
        "",
        "## State table",
        "",
        "| State | N | Fraction | Mean oracle crop gain | Mean fixed 1.2 delta | Mean Box IoU | Mean GT-box support |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in state_names:
        s = summary["states"][name]
        od = s["oracle_delta"]["mean"] if s["oracle_delta"] else float("nan")
        d12 = s["delta_1.2"]["mean"] if s["delta_1.2"] else float("nan")
        bi = s["box_iou"]["mean"] if s["box_iou"] else float("nan")
        gs = s["gt_box_support"]["mean"] if s["gt_box_support"] else float("nan")
        md.append(f"| {name} | {s['count']} | {s['fraction']:.3f} | {od:.4f} | {d12:.4f} | {bi:.3f} | {gs:.3f} |")
    md += ["", "## Interpretation guardrails", "", "- Oracle crop gain is a diagnostic upper bound for changing only the crop scale; it is not an end-to-end method result.", "- A concentration of oracle gain in `box_good_support_sufficient_mask_bad` would implicate crop sensitivity after adequate geometric support. A concentration in `box_good_support_low_mask_bad` would instead implicate box coverage.", "- A large `box_good_mask_good` fraction among high-ICI targets means high-I CI does not automatically imply a mask failure for this paired candidate.", "- These labels are retrospective and use GT; they cannot by themselves establish a deployable no-GT selector."]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
