"""Check whether the reproduced empty-category bug is triggered by saved Q1/Q4 draws.

Uses only GT/detection presence, never computes or changes AP/CI.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


gt = read("gemini/data/coco_dense/coco_dense_val_gt.json")
quartiles = read("gemini/data/coco_dense/coco_dense_quartiles.json")["quartiles"]
cats = sorted(c["id"] for c in gt["categories"])
cat_index = {cid: i for i, cid in enumerate(cats)}
ids = {q: sorted(quartiles[q]["image_ids"]) for q in ("Q4", "Q1")}
image_index = {q: {iid: i for i, iid in enumerate(values)} for q, values in ids.items()}
gt_presence = {q: np.zeros((len(values), len(cats)), dtype=np.int32) for q, values in ids.items()}
for a in gt["annotations"]:
    if a.get("iscrowd", 0) or a["area"] < 0 or a["area"] > 1e10:
        continue
    for q in ids:
        if a["image_id"] in image_index[q]:
            gt_presence[q][image_index[q][a["image_id"]], cat_index[a["category_id"]]] = 1

# Identical draw sequence to the old script: Q4 then Q1, 1000 iterations, seed 42.
rng = np.random.RandomState(42)
draw_counts = {q: np.zeros((1000, len(values)), dtype=np.int32) for q, values in ids.items()}
for b in range(1000):
    for q in ("Q4", "Q1"):
        n = len(ids[q])
        draw_counts[q][b] = np.bincount(rng.choice(n, size=n, replace=True), minlength=n)

report = {"scope": "Checks absence-of-detections branch only, not complete AP/bootstrap correctness.",
          "B": 1000, "seed": 42, "groups": {}}
for model in ("baseline", "ccl01"):
    preds = read(f"gemini/results/predictions_{model}_stage2.json")
    for q in ids:
        dt_presence = np.zeros_like(gt_presence[q])
        for p in preds:
            if p["image_id"] in image_index[q]:
                dt_presence[image_index[q][p["image_id"]], cat_index[p["category_id"]]] = 1
        missing = (draw_counts[q] @ gt_presence[q] > 0) & (draw_counts[q] @ dt_presence == 0)
        full_missing = (gt_presence[q].sum(axis=0) > 0) & (dt_presence.sum(axis=0) == 0)
        report["groups"][f"{q}_{model}"] = {
            "original_sample_omitted_categories": [cid for cid, present in zip(cats, full_missing) if present],
            "bootstrap_draws_with_omitted_positive_category": int(missing.any(axis=1).sum()),
            "total_omitted_category_draws": int(missing.sum()),
            "category_omission_frequency": {str(cid): int(missing[:, k].sum()) for k, cid in enumerate(cats) if missing[:, k].any()},
        }
    del preds
path = OUT / ("BOOTSTRAP_SUPPORT_AUDIT_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"path": str(path), **report}, indent=2))
