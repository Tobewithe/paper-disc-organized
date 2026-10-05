from __future__ import annotations
import json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Dpan\codexproject\paper-disc-organized")
SRC = ROOT / "experiments/coco_raw_capability_5000_20260916/runs/RUN_180459232c434a8ebc385f51dec06f63"
ANN = ROOT / "assets/datasets/coco/annotations/instances_val2017.json"
OUT = ROOT / "experiments/coco_raw_argmax_conditioned_20261005/runs/RUN_83e9f6f1f0de4d969868fd9a04321786"

def load_coco():
    data = json.loads(ANN.read_text(encoding="utf-8"))
    cats = sorted(c["id"] for c in data["categories"])
    cat_cls = {c:i for i,c in enumerate(cats)}
    images = {int(x["id"]): x for x in data["images"]}
    anns = defaultdict(list)
    for a in data["annotations"]:
        if int(a.get("iscrowd", 0)) == 0:
            anns[int(a["image_id"])].append(a)
    for iid in anns:
        anns[iid].sort(key=lambda x: int(x["id"]))
    return cat_cls, anns, images

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat_cls, anns, images = load_coco()
    files = sorted((SRC / "images").glob("*.npz"))
    counts = Counter(); by_size = defaultdict(Counter); by_class = defaultdict(Counter)
    rows = []; image_count = 0; gt_count = 0; t0 = time.time()
    for pos, path in enumerate(files, 1):
        iid = int(path.stem)
        z = np.load(path, allow_pickle=False)
        gt_ids = z["gt_ids"].astype(np.int64)
        if len(gt_ids) != len(anns[iid]):
            raise RuntimeError(f"{iid}: GT identity count {len(gt_ids)} != annotation count {len(anns[iid])}")
        scores = z["scores"]
        argmax = scores.argmax(axis=1)
        box = z["box_ious"]
        mask = z["mask_ious"]
        for gi, ann in enumerate(anns[iid]):
            if int(gt_ids[gi]) != int(ann["id"]):
                raise RuntimeError(f"{iid}: GT identity mismatch at {gi}")
            cls = cat_cls[int(ann["category_id"])]
            good_sem = (argmax == cls)
            box_good = np.isfinite(box[:, gi]) & (box[:, gi] >= 0.75)
            mask_good = np.isfinite(mask[:, gi]) & (mask[:, gi] >= 0.75)
            box_sem = good_sem & box_good
            mask_sem = good_sem & mask_good
            both = box_sem & mask_sem
            if not good_sem.any(): state = "Semantic fail"
            elif not box_sem.any(): state = "Box fail"
            elif both.any(): state = "Success"
            elif mask_sem.any(): state = "Box/Mask misaligned"
            else: state = "Mask fail"
            counts[state] += 1; gt_count += 1
            size = "small" if ann["area"] < 32**2 else ("medium" if ann["area"] < 96**2 else "large")
            by_size[size][state] += 1; by_class[int(ann["category_id"])][state] += 1
            rows.append({"image_id": iid, "annotation_id": int(ann["id"]), "category_id": int(ann["category_id"]), "area_group": size, "state": state, "argmax_correct_raw": int(good_sem.sum()), "argmax_box75_raw": int(box_sem.sum()), "argmax_mask75_raw": int(mask_sem.sum()), "argmax_joint75_raw": int(both.sum())})
        image_count += 1
    (OUT / "PER_GT.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    total = max(gt_count, 1)
    summary = {"study_id":"STUDY_1c1274ba74944f7ab90fcfd7e535dfbb", "run_id":"RUN_83e9f6f1f0de4d969868fd9a04321786", "images":image_count, "gt":gt_count, "counts":dict(counts), "rates":{k:v/total for k,v in counts.items()}, "by_size":{k:dict(v) for k,v in by_size.items()}, "seconds":time.time()-t0, "source_run":"RUN_180459232c434a8ebc385f51dec06f63"}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "COMPLETE.json").write_text(json.dumps({"completed":True,"kind":"derived_diagnostic","gt":gt_count,"images":image_count}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
