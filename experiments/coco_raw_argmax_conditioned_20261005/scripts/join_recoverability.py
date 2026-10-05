from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Dpan\codexproject\paper-disc-organized")
OLD = ROOT / "experiments/coco_raw_capability_5000_20260916/runs/RUN_a7c840bb695d45b4b66bc346715ca1aa/gt_joined.jsonl"
NEW = ROOT / "experiments/coco_raw_argmax_conditioned_20261005/runs/RUN_83e9f6f1f0de4d969868fd9a04321786/PER_GT.jsonl"
OUT = ROOT / "experiments/coco_raw_argmax_conditioned_20261005/runs/RUN_cf9b8ea21c2f41a888c35ac9f922baff"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = {(int(x["image_id"]), int(x["annotation_id"])): x for x in map(json.loads, OLD.open(encoding="utf-8"))}
    rows = [json.loads(x) for x in NEW.read_text(encoding="utf-8").splitlines()]
    def stats(sub):
        n = len(sub)
        return {"gt":n, "argmax_box75":sum(x["argmax_box75_raw"] > 0 for x in sub), "argmax_mask75":sum(x["argmax_mask75_raw"] > 0 for x in sub), "argmax_joint75":sum(x["argmax_joint75_raw"] > 0 for x in sub)}
    state_order = ["normal_success", "raw_geometry_unavailable", "all_good_masks_gt_class_score_le001", "good_mask_class_pair_not_in_head_top300", "coco_category_max100_budget", "coco_matching_competition"]
    by_state = {s:stats([x for x in rows if old[(x["image_id"],x["annotation_id"])].get("normal_mask75_state")==s]) for s in state_order}
    all_pool = stats(rows)
    failed = [x for x in rows if old[(x["image_id"],x["annotation_id"])].get("normal_mask75_state") != "normal_success"]
    bbox_failed = [x for x in rows if old[(x["image_id"],x["annotation_id"])].get("bbox_coco75") is False]
    seg_failed = [x for x in rows if old[(x["image_id"],x["annotation_id"])].get("segm_coco75") is False]
    summary = {"study_id":"STUDY_1c1274ba74944f7ab90fcfd7e535dfbb", "run_id":"RUN_cf9b8ea21c2f41a888c35ac9f922baff", "source_runs":["RUN_180459232c434a8ebc385f51dec06f63","RUN_83e9f6f1f0de4d969868fd9a04321786"], "all_argmax_pool":all_pool, "normal_mask75_failed":stats(failed), "normal_bbox75_failed":stats(bbox_failed), "normal_segm75_failed":stats(seg_failed), "normal_mask75_by_state":by_state}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT / "COMPLETE.json").write_text(json.dumps({"completed":True,"kind":"derived_recoverability_diagnostic"},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT / "run.json").write_text(json.dumps({"schema_version":2,"run_id":"RUN_cf9b8ea21c2f41a888c35ac9f922baff","study_id":"STUDY_1c1274ba74944f7ab90fcfd7e535dfbb","status":"completed","source_kind":"derived_diagnostic","source_runs":summary["source_runs"]},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
