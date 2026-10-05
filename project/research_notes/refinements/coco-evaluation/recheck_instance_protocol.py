"""Read-only verification of original GT identity and instance-stratum support."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
full_path = ROOT/"datasets/coco/annotations/instances_val2017.json"
sub_path = ROOT/"gemini/data/coco_dense/coco_dense_val_gt.json"
man_path = ROOT/"refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv"
full = json.loads(full_path.read_text(encoding="utf-8"))
sub = json.loads(sub_path.read_text(encoding="utf-8"))
orig = {a["id"]: a for a in full["annotations"]}
ids = {im["id"] for im in sub["images"]}
expected = {a["id"] for a in full["annotations"] if a["image_id"] in ids}
actual = {a["id"] for a in sub["annotations"]}
with man_path.open(encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))
high = [r for r in manifest if float(r["ici_same"]) > .5]
dense_high = [r for r in high if int(r["image_id"]) in ids]
run = json.loads((ROOT/"gemini/results/coco_instance_eval/20260911_034617/run.json").read_text(encoding="utf-8"))
result = {
    "full_image_count": len(full["images"]),
    "full_noncrowd": sum(not a.get("iscrowd", 0) for a in full["annotations"]),
    "dense_image_count": len(ids),
    "dense_noncrowd": sum(not a.get("iscrowd", 0) for a in sub["annotations"]),
    "dense_annotations_exact_original_subset": expected == actual and all(orig[a["id"]] == a for a in sub["annotations"]),
    "multipolygon_instances_dense": sum(not a.get("iscrowd", 0) and isinstance(a.get("segmentation"), list)
                                       and len(a["segmentation"]) > 1 for a in sub["annotations"]),
    "manifest_instances": len(manifest),
    "full_high_ici": len(high), "dense_high_ici": len(dense_high),
    "high_outside_dense": len(high)-len(dense_high),
    "high_coverage_percent": 100*len(dense_high)/len(high),
    "latest_evaluation_gt_hash_matches": hashlib.sha256(sub_path.read_bytes()).hexdigest() == run["gt_sha256"],
    "latest_evaluation_manifest_hash_matches": hashlib.sha256(man_path.read_bytes()).hexdigest() == run["manifest_sha256"],
    "scope": "Evaluation GT identity only; training label conversion not certified by this check."
}
assert result["dense_annotations_exact_original_subset"]
assert result["latest_evaluation_gt_hash_matches"]
assert result["latest_evaluation_manifest_hash_matches"]
out = ROOT/"refine-logs/coco-evaluation/INSTANCE_PROTOCOL_RECHECK_20260911.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
