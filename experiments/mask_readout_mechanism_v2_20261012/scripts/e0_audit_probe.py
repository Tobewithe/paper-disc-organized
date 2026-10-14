"""Read-only E0 audit probe on the laptop (2026-10-12): script hashes and cache identity."""
import hashlib
from pathlib import Path

scripts = Path(r"D:\coco_wire\experiments\coefficient_shared_affine_alignment_audit_20261011\scripts")
for p in sorted(scripts.glob("*.py")):
    print(p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16])

index = Path(r"D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache\INDEX.json")
print("INDEX_sha256", hashlib.sha256(index.read_bytes()).hexdigest())

runs = Path(r"D:\coco_wire\experiments\coefficient_shared_affine_alignment_audit_20261011\runs")
for name in ("RUN_CROSS_OLD160_20261011_053905_129", "RUN_CROSS_OFFICIAL640_FULL_20261011"):
    model = runs / name / "SHARED_old160.pt"
    print(name, "model_exists", model.is_file())
print("done")
