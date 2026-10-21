"""Independent standard-library aggregation; does not import audit implementation."""
import csv
import json
import math
import hashlib
import statistics
import sys
from pathlib import Path
from collections import defaultdict

run = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
summary = read(run / "SUMMARY.json")
with (run / "PER_INSTANCE.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == summary["paired_instances"]
identities = {(r["image_id"], r["annotation_id"], r["output_detection_id"]) for r in rows}
assert len(identities) == len(rows)
max_difference = 0.
parseval_max = 0.
for r in rows:
    assert (float(r["iou"]) >= .75) == bool(int(r["own_mask75"]))
    expected_size = "S" if float(r["gt_area"]) < 1024 else "M" if float(r["gt_area"]) < 9216 else "L"
    assert r["size_group"] == expected_size
    e = sum(float(r[f"error_{b}_energy"]) for b in ("low", "mid", "high")) + float(r["error_dc"])
    parseval_max = max(parseval_max, abs(e - int(r["error_pixels"])))
    assert abs(e - int(r["error_pixels"])) < 1e-7
    for prefix in ["error", "error_hann", "gt", "pred", "logit"]:
        fractions = [float(r[f"{prefix}_{b}_fraction"]) for b in ("low", "mid", "high")]
        if all(math.isfinite(v) for v in fractions):
            assert abs(sum(fractions) - 1) < 1e-10
for group_name, g in summary["groups"].items():
    status, size = group_name.split("_")
    sub = [r for r in rows if (status == "all" or int(r["own_mask75"]) == (status == "success"))
           and (size == "all" or r["size_group"] == size)]
    assert len(sub) == g["pairs"]
    assert len({r["image_id"] for r in sub}) == g["images"]
    for prefix in ["error", "error_hann", "gt", "pred", "logit"]:
        ci = g[prefix]["ci95"]
        assert len(ci) == 3 and all(len(v) == 2 and all(math.isfinite(x) for x in v) and v[0] <= v[1] for v in ci)
        for b, band in enumerate(["low", "mid", "high"]):
            by_image = defaultdict(list)
            for r in sub:
                v = float(r[f"{prefix}_{band}_fraction"])
                if math.isfinite(v):
                    by_image[r["image_id"]].append(v)
            independent = statistics.mean(statistics.mean(v) for v in by_image.values())
            difference = abs(independent - g[prefix]["image_macro_fraction"][b])
            max_difference = max(max_difference, difference)
            assert difference < 1e-11
    for b, band in enumerate(["low", "mid", "high"]):
        ratio = sum(float(r[f"pred_{band}_energy"]) for r in sub) / sum(float(r[f"gt_{band}_energy"]) for r in sub)
        assert abs(ratio - g["pooled_pred_gt_energy_ratio"][b]) < 1e-11
result = dict(status="passed", paired_instances=len(rows), group_count=len(summary["groups"]),
              verified_image_macro_fields=len(summary["groups"])*5*3,
              maximum_mean_difference=max_difference, maximum_absolute_parseval_difference=parseval_max,
              bootstrap="not independently recomputed; audited image aggregation and finite CI structure",
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [run/"SUMMARY.json",run/"PER_INSTANCE.csv"]})
(run / "INDEPENDENT_LOCAL_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
