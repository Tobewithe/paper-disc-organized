"""Independent stdlib-only audit; does not import the server implementation."""
import csv, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
run = root/"runs"/json.loads((root/"LATEST_RUN.json").read_text())["run_id"]
summary = json.loads((run/"SUMMARY.json").read_text())
geometry = {r["image_id"]:r for r in json.loads((root/"inputs/PER_IMAGE.json").read_text())}
with (root/"inputs/PER_INSTANCE.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
with (run/"DERIVED.csv").open(encoding="utf-8-sig", newline="") as f:
    derived = list(csv.DictReader(f))
assert len(rows) == len(derived) == 28377
bands = ["low", "mid", "high"]
max_row_diff = 0.
max_summary_diff = 0.
max_adjusted_diff = 0.
scalar_checks = 0
metrics = []
for r, d in zip(rows, derived):
    for k in ("image_id", "annotation_id", "output_detection_id", "own_mask75"):
        assert r[k] == d[k]
    area = float(r["gt_area"])*math.prod(geometry[int(r["image_id"])]["ratio_pad"][0])
    assert math.isclose(area, float(d["input_area"]), rel_tol=1e-14)
    energy = [float(r["error_"+b+"_energy"]) for b in bands]
    vals = {"error_fraction":[], "hann_fraction":[], "direction_error_fraction":[],
            "amplitude_error_fraction":[], "spectral_cosine":[],
            "absolute_error_energy":energy, "error_over_gt_area":[]}
    for b, e in zip(bands, energy):
        p, g = float(r["pred_"+b+"_energy"]), float(r["gt_"+b+"_energy"])
        c = (p+g-e)/2
        amp = (math.sqrt(p)-math.sqrt(g))**2
        direction = max(0, e-amp)
        rho = c/math.sqrt(p*g) if p*g>0 else None
        for key, value in [("amplitude_error",amp),("direction_error",direction),("spectral_cosine",rho)]:
            if value is not None:
                scalar_checks += 1
                diff = abs(value-float(d[b+"_"+key]))
                assert diff < 1e-7
                max_row_diff = max(max_row_diff,diff)
        vals["error_fraction"].append(e/sum(energy))
        vals["hann_fraction"].append(float(r["error_hann_"+b+"_fraction"]))
        vals["direction_error_fraction"].append(direction/e)
        vals["amplitude_error_fraction"].append(amp/e)
        vals["spectral_cosine"].append(rho)
        vals["error_over_gt_area"].append(e/float(r["gt_area"]))
    metrics.append((r,area,vals))

for name, expected in summary["groups"].items():
    status, size = name.split("_")
    selected = [(r,v) for r,_,v in metrics if
                (r["own_mask75"]=="0") == (status=="failure") and (size=="all" or r["size_group"]==size)]
    assert len(selected) == expected["pairs"]
    for key in selected[0][1]:
        per_image = defaultdict(list)
        for r,v in selected:
            per_image[r["image_id"]].append(v[key])
        means = [statistics.fmean(statistics.fmean(v[j] for v in vv if v[j] is not None)
                 for vv in per_image.values() if any(v[j] is not None for v in vv)) for j in range(3)]
        for x,y in zip(means,expected[key]["mean"]):
            max_summary_diff=max(max_summary_diff,abs(x-y))
            assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-11),(name,key,x,y)

for size in "SML":
    strata=defaultdict(lambda:[[],[]])
    for r,area,v in metrics:
        if r["size_group"] != size:
            continue
        key=(int(r["category_id"]),math.floor(math.log2(area)/.5))
        strata[key][int(r["own_mask75"]=="0")].append(v["error_fraction"]+v["hann_fraction"])
    common=[x for x in strata.values() if x[0] and x[1]]
    denom=sum(min(len(a),len(b)) for a,b in common)
    values=[sum(min(len(a),len(b))*(statistics.fmean(t[j] for t in b)-statistics.fmean(t[j] for t in a))
                for a,b in common)/denom for j in range(6)]
    expected=summary["adjusted"][size]
    assert len(common)==expected["strata"]
    assert sum(len(b) for a,b in common)==expected["failure_pairs"]
    for x,y in zip(values,expected["means"]):
        max_adjusted_diff=max(max_adjusted_diff,abs(x-y))
        assert abs(x-y)<1e-12

receipt={"status":"passed","rows":len(rows),"exact_identity_rows":len(rows),
         "decomposition_scalar_checks":scalar_checks,"group_means_checked":8*7*3,
         "adjusted_means_checked":18,"max_derived_abs":max_row_diff,
         "max_group_mean_abs":max_summary_diff,"max_adjusted_mean_abs":max_adjusted_diff,
         "implementation":"Python standard library, independent of analyze.py",
         "bootstrap":"CI structure reviewed; random multiplier resampling not independently repeated",
         "source_sha256":hashlib.sha256((root/"inputs/PER_INSTANCE.csv").read_bytes()).hexdigest()}
(root/"INDEPENDENT_VERIFICATION.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
