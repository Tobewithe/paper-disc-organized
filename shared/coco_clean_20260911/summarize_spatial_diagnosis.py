"""Aggregate and summarize GT-matched spatial diagnosis across models and ICI strata."""
import json
import numpy as np
from pathlib import Path

root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
diag_dir = root / "spatial_diagnosis_results"

models = [
    ("Stock Official", diag_dir / "official_stock_diag.json"),
    ("Baseline (5ep)", diag_dir / "baseline_s0_e5_diag.json"),
    ("CCL-O2M (5ep)", diag_dir / "ccl_o2m_s0_e5_diag.json"),
    ("Dilated-0.1 (5ep)", diag_dir / "dilated_0.1_s0_e5_diag.json"),
]

def load_and_group(path):
    with open(path) as f:
        data = json.load(f)
    records = data["diagnostics"]
    
    # Structure: [alpha][strata] -> list of records
    grouped = {
        0.0: {"Overall": [], "High_ICI (>=0.5)": [], "Med_ICI (0.1-0.5)": [], "Isolated (<0.1)": []},
        0.2: {"Overall": [], "High_ICI (>=0.5)": [], "Med_ICI (0.1-0.5)": [], "Isolated (<0.1)": []},
    }
    
    for r in records:
        a = r["alpha"]
        ici = r["ici"]
        grouped[a]["Overall"].append(r)
        if ici >= 0.5:
            grouped[a]["High_ICI (>=0.5)"].append(r)
        elif ici >= 0.1:
            grouped[a]["Med_ICI (0.1-0.5)"].append(r)
        else:
            grouped[a]["Isolated (<0.1)"].append(r)
            
    return grouped, data["total_missed_gts"]

results = {}
for name, p in models:
    if p.exists():
        results[name] = load_and_group(p)

if not results:
    print("No diagnosis files found.")
    exit(1)

print("=" * 115)
print("GT-MATCHED SPATIAL DIAGNOSIS SUMMARY (Self Coverage, Neighbor Leakage, Background FP)")
print("=" * 115)

for alpha in [0.0, 0.2]:
    print(f"\n>>> DILATION ALPHA = {alpha} ({'Tight Box' if alpha==0.0 else '20% Expanded Box (+40% W/H)'}) <<<")
    print(f"{'Model':<20} | {'Stratum':<18} | {'Self Precision':<15} | {'Self Recall':<15} | {'Neighbor Leak':<15} | {'Bg FP Rate':<15}")
    print("-" * 115)
    for name in results.keys():
        grouped, missed = results[name]
        for stratum in ["Overall", "High_ICI (>=0.5)", "Isolated (<0.1)"]:
            recs = grouped[alpha][stratum]
            if len(recs) == 0:
                continue
            prec = np.mean([r["self_precision"] for r in recs]) * 100
            rec = np.mean([r["self_recall"] for r in recs]) * 100
            n_leak = np.mean([r["neighbor_leak_ratio"] for r in recs]) * 100
            bg_fp = np.mean([r["bg_error_ratio"] for r in recs]) * 100
            print(f"{name:<20} | {stratum:<18} | {prec:6.2f}%         | {rec:6.2f}%         | {n_leak:6.2f}%         | {bg_fp:6.2f}%")
        print("-" * 115)
