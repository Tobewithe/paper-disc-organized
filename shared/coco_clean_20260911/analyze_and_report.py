"""Analyze and tabulate dual-branch dilation benchmark results dynamically across all runs."""
import json
from pathlib import Path

root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")

aligned_dir = root / "aligned_benchmark_results"
candidates = [
    ("Stock Official", aligned_dir / "official_stock_aligned.json"),
    ("Baseline (5ep)", aligned_dir / "baseline_s0_e5_aligned.json"),
    ("CCL-O2M (5ep)", aligned_dir / "ccl_o2m_s0_e5_aligned.json"),
    ("Dilated-0.1 (5ep)", aligned_dir / "dilated_0.1_s0_e5_aligned.json"),
]

runs = {}
for name, p in candidates:
    if p.exists():
        with open(p) as f:
            runs[name] = json.load(f)

if not runs:
    print("No evaluation runs found.")
    exit(1)

col_width = 17
header = f"{'Branch':<6} | {'Condition':<15} | " + " | ".join([f"{name:<{col_width}}" for name in runs.keys()])
sep = "=" * len(header)
sub_sep = "-" * len(header)

print(sep)
print(header)
print(sep)

for branch in ["o2m", "o2o"]:
    # Box AP
    b_k = f"{branch}_alpha_0.0"
    b_vals = [f"{runs[name][b_k]['box_ap'] * 100:6.2f}%" for name in runs.keys()]
    print(f"{branch.upper():<6} | {'Box AP':<15} | " + " | ".join([f"{v:<{col_width}}" for v in b_vals]))
    
    # Mask AP across alphas
    for alpha in [0.0, 0.1, 0.2]:
        k = f"{branch}_alpha_{alpha}"
        m_vals = [f"{runs[name][k]['mask_ap'] * 100:6.2f}%" for name in runs.keys()]
        print(f"{branch.upper():<6} | {f'Mask AP a={alpha}':<15} | " + " | ".join([f"{v:<{col_width}}" for v in m_vals]))
        
    # Mask drop from alpha=0.0 to alpha=0.2
    drops = []
    for name in runs.keys():
        d = (runs[name][f"{branch}_alpha_0.0"]["mask_ap"] - runs[name][f"{branch}_alpha_0.2"]["mask_ap"]) * 100
        drops.append(f"-{d:5.2f}%")
    print(f"{branch.upper():<6} | {'Drop a=0->0.2':<15} | " + " | ".join([f"{v:<{col_width}}" for v in drops]))
    print(sub_sep)

# AP75 section
print("\n" + sep)
print(f"{'Branch':<6} | {'Metric (Strict)':<15} | " + " | ".join([f"{name:<{col_width}}" for name in runs.keys()]))
print(sep)
for branch in ["o2m", "o2o"]:
    for alpha in [0.0, 0.1, 0.2]:
        k = f"{branch}_alpha_{alpha}"
        vals = [f"{runs[name][k]['mask_ap75'] * 100:6.2f}%" for name in runs.keys()]
        print(f"{branch.upper():<6} | {f'AP75 a={alpha}':<15} | " + " | ".join([f"{v:<{col_width}}" for v in vals]))
    print(sub_sep)
