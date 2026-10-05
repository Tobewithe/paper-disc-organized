"""Run official-aligned evaluation across all 4 models sequentially on full 1576 COCO-Dense val images.
Saves metrics and prediction RLEs for downstream GT-matched spatial diagnosis.
"""
import subprocess
import sys
import time
from pathlib import Path

python_exe = sys.executable
root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
eval_script = root / "eval_official_aligned_dilation.py"
out_dir = root / "aligned_benchmark_results"
out_dir.mkdir(parents=True, exist_ok=True)

models = [
    {
        "name": "Stock Official",
        "weights": root / "weights/yolo26m-seg.pt",
        "output": out_dir / "official_stock_aligned.json",
    },
    {
        "name": "Baseline (5ep)",
        "weights": root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt",
        "output": out_dir / "baseline_s0_e5_aligned.json",
    },
    {
        "name": "CCL-O2M (5ep)",
        "weights": root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt",
        "output": out_dir / "ccl_o2m_s0_e5_aligned.json",
    },
    {
        "name": "Dilated-0.1 (5ep)",
        "weights": root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt",
        "output": out_dir / "dilated_0.1_s0_e5_aligned.json",
    },
]

print(f"=== Starting Aligned Full-Val Benchmark ({len(models)} models) ===")
t_start = time.time()

for idx, m in enumerate(models):
    print(f"\n[{idx+1}/{len(models)}] Running: {m['name']}")
    print(f"  Weights: {m['weights']}")
    print(f"  Output:  {m['output']}")
    
    cmd = [
        python_exe,
        "-u",
        str(eval_script),
        "--weights", str(m["weights"]),
        "--output", str(m["output"]),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"--> Finished {m['name']} in {dt/60:.1f} min (Exit Code {res.returncode})")

total_dt = time.time() - t_start
print(f"\n=======================================================")
print(f"ALL ALIGNED BENCHMARKS COMPLETED in {total_dt/60:.1f} min!")
print(f"Results saved in {out_dir}")
print(f"=======================================================")
