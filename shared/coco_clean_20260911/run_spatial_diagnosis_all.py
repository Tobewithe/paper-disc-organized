"""Run GT-matched spatial error diagnosis on all 4 models across 100 dense val images."""
import subprocess
import sys
import time
from pathlib import Path

python_exe = sys.executable
root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
diag_script = root / "probe_gt_matched_spatial.py"
out_dir = root / "spatial_diagnosis_results"
out_dir.mkdir(parents=True, exist_ok=True)

models = [
    {
        "name": "Stock Official",
        "weights": root / "weights/yolo26m-seg.pt",
        "output": out_dir / "official_stock_diag.json",
    },
    {
        "name": "Baseline (5ep)",
        "weights": root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt",
        "output": out_dir / "baseline_s0_e5_diag.json",
    },
    {
        "name": "CCL-O2M (5ep)",
        "weights": root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt",
        "output": out_dir / "ccl_o2m_s0_e5_diag.json",
    },
    {
        "name": "Dilated-0.1 (5ep)",
        "weights": root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt",
        "output": out_dir / "dilated_0.1_s0_e5_diag.json",
    },
]

for idx, m in enumerate(models):
    print(f"\n[{idx+1}/{len(models)}] Diagnosing: {m['name']} ...")
    cmd = [
        python_exe,
        "-u",
        str(diag_script),
        "--weights", str(m["weights"]),
        "--output", str(m["output"]),
        "--limit", "100"
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"Finished in {time.time() - t0:.1f}s")

print("\nAll spatial diagnoses completed!")
