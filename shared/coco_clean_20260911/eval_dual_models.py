"""Sequentially evaluate baseline and CCL-O2M models on COCO-Dense full val (1576 images)."""
import subprocess
import sys
from pathlib import Path

python_exe = sys.executable
root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
eval_script = root / "eval_dual_paths_dilation.py"

models = [
    {
        "name": "Baseline (Dual-branch, 5 ep)",
        "weights": root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt",
        "output": root / "runs/pilot_dual_branch/baseline_s0_e5/full_dilation_eval.json",
    },
    {
        "name": "CCL-O2M (Dual-branch, 5 ep)",
        "weights": root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt",
        "output": root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/full_dilation_eval.json",
    },
]

for m in models:
    print(f"\n=======================================================", flush=True)
    print(f"STARTING EVALUATION: {m['name']}", flush=True)
    print(f"Weights: {m['weights']}", flush=True)
    print(f"Output:  {m['output']}", flush=True)
    print(f"=======================================================", flush=True)
    cmd = [
        python_exe,
        "-u",
        str(eval_script),
        "--weights", str(m["weights"]),
        "--output", str(m["output"])
    ]
    res = subprocess.run(cmd, check=True)
    print(f"Finished {m['name']} with code {res.returncode}", flush=True)

print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY!", flush=True)
