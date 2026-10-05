"""Evaluate Dilated-0.1 model on full COCO-Dense val (1576 images)."""
import subprocess
import sys
from pathlib import Path

python_exe = sys.executable
root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
eval_script = root / "eval_dual_paths_dilation.py"
weights = root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt"
output = root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/full_dilation_eval.json"

print(f"Starting evaluation of Dilated-0.1 on 1576 images...")
cmd = [
    python_exe,
    "-u",
    str(eval_script),
    "--weights", str(weights),
    "--output", str(output)
]
res = subprocess.run(cmd, check=True)
print(f"Evaluation finished with code {res.returncode}")
