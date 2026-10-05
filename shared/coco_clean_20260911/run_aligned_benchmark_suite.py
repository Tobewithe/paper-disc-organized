"""Driver script to sequentially run the aligned benchmark suite:
1. Checks for dms_unconfounded_s0_e5_mr4 completion.
2. Trains control_weight_s0_e5_mr4 (Seed 0, 5 epochs, mask_ratio=4).
3. Trains baseline_s0_e5_mr4 (Seed 0, 5 epochs, mask_ratio=4).
4. Evaluates all arms using eval_models_suite.py on 500 images.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
py_exe = sys.executable

def run_cmd(cmd, desc):
    print(f"\n{'='*70}\n[STARTING] {desc}\nCMD: {' '.join(cmd)}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(root))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[FAILED] {desc} returned non-zero exit code: {res.returncode}")
        sys.exit(res.returncode)
    print(f"[FINISHED] {desc} in {elapsed:.1f}s")

def is_training_complete(exp_dir, target_epochs=5):
    res_file = exp_dir / "results.csv"
    if not res_file.exists():
        return False
    try:
        lines = [line.strip() for line in res_file.read_text().splitlines() if line.strip()]
        if len(lines) < target_epochs + 1:
            return False
        return lines[-1].startswith(f"{target_epochs},")
    except Exception:
        return False

def main():
    dms_dir = root / "runs/aligned_benchmark/dms_unconfounded_s0_e5_mr4"
    print(f"Waiting for DMS unconfounded to complete 5 epochs in {dms_dir} ...")
    while not is_training_complete(dms_dir, 5):
        time.sleep(10)
        
    print(f"DMS unconfounded 5-epoch training complete!")
    time.sleep(5)  # brief grace period for file flush
    
    # 2. Train control_weight
    cw_dir = root / "runs/aligned_benchmark/control_weight_s0_e5_mr4"
    if not is_training_complete(cw_dir, 5):
        run_cmd([py_exe, "-u", "train_aligned_benchmark.py", "--arm", "control_weight", "--seed", "0", "--epochs", "5", "--mask_ratio", "4"],
                "Training Control Weight Arm (Seed 0, 5 epochs, mask_ratio=4)")
    else:
        print(f"Control weight already trained in {cw_dir}")
        
    # 3. Train baseline
    base_dir = root / "runs/aligned_benchmark/baseline_s0_e5_mr4"
    if not is_training_complete(base_dir, 5):
        run_cmd([py_exe, "-u", "train_aligned_benchmark.py", "--arm", "baseline", "--seed", "0", "--epochs", "5", "--mask_ratio", "4"],
                "Training Baseline Arm (Seed 0, 5 epochs, mask_ratio=4)")
    else:
        print(f"Baseline already trained in {base_dir}")
        
    # 4. Run canonical evaluation suite
    run_cmd([py_exe, "-u", "eval_models_suite.py", "--limit", "500", "--out", "aligned_benchmark_eval_records.json"],
            "Evaluating Aligned Benchmark Suite on 500 Images (Canonical RGB + Hungarian + Dual-Track)")
            
    print("\nALL ALIGNED BENCHMARK RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
