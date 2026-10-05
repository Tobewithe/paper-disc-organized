"""Detailed analysis and report generator for the aligned benchmark.
"""
import json
from pathlib import Path
import numpy as np

def main():
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    records_file = root / "aligned_benchmark_eval_records.json"
    data = json.loads(records_file.read_text())
    
    models = list(data.keys())
    print(f"Loaded {len(models)} models from {records_file}")
    
    # -------------------------------------------------------------
    # TRACK 1: FULL FIXED COHORT (N=3875)
    # -------------------------------------------------------------
    print("\n" + "="*140)
    print("TRACK 1: PRIMARY BENCHMARK (Full Fixed GT Cohort, N=3875, Missed = 0 IoU / 0 Rec / 0 Prec)")
    print("="*140)
    
    cohorts = ["ALL_INSTANCES", "Naturally_Oversized", "Accurate_Box", "Naturally_Shifted"]
    
    for ch in cohorts:
        print(f"\n--- Cohort: {ch} ---")
        header = f"{'Model':<26} | {'N':<6} | {'Missed':<7} | {'Mask IoU':<10} | {'Coverage':<10} | {'Precision':<10} | {'Area Ratio':<12} | {'Bg FP':<8} | {'Recall@0.75':<11}"
        print(header)
        print("-" * len(header))
        for m in models:
            sub = [r for r in data[m] if (r["cohort"] == ch or ch == "ALL_INSTANCES")]
            n = len(sub)
            missed = int(sum(r["missed"] for r in sub))
            m_iou = np.mean([r["mask_iou"] for r in sub]) * 100
            cov = np.mean([r["self_recall"] for r in sub]) * 100
            prec = np.mean([r["self_precision"] for r in sub]) * 100
            ar = np.mean([r["area_ratio"] for r in sub])
            bg = np.mean([r["bg_fp"] for r in sub]) * 100
            r75 = np.mean([r["hit_75"] for r in sub]) * 100
            print(f"{m:<26} | {n:<6} | {missed:<7} | {m_iou:6.2f}%   | {cov:6.2f}%   | {prec:6.2f}%   | {ar:6.3f}x     | {bg:6.2f}%  | {r75:6.2f}%")
        print("-" * len(header))

    # -------------------------------------------------------------
    # INSTANCE-LEVEL TRADEOFF AUDIT (Track 1)
    # -------------------------------------------------------------
    print("\n" + "="*140)
    print("INSTANCE-LEVEL TRADEOFF AUDIT (Full Fixed Cohort: DMS vs Baseline mr4 s0, and Control vs Baseline mr4 s0)")
    print("="*140)
    base_m = "Baseline (mr4 s0)"
    for comp_m in ["DMS GuardBand (mr4 s0)", "DMS Unconfounded (mr4 s0)", "Control Weight (mr4 s0)", "Dilated-0.1 (Pilot 5ep)"]:
        print(f"\n>>> Comparison: {comp_m} vs {base_m} <<<")
        b_all = data[base_m]
        c_all = data[comp_m]
        for ch in cohorts:
            b_sub = [r for r in b_all if (r["cohort"] == ch or ch == "ALL_INSTANCES")]
            c_sub = [r for r in c_all if (r["cohort"] == ch or ch == "ALL_INSTANCES")]
            n = len(b_sub)
            improved = sum(1 for b, c in zip(b_sub, c_sub) if c["mask_iou"] > b["mask_iou"] + 0.05)
            degraded = sum(1 for b, c in zip(b_sub, c_sub) if c["mask_iou"] < b["mask_iou"] - 0.05)
            b_miss = int(sum(b["missed"] for b in b_sub))
            c_miss = int(sum(c["missed"] for c in c_sub))
            net_miss = c_miss - b_miss
            print(f"Cohort [{ch:<19}] (N={n:<4}): Improved (+0.05): {improved:4d} ({improved/n*100:5.2f}%) | Degraded (-0.05): {degraded:4d} ({degraded/n*100:5.2f}%) | Base Miss: {b_miss:3d} | Comp Miss: {c_miss:3d} (Net: {net_miss:+d})")

    # -------------------------------------------------------------
    # TRACK 2: MECHANISM AUDIT (Mutually Detected Instances)
    # -------------------------------------------------------------
    print("\n" + "="*140)
    print("TRACK 2: MECHANISM AUDIT (Mutually Detected Instances vs Baseline mr4 s0)")
    print("="*140)
    for eval_arm, label_arm in [("DMS GuardBand (mr4 s0)", "DMS GuardBand"), ("DMS Unconfounded (mr4 s0)", "DMS Unconf (Hard)"), ("Control Weight (mr4 s0)", "Control (2.0x)")]:
        print(f"\n>>> Arm: {label_arm} vs {base_m} <<<")
        for ch in ["Naturally_Oversized", "Accurate_Box", "Naturally_Shifted", "ALL_INSTANCES"]:
            b_sub = data[base_m]
            e_sub = data[eval_arm]
            paired = [(b, e) for b, e in zip(b_sub, e_sub) if (b["cohort"] == ch or ch == "ALL_INSTANCES") and b["missed"] == 0 and e["missed"] == 0]
            cnt = len(paired)
            print(f"\n--- Cohort: {ch} (Mutually Detected N={cnt}) ---")
            header = f"{'Metric':<16} | {'Baseline (mr4)':<14} | {label_arm:<16} | {'Delta':<12}"
            print(header)
            print("-" * len(header))
            metrics = [
                ("mask_iou", "Mask IoU", "%"),
                ("self_recall", "Coverage (Rec)", "%"),
                ("self_precision", "Precision", "%"),
                ("area_ratio", "Area Ratio", "x"),
                ("bg_fp", "Bg FP Rate", "%"),
                ("neighbor_leak_same", "Leak SameCls", "%"),
                ("neighbor_leak_diff", "Leak DiffCls", "%"),
                ("hit_75", "Recall@0.75", "%")
            ]
            for k, label, unit in metrics:
                if unit == "x":
                    vb = np.mean([p[0][k] for p in paired])
                    ve = np.mean([p[1][k] for p in paired])
                    delta = ve - vb
                    print(f"{label:<16} | {vb:6.3f}x         | {ve:6.3f}x           | {delta:+6.3f}x")
                else:
                    vb = np.mean([p[0][k] for p in paired]) * 100
                    ve = np.mean([p[1][k] for p in paired]) * 100
                    delta = ve - vb
                    print(f"{label:<16} | {vb:6.2f}%         | {ve:6.2f}%           | {delta:+6.2f}%")
            print("-" * len(header))
        print("-" * len(header))

    # -------------------------------------------------------------
    # CROWDING DENSITY AUDIT (Cumulative ICI Stratification)
    # -------------------------------------------------------------
    print("\n" + "="*140)
    print("CROWDING DENSITY AUDIT: Cumulative Overlap ICI Stratification (Mutually Detected)")
    print("="*140)
    for stratum, flag in [("High ICI (>=0.5)", "is_high_ici"), ("Low ICI (<0.1)", "is_low_ici")]:
        print(f"\n--- Stratum: {stratum} ---")
        for ch in ["Naturally_Oversized", "ALL_INSTANCES"]:
            b_sub = data[base_m]
            d_sub = data[dms_m]
            c_sub = data[cw_m]
            paired = [(b, d, c) for b, d, c in zip(b_sub, d_sub, c_sub) 
                      if (b["cohort"] == ch or ch == "ALL_INSTANCES") and b[flag] and b["missed"] == 0 and d["missed"] == 0 and c["missed"] == 0]
            cnt = len(paired)
            if cnt == 0:
                continue
            print(f"  Cohort [{ch}] (N={cnt}):")
            print(f"    Baseline IoU: {np.mean([p[0]['mask_iou'] for p in paired])*100:6.2f}% | DMS IoU: {np.mean([p[1]['mask_iou'] for p in paired])*100:6.2f}% (Delta: {np.mean([p[1]['mask_iou'] for p in paired])*100 - np.mean([p[0]['mask_iou'] for p in paired])*100:+6.2f}%)")
            print(f"    Baseline BgFP: {np.mean([p[0]['bg_fp'] for p in paired])*100:6.2f}% | DMS BgFP: {np.mean([p[1]['bg_fp'] for p in paired])*100:6.2f}% (Delta: {np.mean([p[1]['bg_fp'] for p in paired])*100 - np.mean([p[0]['bg_fp'] for p in paired])*100:+6.2f}%)")
            print(f"    Baseline Rec@75: {np.mean([p[0]['hit_75'] for p in paired])*100:6.2f}% | DMS Rec@75: {np.mean([p[1]['hit_75'] for p in paired])*100:6.2f}% (Delta: {np.mean([p[1]['hit_75'] for p in paired])*100 - np.mean([p[0]['hit_75'] for p in paired])*100:+6.2f}%)")

if __name__ == "__main__":
    main()
