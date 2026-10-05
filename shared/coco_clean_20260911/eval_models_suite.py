"""Comprehensive Evaluation Suite for YOLO26-seg Aligned Benchmark.

Evaluates arbitrary model checkpoints against the canonical fixed GT cohort
using the 5 rigorous review guidelines:
1. RGB color space alignment
2. Cumulative Overlap ICI
3. Whole-image Hungarian 1-to-1 matching
4. Disjoint spatial error decomposition (Self TP, Same-class leak, Diff-class leak, Background FP, crowd-safe)
5. Dual-Track reporting (Full Fixed Cohort with missed penalties vs Mutually Detected mechanism audit).
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from ultralytics import YOLO

from canonical_eval_pipeline import (
    load_image_rgb,
    get_coco_instances,
    compute_canonical_cumulative_ici,
    match_image_hungarian,
    run_model_inference_on_dataset
)

def evaluate_models(models_dict, limit=500, out_json="aligned_benchmark_eval_records.json"):
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading COCO ground truth from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    all_img_ids = sorted(coco_gt.getImgIds())
    img_ids = all_img_ids[:limit] if limit > 0 else all_img_ids
    print(f"Total images in benchmark: {len(img_ids)}")
    
    # 1. Reference model to define fixed GT cohort
    ref_weight = root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt"
    if not ref_weight.exists():
        ref_weight = root / "weights/yolo26m-seg.pt"
    print(f"\n[Step 1] Loading Reference Model: {ref_weight} ...")
    ref_yolo = YOLO(str(ref_weight))
    ref_model = ref_yolo.model.eval().cuda()
    ref_model.model[-1].end2end = False
    
    print(f"Running reference inference on {len(img_ids)} images with RGB alignment...")
    t0 = time.time()
    ref_preds = run_model_inference_on_dataset(ref_model, coco_gt, img_ids, img_dir, cat_ids)
    print(f"Reference inference done in {time.time() - t0:.1f}s")
    del ref_model, ref_yolo
    torch.cuda.empty_cache()
    
    # 2. Build Canonical Fixed GT Instances and Natural Box Error Cohorts
    print("\n[Step 2] Building Fixed GT Cohorts with Cumulative Overlap ICI...")
    fixed_gts = []
    image_gt_cache = {}
    
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        
        gt_list, crowd_mask = get_coco_instances(coco_gt, img_id, (orig_h, orig_w))
        image_gt_cache[img_id] = (gt_list, crowd_mask)
        if len(gt_list) == 0:
            continue
            
        icis = compute_canonical_cumulative_ici(gt_list)
        p_list = ref_preds.get(img_id, [])
        matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
        
        for g_idx, (p_idx, b_iou) in matches.items():
            gt = gt_list[g_idx]
            p = p_list[p_idx]
            ici = icis[g_idx]
            
            bx, by, bw, bh = gt["bbox"]
            gt_cx = bx + bw / 2.0
            gt_cy = by + bh / 2.0
            
            pb = p["bbox"]
            p_bw = pb[2] - pb[0]
            p_bh = pb[3] - pb[1]
            p_cx = pb[0] + p_bw / 2.0
            p_cy = pb[1] + p_bh / 2.0
            
            area_ratio = (p_bw * p_bh) / float(gt["box_area"]) if gt["box_area"] > 0 else 1.0
            diag = np.sqrt(bw**2 + bh**2) if (bw > 0 and bh > 0) else 1.0
            center_shift = np.sqrt((p_cx - gt_cx)**2 + (p_cy - gt_cy)**2) / diag
            
            if b_iou >= 0.85 and (0.90 <= area_ratio <= 1.15):
                cohort = "Accurate_Box"
            elif area_ratio >= 1.25 and b_iou >= 0.50:
                cohort = "Naturally_Oversized"
            elif center_shift >= 0.15 and (0.50 <= b_iou <= 0.75):
                cohort = "Naturally_Shifted"
            else:
                cohort = "Other"
                
            fixed_gts.append({
                "image_id": img_id,
                "gt_id": gt["id"],
                "category_id": gt["category_id"],
                "cohort": cohort,
                "ici": float(ici),
                "is_high_ici": bool(ici >= 0.50),
                "is_low_ici": bool(ici < 0.10),
                "box_iou_ref": float(b_iou),
                "area_ratio_ref": float(area_ratio),
                "center_shift_ref": float(center_shift),
                "gt_mask_area": int(gt["area"])
            })
            
    print(f"Total Fixed Matched GT Instances: {len(fixed_gts)}")
    cohort_counts = defaultdict(int)
    for g in fixed_gts:
        cohort_counts[g["cohort"]] += 1
    for k, v in cohort_counts.items():
        print(f"  Cohort [{k}]: {v} instances")
        
    # 3. Evaluate Models
    cohort_eval_data = {}
    for model_name, weight_file in models_dict.items():
        wf = Path(weight_file)
        if not wf.exists():
            print(f"Skipping {model_name}: {wf} not found")
            continue
        print(f"\nEvaluating: {model_name} ({wf}) ...")
        yolo = YOLO(str(wf))
        m = yolo.model.eval().cuda()
        m.model[-1].end2end = False
        
        preds_dict = run_model_inference_on_dataset(m, coco_gt, img_ids, img_dir, cat_ids)
        del m, yolo
        torch.cuda.empty_cache()
        
        eval_records = []
        for g_meta in fixed_gts:
            img_id = g_meta["image_id"]
            gt_id = g_meta["gt_id"]
            gt_list, crowd_mask = image_gt_cache[img_id]
            
            gt_idx = next((i for i, x in enumerate(gt_list) if x["id"] == gt_id), None)
            if gt_idx is None:
                continue
            gt = gt_list[gt_idx]
            p_list = preds_dict.get(img_id, [])
            matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
            
            if gt_idx not in matches:
                eval_records.append({
                    "gt_id": gt_id,
                    "image_id": img_id,
                    "cohort": g_meta["cohort"],
                    "ici": g_meta["ici"],
                    "is_high_ici": g_meta["is_high_ici"],
                    "is_low_ici": g_meta["is_low_ici"],
                    "missed": 1.0,
                    "mask_iou": 0.0,
                    "self_recall": 0.0,
                    "self_precision": 0.0,
                    "area_ratio": 0.0,
                    "neighbor_leak_same": 0.0,
                    "neighbor_leak_diff": 0.0,
                    "bg_fp": 0.0,
                    "hit_75": 0.0
                })
            else:
                p_idx, b_iou = matches[gt_idx]
                p = p_list[p_idx]
                
                m_pred = p["mask"]
                m_gt = gt["mask"]
                gt_area = float(gt["area"])
                
                g_all_noncrowd = np.zeros_like(m_gt)
                g_same_pure = np.zeros_like(m_gt)
                g_diff_pure = np.zeros_like(m_gt)
                
                for other in gt_list:
                    g_all_noncrowd = np.logical_or(g_all_noncrowd, other["mask"])
                    if other["id"] != gt_id:
                        if other["category_id"] == gt["category_id"]:
                            g_same_pure = np.logical_or(g_same_pure, other["mask"])
                        else:
                            g_diff_pure = np.logical_or(g_diff_pure, other["mask"])
                            
                g_same_pure = np.logical_and(g_same_pure, np.logical_not(m_gt))
                g_diff_pure = np.logical_and(g_diff_pure, np.logical_not(m_gt))
                
                tp = float(np.logical_and(m_pred, m_gt).sum())
                union = float(np.logical_or(m_pred, m_gt).sum())
                pred_area = float(m_pred.sum())
                
                mask_iou = tp / union if union > 0 else 0.0
                self_recall = tp / gt_area if gt_area > 0 else 0.0
                self_precision = tp / pred_area if pred_area > 0 else 0.0
                area_ratio = pred_area / gt_area if gt_area > 0 else 0.0
                
                leak_same = float(np.logical_and(m_pred, g_same_pure).sum()) / gt_area if gt_area > 0 else 0.0
                leak_diff = float(np.logical_and(m_pred, g_diff_pure).sum()) / gt_area if gt_area > 0 else 0.0
                bg_mask = np.logical_and(m_pred, np.logical_not(np.logical_or(g_all_noncrowd, crowd_mask)))
                bg_fp = float(bg_mask.sum()) / gt_area if gt_area > 0 else 0.0
                
                eval_records.append({
                    "gt_id": gt_id,
                    "image_id": img_id,
                    "cohort": g_meta["cohort"],
                    "ici": g_meta["ici"],
                    "is_high_ici": g_meta["is_high_ici"],
                    "is_low_ici": g_meta["is_low_ici"],
                    "missed": 0.0,
                    "mask_iou": float(mask_iou),
                    "self_recall": float(self_recall),
                    "self_precision": float(self_precision),
                    "area_ratio": float(area_ratio),
                    "neighbor_leak_same": float(leak_same),
                    "neighbor_leak_diff": float(leak_diff),
                    "bg_fp": float(bg_fp),
                    "hit_75": float(1.0 if mask_iou >= 0.75 else 0.0)
                })
        cohort_eval_data[model_name] = eval_records
        
    # 4. Generate Reports
    print("\n" + "=" * 140)
    print("TRACK 1: PRIMARY BENCHMARK (Full Fixed GT Cohort, Missed Penalized as IoU=0, Rec=0, Prec=0)")
    print("=" * 140)
    for cohort_name in ["ALL_INSTANCES", "Naturally_Oversized", "Accurate_Box", "Naturally_Shifted"]:
        print(f"\n--- Cohort: {cohort_name} ---")
        print(f"{'Model':<26} | {'N':<6} | {'Missed':<7} | {'Mask IoU':<10} | {'Coverage':<10} | {'Precision':<10} | {'Area Ratio':<12} | {'Bg FP':<8} | {'Recall@0.75':<11}")
        print("-" * 125)
        for name, records in cohort_eval_data.items():
            sub = [r for r in records if (r["cohort"] == cohort_name or cohort_name == "ALL_INSTANCES")]
            cnt = len(sub)
            if cnt == 0:
                continue
            missed = int(sum(r["missed"] for r in sub))
            m_iou = np.mean([r["mask_iou"] for r in sub]) * 100
            cov = np.mean([r["self_recall"] for r in sub]) * 100
            prec = np.mean([r["self_precision"] for r in sub]) * 100
            ar = np.mean([r["area_ratio"] for r in sub])
            bg = np.mean([r["bg_fp"] for r in sub]) * 100
            r75 = np.mean([r["hit_75"] for r in sub]) * 100
            print(f"{name:<26} | {cnt:<6} | {missed:<7} | {m_iou:6.2f}%   | {cov:6.2f}%   | {prec:6.2f}%   | {ar:6.3f}x     | {bg:6.2f}%  | {r75:6.2f}%")
        print("-" * 125)
        
    # Track 2: Mechanism Analysis for each comparison against Baseline
    # Look for the best baseline available: "Baseline (mr4 s0)" or "Baseline (Pilot 5ep)"
    baseline_key = "Baseline (mr4 s0)" if "Baseline (mr4 s0)" in cohort_eval_data else "Baseline (Pilot 5ep)"
    if baseline_key in cohort_eval_data:
        print("\n" + "=" * 140)
        print(f"TRACK 2: MECHANISM AUDIT (Mutually Detected Instances vs {baseline_key})")
        print("=" * 140)
        for compare_key in cohort_eval_data.keys():
            if compare_key == baseline_key or compare_key == "Stock Official":
                continue
            print(f"\n>>> Comparison: {compare_key} vs {baseline_key} <<<")
            print(f"{'Cohort':<22} | {'N':<6} | {'Metric':<14} | {baseline_key:<20} | {compare_key:<20} | {'Delta':<10}")
            print("-" * 105)
            for ch in ["Naturally_Oversized", "Accurate_Box", "Naturally_Shifted", "ALL_INSTANCES"]:
                b_sub = cohort_eval_data[baseline_key]
                c_sub = cohort_eval_data[compare_key]
                paired = [(b, c) for b, c in zip(b_sub, c_sub) if (b["cohort"] == ch or ch == "ALL_INSTANCES") and b["missed"] == 0 and c["missed"] == 0]
                cnt = len(paired)
                if cnt == 0:
                    continue
                for m_key, m_label in [("mask_iou", "Mask IoU"), ("self_recall", "Coverage"), ("self_precision", "Precision"),
                                       ("area_ratio", "Area Ratio"), ("bg_fp", "Bg FP Rate"), ("neighbor_leak_same", "Leak SameCls"),
                                       ("neighbor_leak_diff", "Leak DiffCls"), ("hit_75", "Recall@0.75")]:
                    if m_key == "area_ratio":
                        vb = np.mean([p[0][m_key] for p in paired])
                        vc = np.mean([p[1][m_key] for p in paired])
                        delta = vc - vb
                        print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {vb:6.3f}x              | {vc:6.3f}x                | {delta:+6.3f}x")
                    else:
                        vb = np.mean([p[0][m_key] for p in paired]) * 100
                        vc = np.mean([p[1][m_key] for p in paired]) * 100
                        delta = vc - vb
                        print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {vb:6.2f}%              | {vc:6.2f}%                | {delta:+6.2f}%")
                print("-" * 105)

    # Save to disk
    out_path = root / out_json
    out_path.write_text(json.dumps(cohort_eval_data, indent=2))
    print(f"\nSaved evaluation records to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", type=str, default="aligned_benchmark_eval_records.json")
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    models = {
        "Stock Official": root / "weights/yolo26m-seg.pt",
        "Baseline (Pilot 5ep)": root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt",
        "Dilated-0.1 (Pilot 5ep)": root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt",
        "Baseline (mr4 s0)": root / "runs/aligned_benchmark/baseline_s0_e5_mr4/weights/last.pt",
        "Control Weight (mr4 s0)": root / "runs/aligned_benchmark/control_weight_s0_e5_mr4/weights/last.pt",
        "DMS Unconfounded (mr4 s0)": root / "runs/aligned_benchmark/dms_unconfounded_s0_e5_mr4/weights/last.pt",
        "DMS GuardBand (mr4 s0)": root / "runs/aligned_benchmark/dms_guardband_s0_e5_mr4/weights/last.pt",
    }
    evaluate_models(models, limit=args.limit, out_json=args.out)
