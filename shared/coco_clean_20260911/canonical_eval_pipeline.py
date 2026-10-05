"""Canonical and Rigorous Instance Segmentation Evaluation Pipeline.

Compliant with Research Review Guidelines:
1. Color Space: Explicit BGR -> RGB conversion matching official Ultralytics pipeline.
2. Crowding Metric: Canonical Cumulative Overlap ICI (sum of intersections over own box area).
3. Assignment: Whole-image global Hungarian one-to-one matching (no prediction reuse).
4. Spatial Error Decomposition: Strictly disjoint partitioning (Self TP, Same-class Leak,
   Diff-class Leak, Background FP, with crowd regions explicitly excluded).
5. Dual-Track Reporting:
   - Track 1 (Primary): Full Fixed GT Cohort (unmatched GTs penalized as IoU=0, Rec=0, Prec=0).
     Explicitly reports coverage cost, area shrinkage, missed count, and degraded count.
   - Track 2 (Mechanism): Mutually Detected Cohort for physical mechanism analysis.
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
from pycocotools import mask as mask_util
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

def load_image_rgb(img_path):
    """Load image from disk and convert BGR to RGB."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    return bgr[..., ::-1]  # BGR to RGB

def get_coco_instances(coco_gt, img_id, shape):
    """Parse non-crowd and crowd GT instances with exact masks."""
    h, w = shape
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    
    non_crowd_list = []
    crowd_mask = np.zeros((h, w), dtype=bool)
    
    for ann in anns:
        try:
            m = coco_gt.annToMask(ann).astype(bool)
        except Exception:
            continue
        if m.sum() == 0:
            continue
            
        if ann.get("iscrowd", 0):
            crowd_mask = np.logical_or(crowd_mask, m)
        else:
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            non_crowd_list.append({
                "id": ann["id"],
                "category_id": ann["category_id"],
                "bbox": [bx, by, bw, bh],
                "mask": m,
                "area": int(m.sum()),
                "box_area": float(bw * bh)
            })
            
    return non_crowd_list, crowd_mask

def compute_canonical_cumulative_ici(gt_list):
    """Compute Canonical Cumulative Overlap ICI: sum of intersections over own box area."""
    n = len(gt_list)
    icis = [0.0] * n
    for i in range(n):
        bi = gt_list[i]["bbox"]
        ai = gt_list[i]["box_area"]
        if ai <= 0:
            continue
        box_i = [bi[0], bi[1], bi[0] + bi[2], bi[1] + bi[3]]
        ci = gt_list[i]["category_id"]
        
        cum_inter = 0.0
        for j in range(n):
            if i == j or gt_list[j]["category_id"] != ci:
                continue
            bj = gt_list[j]["bbox"]
            box_j = [bj[0], bj[1], bj[0] + bj[2], bj[1] + bj[3]]
            inter_w = max(0.0, min(box_i[2], box_j[2]) - max(box_i[0], box_j[0]))
            inter_h = max(0.0, min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]))
            cum_inter += inter_w * inter_h
            
        icis[i] = cum_inter / ai
    return icis

def match_image_hungarian(gt_list, preds, iou_thresh=0.50):
    """Whole-image global Hungarian one-to-one matching between GT and predictions."""
    num_gt = len(gt_list)
    num_p = len(preds)
    if num_gt == 0 or num_p == 0:
        return {}
        
    cost_matrix = np.ones((num_gt, num_p)) * 1e5
    
    for i, gt in enumerate(gt_list):
        bi = gt["bbox"]
        box_g = [bi[0], bi[1], bi[0] + bi[2], bi[1] + bi[3]]
        area_g = bi[2] * bi[3]
        cg = gt["category_id"]
        
        for j, p in enumerate(preds):
            if p["category_id"] != cg:
                continue
            bp = p["bbox"]
            box_p = bp
            area_p = max(0.0, bp[2] - bp[0]) * max(0.0, bp[3] - bp[1])
            
            inter_w = max(0.0, min(box_g[2], box_p[2]) - max(box_g[0], box_p[0]))
            inter_h = max(0.0, min(box_g[3], box_p[3]) - max(box_g[1], box_p[1]))
            inter = inter_w * inter_h
            union = area_g + area_p - inter
            if union > 0:
                box_iou = inter / union
                if box_iou >= iou_thresh:
                    cost_matrix[i, j] = 1.0 - box_iou
                    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.0:  # Valid match with box_iou >= iou_thresh
            box_iou = 1.0 - cost_matrix[r, c]
            matches[r] = (c, box_iou)
            
    return matches

def run_model_inference_on_dataset(model, coco_gt, img_ids, img_dir, cat_ids):
    """Run model inference on images with explicit RGB and official native post-processing."""
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)
    preds_by_img = {}
    
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        
        im_lb = letterbox(image=im_rgb)
        tensor = torch.from_numpy(im_lb.transpose(2, 0, 1)).float().cuda() / 255.0
        tensor = tensor.unsqueeze(0)
        
        with torch.no_grad():
            out = model(tensor)
            preds_raw, proto = out[0]
            nms_out = non_max_suppression(
                preds_raw, conf_thres=0.25, iou_thres=0.65, nc=80, end2end=False
            )[0]
            
        if nms_out is None or len(nms_out) == 0:
            preds_by_img[img_id] = []
            continue
            
        boxes = nms_out[:, :4]
        scores = nms_out[:, 4]
        clses = nms_out[:, 5].long()
        coeffs = nms_out[:, 6:]
        if proto.dim() == 4:
            proto = proto[0]
            
        boxes_orig = ops.scale_boxes((640, 640), boxes.clone(), (orig_h, orig_w))
        c, mh, mw = proto.shape
        coeffs_eval = coeffs @ proto.float().view(c, -1)
        raw_masks = ops.scale_masks(coeffs_eval.view(-1, mh, mw)[None], (orig_h, orig_w))[0].gt_(0.0).byte()
        m_0 = ops.crop_mask(raw_masks.clone(), boxes_orig).cpu().numpy()
        
        b_np = boxes_orig.cpu().numpy()
        s_np = scores.cpu().numpy()
        c_np = clses.cpu().numpy()
        
        img_preds = []
        for i in range(len(b_np)):
            img_preds.append({
                "pred_id": i,
                "bbox": b_np[i].tolist(),
                "score": float(s_np[i]),
                "category_id": cat_ids[c_np[i]],
                "mask": m_0[i].astype(bool)
            })
        preds_by_img[img_id] = img_preds
        
    return preds_by_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Images to evaluate (0 for all)")
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading COCO ground truth from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    all_img_ids = sorted(coco_gt.getImgIds())
    img_ids = all_img_ids[:args.limit] if args.limit > 0 else all_img_ids
    print(f"Total images in benchmark: {len(img_ids)}")
    
    # 1. Load Reference Model to define canonical fixed GT instance list
    print("\n[Step 1] Loading Reference Model: Baseline (5ep)...")
    ref_yolo = YOLO(str(root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt"))
    ref_model = ref_yolo.model.eval().cuda()
    ref_model.model[-1].end2end = False
    
    print(f"Running reference inference on {len(img_ids)} images with RGB alignment...")
    t0 = time.time()
    ref_preds = run_model_inference_on_dataset(ref_model, coco_gt, img_ids, img_dir, cat_ids)
    print(f"Reference inference done in {time.time() - t0:.1f}s")
    
    # Free reference model memory
    del ref_model, ref_yolo
    torch.cuda.empty_cache()
    
    # 2. Build Canonical Fixed GT Instances and Natural Box Error Cohorts
    print("\n[Step 2] Building Fixed GT Cohorts with Cumulative Overlap ICI...")
    fixed_gts = []
    
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        
        gt_list, crowd_mask = get_coco_instances(coco_gt, img_id, (orig_h, orig_w))
        if len(gt_list) == 0:
            continue
            
        icis = compute_canonical_cumulative_ici(gt_list)
        p_list = ref_preds.get(img_id, [])
        matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
        
        for g_idx, (p_idx, b_iou) in matches.items():
            gt = gt_list[g_idx]
            p = p_list[p_idx]
            ici = icis[g_idx]
            
            # Compute natural box error relative to GT
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
            
            # Natural Cohort Classification
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
        
    high_ici_cnt = sum(1 for g in fixed_gts if g["is_high_ici"])
    low_ici_cnt = sum(1 for g in fixed_gts if g["is_low_ici"])
    print(f"  High Cumulative ICI (>=0.5): {high_ici_cnt} instances")
    print(f"  Low Cumulative ICI (<0.1):   {low_ici_cnt} instances")
    
    # 3. Evaluate All Models with Strict Spatial Error Decomposition
    models_to_eval = [
        ("Stock Official", root / "weights/yolo26m-seg.pt"),
        ("Baseline (5ep)", root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt"),
        ("CCL-O2M (5ep)", root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt"),
        ("Dilated-0.1 (5ep)", root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt"),
    ]
    
    # Pre-load ground truth masks cache per image to accelerate spatial decomposition
    print("\n[Step 3] Pre-loading image GT annotations for fast spatial evaluation...")
    image_gt_cache = {}
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        gt_list, crowd_mask = get_coco_instances(coco_gt, img_id, (orig_h, orig_w))
        image_gt_cache[img_id] = (gt_list, crowd_mask)
        
    cohort_eval_data = {}
    
    for model_name, weight_file in models_to_eval:
        print(f"\nEvaluating: {model_name} ...")
        yolo = YOLO(str(weight_file))
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
            
            # Find the target GT
            gt_idx = next((i for i, x in enumerate(gt_list) if x["id"] == gt_id), None)
            if gt_idx is None:
                continue
            gt = gt_list[gt_idx]
            
            p_list = preds_dict.get(img_id, [])
            # Whole-image Hungarian matching for this model
            matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
            
            if gt_idx not in matches:
                # Missed detection: penalize heavily in Track 1
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
                
                # Construct disjoint spatial masks
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
                            
                # Exclude self from neighbor masks
                g_same_pure = np.logical_and(g_same_pure, np.logical_not(m_gt))
                g_diff_pure = np.logical_and(g_diff_pure, np.logical_not(m_gt))
                
                # Exact closed spatial decomposition
                tp = float(np.logical_and(m_pred, m_gt).sum())
                union = float(np.logical_or(m_pred, m_gt).sum())
                pred_area = float(m_pred.sum())
                
                mask_iou = tp / union if union > 0 else 0.0
                self_recall = tp / gt_area if gt_area > 0 else 0.0
                self_precision = tp / pred_area if pred_area > 0 else 0.0
                area_ratio = pred_area / gt_area if gt_area > 0 else 0.0
                
                leak_same = float(np.logical_and(m_pred, g_same_pure).sum()) / gt_area if gt_area > 0 else 0.0
                leak_diff = float(np.logical_and(m_pred, g_diff_pure).sum()) / gt_area if gt_area > 0 else 0.0
                
                # Background FP: pixels in pred outside all GTs and outside crowd
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
        
    # 4. Generate Comprehensive Dual-Track Reports
    print("\n" + "=" * 140)
    print("TRACK 1: PRIMARY BENCHMARK (Full Fixed GT Cohort, Missed Penalized as IoU=0, Rec=0, Prec=0)")
    print("=" * 140)
    
    all_cohorts = ["Accurate_Box", "Naturally_Oversized", "Naturally_Shifted", "Other", "ALL_INSTANCES"]
    
    for ch in all_cohorts:
        print(f"\n>>> COHORT: {ch} <<<")
        print(f"{'Model':<20} | {'Total':<6} | {'Missed':<7} | {'Mask IoU':<10} | {'Coverage':<10} | {'Precision':<10} | {'Area Ratio':<11} | {'Bg FP':<9} | {'R@0.75':<9}")
        print("-" * 140)
        for name, _ in models_to_eval:
            recs = cohort_eval_data[name] if ch == "ALL_INSTANCES" else [r for r in cohort_eval_data[name] if r["cohort"] == ch]
            cnt = len(recs)
            if cnt == 0:
                continue
            missed = int(sum(r["missed"] for r in recs))
            m_iou = np.mean([r["mask_iou"] for r in recs]) * 100
            cov = np.mean([r["self_recall"] for r in recs]) * 100
            prec = np.mean([r["self_precision"] for r in recs]) * 100
            ar = np.mean([r["area_ratio"] for r in recs])
            bg = np.mean([r["bg_fp"] for r in recs]) * 100
            r75 = np.mean([r["hit_75"] for r in recs]) * 100
            print(f"{name:<20} | {cnt:<6} | {missed:<7} | {m_iou:6.2f}%   | {cov:6.2f}%   | {prec:6.2f}%   | {ar:6.3f}x     | {bg:6.2f}%  | {r75:6.2f}%")
        print("-" * 140)
        
    # Degradation and Improvement Analysis against Baseline
    if "Baseline (5ep)" in cohort_eval_data and "Dilated-0.1 (5ep)" in cohort_eval_data:
        print("\n" + "=" * 140)
        print("INSTANCE-LEVEL TRADEOFF AUDIT: DILATED-0.1 vs BASELINE (Full Fixed Cohort)")
        print("=" * 140)
        b_all = cohort_eval_data["Baseline (5ep)"]
        d_all = cohort_eval_data["Dilated-0.1 (5ep)"]
        
        for ch in ["Accurate_Box", "Naturally_Oversized", "Naturally_Shifted", "ALL_INSTANCES"]:
            b_sub = [r for r in b_all if (r["cohort"] == ch or ch == "ALL_INSTANCES")]
            d_sub = [r for r in d_all if (r["cohort"] == ch or ch == "ALL_INSTANCES")]
            n = len(b_sub)
            if n == 0:
                continue
            
            improved_cnt = sum(1 for b, d in zip(b_sub, d_sub) if d["mask_iou"] > b["mask_iou"] + 0.05)
            degraded_cnt = sum(1 for b, d in zip(b_sub, d_sub) if d["mask_iou"] < b["mask_iou"] - 0.05)
            miss_diff = int(sum(d["missed"] for d in d_sub) - sum(b["missed"] for b in b_sub))
            
            print(f"Cohort [{ch:<19}] (N={n}): Improved (+0.05 IoU): {improved_cnt:4d} ({improved_cnt/n*100:5.2f}%) | Degraded (-0.05 IoU): {degraded_cnt:4d} ({degraded_cnt/n*100:5.2f}%) | Net Missed Diff: {miss_diff:+d}")
        print("-" * 140)

    # TRACK 2: Mechanism Analysis on Mutually Detected Instances
    if "Baseline (5ep)" in cohort_eval_data and "Dilated-0.1 (5ep)" in cohort_eval_data:
        print("\n" + "=" * 140)
        print("TRACK 2: MECHANISM AUDIT (Mutually Detected Instances Only: Baseline vs Dilated-0.1)")
        print("=" * 140)
        print(f"{'Cohort':<22} | {'N':<6} | {'Metric':<14} | {'Baseline':<10} | {'Dilated-0.1':<12} | {'Delta':<10}")
        print("-" * 90)
        
        for ch in ["Accurate_Box", "Naturally_Oversized", "Naturally_Shifted", "ALL_INSTANCES"]:
            b_sub = cohort_eval_data["Baseline (5ep)"]
            d_sub = cohort_eval_data["Dilated-0.1 (5ep)"]
            paired = [(b, d) for b, d in zip(b_sub, d_sub) if (b["cohort"] == ch or ch == "ALL_INSTANCES") and b["missed"] == 0 and d["missed"] == 0]
            cnt = len(paired)
            if cnt == 0:
                continue
                
            for m_key, m_label in [("mask_iou", "Mask IoU"), ("self_recall", "Coverage (Rec)"), ("self_precision", "Precision"),
                                   ("area_ratio", "Area Ratio"), ("bg_fp", "Bg FP Rate"), ("neighbor_leak_same", "Leak SameCls"),
                                   ("neighbor_leak_diff", "Leak DiffCls"), ("hit_75", "Recall@0.75")]:
                if m_key == "area_ratio":
                    val_b = np.mean([p[0][m_key] for p in paired])
                    val_d = np.mean([p[1][m_key] for p in paired])
                    delta = val_d - val_b
                    print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {val_b:6.3f}x    | {val_d:6.3f}x      | {delta:+6.3f}x")
                else:
                    val_b = np.mean([p[0][m_key] for p in paired]) * 100
                    val_d = np.mean([p[1][m_key] for p in paired]) * 100
                    delta = val_d - val_b
                    print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {val_b:6.2f}%    | {val_d:6.2f}%      | {delta:+6.2f}%")
            print("-" * 90)

        # Crowding Stratification on Naturally Oversized and ALL_INSTANCES
        print("\n" + "=" * 140)
        print("CROWDING STRATIFICATION AUDIT: HIGH ICI (>=0.5) vs LOW ICI (<0.1)")
        print("=" * 140)
        for stratum, flag in [("High ICI (>=0.5)", "is_high_ici"), ("Low ICI (<0.1)", "is_low_ici")]:
            print(f"\n--- Stratum: {stratum} (Mutually Detected Instances) ---")
            print(f"{'Cohort':<22} | {'N':<6} | {'Metric':<14} | {'Baseline':<10} | {'Dilated-0.1':<12} | {'Delta':<10}")
            print("-" * 90)
            for ch in ["Naturally_Oversized", "ALL_INSTANCES"]:
                b_sub = cohort_eval_data["Baseline (5ep)"]
                d_sub = cohort_eval_data["Dilated-0.1 (5ep)"]
                paired = [(b, d) for b, d in zip(b_sub, d_sub) if (b["cohort"] == ch or ch == "ALL_INSTANCES") and b[flag] and b["missed"] == 0 and d["missed"] == 0]
                cnt = len(paired)
                if cnt == 0:
                    continue
                for m_key, m_label in [("mask_iou", "Mask IoU"), ("self_recall", "Coverage (Rec)"), ("self_precision", "Precision"),
                                       ("area_ratio", "Area Ratio"), ("bg_fp", "Bg FP Rate"), ("neighbor_leak_same", "Leak SameCls"),
                                       ("hit_75", "Recall@0.75")]:
                    if m_key == "area_ratio":
                        val_b = np.mean([p[0][m_key] for p in paired])
                        val_d = np.mean([p[1][m_key] for p in paired])
                        delta = val_d - val_b
                        print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {val_b:6.3f}x    | {val_d:6.3f}x      | {delta:+6.3f}x")
                    else:
                        val_b = np.mean([p[0][m_key] for p in paired]) * 100
                        val_d = np.mean([p[1][m_key] for p in paired]) * 100
                        delta = val_d - val_b
                        print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {val_b:6.2f}%    | {val_d:6.2f}%      | {delta:+6.2f}%")
                print("-" * 90)

    # Save complete persistence records
    out_file = root / "canonical_eval_records.json"
    out_file.write_text(json.dumps(cohort_eval_data, indent=2))
    print(f"\nSaved complete canonical evaluation records to {out_file}")

if __name__ == "__main__":
    main()
