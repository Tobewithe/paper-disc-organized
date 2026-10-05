"""Evaluate models on fixed GT instances categorized by NATURAL bounding box errors (at alpha=0.0).

Experimental design compliant with user specification:
1. Completely unperturbed standard inference (alpha=0.0).
2. Uses Baseline model to define fixed GT instance cohorts:
   - Accurate Box Group: IoU_box >= 0.85 and 0.9 <= Area_ratio <= 1.15
   - Naturally Oversized Box Group: Area_ratio >= 1.25 and IoU_box >= 0.50
   - Naturally Shifted Box Group: Relative center shift >= 0.15 and 0.50 <= IoU_box <= 0.75
3. Cross-stratified by crowding (High ICI >= 0.5 vs Low ICI < 0.1).
4. Evaluates all models on the EXACT SAME fixed GT list:
   - Mask IoU
   - Self Precision
   - Self Recall
   - Neighbor Leakage Rate (same-class & diff-class)
   - Background False Positive Rate
   - Recall@0.75 (fraction of instances with Mask IoU >= 0.75)
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

def get_instance_masks_and_boxes(coco_gt, img_id, shape):
    h, w = shape
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_list = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rles = mask_util.frPyObjects(seg, h, w)
            rle = mask_util.merge(rles)
        elif isinstance(seg, dict):
            rle = seg
        else:
            continue
        m = mask_util.decode(rle)
        if m.sum() == 0:
            continue
        gt_list.append({
            "id": ann["id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],  # [x, y, w, h]
            "mask": m.astype(bool),
            "area": int(m.sum())
        })
    return gt_list

def compute_ici_for_gts(gt_list):
    n = len(gt_list)
    icis = [0.0] * n
    for i in range(n):
        bi = gt_list[i]["bbox"]
        box_i = [bi[0], bi[1], bi[0] + bi[2], bi[1] + bi[3]]
        ci = gt_list[i]["category_id"]
        max_iou = 0.0
        for j in range(n):
            if i == j or gt_list[j]["category_id"] != ci:
                continue
            bj = gt_list[j]["bbox"]
            box_j = [bj[0], bj[1], bj[0] + bj[2], bj[1] + bj[3]]
            x1, y1 = max(box_i[0], box_j[0]), max(box_i[1], box_j[1])
            x2, y2 = min(box_i[2], box_j[2]), min(box_i[3], box_j[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = bi[2] * bi[3] + bj[2] * bj[3] - inter
            if union > 0:
                max_iou = max(max_iou, inter / union)
        icis[i] = max_iou
    return icis

def match_gt_to_preds_by_box(gt_list, preds):
    """Match GT instances to predictions using Hungarian algorithm on Box IoU."""
    if len(gt_list) == 0 or len(preds) == 0:
        return {}
    num_gt = len(gt_list)
    num_p = len(preds)
    cost = np.ones((num_gt, num_p)) * 1e5
    
    for i, gt in enumerate(gt_list):
        bi = gt["bbox"]
        box_g = [bi[0], bi[1], bi[0] + bi[2], bi[1] + bi[3]]
        cg = gt["category_id"]
        for j, p in enumerate(preds):
            if p["category_id"] != cg:
                continue
            bp = p["bbox"]
            x1, y1 = max(box_g[0], bp[0]), max(box_g[1], bp[1])
            x2, y2 = min(box_g[2], bp[2]), min(box_g[3], bp[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_g = bi[2] * bi[3]
            area_p = (bp[2] - bp[0]) * (bp[3] - bp[1])
            union = area_g + area_p - inter
            if union > 0:
                iou = inter / union
                cost[i, j] = 1.0 - iou
                
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 0.8:  # Accept if Box IoU > 0.2
            matches[r] = (c, 1.0 - cost[r, c])
    return matches

def run_model_inference(model, coco_gt, img_ids, img_dir, cat_ids):
    """Run standard inference (alpha=0.0) and return predictions per image."""
    letterbox = LetterBox(new_shape=(640, 640), scaleup=False)
    preds_by_img = {}
    
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        p = img_dir / fn
        if not p.exists():
            continue
        im0 = cv2.imread(str(p))
        orig_h, orig_w = im0.shape[:2]
        im_lb = letterbox(image=im0)
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
                "bbox": b_np[i].tolist(),
                "score": float(s_np[i]),
                "category_id": cat_ids[c_np[i]],
                "mask": m_0[i].astype(bool)
            })
        preds_by_img[img_id] = img_preds
        
    return preds_by_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Number of images to evaluate")
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading GT from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    all_img_ids = sorted(coco_gt.getImgIds())
    img_ids = all_img_ids[:args.limit] if args.limit > 0 else all_img_ids
    
    # 1. Load Reference Model (Baseline) to define fixed GT cohorts
    print(f"Loading Reference Model: Baseline (5ep)...")
    ref_yolo = YOLO(str(root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt"))
    ref_model = ref_yolo.model.eval().cuda()
    ref_model.model[-1].end2end = False
    
    print(f"Generating reference predictions on {len(img_ids)} images...")
    ref_preds = run_model_inference(ref_model, coco_gt, img_ids, img_dir, cat_ids)
    
    # 2. Build Fixed GT Instance Cohorts based on Reference Model's natural box error
    print("Classifying GT instances into natural box error cohorts...")
    fixed_gts = []
    
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im0 = cv2.imread(str(img_dir / fn))
        orig_h, orig_w = im0.shape[:2]
        gt_list = get_instance_masks_and_boxes(coco_gt, img_id, (orig_h, orig_w))
        if len(gt_list) == 0:
            continue
        icis = compute_ici_for_gts(gt_list)
        p_list = ref_preds.get(img_id, [])
        matches = match_gt_to_preds_by_box(gt_list, p_list)
        
        for g_idx, (p_idx, b_iou) in matches.items():
            gt = gt_list[g_idx]
            p = p_list[p_idx]
            bg = gt["bbox"]
            bp = p["bbox"]
            
            area_g = bg[2] * bg[3]
            area_p = (bp[2] - bp[0]) * (bp[3] - bp[1])
            area_ratio = area_p / max(area_g, 1.0)
            
            center_g = np.array([bg[0] + bg[2]/2, bg[1] + bg[3]/2])
            center_p = np.array([(bp[0] + bp[2])/2, (bp[1] + bp[3])/2])
            center_shift = np.linalg.norm(center_p - center_g) / max(np.sqrt(area_g), 1.0)
            
            ici = icis[g_idx]
            
            # Determine Cohort
            cohort = "Other"
            if b_iou >= 0.85 and 0.90 <= area_ratio <= 1.15:
                cohort = "Accurate_Box"
            elif area_ratio >= 1.25 and b_iou >= 0.50:
                cohort = "Naturally_Oversized"
            elif center_shift >= 0.15 and 0.50 <= b_iou <= 0.75:
                cohort = "Naturally_Shifted"
                
            fixed_gts.append({
                "img_id": img_id,
                "gt_id": gt["id"],
                "category_id": gt["category_id"],
                "cohort": cohort,
                "ici": float(ici),
                "is_high_ici": (ici >= 0.5),
                "is_low_ici": (ici < 0.1),
                "box_iou_ref": float(b_iou),
                "area_ratio_ref": float(area_ratio),
                "center_shift_ref": float(center_shift),
                "gt_area": gt["area"]
            })
            
    print(f"Total fixed matched GT instances: {len(fixed_gts)}")
    cohort_counts = defaultdict(int)
    for g in fixed_gts:
        cohort_counts[g["cohort"]] += 1
    for k, v in cohort_counts.items():
        print(f"  Cohort [{k}]: {v} instances")
        
    # Free reference model memory
    del ref_model, ref_yolo
    torch.cuda.empty_cache()
    
    # 3. Evaluate Models on this EXACT SAME fixed GT list
    models_to_eval = [
        ("Stock Official", root / "weights/yolo26m-seg.pt"),
        ("Baseline (5ep)", root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt"),
        ("CCL-O2M (5ep)", root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt"),
        ("Dilated-0.1 (5ep)", root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt"),
    ]
    
    cohort_results = {}
    
    for name, weight_path in models_to_eval:
        print(f"\nEvaluating: {name} ...")
        yolo = YOLO(str(weight_path))
        m = yolo.model.eval().cuda()
        m.model[-1].end2end = False
        
        preds_dict = run_model_inference(m, coco_gt, img_ids, img_dir, cat_ids)
        
        # Evaluate on each fixed GT
        eval_records = []
        for g_meta in fixed_gts:
            img_id = g_meta["img_id"]
            fn = coco_gt.loadImgs(img_id)[0]["file_name"]
            im0 = cv2.imread(str(img_dir / fn))
            orig_h, orig_w = im0.shape[:2]
            gt_list = get_instance_masks_and_boxes(coco_gt, img_id, (orig_h, orig_w))
            
            # Find target GT
            gt = next((x for x in gt_list if x["id"] == g_meta["gt_id"]), None)
            if gt is None:
                continue
                
            all_gt_union = np.zeros_like(gt["mask"])
            neighbor_mask = np.zeros_like(gt["mask"])
            for other in gt_list:
                all_gt_union = np.logical_or(all_gt_union, other["mask"])
                if other["id"] != gt["id"]:
                    neighbor_mask = np.logical_or(neighbor_mask, other["mask"])
                    
            p_list = preds_dict.get(img_id, [])
            # Match prediction for this model
            matches = match_gt_to_preds_by_box([gt], p_list)
            
            if len(matches) == 0:
                # Missed
                eval_records.append({
                    "cohort": g_meta["cohort"],
                    "is_high_ici": g_meta["is_high_ici"],
                    "is_low_ici": g_meta["is_low_ici"],
                    "mask_iou": 0.0,
                    "self_recall": 0.0,
                    "self_precision": 0.0,
                    "neighbor_leak": 0.0,
                    "bg_fp": 0.0,
                    "hit_75": 0.0,
                    "missed": 1.0
                })
            else:
                p_idx, _ = matches[0]
                p = p_list[p_idx]
                m_pred = p["mask"]
                m_gt = gt["mask"]
                
                inter = np.logical_and(m_pred, m_gt).sum()
                union = np.logical_or(m_pred, m_gt).sum()
                mask_iou = inter / max(union, 1.0)
                
                p_total = m_pred.sum()
                if p_total == 0:
                    eval_records.append({
                        "cohort": g_meta["cohort"],
                        "is_high_ici": g_meta["is_high_ici"],
                        "is_low_ici": g_meta["is_low_ici"],
                        "mask_iou": 0.0,
                        "self_recall": 0.0,
                        "self_precision": 0.0,
                        "neighbor_leak": 0.0,
                        "bg_fp": 0.0,
                        "hit_75": 0.0,
                        "missed": 0.0
                    })
                else:
                    self_recall = inter / max(gt["area"], 1.0)
                    self_precision = inter / max(p_total, 1.0)
                    neighbor_leak = np.logical_and(m_pred, neighbor_mask).sum() / max(p_total, 1.0)
                    bg_fp = np.logical_and(m_pred, ~all_gt_union).sum() / max(p_total, 1.0)
                    hit_75 = 1.0 if mask_iou >= 0.75 else 0.0
                    
                    eval_records.append({
                        "cohort": g_meta["cohort"],
                        "is_high_ici": g_meta["is_high_ici"],
                        "is_low_ici": g_meta["is_low_ici"],
                        "mask_iou": float(mask_iou),
                        "self_recall": float(self_recall),
                        "self_precision": float(self_precision),
                        "neighbor_leak": float(neighbor_leak),
                        "bg_fp": float(bg_fp),
                        "hit_75": float(hit_75),
                        "missed": 0.0
                    })
                    
        cohort_results[name] = eval_records
        del m, yolo
        torch.cuda.empty_cache()
        
    # 4. Aggregate and Report Results Across Natural Cohorts
    print("\n" + "=" * 125)
    print("NATURAL BOX ERROR COHORT BENCHMARK (Fixed Reference-Matched GT Instances at alpha=0.0)")
    print("=" * 125)
    
    cohorts_to_show = ["Accurate_Box", "Naturally_Oversized", "Naturally_Shifted"]
    
    for ch in cohorts_to_show:
        print(f"\n>>> COHORT: {ch} <<<")
        print(f"{'Model':<20} | {'Count':<6} | {'Mask IoU':<12} | {'Precision':<12} | {'Recall':<12} | {'Neighbor Leak':<15} | {'Bg FP':<12} | {'Recall@0.75':<12}")
        print("-" * 125)
        for name, _ in models_to_eval:
            recs = [r for r in cohort_results[name] if r["cohort"] == ch]
            cnt = len(recs)
            if cnt == 0:
                continue
            m_iou = np.mean([r["mask_iou"] for r in recs]) * 100
            prec = np.mean([r["self_precision"] for r in recs]) * 100
            rec = np.mean([r["self_recall"] for r in recs]) * 100
            n_leak = np.mean([r["neighbor_leak"] for r in recs]) * 100
            bg_fp = np.mean([r["bg_fp"] for r in recs]) * 100
            hit_75 = np.mean([r["hit_75"] for r in recs]) * 100
            print(f"{name:<20} | {cnt:<6} | {m_iou:6.2f}%     | {prec:6.2f}%     | {rec:6.2f}%     | {n_leak:6.2f}%         | {bg_fp:6.2f}%   | {hit_75:6.2f}%")
        print("-" * 125)
        
    # High vs Low ICI comparison on Naturally Oversized Boxes
    print("\n" + "=" * 125)
    print("NATURALLY OVERSIZED BOXES: HIGH-CROWD vs LOW-CROWD STRATIFICATION")
    print("=" * 125)
    for stratum, flag in [("High_ICI (>=0.5)", "is_high_ici"), ("Low_ICI (<0.1)", "is_low_ici")]:
        print(f"\n--- Stratum: {stratum} ---")
        print(f"{'Model':<20} | {'Count':<6} | {'Mask IoU':<12} | {'Precision':<12} | {'Recall':<12} | {'Neighbor Leak':<15} | {'Bg FP':<12} | {'Recall@0.75':<12}")
        print("-" * 125)
        for name, _ in models_to_eval:
            recs = [r for r in cohort_results[name] if r["cohort"] == "Naturally_Oversized" and r[flag]]
            cnt = len(recs)
            if cnt == 0:
                continue
            m_iou = np.mean([r["mask_iou"] for r in recs]) * 100
            prec = np.mean([r["self_precision"] for r in recs]) * 100
            rec = np.mean([r["self_recall"] for r in recs]) * 100
            n_leak = np.mean([r["neighbor_leak"] for r in recs]) * 100
            bg_fp = np.mean([r["bg_fp"] for r in recs]) * 100
            hit_75 = np.mean([r["hit_75"] for r in recs]) * 100
            print(f"{name:<20} | {cnt:<6} | {m_iou:6.2f}%     | {prec:6.2f}%     | {rec:6.2f}%     | {n_leak:6.2f}%         | {bg_fp:6.2f}%   | {hit_75:6.2f}%")
        print("-" * 125)
        
    # Paired detected comparison between Baseline and Dilated-0.1
    if "Baseline (5ep)" in cohort_results and "Dilated-0.1 (5ep)" in cohort_results:
        print("\n" + "=" * 125)
        print("PAIRED COMPARISON: BASELINE vs DILATED-0.1 ON MUTUALLY DETECTED INSTANCES (alpha=0.0)")
        print("=" * 125)
        b_recs = cohort_results["Baseline (5ep)"]
        d_recs = cohort_results["Dilated-0.1 (5ep)"]
        
        all_cohorts = ["Accurate_Box", "Naturally_Oversized", "Naturally_Shifted", "Other"]
        print(f"{'Cohort':<22} | {'N':<6} | {'Metric':<14} | {'Baseline':<10} | {'Dilated-0.1':<12} | {'Delta':<10}")
        print("-" * 85)
        for ch in all_cohorts:
            paired = [(b, d) for b, d in zip(b_recs, d_recs) if b["cohort"] == ch and b["missed"] == 0 and d["missed"] == 0]
            if len(paired) == 0:
                continue
            cnt = len(paired)
            for m_key, m_label in [("mask_iou", "Mask IoU"), ("self_precision", "Precision"), ("self_recall", "Recall"), 
                                   ("bg_fp", "Bg FP Rate"), ("neighbor_leak", "Neigh Leak"), ("hit_75", "Recall@0.75")]:
                val_b = np.mean([p[0][m_key] for p in paired]) * 100
                val_d = np.mean([p[1][m_key] for p in paired]) * 100
                delta = val_d - val_b
                print(f"{ch:<22} | {cnt:<6} | {m_label:<14} | {val_b:6.2f}%    | {val_d:6.2f}%      | {delta:+6.2f}%")
            print("-" * 85)
            
    out_file = root / "natural_box_error_cohort_results.json"
    out_file.write_text(json.dumps(cohort_results, indent=2))
    print(f"\nSaved natural error cohort results to {out_file}")

if __name__ == "__main__":
    main()
