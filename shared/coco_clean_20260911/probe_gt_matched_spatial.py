"""GT-matched spatial error decomposition and ICI-stratified diagnosis.

For each GT instance, performs bipartite matching with predictions, then decomposes
the predicted mask pixels into:
1. Self True Positive (pixels covering the target GT instance)
2. Neighbor Leakage (pixels spilling into adjacent GT instances, separated into same-class vs diff-class)
3. Background False Positive (pixels spilling into pure background)
4. Missed GT instances (False Negatives)

Stratified by instance ICI (Inter-Cluster Interference / IoU with same-class neighbors).
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
    """Extract GT binary masks and boxes for an image."""
    h, w = shape
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    
    gt_list = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        # Decode segmentation
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
    """Compute maximum same-class bounding box IoU for each GT instance."""
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
            # Box IoU
            x1 = max(box_i[0], box_j[0])
            y1 = max(box_i[1], box_j[1])
            x2 = min(box_i[2], box_j[2])
            y2 = min(box_i[3], box_j[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = bi[2] * bi[3]
            area_j = bj[2] * bj[3]
            union = area_i + area_j - inter
            if union > 0:
                iou = inter / union
                if iou > max_iou:
                    max_iou = iou
        icis[i] = max_iou
    return icis

def match_gt_to_preds(gt_list, preds):
    """Match GT instances to predictions using Hungarian algorithm on Mask IoU."""
    if len(gt_list) == 0 or len(preds) == 0:
        return {}, set(range(len(gt_list))), set(range(len(preds)))
        
    num_gt = len(gt_list)
    num_p = len(preds)
    cost_matrix = np.ones((num_gt, num_p)) * 1e5
    
    for i, gt in enumerate(gt_list):
        m_gt = gt["mask"]
        c_gt = gt["category_id"]
        for j, p in enumerate(preds):
            if p["category_id"] != c_gt:
                continue
            m_p = p["mask_0.0"]
            inter = np.logical_and(m_gt, m_p).sum()
            union = np.logical_or(m_gt, m_p).sum()
            if union > 0 and inter > 0:
                cost_matrix[i, j] = 1.0 - (inter / union)
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    matched_gts = set()
    matched_preds = set()
    
    for r, c in zip(row_ind, col_ind):
        # Only accept valid matches with cost < 1.0 (IoU > 0)
        if cost_matrix[r, c] < 1.0:
            matches[r] = c
            matched_gts.add(r)
            matched_preds.add(c)
            
    unmatched_gts = set(range(num_gt)) - matched_gts
    unmatched_preds = set(range(num_p)) - matched_preds
    return matches, unmatched_gts, unmatched_preds

def diagnose_image(gt_list, preds, alphas=(0.0, 0.2)):
    """Diagnose spatial pixel allocations for matched pairs across alphas."""
    matches, unmatched_gts, unmatched_preds = match_gt_to_preds(gt_list, preds)
    icis = compute_ici_for_gts(gt_list)
    
    # Pre-aggregate neighbor masks
    all_gt_union = np.zeros_like(gt_list[0]["mask"]) if len(gt_list) > 0 else None
    for gt in gt_list:
        all_gt_union = np.logical_or(all_gt_union, gt["mask"])
        
    records = []
    
    for gt_idx, p_idx in matches.items():
        gt = gt_list[gt_idx]
        p = preds[p_idx]
        ici = icis[gt_idx]
        m_gt = gt["mask"]
        c_gt = gt["category_id"]
        
        # Neighbor GT masks excluding self
        neighbor_same = np.zeros_like(m_gt)
        neighbor_diff = np.zeros_like(m_gt)
        for other_idx, other_gt in enumerate(gt_list):
            if other_idx == gt_idx:
                continue
            if other_gt["category_id"] == c_gt:
                neighbor_same = np.logical_or(neighbor_same, other_gt["mask"])
            else:
                neighbor_diff = np.logical_or(neighbor_diff, other_gt["mask"])
                
        for a in alphas:
            m_p = p[f"mask_{a}"]
            p_total = int(m_p.sum())
            if p_total == 0:
                continue
                
            p_self = int(np.logical_and(m_p, m_gt).sum())
            p_neighbor_same = int(np.logical_and(m_p, neighbor_same).sum())
            p_neighbor_diff = int(np.logical_and(m_p, neighbor_diff).sum())
            p_bg = int(np.logical_and(m_p, ~all_gt_union).sum())
            
            records.append({
                "gt_id": gt["id"],
                "category_id": c_gt,
                "ici": float(ici),
                "alpha": a,
                "gt_area": gt["area"],
                "pred_area": p_total,
                "self_tp": p_self,
                "neighbor_leak_same": p_neighbor_same,
                "neighbor_leak_diff": p_neighbor_diff,
                "bg_fp": p_bg,
                "self_recall": p_self / max(gt["area"], 1),
                "self_precision": p_self / max(p_total, 1),
                "neighbor_leak_ratio": (p_neighbor_same + p_neighbor_diff) / max(p_total, 1),
                "bg_error_ratio": p_bg / max(p_total, 1)
            })
            
    # Missed GTs
    missed_records = []
    for gt_idx in unmatched_gts:
        gt = gt_list[gt_idx]
        missed_records.append({
            "gt_id": gt["id"],
            "category_id": gt["category_id"],
            "ici": float(icis[gt_idx]),
            "area": gt["area"]
        })
        
    return records, missed_records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100, help="Number of images to diagnose")
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading GT from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    img_ids = sorted(coco_gt.getImgIds())[:args.limit]
    
    print(f"Loading model from {args.weights}...")
    yolo = YOLO(str(args.weights))
    model = yolo.model.eval().cuda()
    model.model[-1].end2end = False  # O2M branch with NMS
    
    letterbox = LetterBox(new_shape=(640, 640), scaleup=False)
    
    all_matched = []
    all_missed = []
    
    t0 = time.time()
    for idx, img_id in enumerate(img_ids):
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        p = img_dir / fn
        if not p.exists():
            continue
        im0 = cv2.imread(str(p))
        orig_h, orig_w = im0.shape[:2]
        
        gt_list = get_instance_masks_and_boxes(coco_gt, img_id, (orig_h, orig_w))
        if len(gt_list) == 0:
            continue
            
        im_lb = letterbox(image=im0)
        tensor = torch.from_numpy(im_lb.transpose(2, 0, 1)).float().cuda() / 255.0
        tensor = tensor.unsqueeze(0)
        
        with torch.no_grad():
            out = model(tensor)
            preds_raw, proto = out[0]
            nms_out = non_max_suppression(
                preds_raw,
                conf_thres=0.25,
                iou_thres=0.65,
                max_det=300,
                nc=80,
                end2end=False
            )[0]
            
        if nms_out is None or len(nms_out) == 0:
            for gt in gt_list:
                all_missed.append({"gt_id": gt["id"], "category_id": gt["category_id"], "ici": 0.0, "area": gt["area"]})
            continue
            
        boxes = nms_out[:, :4]
        scores = nms_out[:, 4]
        clses = nms_out[:, 5].long()
        coeffs = nms_out[:, 6:]
        if proto.dim() == 4:
            proto = proto[0]
            
        boxes_orig = ops.scale_boxes((640, 640), boxes.clone(), (orig_h, orig_w))
        
        # Native process_mask for alpha=0.0 and alpha=0.2
        c, mh, mw = proto.shape
        coeffs_eval = coeffs @ proto.float().view(c, -1)
        raw_masks = ops.scale_masks(coeffs_eval.view(-1, mh, mw)[None], (orig_h, orig_w))[0].gt_(0.0).byte()
        
        # Alpha=0.0
        m_0 = ops.crop_mask(raw_masks.clone(), boxes_orig).cpu().numpy()
        
        # Alpha=0.2
        bw = boxes_orig[:, 2] - boxes_orig[:, 0]
        bh = boxes_orig[:, 3] - boxes_orig[:, 1]
        x1 = torch.clamp(boxes_orig[:, 0] - 0.2 * bw, min=0.0)
        y1 = torch.clamp(boxes_orig[:, 1] - 0.2 * bh, min=0.0)
        x2 = torch.clamp(boxes_orig[:, 2] + 0.2 * bw, max=float(orig_w))
        y2 = torch.clamp(boxes_orig[:, 3] + 0.2 * bh, max=float(orig_h))
        boxes_02 = torch.stack([x1, y1, x2, y2], dim=-1)
        m_02 = ops.crop_mask(raw_masks.clone(), boxes_02).cpu().numpy()
        
        parsed_preds = []
        for i in range(len(boxes_orig)):
            parsed_preds.append({
                "category_id": cat_ids[clses[i]],
                "score": float(scores[i]),
                "mask_0.0": m_0[i].astype(bool),
                "mask_0.2": m_02[i].astype(bool),
            })
            
        matched_rec, missed_rec = diagnose_image(gt_list, parsed_preds, alphas=(0.0, 0.2))
        all_matched.extend(matched_rec)
        all_missed.extend(missed_rec)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(img_ids):
            print(f"[{idx+1}/{len(img_ids)}] Diagnosed: {len(all_matched)} matched evaluations", flush=True)
            
    summary = {
        "model": str(args.weights),
        "total_matched_evaluations": len(all_matched),
        "total_missed_gts": len(all_missed),
        "diagnostics": all_matched,
        "missed": all_missed
    }
    
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"Saved diagnosis to {out_p} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
