"""Stress-test bounding box uncertainty and robustness for Baseline vs DMS.

Tests:
1. Confidence threshold sweep (conf in [0.05, 0.10, 0.15, 0.20, 0.25]):
   Tests performance under increasing box uncertainty and looser candidate proposals.
2. Controlled Box Dilation perturbation sweep (alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
   Measures degradation curves of Mask IoU, Background False Positive Rate, and Precision.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from canonical_eval_pipeline import (
    load_image_rgb,
    get_coco_instances,
    compute_canonical_cumulative_ici,
    match_image_hungarian
)

def run_inference_with_conf(model, coco_gt, img_ids, img_dir, cat_ids, conf_thres=0.25):
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
                preds_raw, conf_thres=conf_thres, iou_thres=0.65, nc=80, end2end=False
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
                "mask": m_0[i].astype(bool),
                "raw_mask": raw_masks[i].cpu().numpy().astype(bool)
            })
        preds_by_img[img_id] = img_preds
        
    return preds_by_img

def dilate_box(bbox, alpha, shape):
    """Dilate bbox [x1, y1, x2, y2] by alpha margin clamped to image boundaries."""
    h, w = shape
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    nx1 = max(0.0, x1 - alpha * bw)
    ny1 = max(0.0, y1 - alpha * bh)
    nx2 = min(float(w), x2 + alpha * bw)
    ny2 = min(float(h), y2 + alpha * bh)
    return [nx1, ny1, nx2, ny2]

def evaluate_crop_at_dilation(preds_dict, coco_gt, img_ids, image_gt_cache, alpha=0.0):
    """Re-crop raw predicted masks with dilated bounding boxes."""
    all_ious = []
    all_covs = []
    all_precs = []
    all_bg_fps = []
    all_hit75 = []
    
    for img_id in img_ids:
        gt_list, crowd_mask = image_gt_cache[img_id]
        p_list = preds_dict.get(img_id, [])
        matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
        orig_shape = crowd_mask.shape
        
        for gt_idx, (p_idx, b_iou) in matches.items():
            gt = gt_list[gt_idx]
            p = p_list[p_idx]
            
            # Dilate crop box
            dil_b = dilate_box(p["bbox"], alpha, orig_shape)
            x1, y1, x2, y2 = [int(round(v)) for v in dil_b]
            
            # Crop raw mask
            m_raw = p["raw_mask"]
            m_crop = np.zeros_like(m_raw)
            m_crop[y1:y2, x1:x2] = m_raw[y1:y2, x1:x2]
            
            gt_m = gt["mask"]
            gt_area = float(gt["area"])
            pred_area = float(m_crop.sum())
            tp = float(np.logical_and(m_crop, gt_m).sum())
            union = float(np.logical_or(m_crop, gt_m).sum())
            
            iou = tp / union if union > 0 else 0.0
            cov = tp / gt_area if gt_area > 0 else 0.0
            prec = tp / pred_area if pred_area > 0 else 0.0
            
            g_all = np.zeros_like(gt_m)
            for other in gt_list:
                g_all = np.logical_or(g_all, other["mask"])
            bg_mask = np.logical_and(m_crop, np.logical_not(np.logical_or(g_all, crowd_mask)))
            bg_fp = float(bg_mask.sum()) / gt_area if gt_area > 0 else 0.0
            
            all_ious.append(iou)
            all_covs.append(cov)
            all_precs.append(prec)
            all_bg_fps.append(bg_fp)
            all_hit75.append(1.0 if iou >= 0.75 else 0.0)
            
    return {
        "N": len(all_ious),
        "mask_iou": float(np.mean(all_ious) * 100),
        "coverage": float(np.mean(all_covs) * 100),
        "precision": float(np.mean(all_precs) * 100),
        "bg_fp": float(np.mean(all_bg_fps) * 100),
        "hit_75": float(np.mean(all_hit75) * 100)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    all_img_ids = sorted(coco_gt.getImgIds())
    img_ids = all_img_ids[:args.limit] if args.limit > 0 else all_img_ids
    print(f"Evaluating {len(img_ids)} images for box uncertainty and robustness stress-test...")
    
    image_gt_cache = {}
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        gt_list, crowd_mask = get_coco_instances(coco_gt, img_id, (orig_h, orig_w))
        image_gt_cache[img_id] = (gt_list, crowd_mask)
        
    models = {
        "Baseline (mr4 s0)": root / "runs/aligned_benchmark/baseline_s0_e5_mr4/weights/last.pt",
        "Control (2.0x mr4)": root / "runs/aligned_benchmark/control_weight_s0_e5_mr4/weights/last.pt",
        "DMS Unconf (mr4 s0)": root / "runs/aligned_benchmark/dms_unconfounded_s0_e5_mr4/weights/last.pt"
    }
    
    # -------------------------------------------------------------
    # EXPERIMENT 1: CONTROLLED BOX PERTURBATION SWEEP (alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    # -------------------------------------------------------------
    print("\n" + "="*120)
    print("EXPERIMENT 1: CONTROLLED BOX PERTURBATION SWEEP (Crop Boundary Expansion alpha)")
    print("Simulates realistic box jitter and over-extension under proposal noise")
    print("="*120)
    
    # Cache inference at conf=0.25
    model_preds = {}
    for m_name, wf in models.items():
        print(f"Running base inference for {m_name} ...")
        yolo = YOLO(str(wf))
        m = yolo.model.eval().cuda()
        m.model[-1].end2end = False
        model_preds[m_name] = run_inference_with_conf(m, coco_gt, img_ids, img_dir, cat_ids, conf_thres=0.25)
        del m, yolo
        torch.cuda.empty_cache()
        
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    pert_results = defaultdict(dict)
    
    header = f"{'Alpha':<8} | {'Metric':<14} | {'Baseline (mr4)':<14} | {'Control (2.0x)':<14} | {'DMS Unconf':<14} | {'DMS Delta':<12}"
    print("\n" + header)
    print("-" * len(header))
    
    for a in alphas:
        for m_name in models.keys():
            pert_results[a][m_name] = evaluate_crop_at_dilation(model_preds[m_name], coco_gt, img_ids, image_gt_cache, alpha=a)
        
        rb = pert_results[a]["Baseline (mr4 s0)"]
        rc = pert_results[a]["Control (2.0x mr4)"]
        rd = pert_results[a]["DMS Unconf (mr4 s0)"]
        
        for m_key, m_label in [("mask_iou", "Mask IoU"), ("precision", "Precision"), ("bg_fp", "Bg FP Rate"), ("hit_75", "Recall@0.75")]:
            vb = rb[m_key]
            vc = rc[m_key]
            vd = rd[m_key]
            delta = vd - vb
            print(f"{a:<8.2f} | {m_label:<14} | {vb:6.2f}%         | {vc:6.2f}%         | {vd:6.2f}%         | {delta:+6.2f}%")
        print("-" * len(header))
        
    # -------------------------------------------------------------
    # EXPERIMENT 2: CONFIDENCE OPERATING POINT SWEEP (conf in [0.05, 0.10, 0.15, 0.20, 0.25])
    # -------------------------------------------------------------
    print("\n" + "="*120)
    print("EXPERIMENT 2: CONFIDENCE OPERATING POINT SWEEP (Looser / High-Recall Detector Operating Points)")
    print("="*120)
    conf_results = defaultdict(dict)
    confs = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    conf_header = f"{'Conf':<8} | {'Metric':<14} | {'Baseline (mr4)':<14} | {'Control (2.0x)':<14} | {'DMS Unconf':<14} | {'DMS Delta':<12}"
    print("\n" + conf_header)
    print("-" * len(conf_header))
    
    for c_thresh in confs:
        for m_name, wf in models.items():
            yolo = YOLO(str(wf))
            m = yolo.model.eval().cuda()
            m.model[-1].end2end = False
            p_dict = run_inference_with_conf(m, coco_gt, img_ids, img_dir, cat_ids, conf_thres=c_thresh)
            del m, yolo
            torch.cuda.empty_cache()
            conf_results[c_thresh][m_name] = evaluate_crop_at_dilation(p_dict, coco_gt, img_ids, image_gt_cache, alpha=0.0)
            
        rb = conf_results[c_thresh]["Baseline (mr4 s0)"]
        rc = conf_results[c_thresh]["Control (2.0x mr4)"]
        rd = conf_results[c_thresh]["DMS Unconf (mr4 s0)"]
        
        for m_key, m_label in [("mask_iou", "Mask IoU"), ("precision", "Precision"), ("bg_fp", "Bg FP Rate"), ("hit_75", "Recall@0.75")]:
            vb = rb[m_key]
            vc = rc[m_key]
            vd = rd[m_key]
            delta = vd - vb
            print(f"{c_thresh:<8.2f} | {m_label:<14} | {vb:6.2f}%         | {vc:6.2f}%         | {vd:6.2f}%         | {delta:+6.2f}%")
        print("-" * len(conf_header))
        
    out_file = root / "box_uncertainty_robustness_results.json"
    out_file.write_text(json.dumps({
        "perturbation_sweep": pert_results,
        "confidence_sweep": conf_results
    }, indent=2))
    print(f"\nSaved stress-test results to {out_file}")

if __name__ == "__main__":
    main()
