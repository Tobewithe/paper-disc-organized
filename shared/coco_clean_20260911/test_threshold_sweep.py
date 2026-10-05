"""Sweep mask binarization threshold to test calibration hypothesis.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import math
from pathlib import Path
import cv2
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

def run_inference_with_threshold(model, coco_gt, img_ids, img_dir, cat_ids, prob_thresh=0.50):
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)
    logit_thresh = math.log(prob_thresh / (1.0 - prob_thresh))
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
        raw_masks = ops.scale_masks(coeffs_eval.view(-1, mh, mw)[None], (orig_h, orig_w))[0].gt_(logit_thresh).byte()
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
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    img_ids = sorted(coco_gt.getImgIds())[:100]
    
    # Pre-cache GTs
    image_gt_cache = {}
    for img_id in img_ids:
        fn = coco_gt.loadImgs(img_id)[0]["file_name"]
        im_path = img_dir / fn
        im_rgb = load_image_rgb(im_path)
        orig_h, orig_w = im_rgb.shape[:2]
        gt_list, crowd_mask = get_coco_instances(coco_gt, img_id, (orig_h, orig_w))
        image_gt_cache[img_id] = (gt_list, crowd_mask)
        
    models = {
        "Baseline": root / "runs/aligned_benchmark/baseline_s0_e5_mr4/weights/last.pt",
        "DMS Unconf": root / "runs/aligned_benchmark/dms_unconfounded_s0_e5_mr4/weights/last.pt"
    }
    
    for prob in [0.50, 0.45, 0.40]:
        print(f"\n=======================================================")
        print(f"Testing Probability Threshold: {prob:.2f} (logit={math.log(prob/(1-prob)):.3f})")
        print(f"=======================================================")
        for m_name, wf in models.items():
            yolo = YOLO(str(wf))
            m = yolo.model.eval().cuda()
            m.model[-1].end2end = False
            preds_dict = run_inference_with_threshold(m, coco_gt, img_ids, img_dir, cat_ids, prob_thresh=prob)
            del m, yolo
            torch.cuda.empty_cache()
            
            ious = []
            recalls = []
            precisions = []
            bg_fps = []
            
            for img_id in img_ids:
                gt_list, crowd_mask = image_gt_cache[img_id]
                p_list = preds_dict.get(img_id, [])
                matches = match_image_hungarian(gt_list, p_list, iou_thresh=0.50)
                
                for gt_idx, (p_idx, b_iou) in matches.items():
                    gt = gt_list[gt_idx]
                    p = p_list[p_idx]
                    tp = float(np.logical_and(p["mask"], gt["mask"]).sum())
                    union = float(np.logical_or(p["mask"], gt["mask"]).sum())
                    pred_area = float(p["mask"].sum())
                    gt_area = float(gt["area"])
                    
                    iou = tp / union if union > 0 else 0.0
                    rec = tp / gt_area if gt_area > 0 else 0.0
                    prec = tp / pred_area if pred_area > 0 else 0.0
                    
                    # Background FP
                    g_all = np.zeros_like(gt["mask"])
                    for other in gt_list:
                        g_all = np.logical_or(g_all, other["mask"])
                    bg_mask = np.logical_and(p["mask"], np.logical_not(np.logical_or(g_all, crowd_mask)))
                    bg_fp = float(bg_mask.sum()) / gt_area if gt_area > 0 else 0.0
                    
                    ious.append(iou)
                    recalls.append(rec)
                    precisions.append(prec)
                    bg_fps.append(bg_fp)
                    
            print(f"{m_name:<12} (p={prob:.2f}) | N={len(ious):<4} | IoU: {np.mean(ious)*100:6.2f}% | Rec: {np.mean(recalls)*100:6.2f}% | Prec: {np.mean(precisions)*100:6.2f}% | Bg FP: {np.mean(bg_fps)*100:6.2f}%")

if __name__ == "__main__":
    main()
