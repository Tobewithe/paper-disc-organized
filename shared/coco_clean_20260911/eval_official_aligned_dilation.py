"""Evaluation script fully aligned with official Ultralytics ops.process_mask_native.
Evaluates O2M and O2O branches under crop dilation alpha in [0.0, 0.1, 0.2].
At alpha=0.0, this matches Ultralytics official postprocessing bit-for-bit.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

def process_mask_native_dilated(protos, masks_in, bboxes, shape, alphas=(0.0, 0.1, 0.2)):
    """Apply masks to bounding boxes matching ops.process_mask_native, with optional alpha dilation.
    
    Args:
        protos (torch.Tensor): Mask prototypes (C, mh, mw).
        masks_in (torch.Tensor): Mask coefficients (N, C).
        bboxes (torch.Tensor): Bounding boxes in original image coordinates (N, 4).
        shape (tuple): Target original image shape (height, width).
        alphas (tuple): Dilation factors to evaluate.
        
    Returns:
        dict[float, np.ndarray]: Dict mapping alpha to binary uint8 masks (N, H, W).
    """
    c, mh, mw = protos.shape
    h, w = shape
    if masks_in.shape[0] == 0:
        empty = np.zeros((0, h, w), dtype=np.uint8)
        return {a: empty for a in alphas}
        
    # Prototype-resolution mask logits (N, mh*mw)
    coeffs = masks_in @ protos.float().view(c, -1)
    
    # Official chunked upsampling bounded by pixel budget to avoid OOM
    step = max(1, 32_000_000 // (h * w))
    raw_masks = [
        ops.scale_masks(coeffs[i : i + step].view(-1, mh, mw)[None], shape)[0].gt_(0.0).byte()
        for i in range(0, coeffs.shape[0], step)
    ]
    raw_masks_cat = torch.cat(raw_masks)  # (N, H, W) binary uint8
    
    res = {}
    for a in alphas:
        if a > 0:
            bw = bboxes[:, 2] - bboxes[:, 0]
            bh = bboxes[:, 3] - bboxes[:, 1]
            x1 = torch.clamp(bboxes[:, 0] - a * bw, min=0.0)
            y1 = torch.clamp(bboxes[:, 1] - a * bh, min=0.0)
            x2 = torch.clamp(bboxes[:, 2] + a * bw, max=float(w))
            y2 = torch.clamp(bboxes[:, 3] + a * bh, max=float(h))
            boxes_dilated = torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            boxes_dilated = bboxes
        # Official crop_mask applied to native-resolution binary masks (clone to avoid in-place mutation)
        cropped = ops.crop_mask(raw_masks_cat.clone(), boxes_dilated)
        res[a] = cropped.cpu().numpy()
        
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save-rle", action="store_true", help="Save predictions with RLEs")
    parser.add_argument("--limit", type=int, default=0, help="0 = full 1576 images")
    args = parser.parse_args()
    
    weight_path = Path(args.weights)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading GT from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit > 0:
        img_ids = img_ids[:args.limit]
    print(f"Evaluating on {len(img_ids)} images...")
    
    print(f"Loading model from {weight_path}...")
    yolo = YOLO(str(weight_path))
    model = yolo.model.eval().cuda()
    head = model.model[-1]
    
    letterbox = LetterBox(new_shape=(640, 640), scaleup=False)
    branches = ["o2m", "o2o"]
    alphas = [0.0, 0.1, 0.2]
    
    results_table = {}
    saved_rles = {}
    
    for branch in branches:
        print(f"\n==================================================")
        print(f"Running Branch: {branch.upper()} ({'One-to-Many + NMS' if branch=='o2m' else 'One-to-One End2End (No NMS)'})")
        print(f"==================================================")
        head.end2end = (branch == "o2o")
        
        t0 = time.time()
        coco_results_box = []
        coco_results_mask_by_alpha = {a: [] for a in alphas}
        
        for idx, img_id in enumerate(img_ids):
            im_info = coco_gt.loadImgs(img_id)[0]
            fn = im_info["file_name"]
            p = img_dir / fn
            if not p.exists():
                continue
                
            # Read via cv2 to match official LetterBox transform
            im0 = cv2.imread(str(p))
            orig_h, orig_w = im0.shape[:2]
            
            im_lb = letterbox(image=im0)
            tensor = torch.from_numpy(im_lb.transpose(2, 0, 1)).float().cuda() / 255.0
            tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                out = model(tensor)
                
            if branch == "o2m":
                preds, proto = out[0]
                nms_out = non_max_suppression(
                    preds,
                    conf_thres=0.001,
                    iou_thres=0.7,
                    max_det=300,
                    nc=80,
                    end2end=False
                )[0]
                if len(nms_out) == 0:
                    continue
                boxes = nms_out[:, :4]
                scores = nms_out[:, 4]
                clses = nms_out[:, 5].long()
                coeffs = nms_out[:, 6:]
            else:
                res, proto = out[0]
                res = res[0]
                mask_conf = res[:, 4] > 0.001
                res = res[mask_conf]
                if len(res) == 0:
                    continue
                boxes = res[:, :4]
                scores = res[:, 4]
                clses = res[:, 5].long()
                coeffs = res[:, 6:]
                
            if proto.dim() == 4:
                proto = proto[0]
                
            # Scale boxes from 640x640 letterbox back to original image resolution
            boxes_orig = ops.scale_boxes((640, 640), boxes.clone(), (orig_h, orig_w))
            
            # Generate masks matching official process_mask_native
            masks_by_alpha = process_mask_native_dilated(proto, coeffs, boxes_orig, (orig_h, orig_w), alphas=alphas)
            
            boxes_np = boxes_orig.cpu().numpy()
            scores_np = scores.cpu().numpy()
            clses_np = clses.cpu().numpy()
            
            for i in range(len(boxes_np)):
                bx, by, bx2, by2 = boxes_np[i]
                bw, bh = bx2 - bx, by2 - by
                score = float(scores_np[i])
                cat_id = cat_ids[clses_np[i]]
                
                coco_results_box.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(float(bx), 2), round(float(by), 2), round(float(bw), 2), round(float(bh), 2)],
                    "score": round(score, 4)
                })
                
                for a in alphas:
                    rle = mask_util.encode(np.asfortranarray(masks_by_alpha[a][i]))
                    rle["counts"] = rle["counts"].decode("ascii")
                    coco_results_mask_by_alpha[a].append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "score": round(score, 4)
                    })
                    
            if (idx + 1) % 200 == 0 or (idx + 1) == len(img_ids):
                fps = (idx + 1) / (time.time() - t0)
                print(f"[{idx+1}/{len(img_ids)}] ({fps:.1f} img/s) | Box preds: {len(coco_results_box)}", flush=True)
                
        # Evaluate Box AP
        print(f"\nEvaluating Box AP for {branch.upper()}...")
        coco_dt_box = coco_gt.loadRes(coco_results_box)
        ev_box = COCOeval(coco_gt, coco_dt_box, "bbox")
        ev_box.params.imgIds = img_ids
        ev_box.evaluate()
        ev_box.accumulate()
        ev_box.summarize()
        
        # Evaluate Mask AP for each alpha
        for a in alphas:
            print(f"\nEvaluating Mask AP for {branch.upper()} (alpha = {a})...")
            coco_dt_mask = coco_gt.loadRes(coco_results_mask_by_alpha[a])
            ev_mask = COCOeval(coco_gt, coco_dt_mask, "segm")
            ev_mask.params.imgIds = img_ids
            ev_mask.evaluate()
            ev_mask.accumulate()
            ev_mask.summarize()
            
            key = f"{branch}_alpha_{a}"
            results_table[key] = {
                "branch": branch,
                "alpha": a,
                "box_ap": float(ev_box.stats[0]),
                "box_ap50": float(ev_box.stats[1]),
                "box_ap75": float(ev_box.stats[2]),
                "mask_ap": float(ev_mask.stats[0]),
                "mask_ap50": float(ev_mask.stats[1]),
                "mask_ap75": float(ev_mask.stats[2]),
            }
            print(f"--> [{key}]: Box AP={ev_box.stats[0]:.4f} | Mask AP={ev_mask.stats[0]:.4f} | AP50={ev_mask.stats[1]:.4f} | AP75={ev_mask.stats[2]:.4f}")
            
        if args.save_rle:
            saved_rles[branch] = coco_results_mask_by_alpha
            
    out_path.write_text(json.dumps(results_table, indent=2))
    print(f"\nSaved aligned evaluation summary to {out_path}")
    
    if args.save_rle:
        rle_path = out_path.parent / f"{out_path.stem}_rles.json"
        print(f"Saving prediction RLEs to {rle_path}...")
        rle_path.write_text(json.dumps(saved_rles))
        print("RLEs saved.")

if __name__ == "__main__":
    main()
