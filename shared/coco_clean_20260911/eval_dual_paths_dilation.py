"""Evaluate official stock yolo26m-seg.pt on full COCO-Dense val (1576 images) under dual branches and crop dilation.
Optimized to run model forward pass once per image, then evaluate alpha = [0.0, 0.1, 0.2] in parallel.
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
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy

def crop_mask_dilated(masks: torch.Tensor, boxes: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    _, h, w = masks.shape
    x1 = boxes[:, 0:1, None]
    y1 = boxes[:, 1:2, None]
    x2 = boxes[:, 2:3, None]
    y2 = boxes[:, 3:4, None]
    
    if alpha > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 = torch.clamp(x1 - alpha * bw, min=0.0)
        y1 = torch.clamp(y1 - alpha * bh, min=0.0)
        x2 = torch.clamp(x2 + alpha * bw, max=float(w))
        y2 = torch.clamp(y2 + alpha * bh, max=float(h))
        
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    
    cropped = masks.clone()
    cropped *= (r >= x1) * (r < x2)
    cropped *= (c >= y1) * (c < y2)
    return cropped

def process_masks_multi_alpha(protos, masks_in, bboxes, orig_shape, img_shape=(640, 640), alphas=(0.0, 0.1, 0.2)):
    c, mh, mw = protos.shape
    ih, iw = img_shape
    orig_h, orig_w = orig_shape
    
    # Calculate mask logits on proto grid (160x160)
    raw_masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    
    scale_x = mw / iw
    scale_y = mh / ih
    boxes_proto = bboxes.clone()
    boxes_proto[:, [0, 2]] *= scale_x
    boxes_proto[:, [1, 3]] *= scale_y
    
    gain = min(ih / orig_h, iw / orig_w)
    pad_x = (iw - orig_w * gain) / 2
    pad_y = (ih - orig_h * gain) / 2
    top, left = int(round(pad_y)), int(round(pad_x))
    bottom, right = int(round(ih - pad_y)), int(round(iw - pad_x))
    
    res = {}
    for alpha in alphas:
        cropped = crop_mask_dilated(raw_masks, boxes_proto, alpha=alpha)
        masks_up = F.interpolate(cropped[None], (ih, iw), mode="bilinear", align_corners=False)[0]
        masks_up = masks_up[:, top:bottom, left:right]
        masks_orig = F.interpolate(masks_up[None], (orig_h, orig_w), mode="bilinear", align_corners=False)[0]
        res[alpha] = (masks_orig > 0.5).cpu().numpy().astype(np.uint8)
    return res

def letterbox_image(im: Image.Image, new_shape=(640, 640)):
    w, h = im.size
    gain = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    dw = (new_shape[1] - nw) / 2
    dh = (new_shape[0] - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    new_im = Image.new("RGB", new_shape, (114, 114, 114))
    new_im.paste(im_resized, (left, top))
    arr = np.array(new_im).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float() / 255.0
    return tensor, (gain, (dw, dh))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = full 1576 images")
    parser.add_argument("--weights", type=str, default=r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\weights\yolo26m-seg.pt")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    
    weight_path = Path(args.weights)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = weight_path.parent / f"{weight_path.stem}_dilation_eval.json"
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    
    print(f"Loading GT from {gt_path}...")
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    
    print(f"Loading model from {weight_path}...")
    yolo = YOLO(str(weight_path))
    model = yolo.model
    head = model.model[-1]
    model.eval().cuda()
    
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit > 0:
        img_ids = img_ids[:args.limit]
    print(f"Evaluating on {len(img_ids)} images...")
    
    branches = ["o2m", "o2o"]
    alphas = [0.0, 0.1, 0.2]
    
    results_table = {}
    
    for branch in branches:
        print(f"\n==================================================")
        print(f"Running Branch: {branch.upper()} ({'One-to-Many + NMS' if branch=='o2m' else 'One-to-One (End2End, No NMS)'})")
        print(f"==================================================")
        
        if branch == "o2m":
            head.end2end = False
        else:
            head.end2end = True
            
        t0 = time.time()
        coco_results_box = []
        coco_results_mask_by_alpha = {a: [] for a in alphas}
        
        for idx, img_id in enumerate(img_ids):
            im_info = coco_gt.loadImgs(img_id)[0]
            fn = im_info["file_name"]
            p = img_dir / fn
            if not p.exists():
                continue
            
            pil_im = Image.open(p).convert("RGB")
            orig_w, orig_h = pil_im.size
            tensor, (gain, (dw, dh)) = letterbox_image(pil_im)
            tensor = tensor.unsqueeze(0).cuda()
            
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
            
            boxes_orig = boxes.clone()
            boxes_orig[:, [0, 2]] -= dw
            boxes_orig[:, [1, 3]] -= dh
            boxes_orig[:, [0, 2]] /= gain
            boxes_orig[:, [1, 3]] /= gain
            boxes_orig[:, [0, 2]] = torch.clamp(boxes_orig[:, [0, 2]], min=0, max=orig_w)
            boxes_orig[:, [1, 3]] = torch.clamp(boxes_orig[:, [1, 3]], min=0, max=orig_h)
            
            # Generate masks for all 3 alphas at once
            masks_by_alpha = process_masks_multi_alpha(
                proto[0], coeffs, boxes,
                orig_shape=(orig_h, orig_w),
                img_shape=(640, 640),
                alphas=alphas
            )
            
            boxes_orig_np = boxes_orig.cpu().numpy()
            scores_np = scores.cpu().numpy()
            clses_np = clses.cpu().numpy()
            
            for i in range(len(boxes_orig_np)):
                cid = cat_ids[clses_np[i]]
                bx, by, bx2, by2 = boxes_orig_np[i]
                bw, bh = bx2 - bx, by2 - by
                score = float(scores_np[i])
                
                coco_results_box.append({
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [round(float(bx), 2), round(float(by), 2), round(float(bw), 2), round(float(bh), 2)],
                    "score": round(score, 4)
                })
                
                for a in alphas:
                    rle = mask_util.encode(np.asfortranarray(masks_by_alpha[a][i]))
                    rle["counts"] = rle["counts"].decode("ascii")
                    coco_results_mask_by_alpha[a].append({
                        "image_id": img_id,
                        "category_id": cid,
                        "segmentation": rle,
                        "score": round(score, 4)
                    })
            
            if (idx + 1) % 200 == 0 or (idx + 1) == len(img_ids):
                fps = (idx + 1) / (time.time() - t0)
                print(f"[{idx+1}/{len(img_ids)}] ({fps:.1f} img/s) | Box preds: {len(coco_results_box)}", flush=True)
                
        # Evaluate Box AP once per branch
        print(f"\nEvaluating Box AP for {branch.upper()}...")
        coco_dt_box = coco_gt.loadRes(coco_results_box)
        ev_box = COCOeval(coco_gt, coco_dt_box, "bbox")
        ev_box.params.imgIds = img_ids
        ev_box.evaluate()
        ev_box.accumulate()
        ev_box.summarize()
        box_ap = ev_box.stats[0]
        box_ap50 = ev_box.stats[1]
        box_ap75 = ev_box.stats[2]
        
        # Evaluate Mask AP for each alpha
        for a in alphas:
            print(f"\nEvaluating Mask AP for {branch.upper()} (alpha = {a})...")
            coco_dt_mask = coco_gt.loadRes(coco_results_mask_by_alpha[a])
            ev_mask = COCOeval(coco_gt, coco_dt_mask, "segm")
            ev_mask.params.imgIds = img_ids
            ev_mask.evaluate()
            ev_mask.accumulate()
            ev_mask.summarize()
            mask_ap = ev_mask.stats[0]
            mask_ap50 = ev_mask.stats[1]
            mask_ap75 = ev_mask.stats[2]
            
            key = f"{branch}_alpha_{a}"
            results_table[key] = {
                "branch": branch,
                "alpha": a,
                "box_ap": float(box_ap),
                "box_ap50": float(box_ap50),
                "box_ap75": float(box_ap75),
                "mask_ap": float(mask_ap),
                "mask_ap50": float(mask_ap50),
                "mask_ap75": float(mask_ap75),
            }
            print(f"--> [{key}]: Box AP={box_ap:.4f} | Mask AP={mask_ap:.4f} | AP50={mask_ap50:.4f} | AP75={mask_ap75:.4f}")
            
    out_path.write_text(json.dumps(results_table, indent=2))
    print(f"\n==================================================")
    print(f"FULL BENCHMARK COMPLETE! Saved to {out_path}")
    print(f"==================================================")
    for k, v in results_table.items():
        print(f"{k:15s}: Box AP={v['box_ap']:.4f} | Mask AP={v['mask_ap']:.4f} | Mask AP50={v['mask_ap50']:.4f} | Mask AP75={v['mask_ap75']:.4f}")

if __name__ == "__main__":
    main()
