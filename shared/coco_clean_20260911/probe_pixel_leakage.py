"""Pixel-level mask leakage diagnostic probe.
Directly measures the empirical false positive activation in the dilated box margin
(annular region between alpha=0.0 and alpha=0.2) across models.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

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

def get_boxes_proto(bboxes, imgsz=(640, 640), protosz=(160, 160)):
    iw, ih = imgsz
    mw, mh = protosz
    scale_x = mw / iw
    scale_y = mh / ih
    bp = bboxes.clone()
    bp[:, [0, 2]] *= scale_x
    bp[:, [1, 3]] *= scale_y
    return bp

def dilate_box(boxes: torch.Tensor, alpha: float, max_w: float, max_h: float):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    bw = x2 - x1
    bh = y2 - y1
    nx1 = torch.clamp(x1 - alpha * bw, min=0.0)
    ny1 = torch.clamp(y1 - alpha * bh, min=0.0)
    nx2 = torch.clamp(x2 + alpha * bw, max=max_w)
    ny2 = torch.clamp(y2 + alpha * bh, max=max_h)
    return torch.stack([nx1, ny1, nx2, ny2], dim=1)

def run_probe(weights_path: Path, img_ids, coco_gt, img_dir, limit=100):
    print(f"Loading {weights_path.name}...")
    yolo = YOLO(str(weights_path))
    model = yolo.model.eval().cuda()
    head = model.model[-1]
    head.end2end = False  # O2M branch with NMS
    
    total_tight_pixels = 0
    total_dilated_margin_pixels = 0  # Total pixels in the 0.0 -> 0.2 band
    total_leaked_mask_pixels = 0     # Positive mask pixels falling into the 0.0 -> 0.2 band
    total_instances = 0
    
    with torch.no_grad():
        for idx, img_id in enumerate(img_ids[:limit]):
            fn = coco_gt.loadImgs(img_id)[0]["file_name"]
            p = img_dir / fn
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            inp, _ = letterbox_image(im)
            inp = inp.unsqueeze(0).cuda()
            
            out = model(inp)
            raw_box, proto = out[0]
            if proto.dim() == 4:
                proto = proto[0]
                
            nms_out = non_max_suppression(raw_box, conf_thres=0.25, iou_thres=0.65, nc=80, end2end=False)[0]
            if nms_out is None or len(nms_out) == 0:
                continue
                
            boxes = nms_out[:, :4]
            mask_coeffs = nms_out[:, 6:]
            c, mh, mw = proto.shape
            
            # Raw masks: (N, 160, 160)
            raw_logits = mask_coeffs @ proto.float().view(c, -1)
            raw_binary = (raw_logits.sigmoid().view(-1, mh, mw) > 0.5)
            
            bp_tight = get_boxes_proto(boxes, imgsz=(640, 640), protosz=(mw, mh))
            bp_dilated = dilate_box(bp_tight, alpha=0.2, max_w=float(mw), max_h=float(mh))
            
            # Spatial grids
            r = torch.arange(mw, device=proto.device)[None, None, :]
            c_grid = torch.arange(mh, device=proto.device)[None, :, None]
            
            tight_mask = (r >= bp_tight[:, 0:1, None]) & (r < bp_tight[:, 2:3, None]) & \
                         (c_grid >= bp_tight[:, 1:2, None]) & (c_grid < bp_tight[:, 3:4, None])
                         
            dilated_mask = (r >= bp_dilated[:, 0:1, None]) & (r < bp_dilated[:, 2:3, None]) & \
                           (c_grid >= bp_dilated[:, 1:2, None]) & (c_grid < bp_dilated[:, 3:4, None])
                           
            margin_mask = dilated_mask & (~tight_mask)  # The annular band
            
            # Count mask pixels
            tight_active = (raw_binary & tight_mask).sum().item()
            margin_active = (raw_binary & margin_mask).sum().item()
            margin_total = margin_mask.sum().item()
            
            total_tight_pixels += tight_active
            total_leaked_mask_pixels += margin_active
            total_dilated_margin_pixels += margin_total
            total_instances += len(boxes)
            
    leakage_ratio = total_leaked_mask_pixels / max(total_tight_pixels, 1)
    fpr_margin = total_leaked_mask_pixels / max(total_dilated_margin_pixels, 1)
    
    return {
        "instances": total_instances,
        "tight_pixels": total_tight_pixels,
        "leaked_pixels": total_leaked_mask_pixels,
        "margin_pixels": total_dilated_margin_pixels,
        "leakage_ratio": leakage_ratio,
        "fpr_margin": fpr_margin
    }

def main():
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    gt_path = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\coco_dense_val_gt.json")
    img_dir = Path(r"C:\Dpan\codexproject\paper-disc\gemini\data\coco_dense\images\val2017")
    coco_gt = COCO(str(gt_path))
    img_ids = sorted(coco_gt.getImgIds())
    
    models = {
        "Stock Official": root / "weights/yolo26m-seg.pt",
        "Baseline (5ep)": root / "runs/pilot_dual_branch/baseline_s0_e5/weights/last.pt",
        "CCL-O2M (5ep)": root / "runs/pilot_dual_branch/ccl_o2m_s0_e5/weights/last.pt",
        "Dilated-0.1 (5ep)": root / "runs/pilot_dual_branch/dilated_0.1_s0_e5/weights/last.pt",
    }
    
    results = {}
    for name, p in models.items():
        res = run_probe(p, img_ids, coco_gt, img_dir, limit=100)
        results[name] = res
        print(f"\n--- {name} ---")
        print(f"  Total Valid Instances: {res['instances']}")
        print(f"  Mask Pixels (Inside Tight Box): {res['tight_pixels']:,}")
        print(f"  Mask Pixels (Leaked to Dilated Margin): {res['leaked_pixels']:,}")
        print(f"  Leakage Ratio (Leaked / Inside): {res['leakage_ratio'] * 100:.2f}%")
        print(f"  False Positive Rate in Margin Band: {res['fpr_margin'] * 100:.2f}%\n")
        
    out_file = root / "pixel_leakage_probe_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved probe results to {out_file}")

if __name__ == "__main__":
    main()
