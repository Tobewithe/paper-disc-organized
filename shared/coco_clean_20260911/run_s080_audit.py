"""Run S080 Failure Strata Mechanism Audit on trained BSR-Head and S032 models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from pycocotools.coco import COCO
from ultralytics.utils import ops

from bifurcated_spatial_readout import BifurcatedSpatialHead
from rich_pixel_readout import instance_features, GlobalHead

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / 'runs/bsr_head_aligned_s3_e15'
S032_DIR = ROOT / 'diagnostics/shared_label_controls_20260912'
NORMALIZER_PATH = ROOT / 'diagnostics/shared_label_controls_20260912/normalizer.pt'
S080_INSTANCES = ROOT / 'diagnostics/fixed_prototype_split_20260913_v5/instances.csv'
S080_TENSOR_CACHE = ROOT / 'diagnostics/joint_failure_decoder_20260913_v2/tensor_cache'
ANN_JSON = ROOT.parent.parent / 'datasets/coco/annotations/instances_val2017.json'


def norm(x, n):
    return (x - n['mean']) / n['std']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models...", flush=True)
    normalizer = torch.load(NORMALIZER_PATH, map_location=device)
    
    models_dict = {
        'original': {'type': 'original', 'model': None}
    }
    
    for seed in [0, 1, 2]:
        # S032
        s032_ckpt = S032_DIR / f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        s032_model = GlobalHead().to(device)
        s032_model.load_state_dict(torch.load(s032_ckpt, map_location=device)['model'])
        s032_model.eval()
        models_dict[f's032_raw_s{seed}'] = {'type': 's032', 'model': s032_model}
        
        # BSR-Head
        bsr_ckpt = RUN_DIR / f'bsr_head_s{seed}/checkpoints/epoch015.pt'
        bsr_model = BifurcatedSpatialHead().to(device)
        bsr_model.load_state_dict(torch.load(bsr_ckpt, map_location=device)['model'])
        bsr_model.eval()
        models_dict[f'bsr_head_s{seed}'] = {'type': 'bsr_head', 'model': bsr_model}
        
    print(f"Loaded {len(models_dict)} model arms.", flush=True)
    
    gt = COCO(str(ANN_JSON))
    df_s080 = pd.read_csv(S080_INSTANCES)
    results = []
    
    print("Evaluating S080 instances...", flush=True)
    for idx, row in df_s080.iterrows():
        iid = int(row['image_id'])
        aid = int(row['annotation_id'])
        density = row['density']
        residual = row['residual']
        baseline_iou = float(row['baseline_iou'])
        
        tensor_path = S080_TENSOR_CACHE / f"{iid}.npz"
        if not tensor_path.exists():
            continue
        cache = np.load(tensor_path)
        where = np.flatnonzero(cache['annotation_ids'].astype(np.int64) == aid)
        if len(where) != 1:
            continue
        k = int(where[0])
        
        proto = torch.as_tensor(cache['proto'], device=device, dtype=torch.float32)
        coeff = torch.as_tensor(cache['coeff'][k:k+1], device=device, dtype=torch.float32)
        box = torch.as_tensor(cache['boxes'][k:k+1], device=device, dtype=torch.float32)
        h = torch.as_tensor(cache['h'][k:k+1], device=device, dtype=torch.float32)
        source = int(cache['source_indices'][k])
        lvl = 0 if source < 6400 else 1 if source < 8000 else 2
        level = torch.as_tensor([lvl], device=device, dtype=torch.long)
        
        # Build GT mask in 640x640 space
        ann = gt.anns[aid]
        own = gt.annToMask(ann).astype(bool)
        oh, ow = own.shape
        gain = min(640 / oh, 640 / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
        gt_mask = np.zeros((640, 640), dtype=bool)
        gt_mask[top:top + nh, left:left + nw] = cv2.resize(own.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST_EXACT) > 0
        
        # Instance features
        x = norm(instance_features(h, level, box, (640, 640)), normalizer)
        
        instance_res = dict(
            image_id=iid, annotation_id=aid, density=density,
            residual=residual, baseline_iou=baseline_iou
        )
        
        with torch.inference_mode():
            for arm, model_info in models_dict.items():
                m_type = model_info['type']
                m_obj = model_info['model']
                
                if m_type == 'original':
                    bb = ops.process_mask(proto, coeff, box, (640, 640), upsample=True)
                    pred = bb[0].cpu().numpy() > 0
                elif m_type == 's032':
                    delta_c = m_obj(x)
                    bb = ops.process_mask(proto, coeff + delta_c, box, (640, 640), upsample=True)
                    pred = bb[0].cpu().numpy() > 0
                elif m_type == 'bsr_head':
                    z_final, gate = m_obj.forward_inference(x, coeff, proto)
                    bb = ops.crop_mask(F.interpolate(z_final[:, None], (640, 640), mode='bilinear', align_corners=False)[:, 0], box)
                    pred = bb[0].cpu().numpy() > 0.0
                    instance_res[f'{arm}_gate'] = float(gate[0, 0].cpu().item())
                    
                inter = np.count_nonzero(pred & gt_mask)
                union = np.count_nonzero(pred | gt_mask)
                iou = float(inter / max(union, 1))
                instance_res[f'{arm}_iou'] = iou
                instance_res[f'{arm}_reaches_75'] = bool(iou >= 0.75)
                
        results.append(instance_res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(RUN_DIR / 's080_strata_evaluation.csv', index=False)
    
    # Stratified breakdown
    print("\n=== S080 Dual-Mechanism Breakdown Table ===", flush=True)
    report_rows = []
    strata = [
        'requires_target_pixel_recovery',
        'sufficient_true_pixels_but_residual_false_pixels',
        'mask_good_control',
        'all'
    ]
    for residual_type in strata:
        sub = df_res if residual_type == 'all' else df_res[df_res['residual'] == residual_type]
        row_stat = dict(residual=residual_type, n=len(sub))
        for arm in models_dict:
            reaches = int(sub[f'{arm}_reaches_75'].sum()) if f'{arm}_reaches_75' in sub else 0
            mean_iou = float(sub[f'{arm}_iou'].mean()) * 100.0 if f'{arm}_iou' in sub else 0.0
            row_stat[f'{arm}_reaches_75'] = reaches
            row_stat[f'{arm}_mean_iou'] = mean_iou
            if f'{arm}_gate' in sub:
                row_stat[f'{arm}_mean_gate'] = float(sub[f'{arm}_gate'].mean())
        report_rows.append(row_stat)
        
    df_report = pd.DataFrame(report_rows)
    print(df_report.to_string(index=False), flush=True)
    df_report.to_csv(RUN_DIR / 's080_strata_summary.csv', index=False)
    with open(RUN_DIR / 's080_strata_summary.json', 'w') as f:
        json.dump(report_rows, f, indent=2)
    print(f"\nS080 audit complete! Saved to {RUN_DIR / 's080_strata_summary.csv'}")


if __name__ == '__main__':
    main()
