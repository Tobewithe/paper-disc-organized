"""Evaluation Suite for Innovations Suite (Innovations 1-4).

Evaluates:
- original baseline
- s032_raw (seeds 0, 1, 2)
- bsr_head (seeds 0, 1, 2)
- ada_calib (seeds 0, 1, 2)
- contrastive (seeds 0, 1, 2)
- ortho_center (seeds 0, 1, 2)
- full_synergy (seeds 0, 1, 2)

Across:
1. 300 Transfer Images (2,002 GTs) for official COCOeval AP and stratified ICI R75.
2. S080 Failure Strata Audit (140 instances) for dual-mechanism breakdown.
"""
from pathlib import Path
import json
import time
import contextlib
import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops

from bifurcated_spatial_readout import BifurcatedSpatialHead
from ada_calib_readout import AdaCalibHead
from contrastive_spatial_loss import apply_centerness_prior
from rich_pixel_readout import instance_features, GlobalHead
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'diagnostics/readout_input_scale1200_20260912/cache'
NORMALIZER_PATH = ROOT / 'diagnostics/shared_label_controls_20260912/normalizer.pt'
S032_DIR = ROOT / 'diagnostics/shared_label_controls_20260912'
BSR_DIR = ROOT / 'runs/bsr_head_aligned_s3_e15'
INNOV_DIR = ROOT / 'runs/innovations_suite_aligned_s3_e15'
S080_INSTANCES = ROOT / 'diagnostics/fixed_prototype_split_20260913_v5/instances.csv'
S080_TENSOR_CACHE = ROOT / 'diagnostics/joint_failure_decoder_20260913_v2/tensor_cache'
ANN_JSON = ROOT.parent.parent / 'datasets/coco/annotations/instances_val2017.json'


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def norm(x, n):
    return (x - n['mean']) / n['std']


def read_np(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_all_models(device):
    models = {
        'original': {'type': 'original', 'model': None}
    }
    
    # 1. S032
    for seed in [0, 1, 2]:
        ckpt = S032_DIR / f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        m = GlobalHead().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        m.eval()
        models[f's032_raw_s{seed}'] = {'type': 's032', 'model': m}
        
    # 2. BSR-Head (Innovation 1)
    for seed in [0, 1, 2]:
        ckpt = BSR_DIR / f'bsr_head_s{seed}/checkpoints/epoch015.pt'
        if ckpt.exists():
            m = BifurcatedSpatialHead().to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device)['model'])
            m.eval()
            models[f'bsr_head_s{seed}'] = {'type': 'bsr_head', 'model': m}
            
    # 3. Innovations 2, 3, 4 & Full Synergy
    for mode in ['ada_calib', 'contrastive', 'ortho_center', 'full_synergy']:
        for seed in [0, 1, 2]:
            ckpt = INNOV_DIR / f'{mode}_s{seed}/checkpoints/epoch015.pt'
            if not ckpt.exists():
                continue
            if mode in ['ada_calib', 'full_synergy']:
                m = AdaCalibHead().to(device)
            else:
                m = BifurcatedSpatialHead().to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device)['model'])
            m.eval()
            models[f'{mode}_s{seed}'] = {
                'type': mode,
                'model': m,
                'center_prior': (mode in ['ortho_center', 'full_synergy'])
            }
            
    print(f"Successfully loaded {len(models)} model arms for evaluation.", flush=True)
    return models


def evaluate_transfer(models_dict, normalizer, transfer_ids, out_dir):
    print("\n=======================================================", flush=True)
    print("  Evaluating on 300 Transfer Images (2,002 Ordinary GTs) ", flush=True)
    print("=======================================================", flush=True)
    
    subset_json = CACHE_DIR / 'conversion_input/instances_probe.json'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(subset_json))
        
    gt_eval = COCO()
    gt_eval.dataset = dict(
        info=gt.dataset.get('info', {}),
        categories=list(gt.cats.values()),
        images=[gt.imgs[i] for i in transfer_ids],
        annotations=[ann for i in transfer_ids for ann in gt.imgToAnns[i]]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        gt_eval.createIndex()
        
    meta = {
        ann['id']: {
            'ici_same': ici(ann, [b for b in gt_eval.imgToAnns[ann['image_id']] if not b.get('iscrowd', 0)])
        }
        for ann in gt_eval.anns.values() if not ann.get('iscrowd', 0)
    }
    categories = sorted(gt_eval.cats)
    
    arm_names = list(models_dict.keys())
    predictions = {arm: [] for arm in arm_names}
    
    for num, iid in enumerate(transfer_ids, 1):
        path = CACHE_DIR / 'images' / f'{iid}.npz'
        if not path.exists():
            continue
        item = read_np(path)
        c = cuda(torch.from_numpy(item['coeff'])).float()
        p = cuda(torch.from_numpy(item['proto'])).float()
        boxes = cuda(torch.from_numpy(item['boxes'])).float()
        det = cuda(torch.from_numpy(item['detections'])).float()
        h = cuda(torch.from_numpy(item['h'])).float()
        level = cuda(torch.from_numpy(item['level'])).long()
        shape = tuple(item['shape'])
        ishape = tuple(map(int, item['input_shape']))
        
        if len(c) == 0:
            continue
            
        with torch.inference_mode():
            x = norm(instance_features(h, level, boxes, ishape), normalizer)
            
            for arm, model_info in models_dict.items():
                m_type = model_info['type']
                m_obj = model_info['model']
                use_center_prior = model_info.get('center_prior', False)
                
                if m_type == 'original':
                    bb = ops.process_mask(p, c, boxes, ishape, upsample=True)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.5
                elif m_type == 's032' and not use_center_prior:
                    delta_c = m_obj(x)
                    bb = ops.process_mask(p, c + delta_c, boxes, ishape, upsample=True)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.5
                elif m_type == 's032' and use_center_prior:
                    delta_c = m_obj(x)
                    z_final = torch.einsum('bc,chw->bhw', c + delta_c, p)
                    z_final = apply_centerness_prior(
                        z_final, boxes * (160.0 / ishape[0]), (160, 160), margin_scale=0.5
                    )
                    bb = ops.crop_mask(
                        F.interpolate(z_final[:, None], ishape, mode='bilinear', align_corners=False)[:, 0],
                        boxes,
                    )
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.0
                else:
                    z_final, gate = m_obj.forward_inference(x, c, p)
                    if use_center_prior:
                        z_final = apply_centerness_prior(z_final, boxes * (160.0 / ishape[0]), (160, 160), margin_scale=0.5)
                    bb = ops.crop_mask(F.interpolate(z_final[:, None], ishape, mode='bilinear', align_corners=False)[:, 0], boxes)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.0
                    
                for j in range(len(c)):
                    pred = pm[j].cpu().numpy()
                    if bool(bb[j].any()):
                        rle = mu.encode(np.asfortranarray(pred.astype(np.uint8)))
                        rle['counts'] = rle['counts'].decode('ascii')
                        predictions[arm].append(dict(
                            image_id=iid,
                            category_id=categories[int(det[j, 5])],
                            score=float(det[j, 4]),
                            segmentation=rle
                        ))
        if num % 50 == 0 or num == len(transfer_ids):
            print(f"Evaluated predictions for {num}/{len(transfer_ids)} transfer images...", flush=True)
            
    summary_rows = []
    print("\n--- Transfer Results Summary ---", flush=True)
    for arm in arm_names:
        row, records, pairrecords = evaluate(gt_eval, meta, transfer_ids, predictions[arm], arm)
        row['gap'] = row['r75_low'] - row['r75_high']
        summary_rows.append(row)
        print(f"[{arm:22s}] Mask AP: {row['mask_ap']*100:.3f}% | High R75: {row['r75_high']*100:.3f}% | Low R75: {row['r75_low']*100:.3f}% | Gap: {row['gap']*100:.3f} pts", flush=True)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / 'transfer_evaluation_summary.csv', index=False)
    with open(out_dir / 'transfer_evaluation_summary.json', 'w') as f:
        json.dump(summary_rows, f, indent=2)
    return df_summary


def evaluate_s080(models_dict, normalizer, out_dir):
    print("\n=======================================================", flush=True)
    print("  Evaluating on S080 Dual-Mechanism Audit (140 Cases)  ", flush=True)
    print("=======================================================", flush=True)
    
    gt = COCO(str(ANN_JSON))
    df_s080 = pd.read_csv(S080_INSTANCES)
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        
        ann = gt.anns[aid]
        own = gt.annToMask(ann).astype(bool)
        oh, ow = own.shape
        gain = min(640 / oh, 640 / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
        gt_mask = np.zeros((640, 640), dtype=bool)
        gt_mask[top:top + nh, left:left + nw] = cv2.resize(own.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST_EXACT) > 0
        
        x = norm(instance_features(h, level, box, (640, 640)), normalizer)
        instance_res = dict(
            image_id=iid, annotation_id=aid, density=density,
            residual=residual, baseline_iou=baseline_iou
        )
        
        with torch.inference_mode():
            for arm, model_info in models_dict.items():
                m_type = model_info['type']
                m_obj = model_info['model']
                use_center_prior = model_info.get('center_prior', False)
                
                if m_type == 'original':
                    bb = ops.process_mask(proto, coeff, box, (640, 640), upsample=True)
                    pred = bb[0].cpu().numpy() > 0
                elif m_type == 's032':
                    delta_c = m_obj(x)
                    bb = ops.process_mask(proto, coeff + delta_c, box, (640, 640), upsample=True)
                    pred = bb[0].cpu().numpy() > 0
                else:
                    z_final, gate = m_obj.forward_inference(x, coeff, proto)
                    if use_center_prior:
                        z_final = apply_centerness_prior(z_final, box * (160.0 / 640.0), (160, 160), margin_scale=0.5)
                    bb = ops.crop_mask(F.interpolate(z_final[:, None], (640, 640), mode='bilinear', align_corners=False)[:, 0], box)
                    pred = bb[0].cpu().numpy() > 0.0
                    
                inter = np.count_nonzero(pred & gt_mask)
                union = np.count_nonzero(pred | gt_mask)
                iou = float(inter / max(union, 1))
                instance_res[f'{arm}_iou'] = iou
                instance_res[f'{arm}_reaches_75'] = bool(iou >= 0.75)
                
        results.append(instance_res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / 's080_strata_evaluation.csv', index=False)
    
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
        report_rows.append(row_stat)
        
    df_report = pd.DataFrame(report_rows)
    print("\n=== S080 Dual-Mechanism Breakdown Table ===", flush=True)
    print(df_report.to_string(index=False), flush=True)
    df_report.to_csv(out_dir / 's080_strata_summary.csv', index=False)
    with open(out_dir / 's080_strata_summary.json', 'w') as f:
        json.dump(report_rows, f, indent=2)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = INNOV_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    selection = json.loads((CACHE_DIR / 'selection.json').read_text())
    transfer_ids = selection['transfer']
    normalizer = torch.load(NORMALIZER_PATH, map_location=device)
    
    models = load_all_models(device)
    
    # 1. 300 Transfer images evaluation
    evaluate_transfer(models, normalizer, transfer_ids, out_dir)
    
    # 2. S080 Failure strata audit
    evaluate_s080(models, normalizer, out_dir)
    
    print(f"\nAll evaluations complete! Summaries saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
