"""Training and evaluation pipeline for Bifurcated Spatial-Readout Head (BSR-Head).

Adheres strictly to RESEARCH_FACTS.md protocols:
- 1,200 fit images, 7,811 targets from audited S024/S032 cache
- raw COCO independent binary labels
- 3 random seeds (0, 1, 2), 15 epochs each, batch 32 targets, Adam lr 1e-4
- Evaluated on 300 transfer images (2,002 ordinary GTs) with pycocotools COCOeval + ICI stratification
- Evaluated on S080 diagnostic queue (140 instances) for dual-mechanism breakdown
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import contextlib
import io
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops

from bifurcated_spatial_readout import BifurcatedSpatialHead, bsr_loss
from rich_pixel_readout import instance_features, GlobalHead
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'diagnostics/readout_input_scale1200_20260912/cache'
LABELS_DIR = ROOT / 'diagnostics/shared_label_controls_20260912/labels'
NORMALIZER_PATH = ROOT / 'diagnostics/shared_label_controls_20260912/normalizer.pt'
S032_DIR = ROOT / 'diagnostics/shared_label_controls_20260912'
S080_INSTANCES = ROOT / 'diagnostics/fixed_prototype_split_20260913_v5/instances.csv'
S080_TENSOR_CACHE = ROOT / 'diagnostics/joint_failure_decoder_20260913_v2/tensor_cache'


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def norm(x, n):
    return (x - n['mean']) / n['std']


def read_np(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_fit_dataset(fit_ids):
    print("Loading 1,200 fit images dataset...", flush=True)
    normalizer = torch.load(NORMALIZER_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    records = []
    identities = []
    
    for num, iid in enumerate(fit_ids, 1):
        path = CACHE_DIR / 'images' / f'{iid}.npz'
        if not path.exists():
            continue
        item = read_np(path)
        idx = item['prediction_indices']
        if not len(idx):
            continue
        lab_path = LABELS_DIR / f'{iid}.npz'
        if not lab_path.exists():
            continue
        lab_item = np.load(lab_path)
        yc = lab_item['raw_coco']  # [N_targets, 512]
        
        hfeat = cuda(torch.from_numpy(item['h'])).float()
        lv = cuda(torch.from_numpy(item['level'])).long()
        boxes = cuda(torch.from_numpy(item['boxes'])).float()
        ishape = tuple(map(int, item['input_shape']))
        x = instance_features(hfeat, lv, boxes, ishape)[idx]
        
        records.append(dict(
            x=x.cpu(),
            p=torch.tensor(item['sample_p'][:, :512]).float(),
            c=torch.tensor(item['coeff'][idx]).float(),
            factor=torch.tensor(item['loss_factor']).float(),
            raw_coco=torch.tensor(yc).float()
        ))
        identities.extend((iid, int(item['source_index'][j])) for j in idx)
        if num % 300 == 0:
            print(f"Loaded {num}/{len(fit_ids)} images...", flush=True)
            
    data = {k: torch.cat([r[k] for r in records]) for k in records[0]}
    if torch.cuda.is_available():
        data = {k: v.cuda() for k, v in data.items()}
    data['xn'] = norm(data['x'], normalizer)
    print(f"Dataset ready. Total targets: {len(identities)}", flush=True)
    return data, normalizer, identities


def train_bsr_head(data, seed, out_dir, epochs=15, batch_size=32, lr=1e-4, lambda_spatial=0.001):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = BifurcatedSpatialHead(in_dim=73, p_dim=32, hidden=128)
    if torch.cuda.is_available():
        model = model.cuda()
        
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    dest = out_dir / f'bsr_head_s{seed}'
    (dest / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    count = len(data['xn'])
    history = []
    start_time = time.monotonic()
    
    for epoch in range(epochs):
        order = rng.permutation(count)
        total_loss = 0.0
        total_task_loss = 0.0
        total_reg_loss = 0.0
        updates = 0
        model.train()
        
        for first in range(0, count, batch_size):
            ix = cuda(torch.as_tensor(order[first:first + batch_size], dtype=torch.long))
            z_final, info = model.forward_sampled(data['xn'][ix], data['c'][ix], data['p'][ix])
            loss, loss_dict = bsr_loss(z_final, data['raw_coco'][ix], data['factor'][ix], info, lambda_spatial=lambda_spatial)
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, seed {seed}")
                
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            opt.step()
            
            total_loss += float(loss.detach()) * len(ix)
            total_task_loss += loss_dict['task_loss'] * len(ix)
            total_reg_loss += loss_dict['spatial_reg'] * len(ix)
            updates += 1
            
        row = dict(
            seed=seed, epoch=epoch + 1, updates=updates,
            loss=total_loss / count, task_loss=total_task_loss / count,
            spatial_reg=total_reg_loss / count,
            mean_gate=float(info['gate'].mean().detach().cpu().item())
        )
        history.append(row)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Seed {seed} | Epoch {epoch+1:02d}/{epochs:02d} | Loss: {row['loss']:.4f} (Task: {row['task_loss']:.4f}, Reg: {row['spatial_reg']:.4f}) | Gate: {row['mean_gate']:.3f}", flush=True)
            
        torch.save(dict(
            model=model.state_dict(), optimizer=opt.state_dict(),
            epoch=epoch + 1, seed=seed
        ), dest / 'checkpoints' / f'epoch{epoch+1:03d}.pt')
        
    torch.save(dict(
        model=model.state_dict(), seed=seed
    ), dest / 'checkpoints' / 'epoch015.pt')
    
    with open(dest / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    model.eval()
    return model


def evaluate_transfer_dataset(transfer_ids, models_dict, normalizer, out_dir):
    print("\n--- Starting Full Transfer Evaluation (300 images) ---", flush=True)
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
    gate_records = []
    
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
                
                if m_type == 'original':
                    # Baseline Ultralytics
                    bb = ops.process_mask(p, c, boxes, ishape, upsample=True)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.5
                elif m_type == 's032':
                    # S032 pure coefficient head
                    delta_c = m_obj(x)
                    bb = ops.process_mask(p, c + delta_c, boxes, ishape, upsample=True)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.5
                elif m_type == 'bsr_head':
                    # BSR-Head dual stream
                    z_final, gate = m_obj.forward_inference(x, c, p)
                    bb = ops.crop_mask(F.interpolate(z_final[:, None], ishape, mode='bilinear', align_corners=False)[:, 0], boxes)
                    pm = ops.scale_masks(bb[:, None], shape)[:, 0] > 0.0
                    for g_val in gate.cpu().numpy().flatten():
                        gate_records.append(dict(arm=arm, image_id=iid, gate=float(g_val)))
                        
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
    for arm in arm_names:
        row, records, pairrecords = evaluate(gt_eval, meta, transfer_ids, predictions[arm], arm)
        row['gap'] = row['r75_low'] - row['r75_high']
        summary_rows.append(row)
        print(f"Results [{arm}]: Mask AP: {row['mask_ap']*100:.3f}% | High R75: {row['r75_high']*100:.3f}% | Low R75: {row['r75_low']*100:.3f}% | Gap: {row['gap']*100:.3f} pts", flush=True)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / 'transfer_evaluation_summary.csv', index=False)
    with open(out_dir / 'transfer_evaluation_summary.json', 'w') as f:
        json.dump(summary_rows, f, indent=2)
    return df_summary


def evaluate_s080_strata(models_dict, normalizer, out_dir):
    print("\n--- Starting S080 Failure Strata Mechanism Audit (140 instances) ---", flush=True)
    if not S080_INSTANCES.exists():
        print(f"Warning: S080 instances file {S080_INSTANCES} not found, skipping S080 audit.", flush=True)
        return
        
    ann_json = ROOT.parent.parent / 'datasets/coco/annotations/instances_val2017.json'
    if not ann_json.exists():
        print(f"Warning: {ann_json} not found, skipping S080 audit.", flush=True)
        return
        
    gt = COCO(str(ann_json))
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
    df_res.to_csv(out_dir / 's080_strata_evaluation.csv', index=False)
    
    # Stratified breakdown
    print("\n=== S080 Dual-Mechanism Breakdown Table ===", flush=True)
    report_rows = []
    for residual_type in ['requires_target_pixel_recovery', 'residual_false_pixels', 'all']:
        sub = df_res if residual_type == 'all' else df_res[df_res['residual'] == residual_type]
        row_stat = dict(residual=residual_type, n=len(sub))
        for arm in models_dict:
            reaches = int(sub[f'{arm}_reaches_75'].sum()) if f'{arm}_reaches_75' in sub else 0
            mean_iou = float(sub[f'{arm}_iou'].mean()) * 100.0 if f'{arm}_iou' in sub else 0.0
            row_stat[f'{arm}_reaches_75'] = reaches
            row_stat[f'{arm}_mean_iou'] = mean_iou
        report_rows.append(row_stat)
        
    df_report = pd.DataFrame(report_rows)
    print(df_report.to_string(index=False), flush=True)
    df_report.to_csv(out_dir / 's080_strata_summary.csv', index=False)
    with open(out_dir / 's080_strata_summary.json', 'w') as f:
        json.dump(report_rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BSR-Head")
    parser.add_argument('--seeds', type=str, default='0,1,2', help="Comma-separated seeds")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size in targets")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lambda-spatial', type=float, default=0.001, help="Spatial L2 regularization weight")
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/bsr_head_experiment', help="Output directory")
    args = parser.parse_args()
    
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    
    selection = json.loads((CACHE_DIR / 'selection.json').read_text())
    fit_ids = selection['fit']
    transfer_ids = selection['transfer']
    
    # 1. Load fit data
    data, normalizer, identities = load_fit_dataset(fit_ids)
    
    # 2. Train BSR-Head for each seed
    bsr_models = {}
    for seed in seeds:
        print(f"\n>>> Training BSR-Head with seed {seed} <<<", flush=True)
        m = train_bsr_head(data, seed, out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lambda_spatial=args.lambda_spatial)
        bsr_models[f'bsr_head_s{seed}'] = m
        
    # 3. Assemble model suite for evaluation
    models_dict = {
        'original': {'type': 'original', 'model': None}
    }
    # Load S032 checkpoints
    for seed in seeds:
        s032_ckpt = S032_DIR / f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if s032_ckpt.exists():
            s032_model = GlobalHead().cuda() if torch.cuda.is_available() else GlobalHead()
            s032_model.load_state_dict(torch.load(s032_ckpt, weights_only=False)['model'])
            s032_model.eval()
            models_dict[f's032_raw_s{seed}'] = {'type': 's032', 'model': s032_model}
        models_dict[f'bsr_head_s{seed}'] = {'type': 'bsr_head', 'model': bsr_models[f'bsr_head_s{seed}']}
        
    # 4. Transfer evaluation (300 images)
    evaluate_transfer_dataset(transfer_ids, models_dict, normalizer, out_dir)
    
    # 5. S080 failure strata audit (140 instances)
    evaluate_s080_strata(models_dict, normalizer, out_dir)
    
    print(f"\nAll experiments complete! Results saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
