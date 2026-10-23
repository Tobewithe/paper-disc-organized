import math
import time

import numpy as np
import torch

from base_model import build, save, set_seed
from common import dump, read, run_context, sha
from evaluate import evaluate


def lr_fraction(step, total, warmup, minimum=.02):
    if step <= warmup:
        return .1+.9*step/warmup
    progress = min(1., (step-warmup)/max(1, total-warmup))
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))


def smoke(root, store):
    cfg = read(root/'PILOT_CONFIG.json')
    with run_context(root, 'engineering_baseline') as (run, meta):
        model = build(cfg['initial_weights'], 2); model.requires_grad_(True); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
        rows = [r for r in store.rows if r['split'] == 'train'][:12]; batch = store.batch(rows, 0)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                parts, _ = model.loss(batch); loss = parts.sum()/len(rows)
            assert torch.isfinite(loss); loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
            assert np.isfinite(gradient); optimizer.step(); model.criterion.update(); losses.append(float(loss.detach()))
        meta['optimizer_steps'] = 2
        dump(run/'SUMMARY.json', {'losses': losses, 'training_precision': 'bfloat16_autocast', 'parameters': 'fp32',
                                  'gradient_norm': gradient, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'steps_are_engineering_only': True})
        dump(run/'DECISION.json', {'status': 'smoke_passed', 'next_action': 'full_real_baseline_12720_steps'})
        del model, optimizer, batch, loss, parts; torch.cuda.empty_cache()


def train(root, store):
    cfg = read(root/'PILOT_CONFIG.json')
    with run_context(root, 'baseline') as (run, meta):
        set_seed(); assert sha(cfg['initial_weights']) == cfg['initial_weights_sha256']
        steps = cfg['baseline_steps']; model = build(cfg['initial_weights'], steps); model.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['baseline_peak_lr'], weight_decay=cfg['weight_decay'])
        real = [r for r in store.rows if r['split'] == 'train']; assert len(real) == 1272
        rng = np.random.default_rng(231023); schedule = []
        for epoch in range(cfg['baseline_epochs']):
            order = rng.permutation(len(real))
            schedule.extend([[real[int(i)] for i in order[j:j+12]] for j in range(0, len(real), 12)])
        assert len(schedule) == steps
        dump(run/'SAMPLE_SCHEDULE.json', [[r['source_id'] for r in batch] for batch in schedule])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        meta.update(trainable_parameters=trainable, batch_size=12, training_precision='bfloat16_autocast', evaluation_precision='fp32')
        best = -float('inf'); evaluations = []; log = []; start = time.monotonic()
        for step, rows in enumerate(schedule, 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            factor = lr_fraction(step, steps, cfg['baseline_warmup_steps'])
            optimizer.param_groups[0]['lr'] = cfg['baseline_peak_lr']*factor
            batch = store.batch(rows, 10000+step, augment=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values, pieces = model.loss(batch); total = values.sum()/12
            assert torch.isfinite(total), step
            total.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)); assert np.isfinite(gradient)
            optimizer.step(); model.criterion.update()
            log.append({'step': step, 'epoch': step/106, 'images_seen': step*12, 'loss': float(total.detach()),
                        'parts': pieces.detach().cpu().tolist(), 'lr': optimizer.param_groups[0]['lr'],
                        'gradient_norm': gradient, 'elapsed_seconds': time.monotonic()-start})
            if step == 1 or step % 100 == 0:
                meta.update(optimizer_steps=step, epoch=step/106, images_seen=step*12)
                dump(run/'run.json', meta); dump(run/'TRAINING.json', log)
                print('BASELINE', step, steps, 'epoch', round(step/106, 2), 'loss', round(log[-1]['loss'], 4), flush=True)
            if step % cfg['baseline_evaluation_interval'] == 0 or step == steps:
                metric = evaluate(model, root, store, run/'evaluation', f'step_{step:05d}'); metric.update(step=step, epoch=step/106)
                evaluations.append(metric)
                if metric['mask_ap'] > best:
                    best = metric['mask_ap']; selected = metric; save(model, run/'best.pt', 'baseline', step)
                save(model, run/'last.pt', 'baseline', step)
                torch.save({'model_checkpoint': 'last.pt', 'optimizer': optimizer.state_dict(), 'step': step,
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, run/'TRAIN_STATE.pt')
                dump(run/'EVALUATIONS.json', evaluations)
                dump(run/'LIVE_SUMMARY.json', {'selected': selected, 'last': metric, 'steps_completed': step, 'complete': step == steps})
        meta.update(optimizer_steps=steps, epoch=120, images_seen=steps*12)
        dump(run/'SUMMARY.json', {'mode': 'baseline', 'optimizer_steps': steps, 'epochs': 120, 'images_seen': steps*12,
                                  'selected': selected, 'last': evaluations[-1], 'trainable_parameters': trainable,
                                  'best_checkpoint_sha256': sha(run/'best.pt'), 'last_checkpoint_sha256': sha(run/'last.pt'),
                                  'elapsed_seconds': time.monotonic()-start})
        dump(run/'DECISION.json', {'status': 'baseline_completed', 'next_action': 'task_mask_supervised_prototypes_in_frozen_coordinates',
                                   'prototype_utility_claim': False})
        del model, optimizer, batch, total, values; torch.cuda.empty_cache()
