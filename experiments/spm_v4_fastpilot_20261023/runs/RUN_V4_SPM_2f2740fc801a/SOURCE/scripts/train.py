import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from common import dump, read, run_context, sha, state_hash
from evaluate import evaluate
from model import attach, build, load, save, set_seed, train_mode


def stage_checkpoint(root, stage):
    pointer = read(root/(stage.upper()+'_POINTER.json'))
    return root/'runs'/pointer['run_id']/'best.pt'


def train_stage(root, store, mode):
    cfg = read(root/'PILOT_CONFIG.json'); is_base = mode == 'baseline'
    steps = cfg['baseline_steps'] if is_base else cfg['arm_steps']
    with run_context(root, mode) as (run, meta):
        set_seed()
        memory = None
        if is_base:
            assert sha(cfg['initial_weights']) == cfg['initial_weights_sha256']
            model = build(cfg['initial_weights'], steps)
        else:
            path = stage_checkpoint(root, 'baseline'); model = load(path, steps)
            meta['initial_checkpoint_sha256'] = sha(path)
            if mode in ['adapter', 'spm']:
                memrun = root/'runs'/read(root/'PROTOTYPE_POINTER.json')['run_id']
                memory = torch.load(memrun/'memory.pt', map_location='cpu', weights_only=False)
                attach(model, memory, mode)
        model.requires_grad_(True); train_mode(model, not is_base)
        body_before = state_hash(model, include_head=False)
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('spm.')]
        groups = [{'params': head_params, 'lr': cfg['head_lr']}]
        if hasattr(model, 'spm'):
            groups += [{'params': [p for n, p in model.spm.named_parameters() if p.requires_grad and n != 'alpha'],
                        'lr': cfg['branch_lr']}, {'params': [model.spm.alpha], 'lr': cfg['alpha_lr'], 'weight_decay': 0.}]
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        real = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'train']
        syn = [r for r in store.rows if r['domain'] != 'real' and r['split'] == 'train']
        rng = np.random.default_rng(1000 if is_base else 2000)
        schedule = []
        for _ in range(steps):
            batch = [real[int(rng.integers(len(real)))] for _ in range(3 if is_base else 2)]
            if not is_base:
                batch.append(syn[int(rng.integers(len(syn)))])
            schedule.append(batch)
        dump(run/'SAMPLE_SCHEDULE.json', [[r['source_id'] for r in b] for b in schedule])
        meta.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    frozen_body_sha256=body_before, precision='fp32', batch_size=3)
        if memory is not None:
            meta['memory_sha256'] = sha(memrun/'memory.pt')
            # Exact residual-off equality is checked before any optimization, in eval mode.
            reference = load(stage_checkpoint(root, 'baseline')).eval()
            sample = store.batch([real[0]], seed=0, augment=False)['img']
            model.eval()
            with torch.no_grad():
                actual, expected = model(sample)[0], reference(sample)[0]
                err = max((a-b).abs().max().item() for a, b in zip(actual, expected))
            assert err == 0., err
            meta['zero_residual_max_absolute_error'] = err
            del reference, actual, expected, sample
        start = time.monotonic(); log = []; evaluations = []; best = -float('inf')
        for step, rows in enumerate(schedule, 1):
            train_mode(model, not is_base); optimizer.zero_grad(set_to_none=True)
            batch = store.batch(rows, seed=step+3000, augment=True)
            losses, parts = model.loss(batch); total = losses.sum()/3
            assert torch.isfinite(total), f'nonfinite loss step {step}'
            total.backward()
            alpha_grad = float(model.spm.alpha.grad) if hasattr(model, 'spm') else None
            assert alpha_grad is None or np.isfinite(alpha_grad)
            grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.))
            assert np.isfinite(grad)
            optimizer.step()
            if hasattr(model, 'spm'):
                model.spm.alpha.data.clamp_(-.5, .5)
            model.criterion.update()
            item = {'step': step, 'loss': float(total.detach()), 'parts': parts.detach().cpu().tolist(),
                    'gradient_norm': grad, 'alpha_gradient': alpha_grad,
                    'alpha': float(model.spm.alpha.detach()) if hasattr(model, 'spm') else None,
                    'o2m_gain': float(model.criterion.o2m), 'elapsed_seconds': time.monotonic()-start}
            log.append(item)
            if step == 1 or step % 50 == 0:
                print('TRAIN', mode, step, round(item['loss'], 4), item['alpha'], flush=True)
                meta['optimizer_steps'] = step; dump(run/'run.json', meta); dump(run/'TRAINING.json', log)
            if step % 200 == 0 or step == steps:
                metric = evaluate(model, root, store, run/'evaluation', f'step_{step:04d}')
                metric['step'] = step; evaluations.append(metric)
                if metric['mask_ap'] > best:
                    best = metric['mask_ap']; selected = metric
                    save(model, run/'best.pt', mode, step, memory)
                dump(run/'EVALUATIONS.json', evaluations)
        save(model, run/'last.pt', mode, steps, memory)
        body_after = state_hash(model, include_head=False)
        if not is_base:
            assert body_after == body_before, 'Frozen upstream changed'
        summary = {'mode': mode, 'optimizer_steps': steps, 'selected': selected, 'last': evaluations[-1],
                   'selected_checkpoint_sha256': sha(run/'best.pt'), 'last_checkpoint_sha256': sha(run/'last.pt'),
                   'frozen_body_before': body_before, 'frozen_body_after': body_after,
                   'trainable_parameters': meta['trainable_parameters'], 'elapsed_seconds': time.monotonic()-start}
        meta['optimizer_steps'] = steps
        dump(run/'SUMMARY.json', summary)
        dump(run/'DECISION.json', {'status': 'evaluated_single_seed_development',
                                   'next_action': 'prototype_in_baseline_coordinates' if is_base else 'compare_all_frozen_arms',
                                   'mechanism_proven': False})
        del model, optimizer; torch.cuda.empty_cache()
    return summary
