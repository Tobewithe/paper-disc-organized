from collections import OrderedDict
import time

import cv2
import numpy as np
import torch

from common import dump, read, sha, state_hash
from data import Store
from evaluate import evaluate
from proto_train import checkpoint
from task_context import task_context
from task_model import augment, load, memory_module, save, train_mode
from train_baseline import lr_fraction


class TrainingStore(Store):
    """Avoid decoding full 640 ID/overlap maps in native training batches."""
    def __init__(self, rows):
        super().__init__(rows, max_cache=1536)

    def get(self, row):
        key = row['source_id']
        if key not in self.cache:
            with np.load(row['label']) as label:
                self.cache[key] = {'rgb': cv2.imread(row['image']), 'masks': label['masks'], 'boxes': label['boxes']}
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def schedule(rows, steps):
    rng = np.random.default_rng(23102331)
    pools = {d: [r for r in rows if r['split'] == 'train' and r['domain'] == d] for d in ['real', 'cutout', 'synthetic']}
    shuffled = {d: [] for d in pools}; cursors = {d: 0 for d in pools}
    def draw(domain, n):
        result = []
        for _ in range(n):
            if cursors[domain] >= len(shuffled[domain]):
                shuffled[domain] = rng.permutation(len(pools[domain])).tolist(); cursors[domain] = 0
            result.append(pools[domain][shuffled[domain][cursors[domain]]]); cursors[domain] += 1
        return result
    return [draw('real', 8)+draw('cutout', 2)+draw('synthetic', 2) for _ in range(steps)]


def make_model(root, mode, steps):
    model = load(checkpoint(root, 'baseline'), steps=steps)
    memory = None
    if mode != 'continued':
        memrun = root/'runs'/read(root/'PROTOTYPE_FINAL_POINTER.json')['run_id']
        memory = torch.load(memrun/'memory.pt', map_location='cpu', weights_only=False)
        assert memory['upstream_checkpoint_sha256'] == sha(checkpoint(root, 'baseline'))
        augment(model, memory, mode)
    return model, memory


def smoke(root):
    cfg = read(root/'TASK_CONFIG_V3.json')
    with task_context(root, 'engineering_task_gpu') as (run, meta):
        torch.manual_seed(0); model, memory = make_model(root, 'task_spm', 2)
        store = TrainingStore(read(root/'ALL_DATASET.json')['rows']); rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'train'][:12]
        train_mode(model, True); body = state_hash(model, False); bank_hash = memory_module(model).frozen_hash()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
        gradients = []
        for step in range(2):
            batch = store.batch(rows, 0); optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values, _ = model.loss(batch); loss = values.sum()/12
            assert torch.isfinite(loss); loss.backward()
            extra = sum(float(level[-1].extra.weight.grad.norm()) for name in ['cv4', 'one2one_cv4'] for level in getattr(model.model[-1], name) if level[-1].extra.weight.grad is not None)
            mixing = float(memory_module(model).mix.weight.grad.norm())
            assert np.isfinite(extra+mixing)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.); optimizer.step()
            gradients.append({'step': step+1, 'loss': float(loss.detach()), 'coefficient_gradient_norm': extra, 'basis_mixer_gradient_norm': mixing})
        assert gradients[0]['coefficient_gradient_norm'] > 0 and gradients[1]['basis_mixer_gradient_norm'] > 0
        assert state_hash(model, False) == body and memory_module(model).frozen_hash() == bank_hash
        meta['optimizer_steps'] = 2
        dump(run/'SUMMARY.json', {'gradients': gradients, 'actual_learned_memory': True, 'precision': 'bfloat16_autocast',
                                  'frozen_coordinates_and_memory_unchanged': True, 'formal_training_steps': 0})
        dump(run/'DECISION.json', {'status': 'gpu_smoke_passed', 'next_action': 'matched_four_arm_training'})


def train(root, mode):
    cfg = read(root/'TASK_CONFIG_V3.json')
    with task_context(root, mode) as (run, meta):
        torch.manual_seed(0); steps = cfg['arm_steps']; model, memory = make_model(root, mode, steps)
        store = TrainingStore(read(root/'ALL_DATASET.json')['rows']); batches = schedule(store.rows, steps)
        dump(run/'SAMPLE_SCHEDULE.json', [[r['source_id'] for r in b] for b in batches])
        train_mode(model, mask_only=False)
        native_params, branch_params = [], []
        for name, parameter in model.model[-1].named_parameters():
            if not parameter.requires_grad:
                continue
            (branch_params if '.extra.' in name or name.startswith('proto.memory.mix.') else native_params).append(parameter)
        groups = [{'params': native_params, 'lr': cfg['head_peak_lr'], 'peak_lr': cfg['head_peak_lr']}]
        if branch_params:
            groups.append({'params': branch_params, 'lr': cfg['branch_peak_lr'], 'peak_lr': cfg['branch_peak_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        body = state_hash(model, False); module = memory_module(model); frozen = module.frozen_hash() if module else None
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        meta.update(initial_checkpoint_sha256=sha(checkpoint(root, 'baseline')), full_head_trainable_parameters=parameters,
                    added_path_trainable_parameters=sum(p.numel() for p in branch_params), training_precision='bfloat16_autocast',
                    frozen_body_sha256=body, frozen_memory_sha256=frozen, batch_size=12)
        if memory is not None:
            meta['memory_sha256'] = sha(root/'runs'/read(root/'PROTOTYPE_FINAL_POINTER.json')['run_id']/'memory.pt')
        best = -float('inf'); evaluations = []; trace = []; started = time.monotonic()
        for step, rows in enumerate(batches, 1):
            warm = step <= cfg['mask_path_only_warmup_steps']; train_mode(model, mask_only=warm)
            for group in optimizer.param_groups:
                group['lr'] = group['peak_lr']*lr_fraction(step, steps, cfg['lr_warmup_steps'])
            optimizer.zero_grad(set_to_none=True); batch = store.batch(rows, 20000+step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values, pieces = model.loss(batch); loss = values.sum()/12
            assert torch.isfinite(loss), step
            loss.backward()
            extra_gradient = sum(float(p.grad.detach().norm()) for p in branch_params if p.grad is not None)
            grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.)); assert np.isfinite(grad)
            optimizer.step(); model.criterion.update()
            trace.append({'step': step, 'loss': float(loss.detach()), 'parts': pieces.detach().cpu().tolist(),
                          'gradient_norm': grad, 'added_path_gradient_norm': extra_gradient, 'mask_only': warm,
                          'native_lr': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.monotonic()-started})
            if step == 1 or step % 100 == 0:
                print('METHOD', mode, step, steps, round(trace[-1]['loss'], 4), 'branch_grad', round(extra_gradient, 5), flush=True)
                meta.update(optimizer_steps=step, images_seen=step*12, real_images_seen=step*8)
                dump(run/'run.json', meta); dump(run/'TRAINING.json', trace)
            if step % cfg['evaluation_interval'] == 0 or step == steps:
                metric = evaluate(model, root, store, run/'evaluation', f'step_{step:05d}'); metric['step'] = step; evaluations.append(metric)
                if metric['mask_ap'] > best:
                    best = metric['mask_ap']; selected = metric; save(model, run/'best.pt', mode, step, memory)
                save(model, run/'last.pt', mode, step, memory)
                torch.save({'model_checkpoint': 'last.pt', 'optimizer': optimizer.state_dict(), 'step': step,
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, run/'TRAIN_STATE.pt')
                dump(run/'EVALUATIONS.json', evaluations); dump(run/'LIVE_SUMMARY.json', {'selected': selected, 'last': metric, 'steps_completed': step})
                assert state_hash(model, False) == body
                if module:
                    assert module.frozen_hash() == frozen
        meta.update(optimizer_steps=steps, images_seen=steps*12, real_images_seen=steps*8)
        dump(run/'SUMMARY.json', {'mode': mode, 'optimizer_steps': steps, 'selected': selected, 'last': evaluations[-1],
                                  'full_head_trainable_parameters': parameters, 'added_path_trainable_parameters': sum(p.numel() for p in branch_params),
                                  'frozen_body_before': body, 'frozen_body_after': state_hash(model, False),
                                  'frozen_memory_before': frozen, 'frozen_memory_after': module.frozen_hash() if module else None,
                                  'best_checkpoint_sha256': sha(run/'best.pt'), 'last_checkpoint_sha256': sha(run/'last.pt'),
                                  'sample_schedule_sha256': sha(run/'SAMPLE_SCHEDULE.json'), 'elapsed_seconds': time.monotonic()-started})
        dump(run/'DECISION.json', {'status': 'method_arm_evaluated', 'next_action': 'four_arm_comparison_and_prototype_interventions',
                                   'five_point_goal_status': 'pending_all_arms'})
