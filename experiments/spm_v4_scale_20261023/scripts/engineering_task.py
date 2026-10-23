import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import dump, read, sha, state_hash
from data import Store
from proto_data import sample_episode
from proto_train import task_loss
from task_context import task_context
from task_model import Projector, augment, load, memory_module, query_features, save, train_mode


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(4); torch.manual_seed(0)
    source = Path('/srv/gpu-lab/users/a25023104/projects/spm-v4-fastpilot-20261023/runs/RUN_V4_BASELINE_013ffd51f526/best.pt')
    with task_context(root, 'engineering_task_cpu', freeze='ENGINEERING_TASK_FREEZE.json') as (run, meta):
        model = load(source, steps=2, device='cpu').eval()
        rows = [r for r in read(root/'REAL_DATASET.json')['rows'] if r['split'] == 'train'][:2]
        store = Store(rows); batch = store.batch(rows, 0, augment=False, device='cpu')
        captured = {}
        h1 = model.model[2].register_forward_hook(lambda layer, inputs, output: captured.update(detail=output.detach()))
        h2 = model.model[-1].register_forward_pre_hook(lambda layer, inputs: captured.update(semantic=inputs[0][0].detach()))
        with torch.no_grad():
            native_det, native_proto = model(batch['img'])[0]
            reference_query = torch.cat([captured['detail'], F.interpolate(captured['semantic'], (160, 160), mode='bilinear', align_corners=False)], dim=1)
            fast_query = query_features(model, batch['img'])
            query_error = float((fast_query-reference_query).abs().max())
        h1.remove(); h2.remove(); assert query_error == 0., query_error
        projector = Projector(); memory = {'mean': torch.zeros(512), 'std': torch.ones(512),
                                           'projector': projector.state_dict(), 'centers': F.normalize(torch.randn(32, 128), dim=1)}
        augment(model, memory, 'task_spm'); model.eval()
        with torch.no_grad():
            det, proto = model(batch['img'])[0]
            assert torch.equal(det[:, :, :6], native_det[:, :, :6])
            assert torch.equal(det[:, :, 6:38], native_det[:, :, 6:])
            assert torch.equal(proto[:, :32], native_proto)
            errors = []
            for i in range(len(rows)):
                old_logits = native_det[i, :20, 6:]@native_proto[i].flatten(1)
                new_logits = det[i, :20, 6:]@proto[i].flatten(1)
                errors.append(float((old_logits-new_logits).abs().max()))
        assert max(errors) < 1e-4
        body = state_hash(model, include_head=False); frozen = memory_module(model).frozen_hash()
        train_mode(model, mask_only=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
        gradients = []
        for step in range(2):
            train_mode(model, True); optimizer.zero_grad()
            values, _ = model.loss(batch); loss = values.sum()/len(rows); assert torch.isfinite(loss)
            loss.backward()
            coefficient_gradient = sum(float(level[-1].extra.weight.grad.norm()) for name in ['cv4', 'one2one_cv4'] for level in getattr(model.model[-1], name) if level[-1].extra.weight.grad is not None)
            mixer_gradient = float(memory_module(model).mix.weight.grad.norm())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.); optimizer.step()
            gradients.append({'step': step+1, 'loss': float(loss.detach()), 'extra_coefficient_grad_norm': coefficient_gradient,
                              'prototype_basis_mixer_grad_norm': mixer_gradient})
        assert gradients[0]['extra_coefficient_grad_norm'] > 0 and gradients[1]['prototype_basis_mixer_grad_norm'] > 0
        assert state_hash(model, include_head=False) == body and memory_module(model).frozen_hash() == frozen
        save(model, run/'engineering_only.pt', 'task_spm', 2, memory)
        model.eval(); reloaded = load(run/'engineering_only.pt', device='cpu').eval()
        with torch.no_grad():
            first, second = model(batch['img'])[0], reloaded(batch['img'])[0]
            reload_error = max(float((a-b).abs().max()) for a, b in zip(first, second))
        assert reload_error == 0.
        # Validate instance-mask alignment gradients using real image features and true IDs.
        with torch.no_grad():
            aa = query_features(reloaded, batch['img']).permute(0, 2, 3, 1).reshape(2, 25600, 512)
            bb = query_features(reloaded, (batch['img']*.9+.02).clamp(0, 1)).permute(0, 2, 3, 1).reshape(2, 25600, 512)
        a, b, labels, target, episode_id, pairs = [], [], [], [], [], []
        rng = np.random.default_rng(3)
        for i in range(16):
            episode = sample_episode(store.get(rows[i % 2]), rng); positions = episode['indices']
            a.append(aa[i % 2, positions]); b.append(bb[i % 2, positions]); labels.extend(episode['labels'])
            target.extend(episode['targets'].tolist()); episode_id.extend([i]*len(positions)); pairs.extend((i*9+x, i*9+y) for x, y in episode['contact_pairs'])
        projector = Projector(); fg_head, contact_head = nn.Linear(128, 1), nn.Linear(128, 1)
        aux_loss, pieces = task_loss(projector, fg_head, contact_head, torch.cat(a), torch.cat(b), torch.tensor(labels),
                                     torch.tensor(target), torch.tensor(episode_id), torch.tensor(pairs).reshape(-1, 2),
                                     read(root/'TASK_CONFIG_V3.json'), torch.tensor(10.))
        assert torch.isfinite(aux_loss); aux_loss.backward(); auxiliary_grad = float(projector.first.weight.grad.norm()); assert auxiliary_grad > 0
        meta.update(optimizer_steps=2, baseline_for_engineering_only_sha256=sha(source))
        dump(run/'SUMMARY.json', {'query_subgraph_max_error': query_error, 'initial_mask_logit_errors': errors,
                                  'initial_box_class_score_identity': True, 'gradients': gradients,
                                  'reload_max_error': reload_error, 'frozen_upstream_and_memory_unchanged': True,
                                  'true_instance_alignment_loss': float(aux_loss.detach()), 'alignment_gradient_norm': auxiliary_grad,
                                  'alignment_terms': {k: float(v.detach()) for k, v in pieces.items()},
                                  'memory_is_random_engineering_placeholder': True, 'formal_method_training_steps': 0})
        dump(run/'DECISION.json', {'status': 'engineering_passed', 'next_action': 'full_baseline_coordinates_task_prototype_training',
                                   'five_point_target_achieved': False})


if __name__ == '__main__':
    main()
