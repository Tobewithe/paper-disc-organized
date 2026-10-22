import torch
from torch.nn import functional as F

from common import dump, read, run_context, state_hash
from model import attach, build, train_mode, set_seed


def smoke(root, store):
    cfg = read(root/'PILOT_CONFIG.json')
    with run_context(root, 'engineering') as (run, meta):
        model = build(cfg['initial_weights'], 2)
        rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'train'][:3]
        batch = store.batch(rows, seed=0, augment=False)
        model.eval()
        with torch.no_grad():
            before = tuple(x.clone() for x in model(batch['img'])[0])
        set_seed()
        memory = {'mean': torch.zeros(256), 'std': torch.ones(256), 'projector_weight': torch.randn(128, 256)*.05,
                  'projector_bias': torch.zeros(128), 'centers': F.normalize(torch.randn(32, 128), dim=1)}
        attach(model, memory, 'spm'); model.eval()
        with torch.no_grad():
            after = model(batch['img'])[0]
            difference = max(float((a-b).abs().max()) for a, b in zip(before, after))
        assert difference == 0., difference
        train_mode(model, True); body = state_hash(model, include_head=False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        gradients = []
        for step in range(2):
            train_mode(model, True); opt.zero_grad()
            losses, _ = model.loss(batch); loss = losses.sum()/3
            assert torch.isfinite(loss); loss.backward()
            alpha_grad = float(model.spm.alpha.grad)
            readout_grad = float(model.spm.readout.weight.grad.norm())
            gradients.append({'step': step+1, 'loss': float(loss.detach()),
                              'alpha_gradient': alpha_grad, 'readout_gradient_norm': readout_grad})
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.)
            opt.step(); model.criterion.update()
        assert gradients[0]['alpha_gradient'] != 0. and gradients[1]['readout_gradient_norm'] > 0, gradients
        assert state_hash(model, include_head=False) == body
        meta['optimizer_steps'] = 2
        dump(run/'SUMMARY.json', {'zero_residual_error': difference, 'gradients': gradients, 'body_unchanged': True,
                                  'steps_are_engineering_only': True})
        dump(run/'DECISION.json', {'status': 'smoke_passed', 'next_action': 'baseline_600_steps'})
        del model, opt, before, after, batch, loss, losses; torch.cuda.empty_cache()
