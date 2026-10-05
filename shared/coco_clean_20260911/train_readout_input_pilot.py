"""Fit S023 small readouts on frozen real-COCO cache; keep every epoch.

No transfer labels are loaded by this script. Hyperparameters and final epoch are
fixed before execution; fitting and generalization must be reported separately.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from readout_input_probe import ARMS, Readout, inputs, objective, sha, write_json


def load_fit(cache, device):
    selection = json.loads((cache / 'selection.json').read_text())
    records, identities = [], []
    for iid in selection['fit']:
        with np.load(cache / 'images' / f'{iid}.npz') as q:
            idx = q['prediction_indices']
            if not len(idx):
                continue
            records.append(dict(h=q['h'][idx], region=q['region'][idx], level=q['level'][idx],
                                c=q['coeff'][idx], p=q['sample_p'], y=q['sample_y'], factor=q['loss_factor']))
            identities.extend((iid, int(source)) for source in q['source_index'][idx])
    if not records:
        raise RuntimeError('No fit targets; cannot run a learning probe')
    data = {k: torch.as_tensor(np.concatenate([r[k] for r in records]), device=device,
                              dtype=torch.long if k == 'level' else torch.float32) for k in records[0]}
    return data, identities


def model_input(model, data, mode, identities, seed, epoch, norm):
    if model.linear:
        return (data['h'] - norm[0]) / norm[1]
    return inputs(data['h'], data['region'], data['level'], mode, identities, seed, epoch, norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cache_receipt = json.loads((args.cache / 'COMPLETE.json').read_text())
    for name, digest in cache_receipt['hashes'].items():
        if sha(args.cache / name) != digest:
            raise RuntimeError(f'Cache changed: {name}')
    config = json.loads((args.cache / 'protocol.json').read_text())['config']
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'source').mkdir()
    for name in ['train_readout_input_pilot.py', 'readout_input_probe.py']:
        shutil.copy2(Path(__file__).with_name(name), args.out / 'source' / name)
    data, identities = load_fit(args.cache, device)
    norm = (data['h'].mean(0), data['h'].std(0, unbiased=False).clamp_min(.01))
    torch.save(dict(mean=norm[0].cpu(), std=norm[1].cpu()), args.out / 'normalizer.pt')
    write_json(args.out / 'protocol.json', dict(config=config, cache_complete_sha256=sha(args.cache / 'COMPLETE.json'),
               targets=len(identities), identities=identities,
               source_hashes={p.name: sha(p) for p in (args.out / 'source').iterdir()},
               scope='Frozen cache pure sampled native640 BCE; no transfer targets read during fit.'))
    count = len(identities)
    start = time.monotonic()
    history = []
    parameters = {}
    for seed in config['seeds']:
        for mode in ARMS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = Readout(data['h'].shape[1], linear=mode == 'linear').to(device)
            parameters[mode] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            rng = np.random.default_rng(seed)
            run = args.out / f'{mode}_s{seed}'
            (run / 'checkpoints').mkdir(parents=True)
            x = model_input(model, data, mode, identities, seed, 0, norm)
            with torch.no_grad():
                if torch.count_nonzero(model(x, data['level'])):
                    raise RuntimeError('Nonzero initial residual')
            updates = 0
            for epoch in range(config['epochs']):
                x = model_input(model, data, mode, identities, seed, epoch, norm)
                order = rng.permutation(count)
                total = 0.
                for offset in range(0, count, config['batch_targets']):
                    ix = torch.as_tensor(order[offset:offset + config['batch_targets']], device=device)
                    delta = model(x[ix], data['level'][ix])
                    loss = objective(delta, data['c'][ix], data['p'][ix], data['y'][ix], data['factor'][ix])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss {mode} {seed} {epoch}')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError('Nonfinite parameters')
                    updates += 1
                    total += float(loss.detach()) * len(ix)
                row = dict(mode=mode, seed=seed, epoch=epoch + 1, updates=updates, training_sampled_bce=total / count)
                history.append(row)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                           numpy_rng=rng.bit_generator.state, epoch=epoch + 1, updates=updates,
                           mode=mode, seed=seed), run / 'checkpoints' / f'epoch{epoch+1:03d}.pt')
                if (epoch + 1) % 50 == 0:
                    print(json.dumps(row), flush=True)
            final = run / 'checkpoints' / f'epoch{config["epochs"]:03d}.pt'
            write_json(run / 'COMPLETE.json', dict(status='COMPLETE', final_sha256=sha(final), updates=updates,
                       parameters=parameters[mode], checkpoints=config['epochs']))
            write_json(args.out / 'history.json', history)
    # A separate deterministic convex diagnostic on normalized h. Ridge units
    # are this feature parameterization; no claim of matching historical decay.
    dd = {k: v if k == 'level' else v.double() for k, v in data.items()} if config.get('run_convex',True) else {}
    xx = ((data['h'] - norm[0]) / norm[1]).double() if config.get('run_convex',True) else None
    convex = []
    for ridge in ([1e-4, 1e-6] if config.get('run_convex',True) else []):
        model = Readout(xx.shape[1], linear=True, dtype=torch.float64).to(device)
        opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=500, max_eval=650,
                               tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(set_to_none=True)
            value = objective(model(xx, dd['level']), dd['c'], dd['p'], dd['y'], dd['factor'])
            value = value + ridge * sum(p.square().sum() for p in model.parameters())
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite convex objective')
            value.backward()
            return value
        initial = float(closure().detach())
        opt.step(closure)
        final = float(closure().detach())
        state = opt.state[next(iter(model.parameters()))]
        grad = float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm())
        name = f'convex_ridge_{ridge:.0e}'
        torch.save(dict(model=model.state_dict(), ridge=ridge), args.out / f'{name}.pt')
        row = dict(mode=name, ridge=ridge, objective_initial=initial, objective_final=final,
                   gradient_norm=grad, iterations=state.get('n_iter', 0), evaluations=state.get('func_evals', 0),
                   hit_iteration_cap=state.get('n_iter', 0) >= 500,
                   scope='Finite convex solve on fit sampled BCE; one start, no global optimality certificate')
        convex.append(row)
        print(json.dumps(row), flush=True)
    write_json(args.out / 'convex.json', convex)
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', neural_runs=len(ARMS)*len(config['seeds']),
               convex_runs=len(convex), parameters=parameters, seconds=time.monotonic()-start,
               checkpoints=config['epochs']*len(ARMS)*len(config['seeds']),
               hashes={str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
