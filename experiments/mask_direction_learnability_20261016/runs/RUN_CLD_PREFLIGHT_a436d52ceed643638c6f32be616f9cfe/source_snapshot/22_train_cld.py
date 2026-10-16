"""A single explicit arm. Smoke never chains into formal training."""
import argparse
import copy
import json
import torch
from cld_common import *
from cld_math import context, target, remap, loss, deploy_coefficients, probe_coefficients
from adapter_data import FrozenCache
from adapter_model import CoefficientAdapter
from native_output_bridge import decode_native, encode_rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('smoke', 'full'), required=True)
    parser.add_argument('--arm', choices=('REAL', 'PERM'), required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--smoke-real', type=Path)
    parser.add_argument('--smoke-perm', type=Path)
    args = parser.parse_args()
    preflight = args.preflight.resolve()
    check_completed(preflight)
    authorization = read_json(STUDY / 'OPERATION_AUTHORIZATION.json')
    require(authorization['allow_smoke'], 'smoke not authorized')
    full = args.scope == 'full'
    if full:
        require(authorization.get('allow_formal_training') is True, 'formal training is not authorized by the current operation')
        for path, expected_arm in ((args.smoke_real, 'REAL'), (args.smoke_perm, 'PERM')):
            require(path is not None, 'both smoke arms must be specified')
            check_completed(path.resolve())
            c = read_json(path / 'EVAL_CONTRACT.json')
            require(c['arm'] == expected_arm and c['scope'] == 'smoke', 'wrong smoke dependency')
        for old in (STUDY / 'runs').glob('RUN_CLD_FULL_' + args.arm + '_*/run.json'):
            require(not read_json(old).get('optimization_started'), 'formal arm budget already consumed')
    with Run(args.scope.upper() + '_' + args.arm, preflight, 7200 if full else 900) as run:
        data = inspect_inputs()
        ids = data['split']['fit_image_ids'] if full else data['split']['fit_image_ids'][:8]
        orders = read_json(STUDY / 'IMAGE_ORDERS.json')['epochs'] if full else [ids]
        require(not set(ids) & set(data['split']['heldout_image_ids']), 'heldout image in training')
        amplitude = read_json(preflight / 'AMPLITUDE.json')['value']
        contract = copy.deepcopy(data['inherited']['source_training_contract'])
        contract.update(stage='CLD_training', scope=args.scope, arm=args.arm, architecture_arm='B', epochs=len(orders), fit_images=len(ids),
                        expected_optimizer_steps=1200 if full else 2, expected_image_visits=4800 if full else 8,
                        split_sha256=sha256(STUDY / 'SPLIT.json'), condition_map_sha256=sha256(STUDY / 'CONDITION_MAP.json'),
                        amplitude=amplitude, amplitude_sha256=sha256(preflight / 'AMPLITUDE.json'), preflight_run=preflight.name,
                        original_initial_sha256=sha256(INITIAL / 'initial.pt'), objective='CLD-v1 functional-direction MSE + .1 native-probability BCE',
                        paired_variable='within-image instance correspondence only', environment=data['environment'],
                        operation_authorization=authorization, scientific_gate='not_adjudicated_by_training')
        contract.pop('damage_sample_budget')
        contract['fit_dev_split_sha256'] = sha256(STUDY / 'SPLIT.json')
        write_new(run.path / 'IMAGE_ORDERS.json', orders)
        contract['image_orders_sha256'] = sha256(run.path / 'IMAGE_ORDERS.json')
        contract['order_algorithm'] = 'frozen CLD epoch lists' if full else 'original-order first 8 of frozen fit1600, one pass'
        run.freeze(contract)
        torch.manual_seed(0)
        model = CoefficientAdapter('B').cuda()
        initial = torch.load(INITIAL / 'initial.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(initial['state_dict'], strict=True)
        initial_hash = tensor_state_hash(model)
        require(initial_hash == contract['model_state_initial_sha256'], 'initial state changed')
        whitelist = [n for n, p in model.named_parameters() if p.requires_grad]
        require(whitelist == contract['trainable_whitelist'] and sum(p.numel() for p in model.parameters()) == 167872, 'architecture/whitelist changed')
        optimizer_spec = dict(contract['optimizer'])
        optimizer_spec['betas'] = tuple(optimizer_spec['betas'])
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_spec)
        actual_optimizer = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        require(canonical(actual_optimizer) == canonical(contract['optimizer']) and not optimizer.state, 'optimizer changed or resumed')
        write_new(run.path / 'ACTUAL_OPTIMIZER.json', actual_optimizer)
        shutil.copyfile(INITIAL / 'initial.pt', run.path / 'initial.pt')
        source = FrozenCache(CACHE / 'trace', ids)
        teachers = teacher_rows(ids)
        forwards, visited, steps = 0, 0, 0
        controls = []
        if not full:
            with torch.inference_mode():
                model.eval()
                for iid in ids:
                    cache, rows = source.load(iid)
                    d = model(cache, 'smoke:initial')
                    forwards += 1
                    require(int(torch.count_nonzero(d)) == 0, 'nonzero original initialization')
                    coefficients = deploy_coefficients(remap(d, data['maps'][iid], args.arm), context(cache, d.device), amplitude)
                    decoded = decode_native(cache, coefficients)
                    require(all(encode_rle(decoded[j]) == r['segmentation'] for j, r in enumerate(rows)), 'zero-delta RLE replay differs')
                    controls.append(dict(image_id=iid, output_rows=len(rows), zero_delta_RLE_exact=True))
                    del cache, rows, d, coefficients, decoded
            jsonl(run.path / 'ZERO_DELTA.jsonl', controls)
        model.train()
        run.update(optimization_started=True, phase='smoke_training' if not full else 'formal_training')
        gradient_names, step_rows = set(), []
        with (run.path / 'training.jsonl').open('x', encoding='utf-8') as log:
            for epoch, order in enumerate(orders):
                for offset in range(0, len(order), 4):
                    group = order[offset:offset + 4]
                    require(len(group) == 4, 'partial accumulation group changes budget')
                    optimizer.zero_grad(set_to_none=True)
                    for iid in group:
                        run.check_budget()
                        cache, rows = source.load(iid)
                        targets = {t['rank']: target(cache, t)[0] for t in teachers[iid]}
                        raw = model(cache, 'train:' + str(epoch))
                        forwards += 1
                        direction = remap(raw, data['maps'][iid], args.arm)
                        value, info = loss(direction, context(cache, raw.device), targets, amplitude)
                        require(torch.isfinite(value).item(), 'nonfinite objective')
                        (value / 4).backward()
                        visited += 1
                        log.write(json.dumps(dict(epoch=epoch, image_id=iid, step=steps + 1, loss=float(value.detach()),
                                                 cache_sha256=source.records[iid]['cache_sha256'], **info), allow_nan=False) + '\n')
                        del cache, rows, targets, raw, direction, value
                    grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
                    require(grads and all(torch.isfinite(g).all() for g in grads.values()), 'invalid or missing gradients')
                    gradient_names.update(n for n, g in grads.items() if torch.count_nonzero(g))
                    if not full:
                        torch.save(grads, run.path / f'gradient_{steps + 1:04d}.pt')
                    optimizer.step()
                    steps += 1
                    require(all(torch.isfinite(p).all() for p in model.parameters()), 'nonfinite parameter')
                    if not full:
                        torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), step=steps), run.path / f'step_{steps:04d}.pt')
                    step_rows.append(dict(step=steps, epoch=epoch, image_ids=group, gradient_parameter_names=sorted(grads)))
                    if steps % 25 == 0 or not full:
                        log.flush()
                        run.update(optimizer_steps=steps, images_seen=visited, adapter_forward_calls=forwards)
                        print('TRAIN', args.arm, args.scope, steps, '/', contract['expected_optimizer_steps'], flush=True)
        require(steps == contract['expected_optimizer_steps'] and visited == contract['expected_image_visits'], 'training budget differs')
        require(any(n.startswith('readout.2') for n in gradient_names), 'final head received no gradient')
        require(tensor_state_hash(model) != initial_hash, 'model did not update')
        jsonl(run.path / 'OPTIMIZER_STEPS.jsonl', step_rows)
        torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), arm=args.arm,
                        completed_epochs=len(orders), steps=steps, contract_sha256=sha256(run.path / 'EVAL_CONTRACT.json')), run.path / 'last.pt')
        if not full:
            model.eval()
            post_tensors, mask_rows = {}, []
            from pycocotools import mask as masks
            with torch.inference_mode():
                for iid in ids:
                    cache, rows = source.load(iid)
                    raw = model(cache, 'smoke:after')
                    forwards += 1
                    direction = remap(raw, data['maps'][iid], args.arm)
                    ctx = context(cache, direction.device)
                    coefficients = deploy_coefficients(direction, ctx, amplitude)
                    decoded = decode_native(cache, coefficients)
                    scale_map = {t['rank']: target(cache, t)[1] for t in teachers[iid]}
                    probe = probe_coefficients(direction, ctx, scale_map)
                    post_tensors[iid] = dict(raw=raw.cpu(), coefficients=coefficients.cpu(), probe_coefficients=probe.cpu(), teacher_scales=scale_map)
                    for rank, row in enumerate(rows):
                        gate = int(masks.area(row['segmentation'])) >= 9216
                        mixed = dict(row)
                        mixed['segmentation'] = encode_rle(decoded[rank]) if gate else row['segmentation']
                        mask_rows.append(dict(image_id=iid, global_index=cache['prediction_indices'][rank], rank=rank, gate=gate, prediction=mixed))
                    del cache, rows, raw, direction, ctx, coefficients, decoded, probe
            torch.save(post_tensors, run.path / 'SMOKE_POST.pt')
            jsonl(run.path / 'SMOKE_MASKS.jsonl', mask_rows)
        source.verify()
        inspect_inputs()
        require(canonical({k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}) == canonical(actual_optimizer), 'optimizer changed')
        run.update(optimizer_steps=steps, images_seen=visited, adapter_forward_calls=forwards, phase='configuration_checked')
        write_new(run.path / 'CONFIG_AUDIT.json', dict(status='passed', actual_optimizer_steps=steps, actual_image_visits=visited,
                    same_original_initialization=True, parameter_count=167872, trainable_whitelist=whitelist,
                    fit_only_targets=True, heldout_training_visits=0, frozen_native_inputs_and_runtime=True,
                    amplitude=amplitude, checkpoint_sha256=sha256(run.path / 'last.pt'), adapter_forward_calls=forwards,
                    base_model_forward_calls=0, no_oracle_optimization=True, scientific_gates='not_evaluated'))
        run.finish(dict(arm=args.arm, scope=args.scope, optimizer_steps=steps, images_seen=visited, adapter_forward_calls=forwards,
                        initial_state_sha256=initial_hash, final_state_sha256=tensor_state_hash(model),
                        checkpoint_sha256=sha256(run.path / 'last.pt'), parameters_with_nonzero_gradient=sorted(gradient_names)),
                   scientific='trained_not_evaluated' if full else 'engineering_only', next_action='Independent verification; no automatic continuation')


if __name__ == '__main__':
    main()
