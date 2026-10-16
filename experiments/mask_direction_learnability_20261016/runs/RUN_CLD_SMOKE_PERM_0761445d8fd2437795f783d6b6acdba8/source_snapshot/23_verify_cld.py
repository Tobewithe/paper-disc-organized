"""Independent NumPy target/scale and explicit-gradient/AdamW smoke audit."""
import argparse
from collections import Counter
import math
import traceback
import time
import numpy as np
import torch
from cld_common import *
from adapter_model import CoefficientAdapter
from adapter_data import FrozenCache
from native_output_bridge import decode_native, encode_rle

STARTED = time.monotonic()


def budget():
    require(time.monotonic() - STARTED <= 3600, 'independent verification wall budget exhausted')


def artifact_check(path):
    records = read_json(path / 'ARTIFACT_MANIFEST.json')['artifacts']
    for record in records:
        budget()
        item = path / record['relative_path']
        require(item.resolve().is_relative_to(path), 'artifact escaped Run')
        require(item.stat().st_size == record['size_bytes'] and sha256(item) == record['sha256'], 'Run artifact changed: ' + str(item))
    sources = read_json(path / 'SOURCE_MANIFEST.json')['files']
    mapping = read_json(path / 'SOURCE_SNAPSHOT_MAP.json')
    require(set(sources) == set(mapping), 'incomplete source snapshot')
    for relative, digest in sources.items():
        require(sha256(path / mapping[relative]) == digest, 'source snapshot mismatch')
    return len(records)


def independent_target(cache, row):
    p = cache['proto'].numpy().astype(np.float64)
    delta = np.asarray(row['coefficients_fp32'], dtype=np.float32).astype(np.float64) - cache['c0'][row['rank']].numpy().astype(np.float64)
    response = np.einsum('c,chw->hw', delta, p, optimize=False)
    h, w = p.shape[-2:]
    ih, iw = cache['input_hw']
    b = cache['boxes_input'][row['rank']].numpy().astype(np.float32) * np.asarray([w/iw, h/ih, w/iw, h/ih], dtype=np.float32)
    yy, xx = np.indices((h, w))
    roi = (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
    require(roi.any(), 'independent target empty ROI')
    scale = float(np.sqrt(np.mean(response[roi] ** 2)))
    require(math.isfinite(scale) and scale > 1e-6, 'independent target invalid scale')
    return (response / scale).astype(np.float32), scale


def independent_loss(model, cache, teachers, mapping, arm, amplitude):
    """No call to producer's loss/target/context/remap helpers."""
    raw = model(cache, 'independent:train')
    d = raw if arm == 'REAL' else raw[torch.tensor(mapping, device=raw.device)]
    p = cache['proto'].to(raw).flatten(1)
    c0 = cache['c0'].to(raw)
    h, w = cache['proto'].shape[-2:]
    ih, iw = cache['input_hw']
    boxes = cache['boxes_input'].to(raw) * raw.new_tensor([w/iw, h/ih, w/iw, h/ih])
    x, y = torch.arange(w, device=raw.device)[None, None], torch.arange(h, device=raw.device)[None, :, None]
    roi = (x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) & (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None])
    response = (d @ p).reshape(len(d), h, w)
    count = roi.sum((1, 2))
    squared = torch.where(roi, response.square(), 0).sum((1, 2)) / count.clamp_min(1)
    scale = torch.sqrt(torch.maximum(squared, torch.ones_like(squared)))
    bounded = d / scale[:, None] * (count > 0)[:, None]
    after = ((c0 + amplitude * bounded) @ p).reshape_as(response)
    before = (c0 @ p).reshape_as(response)
    direction_terms = []
    for row in teachers:
        q, _ = independent_target(cache, row)
        rank = row['rank']
        direction_terms.append((response[rank][roi[rank]] - torch.as_tensor(q, device=raw.device)[roi[rank]]).square().mean())
    zero = d.sum() * 0
    direction_loss = torch.stack(direction_terms).mean() if direction_terms else zero
    preserve_terms = [torch.nn.functional.binary_cross_entropy_with_logits(after[i][roi[i]], before[i][roi[i]].sigmoid()) for i in range(len(d)) if int(count[i]) > 0]
    preservation = torch.stack(preserve_terms).mean() if preserve_terms else zero
    return direction_loss + .1 * preservation


def verify_preflight(path, data):
    ids = data['split']['fit_image_ids']
    by_image = teacher_rows(ids)
    recorded = {(r['image_id'], r['annotation_id']): r for r in read_jsonl(path / 'FIT_AMPLITUDES.jsonl')}
    trace = read_json(CACHE / 'trace/TRACE_MANIFEST.json')
    cache_rows = {r['image_id']: r for r in trace['images']}
    values, largest = [], 0.
    for pos, iid in enumerate(ids, 1):
        budget()
        cache_path = CACHE / 'trace' / cache_rows[iid]['cache']
        require(sha256(cache_path) == cache_rows[iid]['cache_sha256'], 'cache changed')
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        for row in by_image[iid]:
            q, scale = independent_target(cache, row)
            saved = recorded[(iid, row['annotation_id'])]
            require(saved['rank'] == row['rank'] and saved['output_detection_id'] == row['output_detection_id'], 'scale row identity mismatch')
            require(math.isclose(scale, saved['scale'], rel_tol=2e-10, abs_tol=2e-10), 'independent scale differs')
            largest = max(largest, abs(scale - saved['scale']))
            values.append(scale)
        if pos % 200 == 0:
            print('VERIFY_AMPLITUDE', pos, '/', len(ids), flush=True)
        del cache
    a = read_json(path / 'AMPLITUDE.json')
    require(len(values) == len(recorded) == 2510, 'fit-only amplitude population mismatch')
    require(math.isclose(float(np.median(values)), a['value'], rel_tol=2e-10, abs_tol=2e-10), 'fit-only median differs')
    require(a['heldout_teachers_used'] == 0 and a['rows_sha256'] == sha256(path / 'FIT_AMPLITUDES.jsonl'), 'amplitude source identity differs')
    replays = read_jsonl(path / 'REPLAY.jsonl')
    require([r['image_id'] for r in replays] == ids and sum(r['native_rows'] for r in replays) == 176042, 'replay coverage mismatch')
    require(sum(r['teacher_rows'] for r in replays) == 2510 and all(r['native_RLE_exact'] and r['teacher_RLE_exact'] for r in replays), 'producer replay failed')
    return dict(kind='preflight', fit_images=1600, teacher_scales_independently_recomputed=2510,
                median=a['value'], maximum_scale_difference=largest, producer_native_and_teacher_replay_coverage_verified=True,
                independent_decoder_reexecution=False, adapter_forward_calls=0, additional_optimizer_steps=0)


def verify_training(path, data):
    c = read_json(path / 'EVAL_CONTRACT.json')
    full = c['scope'] == 'full'
    ids = data['split']['fit_image_ids'] if full else data['split']['fit_image_ids'][:8]
    expected_orders = read_json(STUDY / 'IMAGE_ORDERS.json')['epochs'] if full else [ids]
    require(read_json(path / 'IMAGE_ORDERS.json') == expected_orders, 'actual image order differs')
    logs = read_jsonl(path / 'training.jsonl')
    flat = [iid for order in expected_orders for iid in order]
    require([r['image_id'] for r in logs] == flat and len(logs) == c['expected_image_visits'], 'image visits changed')
    require(not set(flat) & set(data['split']['heldout_image_ids']), 'heldout fitting leaked')
    steps = read_jsonl(path / 'OPTIMIZER_STEPS.jsonl')
    require(len(steps) == c['expected_optimizer_steps'] and [i for step in steps for i in step['image_ids']] == flat, 'optimizer coverage differs')
    for i, step in enumerate(steps, 1):
        budget()
        require(step['step'] == i and len(step['image_ids']) == 4, 'optimizer step sequence differs')
    require(sha256(path / 'initial.pt') == sha256(INITIAL / 'initial.pt') == c['original_initial_sha256'], 'initial bytes differ')
    base = data['inherited']['source_training_contract']
    for key in ('optimizer', 'amp', 'dtype', 'scheduler', 'early_stopping', 'checkpoint_selection', 'augmentation', 'parameters', 'trainable_whitelist', 'preservation_weight', 'environment'):
        require(canonical(c[key]) == canonical(base[key]), 'inherited configuration differs: ' + key)
    last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
    require(last['steps'] == len(steps) and last['contract_sha256'] == sha256(path / 'EVAL_CONTRACT.json'), 'checkpoint identity/budget differs')
    state = last['optimizer_state_dict']
    actual_group = {k: v for k, v in state['param_groups'][0].items() if k != 'params'}
    require(canonical(actual_group) == canonical(c['optimizer']), 'serialized optimizer differs')
    counts = Counter(n for step in steps for n in step['gradient_parameter_names'])
    for index, name in enumerate(c['trainable_whitelist']):
        if counts[name]:
            require(float(state['state'][index]['step']) == counts[name], 'serialized per-parameter step differs')
        else:
            require(index not in state['state'], 'unused parameter acquired optimizer state')
    teachers = teacher_rows(ids)
    for row in logs:
        require(row['teachers'] == len(teachers[row['image_id']]) and math.isfinite(row['loss']), 'actual target count/loss invalid')
    if full:
        return dict(kind='full_training_serialized_audit', image_visits=len(logs), optimizer_steps=len(steps),
                    optimizer_state_steps_verified=True, heldout_training_visits=0, additional_optimizer_steps=0,
                    independent_gradient_reexecution=False, scientific_status='trained_not_evaluated')
    source = FrozenCache(CACHE / 'trace', ids)
    model = CoefficientAdapter('B').cuda()
    np_scale = c['amplitude']
    pnames = list(dict(model.named_parameters()))
    initial = torch.load(INITIAL / 'initial.pt', map_location='cpu', weights_only=True)['state_dict']
    expected_params = {n: p.clone() for n, p in initial.items()}
    moments, counters, gradient_max_error, forward_calls = {}, Counter(), 0., 0
    for i, step in enumerate(steps, 1):
        before = initial if i == 1 else torch.load(path / f'step_{i-1:04d}.pt', map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(before, strict=True)
        model.train()
        accumulated = {}
        for image_id in step['image_ids']:
            cache, _ = source.load(image_id)
            value = independent_loss(model, cache, teachers[image_id], data['maps'][image_id], c['arm'], np_scale)
            forward_calls += 1
            recorded = logs[(i - 1) * 4 + step['image_ids'].index(image_id)]
            require(math.isclose(float(value.detach()), recorded['loss'], rel_tol=1e-5, abs_tol=3e-6), 'independent objective differs')
            grads = torch.autograd.grad(value / 4, tuple(model.parameters()), allow_unused=True)
            for name, grad in zip(pnames, grads):
                if grad is not None:
                    g = grad.detach().cpu()
                    accumulated[name] = accumulated.get(name, torch.zeros_like(g)) + g
            del value, grads, cache
        actual = torch.load(path / f'gradient_{i:04d}.pt', map_location='cpu', weights_only=True)
        require(set(accumulated) == set(actual) == set(step['gradient_parameter_names']), 'gradient whitelist differs')
        for name, grad in actual.items():
            torch.testing.assert_close(accumulated[name], grad, atol=2e-5, rtol=2e-4)
            gradient_max_error = max(gradient_max_error, float((accumulated[name] - grad).abs().max()))
            # Algebraic verification on clones; no optimizer.step or model fitting.
            counters[name] += 1
            m, v = moments.get(name, (torch.zeros_like(grad), torch.zeros_like(grad)))
            beta1, beta2 = c['optimizer']['betas']
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad.square()
            moments[name] = (m, v)
            t = counters[name]
            lr, wd, eps = (c['optimizer'][k] for k in ('lr', 'weight_decay', 'eps'))
            expected_params[name] = expected_params[name] * (1 - lr * wd) - (lr / (1 - beta1**t)) * m / (v.sqrt() / math.sqrt(1 - beta2**t) + eps)
        after = torch.load(path / f'step_{i:04d}.pt', map_location='cpu', weights_only=True)['state_dict']
        for name in expected_params:
            torch.testing.assert_close(expected_params[name], after[name], atol=2e-6, rtol=2e-5)
    for name in expected_params:
        torch.testing.assert_close(after[name], last['state_dict'][name], atol=0, rtol=0)
    post = torch.load(path / 'SMOKE_POST.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(last['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    masks = read_jsonl(path / 'SMOKE_MASKS.jsonl')
    by_image = {iid: [r for r in masks if r['image_id'] == iid] for iid in ids}
    checked_masks = 0
    from pycocotools import mask as coco_mask
    for iid in ids:
        budget()
        cache, rows = source.load(iid)
        saved = post[iid]
        with torch.inference_mode():
            recomputed = model(cache, 'independent:post')
            forward_calls += 1
            torch.testing.assert_close(recomputed.cpu(), saved['raw'], atol=0, rtol=0)
            del recomputed
        d = saved['raw'].cuda()
        if c['arm'] == 'PERM':
            d = d[torch.tensor(data['maps'][iid], device=d.device)]
        p = cache['proto'].cuda().flatten(1)
        response = (d @ p).reshape(len(d), *cache['proto'].shape[-2:])
        coeff = cache['c0'].cuda().clone()
        probe = coeff.clone()
        for rank in range(len(d)):
            roi = __import__('s1_geometry').predicted_roi(cache, rank).cuda()
            rms = float(response[rank][roi].square().mean().sqrt()) if roi.any() else 0.
            if roi.any():
                coeff[rank] += np_scale * d[rank] / max(1., rms)
            if rank in saved['teacher_scales'] and rms > 1e-6:
                probe[rank] += saved['teacher_scales'][rank] * d[rank] / rms
        torch.testing.assert_close(coeff.cpu(), saved['coefficients'], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(probe.cpu(), saved['probe_coefficients'], atol=3e-6, rtol=2e-5)
        native = decode_native(cache)
        changed = decode_native(cache, saved['coefficients'])
        require(len(by_image[iid]) == len(rows), 'smoke output rows missing')
        for rank, (old, stored) in enumerate(zip(rows, by_image[iid])):
            require(encode_rle(native[rank]) == old['segmentation'], 'independent native replay differs')
            gate = int(coco_mask.area(old['segmentation'])) >= 9216
            require(stored['rank'] == rank and stored['global_index'] == cache['prediction_indices'][rank] and stored['gate'] == gate, 'output identity or gate differs')
            expected = dict(old)
            expected['segmentation'] = encode_rle(changed[rank]) if gate else old['segmentation']
            require(stored['prediction'] == expected, 'mixed output or box/class/score identity differs')
            checked_masks += 1
        del cache, rows, d, p, response, coeff, probe, native, changed
    source.verify()
    return dict(kind='smoke', arm=c['arm'], image_visits=len(logs), optimizer_steps=len(steps),
                independently_recomputed_forward_backward_images=len(logs), adapter_forward_calls=forward_calls,
                independent_post_forward_images=len(ids), maximum_gradient_difference=gradient_max_error,
                algebraic_AdamW_updates_verified=len(steps), normal_mixed_masks_verified=checked_masks,
                equal_amplitude_and_bounded_coefficients_verified=True, heldout_training_visits=0,
                additional_optimizer_steps=0, base_model_forward_calls=0, scientific_status='engineering_only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    path = parser.parse_args().run.resolve()
    require(not (path / 'INDEPENDENT.json').exists(), 'independent receipt already exists')
    try:
        resources = resource_check()
        check_completed(path, independent=False)
        artifact_count = artifact_check(path)
        data = inspect_inputs()
        stage = read_json(path / 'EVAL_CONTRACT.json')['stage']
        if stage == 'engineering_tensor_preflight':
            result = verify_preflight(path, data)
        elif stage == 'CLD_evaluation':
            from verify_evaluation import verify
            result = verify(path, data, budget)
        else:
            result = verify_training(path, data)
        inspect_inputs()
        budget()
        write_new(path / 'INDEPENDENT.json', dict(status='passed', at=now(), run_id=path.name,
                   artifact_hash_checks=artifact_count, resources=resources, elapsed_seconds=time.monotonic()-STARTED,
                   protocol_sha256=sha256(path / 'PROTOCOL.md'), contract_sha256=sha256(path / 'EVAL_CONTRACT.json'), **result))
        print('INDEPENDENT_PASSED', path.name, json.dumps(result), flush=True)
    except Exception:
        write_new(path / ('INDEPENDENT_ERROR_' + __import__('uuid').uuid4().hex + '.json'), dict(status='failed', at=now(), error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
