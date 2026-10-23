import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import Residual, loss_for, prior_maps, scatter_delta
from rt import context, dump, read, sha, state_hash
from transfer_io import (coco_score, encode, gpu_bank, load_native, make_bank,
                         prepare_cache, verify_real)


class Balanced:
    def __init__(self, labels, cfg):
        self.rng = np.random.default_rng(cfg['seed']+401)
        self.groups = [np.where(labels)[0], np.where(~labels)[0]]
        assert all(len(x) for x in self.groups), [len(x) for x in self.groups]
        self.orders = [self.rng.permutation(x) for x in self.groups]
        self.positions = [0, 0]
        self.npos = round(cfg['batch_size']*cfg['positive_fraction'])
        self.batch_size = cfg['batch_size']

    def take(self, k, n):
        result = []
        while n:
            pos, order = self.positions[k], self.orders[k]
            use = min(n, len(order)-pos)
            result.extend(order[pos:pos+use]); n -= use; pos += use
            if pos == len(order):
                self.orders[k] = self.rng.permutation(self.groups[k]); pos = 0
            self.positions[k] = pos
        return result

    def next(self):
        ids = self.take(0, self.npos)+self.take(1, self.batch_size-self.npos)
        return np.asarray(ids, dtype=np.int64)


def load_training(cache):
    parts = [torch.load(r['file'], map_location='cpu', weights_only=False) for r in cache['train']]
    assert parts
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def train_arms(run, meta, cfg, cache, bank, engineering=False):
    data = load_training(cache)
    labels = data['positive'].numpy()
    steps = cfg['engineering_steps_per_arm'] if engineering else cfg['steps_per_arm']
    models, results = {}, {}
    initial, init_hash = None, None
    for arm in cfg['arms']:
        torch.manual_seed(cfg['seed'])
        model = Residual(cfg).cuda()
        if initial is None:
            initial = copy.deepcopy(model.state_dict()); init_hash = state_hash(model)
        else:
            model.load_state_dict(initial)
        assert state_hash(model) == init_hash
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        sampler = Balanced(labels, cfg)
        exposure = np.zeros(len(labels), dtype=np.int64)
        grad_total = {name: 0. for name, _ in model.named_parameters()}
        logs, start = [], time.monotonic()
        model.train()
        for step in range(1, steps+1):
            index_np = sampler.next(); np.add.at(exposure, index_np, 1)
            index = torch.from_numpy(index_np)
            common = data['common'][index].cuda().float()
            native = data['native'][index].cuda().float()
            target = data['target'][index].cuda().float()
            maps = prior_maps(bank, data['bank_index'][index].cuda(), arm)
            optimizer.zero_grad(set_to_none=True)
            loss, components = loss_for(model, common, maps, native, target, cfg)
            assert torch.isfinite(loss), (arm, step)
            loss.backward()
            if engineering:
                for name, p in model.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    grad_total[name] += float(p.grad.abs().sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); meta['optimizer_steps'] += 1
            if step == 1 or step % 100 == 0 or step == steps:
                entry = dict(step=step, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in components.items()})
                logs.append(entry); dump(run/(arm+'_PROGRESS.json'), entry); dump(run/'run.json', meta)
                print('TRAIN', arm, step, round(entry['loss'], 6), flush=True)
        if engineering: assert all(v > 0 for v in grad_total.values()), grad_total
        receipt = dict(optimizer_steps=steps, trainable_parameters=params, initialization_state_sha256=init_hash,
                       losses=logs, elapsed_seconds=time.monotonic()-start, sampler_objects=len(labels),
                       positive_objects=int(labels.sum()), min_exposure=int(exposure.min()),
                       max_exposure=int(exposure.max()), exposed_objects=int((exposure > 0).sum()),
                       gradient_absolute_sum=grad_total if engineering else None,
                       checkpoint_selection=cfg['selection'])
        if not engineering:
            assert (exposure > 0).all(), 'Sampler failed to cover every eligible training object'
            path = run/(arm+'_last.pt')
            torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()}, arm=arm,
                            config=cfg, optimizer_steps=steps, trainable_parameters=params,
                            bank_sha256=read(run.parents[1]/'BANK_POINTER.json')['bank_sha256'],
                            protocol_sha256=meta['protocol_sha256'], config_sha256=meta['config_sha256'],
                            native_checkpoint_sha256=read(run.parents[1]/'INVENTORY.json')['external_files'][
                                str(Path(cfg['parent_root'])/cfg['native_checkpoint'])]), path)
            receipt['checkpoint_sha256'] = sha(path)
            np.save(run/(arm+'_exposures.npy'), exposure)
        dump(run/(arm+'_TRAINING.json'), receipt)
        results[arm] = receipt; models[arm] = model.eval()
        del optimizer
    assert len({x['trainable_parameters'] for x in results.values()}) == 1
    assert len({x['initialization_state_sha256'] for x in results.values()}) == 1
    return models, results


def successes(ev):
    result = {}
    t = int(np.where(np.isclose(ev.params.iouThrs, .75))[0][0])
    for row in ev.evalImgs:
        if row is None or row['aRng'] != [0, 1e10] or row['maxDet'] != 100: continue
        for k, gt_id in enumerate(row['gtIds']):
            if not row['gtIgnore'][k]: result[int(gt_id)] = bool(row['gtMatches'][t, k])
    return result


@torch.no_grad()
def evaluate(run, cfg, cache, bank, models, native, ops):
    modes = [('structured', 'structured', 'structured'),
             ('scrambled_control', 'scrambled_control', 'scrambled_control'),
             ('structured_off', 'structured', 'off'),
             ('structured_scrambled', 'structured', 'scrambled')]
    predicted = {tag: [] for tag, _, _ in modes}
    position = 0
    start = time.monotonic()
    for i, item in enumerate(cache['dev']):
        sample = torch.load(item['file'], map_location='cpu', weights_only=False)
        det = sample['det'].cuda(); eligible = sample['eligible'].cuda()
        source = native[position:position+len(det)]; position += len(det)
        assert all(p['image_id'] == item['image_id'] for p in source)
        boxes = det[eligible, :4]
        common = sample['common'].cuda().float()
        for tag, model_name, input_mode in modes:
            output = copy.deepcopy(source)
            if len(eligible):
                delta = models[model_name](common, prior_maps(bank, sample['bank_index'].cuda(), input_mode))
                modified = sample['base_logits'].cuda()+scatter_delta(delta, boxes)
                upsampled = F.interpolate(modified[None], (640, 640), mode='bilinear', align_corners=False)[0]
                masks = ops.crop_mask(upsampled, boxes) > 0
                for k, mask in zip(eligible.cpu().tolist(), masks.cpu().numpy()):
                    output[k]['segmentation'] = encode(mask)
            for a, b in zip(source, output):
                assert all(a[k] == b[k] for k in ('image_id', 'category_id', 'score', 'bbox'))
            predicted[tag] += output
        if (i+1) % 100 == 0: print('MASK_EVAL', i+1, len(cache['dev']), flush=True)
    assert position == len(native)
    gt = Path(cfg['parent_root'])/'DEV_COCO.json'
    base_metrics, base_ev = coco_score(gt, native)
    metrics = {'native': base_metrics}
    original_success = successes(base_ev)
    damage = {}
    for tag, _, _ in modes:
        dump(run/(tag+'_predictions.json'), predicted[tag])
        value, ev = coco_score(gt, predicted[tag]); metrics[tag] = value
        current = successes(ev)
        assert set(current) == set(original_success)
        harm = sum(ok and not current[g] for g, ok in original_success.items())
        repair = sum(not ok and current[g] for g, ok in original_success.items())
        n = sum(original_success.values())
        damage[tag] = dict(native_success_GT=n, success_to_failure=harm, damage_fraction=harm/max(1, n),
                           native_failures=len(current)-n, failure_to_success=repair)
        print('METRIC', tag, value['mask_ap'], flush=True)
    bbox, _ = coco_score(gt, native, 'bbox')
    dump(run/'IDENTITY_CHECK.json', dict(same_box_class_score_order=True, predictions=len(native),
                                       box_ap=bbox['mask_ap'], changed_detection_rows=0))
    result = dict(metrics=metrics, damage=damage, evaluation_images=432, inference_GT_used=False,
                  elapsed_seconds=time.monotonic()-start,
                  source='fixed last checkpoints; complete development set; no test and no model selection')
    dump(run/'EVALUATION.json', result)
    return result


def adjudicate(result, cfg):
    ap = {k: v['mask_ap'] for k, v in result['metrics'].items()}
    dn, dq = ap['structured']-ap['native'], ap['structured']-ap['scrambled_control']
    if dn > 0 and dq > 0:
        status = 'development_pilot_positive_over_native_and_layout_control'
        direction = 'freeze_object_interface_and_plan_held_out_confirmation'
    elif dn > 0:
        status = 'deployment_gain_without_body_layout_increment'
        direction = 'prioritize_local_image_mask_readout_and_close_object_layout_expansion'
    else:
        status = 'current_object_interface_no_positive_real_segmentation_gain'
        direction = ('prioritize_local_image_mask_readout_and_close_object_interface_expansion'
                     if ap['scrambled_control'] > ap['native'] else
                     'prioritize_real_visible_object_alignment_before_any_further_residual_training')
    return dict(status=status, structured_minus_native_pp=dn, structured_minus_control_pp=dq,
                off_drop_pp=ap['structured']-ap['structured_off'],
                scramble_drop_pp=ap['structured']-ap['structured_scrambled'],
                prototype_dependency_observed=(ap['structured'] > ap['structured_off'] and
                                               ap['structured'] > ap['structured_scrambled']),
                positive_development_point_estimate_is_not_independent_confirmation=True,
                five_pp_goal_achieved=dn >= cfg['goal_delta_pp'],
                chosen_research_direction=direction, automatic_followup_training=False)


def resampler_check():
    torch.manual_seed(917)
    delta = torch.randn(2, 32, 32, device='cuda')
    boxes = torch.tensor([[43.2, 17.7, 279.4, 310.6], [4., 301., 123.4, 619.2]], device='cuda')
    got = scatter_delta(delta, boxes).cpu().numpy()
    source, bb = delta.cpu().numpy(), boxes.cpu().numpy()
    expected = np.zeros_like(got)
    y, x = np.meshgrid((np.arange(160)+.5)*4, (np.arange(160)+.5)*4, indexing='ij')
    for k, b in enumerate(bb):
        u, v = (x-b[0])/(b[2]-b[0]), (y-b[1])/(b[3]-b[1])
        sx, sy = u*32-.5, v*32-.5
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        result = np.zeros((160, 160), np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                valid = (ix+dx >= 0) & (ix+dx < 32) & (iy+dy >= 0) & (iy+dy < 32)
                result[valid] += (source[k, np.clip(iy+dy, 0, 31), np.clip(ix+dx, 0, 31)]*
                                  (1-np.abs(sx-(ix+dx)))*(1-np.abs(sy-(iy+dy))))[valid]
        expected[k] = result*((u >= 0) & (u < 1) & (v >= 0) & (v < 1))
    error = float(np.abs(got-expected).max())
    assert error < 2e-5, error
    return error


def execute(root, stage):
    with context(root, stage) as (run, meta, cfg):
        assert os.getenv('SLURM_JOB_ID'), 'GPU work must use one-time Slurm allocation'
        assert torch.cuda.device_count() == 1 and 'A6000' in torch.cuda.get_device_name(0)
        torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
        rows = verify_real(cfg, run)
        model, captured, hooks, ops = load_native(cfg)
        before = state_hash(model)
        smoke = stage == 'engineering'
        if smoke:
            bank = make_bank(root, run, cfg)
        else:
            ep = read(root/'ENGINEERING_POINTER.json')
            assert read(root/'runs'/ep['run_id']/'SUMMARY.json')['engineering_passed']
            bank_source = read(root/'BANK_POINTER.json')
            assert sha(bank_source['file']) == bank_source['bank_sha256']
            bank = torch.load(bank_source['file'], map_location='cpu', weights_only=False)
        bank = gpu_bank(bank)
        cache, native = prepare_cache(root, run, cfg, model, captured, ops, bank, rows, smoke)
        assert state_hash(model) == before
        if smoke:
            sample = torch.load(cache['train'][0]['file'], weights_only=False)
            initial = Residual(cfg).cuda()
            zero = initial(sample['common'].cuda().float(), prior_maps(bank, sample['bank_index'].cuda()))
            assert torch.count_nonzero(zero) == 0
            error = resampler_check()
            _, training = train_arms(run, meta, cfg, cache, bank, engineering=True)
            summary = dict(engineering_passed=True, actual_optimizer_steps=meta['optimizer_steps'],
                           zero_residual_exact=True, zero_replay_native_rows=cache['zero_replay_rows'],
                           independent_resampler_max_abs_error=error, trainable_white_list='Residual.net only',
                           native_state_before=before, native_state_after=state_hash(model),
                           parameters={k: v['trainable_parameters'] for k, v in training.items()},
                           engineering_checkpoints_discarded=True, real_test_used=False)
            dump(run/'SUMMARY.json', summary)
        else:
            metrics, _ = coco_score(Path(cfg['parent_root'])/'DEV_COCO.json', native)
            parity_error = abs(metrics['mask_ap']-cfg['expected_native_mask_ap'])
            dump(run/'BASELINE_PARITY.json', dict(metrics=metrics, absolute_error_pp=parity_error,
                                                  tolerance_pp=cfg['native_parity_tolerance_pp']))
            assert parity_error <= cfg['native_parity_tolerance_pp'], metrics
            models, training = train_arms(run, meta, cfg, cache, bank)
            result = evaluate(run, cfg, cache, bank, models, native, ops)
            assert state_hash(model) == before
            bank_source = read(root/'BANK_POINTER.json')
            assert sha(bank_source['file']) == bank_source['bank_sha256']
            decision = adjudicate(result, cfg)
            dump(run/'DECISION.json', decision)
            dump(run/'SUMMARY.json', dict(**result, **decision, optimizer_steps=meta['optimizer_steps'],
                   training={k: {a: b for a, b in v.items() if a != 'losses'} for k, v in training.items()},
                   native_state_before=before, native_state_after=state_hash(model),
                   peak_GPU_memory_bytes=int(torch.cuda.max_memory_allocated()),
                   cache_summary={k: v for k, v in cache.items() if k not in ('train', 'dev')},
                   real_test_used=False))
        dump(run/'ENVIRONMENT.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                 numpy=np.__version__, GPU=torch.cuda.get_device_name(0), device_count=torch.cuda.device_count(),
                 precision='FP32', trainable_modules='Residual.net only'))
        for h in hooks: h.remove()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['engineering', 'formal'], required=True)
    a = p.parse_args(); execute(a.root.resolve(), a.stage)
