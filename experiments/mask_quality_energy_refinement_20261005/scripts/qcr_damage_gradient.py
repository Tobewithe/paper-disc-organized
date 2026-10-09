"""Fixed-checkpoint local mechanism diagnostic; no training or step selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import traceback

IDENTITY = ('image_id', 'annotation_id', 'branch', 'raw_id', 'pyramid_level', 'target_gt_idx')
PROBES = {'tiny': 1/4096, 'small': 1/128, 'train_inner': 1/8, 'train_outer': 1/4, 'first_nominal': 1/2}


def key(row):
    return tuple(row[name] for name in IDENTITY)


def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if cfg.get('authorized_local_diagnostic') is not True:
        raise RuntimeError('Require explicit local-diagnostic authorization')
    sys.path.insert(0, cfg['source_python'])
    import numpy as np
    import torch
    from online_runtime import dump, sha256
    from qcr_train_fixed import QCR, z_from, soft_iou, LAMBDA
    from qcr_evaluate_complete import native_masks, refine_states
    from qcr_local_streaming_data import LocalStreamingDEV

    if not torch.cuda.is_available():
        raise RuntimeError('The authorized local diagnostic requires CUDA')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PER_CANDIDATE.jsonl').exists() or (out/'COMPLETE.json').exists():
        raise RuntimeError('Keep previous artifacts; use an independent Run ID')
    cfg.update(server_root=str(out), images=str(out/'source_images'),
               cache=str(out/'unused_cache'), assets=str(out/'unused_cache'))
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    if sha256(cfg['quality_checkpoint']) != cfg['quality_checkpoint_sha256']:
        raise AssertionError('Fixed Quality checkpoint hash differs')
    checkpoint = torch.load(cfg['quality_checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint['mode'] != 'quality' or checkpoint['epoch'] != 3:
        raise AssertionError('Require the existing final epoch-3 Quality checkpoint')
    rho = float(checkpoint['rho'])
    model = QCR('quality', rho).to('cuda').float().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state_dict']); del checkpoint
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ids = json.loads(Path(cfg['split']).read_text(encoding='utf-8-sig'))['dev_images'][:args.images]
    if args.images not in (2, 64) or len(ids) != args.images:
        raise AssertionError('Locked samples are first 2 smoke or first 64 DEV images')
    reference = {}
    idset = set(ids)
    with Path(cfg['reference_dev_rows']).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['image_id'] in idset:
                reference[key(row)] = row
    metadata = {
        'purpose': 'mechanism diagnostic, no tuning or new method performance claim',
        'execution_environment': 'authorized local Windows CUDA',
        'python': sys.executable, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'tf32': False, 'dtype': 'float32',
        'historical_torch': '2.8.0+cu128', 'cross_environment_exact_equivalence_claimed': False,
        'config': cfg, 'image_ids': ids, 'requested_images': len(ids),
        'sample_rule': 'first fixed DEV image IDs; all official TAL candidates',
        'quality_checkpoint_sha256': sha256(cfg['quality_checkpoint']),
        'reference_dev_rows_sha256': sha256(cfg['reference_dev_rows']),
        'split_sha256': sha256(cfg['split']), 'protocol_sha256': sha256(cfg['diagnostic_protocol']),
        'script_sha256': sha256(__file__), 'rho': rho, 'lambda': LAMBDA,
        'eta': [rho/2, rho/4], 'probe_displacements': {k: v*rho for k, v in PROBES.items()},
        'parameter_updates': False, 'oracle_used': False, 'GT_enters_model': False,
        'soft_target': 'actual training cropped official overlap mask at 640',
        'hard_target': 'normal original-resolution binary decoder and original COCO GT',
        'counterfactuals': 'same eta1, quality-only and penalty-only second step; no step selection',
    }
    dump(out/'DIAGNOSTIC_CONFIG.json', metadata)
    rows = []
    seen_images = []
    started = time.monotonic()

    def cos(a, b):
        denom = float(a.norm()*b.norm())
        return float(torch.dot(a, b)/denom) if denom > 1e-12 else None

    def project(c, c0):
        d = c-c0
        return c0+d*min(1., rho/max(float(d.norm()), 1e-12))

    def stats(group):
        n = len(group)
        count = lambda fn: sum(bool(fn(r)) for r in group)
        fields = ('cos_gq0_gsoft0', 'gq0_norm', 'gsoft0_norm', 'gq1_over_reg_norm',
                  'cos_ge1_toward_c0', 'cos_gq1_toward_c0', 'delta_D1', 'delta_D',
                  'iou_A', 'iou_D1', 'iou_D', 'iou_D_qonly', 'iou_D_regonly')
        result = {'candidates': n, 'images': len({r['image_id'] for r in group})}
        result['counts'] = {
            'q_first_decrease': count(lambda r: r['q_D1'] < r['q_A']),
            'q_tiny_increase': count(lambda r: r['q_tiny'] > r['q_A']),
            'q_small_increase': count(lambda r: r['q_small'] > r['q_A']),
            'q_tiny_increase_actual_first_decrease': count(lambda r: r['q_tiny'] > r['q_A'] and r['q_D1'] < r['q_A']),
            'q_small_increase_actual_first_decrease': count(lambda r: r['q_small'] > r['q_A'] and r['q_D1'] < r['q_A']),
            'gq_soft_cos_negative': count(lambda r: r['cos_gq0_gsoft0'] is not None and r['cos_gq0_gsoft0'] < 0),
            'soft_small_increase': count(lambda r: r['soft_small'] > r['soft_A']),
            'hard_small_increase': count(lambda r: r['iou_small'] > r['iou_A']),
            'hard_small_decrease': count(lambda r: r['iou_small'] < r['iou_A']),
            'soft_small_increase_hard_small_decrease': count(lambda r: r['soft_small'] > r['soft_A'] and r['iou_small'] < r['iou_A']),
            'q_small_increase_soft_small_decrease': count(lambda r: r['q_small'] > r['q_A'] and r['soft_small'] < r['soft_A']),
            'q_small_increase_hard_small_decrease': count(lambda r: r['q_small'] > r['q_A'] and r['iou_small'] < r['iou_A']),
            'hard_small_nondecrease_actual_first_decrease': count(lambda r: r['iou_small'] >= r['iou_A'] and r['iou_D1'] < r['iou_A']),
            'first_damage': count(lambda r: r['iou_A'] >= .75 and r['iou_D1'] < .75),
            'final_damage': count(lambda r: r['iou_A'] >= .75 and r['iou_D'] < .75),
            'first_damage_recovered': count(lambda r: r['iou_A'] >= .75 and r['iou_D1'] < .75 and r['iou_D'] >= .75),
            'qonly_second_recovers_first_damage': count(lambda r: r['iou_A'] >= .75 and r['iou_D1'] < .75 and r['iou_D_qonly'] >= .75),
            'regonly_second_recovers_first_damage': count(lambda r: r['iou_A'] >= .75 and r['iou_D1'] < .75 and r['iou_D_regonly'] >= .75),
            'penalty_norm_exceeds_quality_norm': count(lambda r: r['gq1_over_reg_norm'] < 1),
            'energy_second_moves_toward_c0': count(lambda r: r['cos_ge1_toward_c0'] is not None and r['cos_ge1_toward_c0'] > 0),
            'quality_second_moves_toward_c0': count(lambda r: r['cos_gq1_toward_c0'] is not None and r['cos_gq1_toward_c0'] > 0),
            'final_exceeds_success_training_radius': count(lambda r: r['delta_D'] > rho/4+1e-5),
        }
        result['distributions'] = {}
        for field in fields:
            values = [r[field] for r in group if r[field] is not None]
            result['distributions'][field] = {
                'n': len(values), 'mean': float(np.mean(values)) if values else None,
                'quantiles_0_25_50_75_100': np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else [],
            }
        result['mean_delta_hard_iou'] = {arm: float(np.mean([r['iou_'+arm]-r['iou_A'] for r in group])) if n else None
                                          for arm in [*PROBES, 'D1', 'D', 'D_qonly', 'D_regonly']}
        result['mean_delta_soft_iou'] = {arm: float(np.mean([r['soft_'+arm]-r['soft_A'] for r in group])) if n else None
                                          for arm in [*PROBES, 'D1', 'D', 'D_qonly', 'D_regonly']}
        return result

    stream = LocalStreamingDEV(cfg, ids, out, split='dev')
    try:
        with (out/'PER_CANDIDATE.jsonl').open('x', encoding='utf-8') as handle, stream:
            for pos, (iid, x) in enumerate(stream.iter_images()):
                observed = {key(r) for r in x['rows']}
                expected = {ident for ident in reference if ident[0] == iid}
                if observed != expected:
                    dump(out/'IDENTITY_FAILURE.json', {'image_id': iid, 'missing': list(expected-observed), 'extra': list(observed-expected)})
                    raise AssertionError('Fresh local official TAL identities differ from historical DEV; no substitution')
                proto = x['proto'].to('cuda').float()
                for k, source in enumerate(x['rows']):
                    c0 = x['c0'][k].to('cuda').float()
                    box = x['boxes'][k].to('cuda').float()
                    h = x['operator']['h0'][k].to('cuda').float()
                    target = (x['masks'].to('cuda') == int(x['owners'][k])+1)
                    gt = torch.as_tensor(stream.coco.annToMask(stream.coco.anns[int(source['annotation_id'])]).astype(bool), device='cuda')
                    with torch.no_grad():
                        evidence = model.encoder(model.evidence_input(h[None], box[None], c0[None], z_from(proto[None], c0[None])))
                    c = c0.detach().clone().requires_grad_(True)
                    q = model.quality_value(evidence, c[None], z_from(proto[None], c[None]))[0]
                    g0 = torch.autograd.grad(q, c)[0].detach()
                    c = c0.detach().clone().requires_grad_(True)
                    soft = soft_iou(z_from(proto[None], c[None])[0], target, box)
                    gs = torch.autograd.grad(soft, c)[0].detach()
                    c1, c2 = refine_states(model, evidence, proto, c0, rho)
                    c = c1.detach().clone().requires_grad_(True)
                    q = model.quality_value(evidence, c[None], z_from(proto[None], c[None]))[0]
                    g1 = torch.autograd.grad(q, c)[0].detach()
                    reg = LAMBDA*(c1-c0)
                    ge = g1-reg
                    cq = project(c1+rho/4*g1/(g1.norm()+1e-8), c0).detach()
                    cr = project(c1-rho/4*reg/(reg.norm()+1e-8), c0).detach()
                    cs = {'A': c0, **{name: c0+fraction*rho*g0/(g0.norm()+1e-8) for name, fraction in PROBES.items()},
                          'D1': c1, 'D': c2, 'D_qonly': cq, 'D_regonly': cr}
                    row = {name: source[name] for name in IDENTITY}
                    row.update(row_index=k, box_iou=source['box_iou'], predicted_class_id=source['predicted_class_id'],
                               gq0_norm=float(g0.norm()), gsoft0_norm=float(gs.norm()), cos_gq0_gsoft0=cos(g0, gs),
                               gq1_norm=float(g1.norm()), reg1_norm=float(reg.norm()), ge1_norm=float(ge.norm()),
                               gq1_over_reg_norm=float(g1.norm()/reg.norm().clamp_min(1e-12)),
                               cos_ge1_toward_c0=cos(ge, c0-c1), cos_gq1_toward_c0=cos(g1, c0-c1),
                               delta_D1=float((c1-c0).norm()), delta_D=float((c2-c0).norm()),
                               second_step_norm=float((c2-c1).norm()),
                               quality_autograd_directional_derivative=float(g0.norm()),
                               soft_autograd_directional_derivative=float(torch.dot(gs, g0)/(g0.norm()+1e-8)))
                    with torch.no_grad():
                        names = list(cs); stacked = torch.stack(list(cs.values()))
                        zs = z_from(proto[None].expand(len(cs), -1, -1, -1), stacked)
                        qvalues = model.quality_value(evidence.expand(len(cs), -1), stacked, zs)
                        _, original = native_masks(proto, stacked, box, x)
                        for j, name in enumerate(names):
                            row['q_'+name] = float(qvalues[j])
                            row['soft_'+name] = float(soft_iou(zs[j], target, box))
                            row['iou_'+name] = int((original[j]&gt).sum())/max(int((original[j]|gt).sum()), 1)
                    old = reference[key(source)]
                    row['reference_iou_error'] = {arm: row['iou_'+arm]-old['iou_'+arm] for arm in ('A', 'D1', 'D')}
                    row['reference_q_error'] = {arm: row['q_'+arm]-old['q_c'+str(j)] for j, arm in enumerate(('A', 'D1', 'D'))}
                    row['reference_mask75_same'] = {arm: (row['iou_'+arm]>=.75)==(old['iou_'+arm]>=.75) for arm in ('A', 'D1', 'D')}
                    if not all(np.isfinite(value) for name, value in row.items() if isinstance(value, (int, float))):
                        raise FloatingPointError('Nonfinite diagnostic record')
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
                    rows.append(row)
                seen_images.append(iid)
                progress = {'images': len(seen_images), 'requested_images': len(ids), 'candidates': len(rows), 'elapsed_s': time.monotonic()-started}
                dump(out/'PROGRESS.json', progress)
                if pos % 8 == 0 or pos+1 == len(ids):
                    print(json.dumps(progress), flush=True)
        if seen_images != ids or {key(r) for r in rows} != set(reference):
            raise AssertionError('Fixed image/candidate membership incomplete')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), model_state[name], atol=0, rtol=0)
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise AssertionError('Quality checkpoint was not immutable')
        summary = {'metadata': metadata, 'elapsed_s': time.monotonic()-started,
                   'quality_model_unchanged': True, 'original_model_buffers_unchanged': True,
                   'candidate_identity_exact': True, 'reference': {
                       'candidate_count': len(reference),
                       'max_abs_iou_error': {arm: max(abs(r['reference_iou_error'][arm]) for r in rows) for arm in ('A', 'D1', 'D')},
                       'max_abs_q_error': {arm: max(abs(r['reference_q_error'][arm]) for r in rows) for arm in ('A', 'D1', 'D')},
                       'mask75_differences': {arm: sum(not r['reference_mask75_same'][arm] for r in rows) for arm in ('A', 'D1', 'D')},
                   }, 'groups': {
                       'all': stats(rows),
                       'baseline_success': stats([r for r in rows if r['iou_A'] >= .75]),
                       'final_damage': stats([r for r in rows if r['iou_A'] >= .75 and r['iou_D'] < .75]),
                       'preserved_success': stats([r for r in rows if r['iou_A'] >= .75 and r['iou_D'] >= .75]),
                       'baseline_failure': stats([r for r in rows if r['iou_A'] < .75]),
                   }}
        dump(out/'SUMMARY.json', summary)
        dump(out/'COMPLETE.json', {'completed': True, 'images': len(ids), 'candidates': len(rows), 'summary_sha256': sha256(out/'SUMMARY.json'),
                                  'parameter_updates': False, 'full_final_reassessment': False})
        print(json.dumps({'completed': True, 'candidates': len(rows), 'reference': summary['reference']}), flush=True)
    finally:
        stream.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--images', required=True, type=int, choices=(2, 64))
    args = ap.parse_args()
    try:
        run(args)
    except BaseException as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out)/'FAILURE.json').write_text(json.dumps({'error': repr(exc), 'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
        raise
