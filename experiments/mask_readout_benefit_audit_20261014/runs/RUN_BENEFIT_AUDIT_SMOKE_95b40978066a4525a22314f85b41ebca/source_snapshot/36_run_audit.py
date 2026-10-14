"""One frozen checkpoint; paired stages and preregistered GT-free signal audit."""
from collections import defaultdict
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_readout_benefit_audit_20261014'
SOFT = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014'
sys.path.insert(0, str(SOFT/'scripts'))
import pilot_common as prior
from posttrain_run import RegisteredRun, require, sha256, read_json, write_new, tensor_state_hash
from posttrain_audit import read_jsonl, compare_outputs
from s1_geometry import geometry, project_gt, full_rle_iou, iter_prediction_array
from audit_statistics import counted, summarize, decide, failure_stage, FEATURES, STAGES

TRAIN = SOFT/'runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
EVAL = SOFT/'runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
CHECKPOINT = TRAIN/'last.pt'
BASE = prior.BASE
RECOVERY = STUDY/'runs/RUN_BENEFIT_AUDIT_FULL_1ab7051a0f39457090f9ba6fdde8a336'
CODE_FILES = list(dict.fromkeys(prior.CODE_FILES + [
    'experiments/mask_readout_benefit_audit_20261014/'+name for name in
    ('PROTOCOL.md', 'PROTOCOL_V2.md', 'INPUT_EXPECTATIONS.json', 'SOURCE_CONTEXT.json', 'SOURCE_DIRTY.patch',
     'scripts/run_audit.py', 'scripts/audit_statistics.py', 'scripts/test_audit.py',
     'scripts/collect_audit.py', 'scripts/launch_audit.ps1')]))


class BenefitRun(RegisteredRun):
    def __init__(self, scope):
        super().__init__('BENEFIT_AUDIT_'+scope.upper(), 'BENEFIT_AUDIT', EVAL, CODE_FILES,
                         900 if scope == 'smoke' else 3600)
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.BENEFIT_AUDIT.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_BENEFIT_AUDIT_20261014', scope=scope,
                         new_training_steps=0, optimization_started=False, baseline_run_id=BASE.name,
                         retry_of=RECOVERY.name, engineering_version='V2_per_image_memory_release')

    def __enter__(self):
        require(not list(ROOT.glob('experiments/*/.*.lock')), 'another stage has an unresolved lock')
        return super().__enter__()

    def freeze(self, contract):
        context = read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources = {r: sha256(ROOT/r) for r in CODE_FILES}
        self.meta.update(code_commit=context['code_commit'], dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json', {'files': self.sources, 'code_commit': context['code_commit'],
                                                   'actual_sources_snapshotted': True})
        for i, rel in enumerate(self.sources):
            target = self.path/'source_snapshot'/(f'{i:02d}_'+Path(rel).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel, target)
        shutil.copyfile(STUDY/'PROTOCOL_V2.md', self.path/'PROTOCOL.md')
        shutil.copyfile(STUDY/'SOURCE_DIRTY.patch', self.path/'dirty.patch')
        write_new(self.path/'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'),
                    input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))


def inspect_inputs():
    data = prior.inspect_inputs()
    expected = read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for rel, digest in expected['files'].items():
        require(sha256(ROOT/rel) == digest, 'new audit input drift: '+rel)
    for rel, digest in expected['previous_frozen_sources'].items():
        require(sha256(ROOT/rel) == digest, 'soft-IoU implementation drift: '+rel)
    for parent in (TRAIN, EVAL):
        require(read_json(parent/'run.json')['execution_status'] == 'completed', 'source run incomplete')
        require(read_json(parent/'CONFIG_AUDIT.json')['status'] == 'passed', 'source config unverified')
    require(read_json(EVAL/'DECISION.json')['status'] == 'failed', 'original S2 status changed')
    contract = read_json(EVAL/'EVAL_CONTRACT.json')
    require(contract['environment'] == data['environment'], 'soft-IoU runtime mismatch')
    require(contract['checkpoints']['B']['sha256'] == sha256(CHECKPOINT), 'different checkpoint')
    require(contract['dev_ids'] == data['split']['dev_ids'], 'dev IDs changed')
    panel = read_json(EVAL/'DAMAGE_PANEL.json')
    require(panel['checkpoint_sha256'] == sha256(CHECKPOINT) and
            panel['inference_contract_sha256'] == sha256(EVAL/'EVAL_CONTRACT.json'), 'panel binding changed')
    require(len(panel['sample']) == 1024 and len(read_json(EVAL/'EVALUATION_IMAGES.json')) == 1314, 'wrong source populations')
    require(read_json(RECOVERY/'run.json')['execution_status'] == 'error', 'recovery source not an interrupted engineering run')
    data.update(audit_inputs=expected['files'], source_evaluation_contract=contract, damage_panel=panel)
    return data


def check_recovery(run, records):
    path = RECOVERY/'PAIR_MEASUREMENTS.jsonl'
    old, omitted = [], 0
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            omitted += 1
            continue
        old.append(json.loads(line))
    key = lambda r: (r['role'], r['image_id'], r['annotation_id'], r['output_id'])
    previous = {key(r): r for r in old}
    actual = {key(r): r for r in records}
    require(len(previous) == len(old), 'duplicate recovery rows')
    matched = previous.keys() & actual.keys()
    if run.meta['scope'] == 'full':
        require(matched == previous.keys(), 'old complete rows missing from recovery')
    maximum = 0.
    for k in matched:
        a, b = previous[k], actual[k]
        for name in a:
            if name != 'stages':
                require(a[name] == b[name], 'recovery changed '+name)
        for stage in STAGES:
            if stage != 'soft_roi' or a['stages'][stage]['delta'] is None:
                require(a['stages'][stage] == b['stages'][stage], 'recovery changed binary counts/IoU')
                continue
            sa, sb = a['stages'][stage], b['stages'][stage]
            values = [(sa['delta'], sb['delta'])]
            values += [(sa[side][name], sb[side][name]) for side in ('before', 'after') for name in sa[side]]
            maximum = max(maximum, max(abs(x-y) for x, y in values))
            require(maximum <= 1e-7, 'recovery changed soft arithmetic')
    write_new(run.path/'RECOVERY_OVERLAP_CHECK.json', dict(status='passed', source_run=RECOVERY.name,
        source_partial_sha256=sha256(path), complete_old_pairs=len(old), matched_pairs=len(matched),
        incomplete_trailing_lines_preserved_and_omitted=omitted, maximum_soft_absolute_error=maximum,
        all_integer_counts_signals_original_IoU_and_stage_classes_equal=True))


def mapped(oid, indices, source_sha):
    return None if oid is None else source_sha+':'+str(indices[int(oid.rsplit(':', 1)[1])])


def prepare_pairs(ids, data):
    selected = set(ids)
    baseline = read_jsonl(EVAL/'BASELINE_SUBSET_GT_LEDGER.jsonl')
    method = read_jsonl(EVAL/'B_GT_LEDGER.jsonl')
    old = {(r['image_id'], r['annotation_id']): r for r in baseline}
    new = {(r['image_id'], r['annotation_id']): r for r in method}
    require(len(old) == len(baseline) and len(new) == len(method) and old.keys() == new.keys(), 'GT coverage mismatch')
    mapping = read_json(EVAL/'B_ROW_MAPPING.json')
    indices, source_sha = mapping['original_global_indices'], mapping['source_prediction_sha256']
    for key, r in old.items():
        require(r['is_ignored'] == new[key]['is_ignored'], 'GT ignore changed')
        require(r['bbox75_match_output_id'] == mapped(new[key]['bbox75_match_output_id'], indices, source_sha), 'bbox matching changed')
    result = defaultdict(list)
    for row in read_jsonl(EVAL/'B_FIXED_PAIR_EFFECTS.jsonl'):
        key = (row['image_id'], row['annotation_id'])
        require(row['baseline_output_detection_id'] == old[key]['bbox75_match_output_id'], 'fixed bbox identity changed')
        if row['image_id'] in selected:
            result[row['image_id']].append(dict(role='dev_bbox75', image_id=key[0], annotation_id=key[1],
                output_id=row['baseline_output_detection_id'], population=row['population'],
                expected_baseline_iou=row['baseline_iou'], expected_method_iou=row['method_iou'],
                task_damaged=old[key]['task_mask75'] == 'success' and new[key]['task_mask75'] == 'failure',
                method_segm_output_id=mapped(new[key]['segm75_match_output_id'], indices, source_sha)))
    for entry in data['damage_panel']['sample']:
        key = (entry['image_id'], entry['annotation_id'])
        require(old[key]['task_mask75'] == 'success' and not old[key]['is_ignored'], 'invalid damage sample')
        oid = old[key]['segm75_match_output_id']
        require(oid is not None, 'successful GT without segm match')
        if key[0] in selected:
            result[key[0]].append(dict(role='success_segm75', image_id=key[0], annotation_id=key[1],
                output_id=oid, population='S', expected_baseline_iou=None, expected_method_iou=None,
                task_damaged=new[key]['task_mask75'] == 'failure',
                method_segm_output_id=mapped(new[key]['segm75_match_output_id'], indices, source_sha)))
    return result, mapping


def tensor_count(prediction, target, roi=None):
    import torch
    require(prediction.shape == target.shape, 'count shape mismatch')
    p, g = prediction.bool(), target.bool()
    roi = torch.ones_like(g) if roi is None else roi.bool()
    counts = [int(v.sum().item()) for v in ((p & g) & roi, (p & ~g) & roi, (~p & g) & roi, (~p & ~g) & roi)]
    return counted(*counts)


def paired_counts(before, after):
    return {'before': before, 'after': after, 'delta': after['iou']-before['iou']}


def input_gt(mask, cache, device):
    import torch
    from torch.nn import functional as F
    info = geometry(cache)
    t = torch.tensor(mask.copy(), dtype=torch.float32)[None, None]
    resized = F.interpolate(t, info['resized_hw'], mode='nearest')
    l, top, r, b = info['ltrb_padding']
    return F.pad(resized, (l, r, top, b))[0, 0].to(device).bool()


def neighbor_score(cache, rank):
    from pycocotools import mask as masks
    boxes = cache['boxes_input'].tolist()
    selected = [i for i in range(len(boxes)) if cache['classes'][i] == cache['classes'][rank]
                and cache['raw_ids'][i] != cache['raw_ids'][rank]]
    if not selected:
        return 0.0
    xywh = lambda b: [b[0], b[1], max(0., b[2]-b[0]), max(0., b[3]-b[1])]
    return float(masks.iou([xywh(boxes[rank])], [xywh(boxes[i]) for i in selected], [0]*len(selected)).max())


def measure(run, source, specifications, data):
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    from ultralytics.utils import ops
    from adapter_model import CoefficientAdapter, loss_grids
    from native_output_bridge import encode_rle
    with contextlib.redirect_stdout(sys.stdout):
        coco = COCO(str(prior.GT))
    mapping = read_json(EVAL/'B_ROW_MAPPING.json')['original_global_indices']
    wanted = set(source.predictions)
    retained = {mapping[i]: row for i, row in enumerate(iter_prediction_array(EVAL/'B_predictions.json')) if mapping[i] in wanted}
    require(set(retained) == wanted, 'method endpoint rows missing')
    model = CoefficientAdapter('B').cuda()
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    require(saved['arm'] == 'B' and saved['completed_epochs'] == 3, 'wrong checkpoint state')
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    state_before = tensor_state_hash(model)
    records, checks = [], []
    started = time.monotonic()
    with (run.path/'PAIR_MEASUREMENTS.jsonl').open('x', encoding='utf8') as out, torch.inference_mode():
        for number, image_id in enumerate(source.records, 1):
            run.check_budget()
            cache, original = source.load(image_id)
            delta = model(cache, 'eval:0')
            require(torch.isfinite(delta).all().item(), 'nonfinite coefficient residual')
            c0 = cache['c0'].to(delta)
            logits, native, roi = loss_grids(cache, c0+delta)
            p, boxes = cache['proto'].to(delta), cache['boxes_input'].to(delta)
            before_input = ops.process_mask_native(p, c0, boxes, shape=cache['native_decode_hw'])
            after_input = ops.process_mask_native(p, c0+delta, boxes, shape=cache['native_decode_hw'])
            before_original = ops.scale_masks(before_input[None], cache['original_hw'], ratio_pad=cache['ratio_pad'])[0].byte()
            after_original = ops.scale_masks(after_input[None], cache['original_hw'], ratio_pad=cache['ratio_pad'])[0].byte()
            before_rles = [encode_rle(m) for m in before_original]
            after_rles = [encode_rle(m) for m in after_original]
            require(before_rles == [r['segmentation'] for r in original], 'zero-delta native RLE replay differs')
            method = [retained[i] for i in cache['prediction_indices']]
            compare_outputs(original, method)
            require(after_rles == [r['segmentation'] for r in method], 'frozen soft-IoU RLE replay differs')
            ranks = {i: k for k, i in enumerate(cache['prediction_indices'])}
            # All three signals are formed before retrieving GT for this image.
            signals = {}
            for rank in {ranks[int(r['output_id'].rsplit(':', 1)[1])] for r in specifications[image_id]}:
                area = int(masks.area(before_rles[rank]))
                intersection = int(masks.area(masks.merge([before_rles[rank], after_rles[rank]], intersect=True)))
                nroi = int(roi[rank].sum().item())
                uncertainty = float(((native[rank].abs() <= 1) & roi[rank]).sum().item()/nroi) if nroi else 0.
                signals[rank] = dict(preserved_foreground=intersection/area if area else 1.,
                    native_uncertainty=uncertainty, same_class_neighbor_iou=neighbor_score(cache, rank),
                    native_foreground_pixels=area, retained_foreground_pixels=intersection)
            for spec in specifications[image_id]:
                rank = ranks[int(spec['output_id'].rsplit(':', 1)[1])]
                require(spec['output_id'].split(':')[0] == source.manifest['predictions_sha256'], 'wrong source output SHA')
                annotation = coco.anns[spec['annotation_id']]
                require(annotation['image_id'] == image_id and annotation['category_id'] == original[rank]['category_id'], 'GT identity mismatch')
                gt_rle = coco.annToRLE(annotation)
                gt_mask = masks.decode(gt_rle)
                target = project_gt(gt_mask, cache).to(delta)
                region = roi[rank]
                nroi = int(region.sum().item())
                stages = {}
                if nroi:
                    soft = []
                    for z in (native[rank], logits[rank]):
                        probabilities = z.sigmoid()[region]
                        g = target[region]
                        intersection = (probabilities*g).sum()
                        union = (probabilities+g-probabilities*g).sum()
                        soft.append({'intersection': float(intersection), 'union': float(union),
                                     'iou': float((intersection+1e-6)/(union+1e-6))})
                    stages['soft_roi'] = paired_counts(*soft)
                    stages['binary_roi'] = paired_counts(tensor_count(native[rank] > 0, target, region),
                                                         tensor_count(logits[rank] > 0, target, region))
                else:
                    stages['soft_roi'] = stages['binary_roi'] = {'before': None, 'after': None, 'delta': None}
                stages['binary_proto_full'] = paired_counts(tensor_count((native[rank] > 0) & region, target),
                                                            tensor_count((logits[rank] > 0) & region, target))
                gt_input = input_gt(gt_mask, cache, delta.device)
                require(list(gt_input.shape) == list(before_input.shape[-2:]), 'native input/GT support differs')
                stages['binary_input_full'] = paired_counts(tensor_count(before_input[rank], gt_input),
                                                            tensor_count(after_input[rank], gt_input))
                gt_original = torch.tensor(gt_mask.copy(), device=delta.device).bool()
                stages['binary_original'] = paired_counts(tensor_count(before_original[rank], gt_original),
                                                          tensor_count(after_original[rank], gt_original))
                for side, rle in [('before', before_rles[rank]), ('after', after_rles[rank])]:
                    require(abs(stages['binary_original'][side]['iou']-full_rle_iou(rle, gt_rle)) <= 1e-12, 'pixel/RLE IoU differs')
                if spec['role'] == 'dev_bbox75':
                    require(abs(stages['binary_original']['before']['iou']-spec['expected_baseline_iou']) <= 1e-12, 'old baseline endpoint differs')
                    require(abs(stages['binary_original']['after']['iou']-spec['expected_method_iou']) <= 1e-12, 'old method endpoint differs')
                else:
                    require(stages['binary_original']['before']['iou'] >= .75, 'baseline segm match not successful')
                row = {**spec, 'output_rank': rank, 'raw_id': cache['raw_ids'][rank], 'signals': signals[rank],
                       'stages': stages, 'empty_roi': nroi == 0, 'ROI_pixels': nroi,
                       'input_hw': cache['input_hw'], 'proto_hw': list(cache['proto'].shape[-2:]),
                       'original_hw': cache['original_hw']}
                row['damage_stage_signature'] = failure_stage(row) if row['role'] == 'success_segm75' else None
                out.write(json.dumps(row, allow_nan=False)+'\n')
                records.append(row)
            checks.append(dict(image_id=image_id, outputs=len(original), zero_RLE_equal=True, method_RLE_equal=True,
                               nonmask_fields_equal=True, pairs=len(specifications[image_id])))
            if number % 50 == 0 or number == len(source.records):
                out.flush()
                run.update(phase='paired_stages_and_GT_free_signals', completed_images=number, total_images=len(source.records))
                print('BENEFIT_AUDIT', number, '/', len(source.records), 'seconds', round(time.monotonic()-started, 1), flush=True)
            del cache, original, method, delta, logits, native, roi, p, boxes, before_input, after_input, before_original, after_original
            # No change to arithmetic, shape, precision or decoding: release
            # lingering per-pair views before reclaiming unused CUDA allocations.
            target = gt_input = gt_original = region = probabilities = g = z = intersection = None
            del c0
            torch.cuda.empty_cache()
    require(len(records) == len({(r['role'], r['image_id'], r['annotation_id'], r['output_id']) for r in records}), 'duplicate pair key')
    require(len(records) == sum(map(len, specifications.values())), 'missing pairs')
    require(tensor_state_hash(model) == state_before and not model.training, 'model state/mode mutated')
    require(not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'gradient path active')
    write_new(run.path/'FROZEN_MODEL_STATE.json', dict(before=state_before, after=tensor_state_hash(model),
              parameters_require_grad=False, parameter_gradients_present=False, new_optimizer_steps=0,
              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    prior.jsonl(run.path/'IMAGE_REPLAY_CHECKS.jsonl', checks)
    del model
    torch.cuda.empty_cache()
    return records, dict(images=len(checks), outputs=sum(r['outputs'] for r in checks), elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('smoke', 'full'))
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        data = inspect_inputs()
        print('PREFLIGHT_PASSED', 'frozen_inputs', len(data['audit_inputs']), flush=True)
        return
    require(args.scope is not None, 'scope required')
    with BenefitRun(args.scope) as run:
        data = inspect_inputs()
        ids = read_json(EVAL/'EVALUATION_IMAGES.json')
        if args.scope == 'smoke': ids = ids[:8]
        dev_ids = data['split']['dev_ids']
        contract = dict(scope=args.scope, image_ids=ids, dev_ids=dev_ids, new_training_steps=0,
            checkpoint_sha256=sha256(CHECKPOINT), source_evaluation_run=EVAL.name,
            source_evaluation_contract_sha256=sha256(EVAL/'EVAL_CONTRACT.json'),
            native_profile_sha256=sha256(prior.PROFILE), environment=data['environment'],
            external_evaluator_sources=data['external_evaluator_sources'], inputs_sha256=data['audit_inputs'],
            signals=list(FEATURES), stages=list(STAGES), original_S2_gate_status='failed_unchanged',
            expected_full_populations=dict(dev_images=512, dev_pairs=2925, target_pairs=655, success_panel=1024,
                                           damaged_success_GT=53, images=1314, outputs=185621),
            AUC_gate=dict(minimum_auc=.60, minimum_corrected_ci_lower_strict=.50, minimum_each_half_auc=.55,
                          resamples=2000, seed=1729, quantiles=[.05/6, 1-.05/6]),
            inference_parameters='exact source evaluation contract including phase eval:0 and full output matrix',
            wall_seconds=run.wall_seconds)
        contract.update(engineering_version='V2_per_image_memory_release', recovery_from=RECOVERY.name,
                        original_scientific_protocol_sha256=sha256(STUDY/'PROTOCOL.md'))
        run.freeze(contract)
        if args.scope == 'full':
            require(args.smoke_run is not None, 'completed smoke required')
            require(read_json(args.smoke_run/'run.json')['execution_status'] == 'completed', 'smoke incomplete')
            require(read_json(args.smoke_run/'CONFIG_AUDIT.json')['status'] == 'passed', 'smoke failed')
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files'] == run.sources, 'sources changed after smoke')
            smoke = read_json(args.smoke_run/'EVAL_CONTRACT.json')
            require(smoke['environment'] == contract['environment'] and smoke['inputs_sha256'] == contract['inputs_sha256'], 'smoke configuration differs')
            run.update(smoke_run_id=args.smoke_run.name)
        specifications, mapping = prepare_pairs(ids, data)
        from adapter_data import FrozenCache
        source = FrozenCache(BASE/'trace', ids)
        records, export = measure(run, source, specifications, data)
        check_recovery(run, records)
        source.verify()
        prior.verify_after(data)
        for rel, digest in data['audit_inputs'].items():
            require(sha256(ROOT/rel) == digest, 'input changed during diagnostic: '+rel)
        summary = summarize(records, dev_ids)
        if args.scope == 'full':
            require(summary['pair_roles'] == {'dev_bbox75': 2925, 'success_segm75': 1024}, 'full pair coverage differs')
            require(sum(r['role'] == 'dev_bbox75' and r['population'] == 'T' for r in records) == 655, 'T changed')
            require(summary['damaged_success_GT'] == 53 and export['outputs'] == 185621, 'source damage/output count differs')
        write_new(run.path/'CONFIG_AUDIT.json', {'status': 'passed', 'checks': dict(
            frozen_inputs_and_cache_before_after=True, environment_and_external_sources=True,
            source_output_IDs_boxes_class_score_order=True, full_zero_and_method_RLE_replay=True,
            existing_dev_endpoint_and_damage_panel_identity=True, independent_bbox_segm_matching=True,
            zero_training_and_frozen_model_state=True, all_expected_pairs_covered=True),
            'checkpoint_sha256': sha256(CHECKPOINT), 'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json')})
        decision = decide(summary) if args.scope == 'full' else {'status': 'engineering_only', 'direction': None, 'next_action': 'Full frozen diagnostic'}
        summary.update(export=export, route_decision=decision, original_S2_decision='failed_unchanged',
                       scientific_scope='existing development data; zero-training ranking and stage diagnosis, no method utility validation')
        run.check_budget()
        run.finish(summary, status='passed' if args.scope == 'smoke' else decision['status'],
                   scientific='engineering_only' if args.scope == 'smoke' else 'diagnostic_'+decision['status'],
                   next_action=decision['next_action'])
        print('BENEFIT_AUDIT_DECISION', json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
