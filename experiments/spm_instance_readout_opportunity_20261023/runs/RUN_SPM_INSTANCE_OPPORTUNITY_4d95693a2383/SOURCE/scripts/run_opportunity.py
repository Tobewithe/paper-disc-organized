import argparse
from collections import defaultdict
import contextlib
import io
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from pycocotools import mask as cm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch

from readout_runtime import context, dump, read, sha
from readout_core import (forward_native, representations, predicted_supports, predicted_readout,
                          gt_supports, oracle_masks, iou_mask, valid_grid)


def encode(mask):
    result = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    result['counts'] = result['counts'].decode('ascii')
    return result


def prediction(row, d, mask):
    x0, y0, x1, y1 = [float(x) for x in d[:4]]
    return dict(image_id=row['image_id'], category_id=1, score=float(d[4]),
                bbox=[x0, y0, x1-x0, y1-y0], segmentation=encode(mask))


def gt_pairs(gt_file, reference, identities):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_file))
        dt = gt.loadRes([{k: v for k, v in p.items() if k != 'segmentation'} for p in reference])
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.iouThrs = np.array([.75])
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ['all']
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
    pairs = defaultdict(list)
    for frame in ev.evalImgs:
        if frame is None:
            continue
        assert frame['aRng'] == [0, 1e10] and frame['maxDet'] == 100
        for i, detection_id in enumerate(frame['dtIds']):
            ann_id = int(frame['dtMatches'][0, i])
            if ann_id and not frame['dtIgnore'][0, i]:
                identity = identities[detection_id-1]
                assert identity['image_id'] == frame['image_id']
                ann = gt.anns[ann_id]
                source = reference[detection_id-1]
                x, y, w, h = source['bbox']; a, b, c, d = ann['bbox']
                inter = max(0, min(x+w, a+c)-max(x, a))*max(0, min(y+h, b+d)-max(y, b))
                box_iou = inter/max(1e-20, w*h+c*d-inter)
                assert box_iou >= .75-1e-7
                pairs[frame['image_id']].append({**identity, 'coco_dt_id': detection_id,
                                                  'annotation_id': ann_id, 'box_iou': box_iou})
    return gt, pairs


def strata(area):
    return 'S' if area < 32**2 else 'M' if area < 96**2 else 'L'


def training_coverage(parent, rows, cfg):
    payload = torch.load(parent/cfg['embedding'], map_location='cpu', weights_only=False)
    episodes = {e['source_id']: e for e in payload['episodes'] if e['domain'] == 'real'}
    totals = {s: dict(total=0, sampled=0) for s in ['S', 'M', 'L']}
    frames = []
    for row in rows:
        if row['domain'] != 'real' or row['split'] != 'train':
            continue
        episode = episodes.get(row['source_id'], {})
        selected = set(episode.get('instance_ids', []))
        assert all(1 <= k <= len(row['rles']) for k in selected)
        for k, rle in enumerate(row['rles'], 1):
            size = strata(float(cm.area(rle)))
            totals[size]['total'] += 1
            totals[size]['sampled'] += int(k in selected)
        frames.append(dict(source_id=row['source_id'], group=row['group'], instances=len(row['rles']),
                           sampled_instances=len(selected), selected_instance_ids=sorted(selected),
                           sampled_token_count=episode.get('stop', 0)-episode.get('start', 0)))
    assert len(frames) == 1272
    for value in totals.values():
        value['sampled_fraction'] = value['sampled']/max(1, value['total'])
    return dict(total_instances=sum(v['total'] for v in totals.values()),
                sampled_instances=sum(v['sampled'] for v in totals.values()), by_size=totals,
                frames=frames, scope='fixed_extraction_episodes_not_frequency_of_optimizer_reuse')


def summarize_opportunity(ledger, representation, cfg):
    failure = [r for r in ledger if r['native_mask_iou'] < .75]
    success = [r for r in ledger if r['native_mask_iou'] >= .75]
    def part(rows):
        n = len(rows)
        delta = [r['oracle'][representation]['mask_iou']-r['native_mask_iou'] for r in rows]
        return dict(objects=n, prototype_coverage=sum(r['oracle'][representation]['prototype_available'] for r in rows)/max(1, n),
                    mean_native_iou=float(np.mean([r['native_mask_iou'] for r in rows])) if n else None,
                    mean_oracle_iou=float(np.mean([r['oracle'][representation]['mask_iou'] for r in rows])) if n else None,
                    mean_iou_delta=float(np.mean(delta)) if n else None,
                    mask75_fraction=sum(r['oracle'][representation]['mask_iou'] >= .75 for r in rows)/max(1, n),
                    decreases_at_least_05=sum(d <= -.05 for d in delta),
                    success_to_failure=sum(r['native_mask_iou'] >= .75 and r['oracle'][representation]['mask_iou'] < .75 for r in rows))
    f = part(failure)
    positive = (f['objects'] >= cfg['min_failures'] and f['prototype_coverage'] >= cfg['min_coverage']
                and f['mean_iou_delta'] >= cfg['min_failure_mean_iou_gain']
                and f['mask75_fraction'] >= cfg['min_failure_repair_fraction'])
    return dict(failure=f, success=part(success), all=part(ledger), opportunity_positive=positive,
                by_size={s: part([r for r in failure if r['size'] == s]) for s in ['S', 'M', 'L']})


def decision(summary, cfg):
    sufficient = all(s['failure']['objects'] >= cfg['min_failures'] and
                     s['failure']['prototype_coverage'] >= cfg['min_coverage'] for s in summary.values())
    if not sufficient:
        status = 'necessary_condition_or_precision_insufficient'
        direction = 'fix_measured_dense_grid_or_instance_sampling_coverage_before_new_readout_training'
    elif summary['T']['opportunity_positive']:
        status = 'trained_instance_mean_readout_opportunity_supported_GT_only'
        direction = 'instance_conditioned_support_generation_and_mask_readout_with_matched_training_controls'
    elif summary['U']['opportunity_positive'] or summary['R']['opportunity_positive']:
        status = 'current_trained_projection_mean_readout_not_supported_control_has_opportunity'
        direction = 'correct_dense_small_instance_supervision_and_projection_compression_before_head_changes'
    else:
        status = 'single_mean_readout_no_practical_opportunity_in_frozen_coordinates'
        direction = 'direct_mask_supervised_query_and_basis_learning_with_matched_input_and_capacity'
    return dict(status=status, chosen_research_direction=direction, new_training_started=False,
                GT_oracle_does_not_satisfy_deployment_goal=True, old_configuration_unchanged=True)


@torch.no_grad()
def execute(root, run, meta, cfg):
    parent = Path(cfg['parent_root'])
    sys.path.insert(0, str(parent/'scripts'))
    import ultralytics
    from common import state_hash
    from task_model import load, Projector, query_features
    from evaluate import predict, score
    assert ultralytics.__version__ == '8.4.100'
    assert torch.__version__ == '2.5.1+cu124'
    rows = read(parent/'ALL_DATASET.json')['rows']
    dev = [r for r in rows if r['domain'] == 'real' and r['split'] == 'dev']
    assert len(dev) == 432 and len({r['image_id'] for r in dev}) == 432
    model = load(parent/cfg['native_checkpoint']).eval().requires_grad_(False)
    before = state_hash(model)
    body = state_hash(model, include_head=False)
    upstream = load(parent/cfg['upstream_checkpoint'], device='cpu').eval()
    assert state_hash(upstream, include_head=False) == body
    del upstream
    memory = torch.load(parent/cfg['memory'], map_location='cpu', weights_only=False)
    assert memory['upstream_checkpoint_sha256'] == cfg['upstream_sha256'] and memory['train_only']
    trained = Projector().cuda().eval().requires_grad_(False)
    trained.load_state_dict(memory['projector'])
    torch.manual_seed(cfg['random_projection_seed'])
    random = Projector().cuda().eval().requires_grad_(False)
    captured = {}
    hooks = [model.model[i].register_forward_hook(lambda layer, inputs, output, index=i: captured.__setitem__(index, output))
             for i in [2, 16]]
    reference = read(parent/cfg['reference_predictions'])
    by_image = defaultdict(list)
    for p in reference:
        by_image[p['image_id']].append(p)
    predictions = {k: [] for k in ['native', 'T', 'U', 'R', 'T_wrong']}
    identities, frame_audit, predicted_rles, timings = [], [], {}, []
    engineering = []
    checked_groups = set()
    start = time.monotonic()
    # First pass: RGB/native prediction only. DEV_COCO/GT masks are not loaded until this pass is complete.
    for frame_index, row in enumerate(dev):
        assert sha(row['image']) == row['image_sha256'], row['source_id']
        rgb = cv2.imread(row['image'])
        assert rgb.shape == (640, 640, 3)
        image = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1).copy()).cuda().float()[None]/255
        torch.cuda.synchronize(); begin = time.perf_counter()
        det, native, logits, head_ids, features = forward_native(model, image, captured)
        if len(engineering) < 2 and row['group'] not in checked_groups:
            checked_groups.add(row['group'])
            ref_det, ref_mask = predict(model, image)
            assert np.array_equal(ref_det, det.cpu().numpy()) and np.array_equal(ref_mask, native.cpu().numpy())
            feature_error = float((query_features(model, image)-features).abs().max())
            assert feature_error == 0
            engineering.append(dict(image_id=row['image_id'], source_id=row['source_id'], native_decoder_exact=True,
                                    feature_subgraph_max_abs_error=feature_error))
        query = representations(features, memory, trained, random)
        valid = valid_grid(row['valid_rect'], image.device)
        supports, confident, mask_iou = predicted_supports(det, logits, native, valid)
        outputs, details = {'native': native}, {}
        for name in ['T', 'U', 'R']:
            outputs[name], details[name] = predicted_readout(query[name], supports, confident, mask_iou, det[:, :4], native)
        outputs['T_wrong'], details['T_wrong'] = predicted_readout(query['T'], supports, confident, mask_iou, det[:, :4], native, wrong=True)
        # Whole-readout deletion is the exact tensor produced by the native decoder.
        assert torch.equal(outputs['native'], native)
        torch.cuda.synchronize()
        timings.append((time.perf_counter()-begin)*1000)
        det_np = det.cpu().numpy()
        per_image = {}
        keep_audit = {}
        for name, masks in outputs.items():
            per_image[name] = {}
            masks = masks.cpu().numpy()
            kept = []
            for output_id, (d, mask) in enumerate(zip(det_np, masks)):
                # Preserve every native row identity, even if the altered mask becomes empty.
                enc = encode(mask)
                per_image[name][output_id] = enc
                if mask.any():
                    predictions[name].append(prediction(row, d, mask))
                    kept.append(output_id)
            keep_audit[name] = kept
        assert predictions['native'][-len(det):] == by_image[row['image_id']], row['image_id']
        for output_id, head_id in enumerate(head_ids.cpu().tolist()):
            identities.append(dict(image_id=row['image_id'], source_id=row['source_id'], group=row['group'],
                                   head_output_rank=head_id, native_output_detection_id=output_id))
        predicted_rles[row['image_id']] = per_image
        frame_audit.append(dict(image_id=row['image_id'], source_id=row['source_id'],
                                native_output_rows=len(det), head_output_ranks=head_ids.cpu().tolist(),
                                kept_native_output_ids=keep_audit, support_audit=details))
        if frame_index % 25 == 0:
            dump(run/'PROGRESS.json', dict(stage='GT_free_readout', images=frame_index+1, total=432,
                                           elapsed_seconds=time.monotonic()-start))
            print('GT_FREE', frame_index+1, 432, flush=True)
    assert len(engineering) == 2
    assert predictions['native'] == reference
    dump(run/'ENGINEERING.json', dict(frames=engineering, whole_readout_deletion_exact=True,
                                      frozen_body_sha256=body, native_prediction_rows=len(reference)))
    dump(run/'OUTPUT_IDENTITIES.json', identities)
    dump(run/'FRAME_READOUT_AUDIT.json', frame_audit)
    for name, p in predictions.items():
        dump(run/f'predictions_{name}.json', p)
    metrics = {}
    for name, p in predictions.items():
        seg = score(parent/'DEV_COCO.json', p, 'segm')
        box = score(parent/'DEV_COCO.json', p, 'bbox')
        metrics[name] = dict(mask_ap=seg[0], mask_ap75=seg[2], segm_stats=seg, bbox_stats=box,
                             images=432, predictions=len(p), split='development_not_independent_test')
        print('AP', name, seg[0], flush=True)
    assert abs(metrics['native']['mask_ap']-cfg['expected_native_mask_ap']) < 1e-7
    dump(run/'METRICS.json', metrics)
    dump(run/'REPLAY.json', dict(prediction_list_exact=True, native_ap_exact=True,
                                original_prediction_file_sha256=sha(parent/cfg['reference_predictions']),
                                current_prediction_file_sha256=sha(run/'predictions_native.json')))
    # Official bbox matches are constructed only after all deployable outputs have been fixed.
    gt, pairs = gt_pairs(parent/'DEV_COCO.json', reference, identities)
    ledger = []
    example_frames = set()
    for frame_index, row in enumerate(dev):
        if not pairs[row['image_id']]:
            continue
        assert sha(row['label']) == row['label_sha256']
        anns = gt.imgToAnns[row['image_id']]
        assert [a['segmentation'] for a in anns] == row['rles']
        local = {ann['id']: i for i, ann in enumerate(anns)}
        gt_masks = np.stack([gt.annToMask(a).astype(bool) for a in anns])
        rgb = cv2.imread(row['image'])
        image = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1).copy()).cuda().float()[None]/255
        det, native, logits, head_ids, features = forward_native(model, image, captured)
        queries = representations(features, memory, trained, random)
        supports = gt_supports(gt_masks, row['valid_rect'], image.device)
        requests = [(p['native_output_detection_id'], local[p['annotation_id']]) for p in pairs[row['image_id']]]
        oracle, audits = {}, {}
        for name, query in queries.items():
            oracle[name], audits[name] = oracle_masks(query, supports, det, native, requests)
        for p in pairs[row['image_id']]:
            output_id = p['native_output_detection_id']; ann = gt.anns[p['annotation_id']]
            mask = gt_masks[local[p['annotation_id']]]
            native_mask = native[output_id].cpu().numpy()
            record = {**p, 'GT_area': ann['area'], 'size': strata(ann['area']),
                      'native_mask_iou': iou_mask(native_mask, mask), 'oracle': {}, 'GT_free': {}}
            for name in ['T', 'U', 'R']:
                value = oracle[name][output_id].cpu().numpy()
                record['oracle'][name] = {**audits[name][output_id], 'mask_iou': iou_mask(value, mask),
                                          'segmentation': encode(value)}
            for name in ['T', 'U', 'R', 'T_wrong']:
                value = cm.decode(predicted_rles[row['image_id']][name][output_id]).astype(bool)
                record['GT_free'][name] = dict(mask_iou=iou_mask(value, mask))
            ledger.append(record)
        if row['group'] not in example_frames:
            example_frames.add(row['group'])
            np.savez_compressed(run/f'example_{row["image_id"]}.npz', rgb=rgb, gt_masks=gt_masks,
                                native=native.cpu().numpy(), gt_free=cm.decode([predicted_rles[row['image_id']]['T'][k] for k in range(len(native))]).transpose(2, 0, 1),
                                oracle_T=np.stack([oracle['T'][k].cpu().numpy() for k, _ in requests]),
                                boxes=det[:, :4].cpu().numpy(),
                                pairs=np.asarray(requests), valid_rect=np.asarray(row['valid_rect']),
                                T_query=queries['T'].cpu().numpy())
        if frame_index % 25 == 0:
            dump(run/'PROGRESS.json', dict(stage='GT_only_oracle', images=frame_index+1, total=432))
            print('GT_ORACLE', frame_index+1, 432, flush=True)
    for hook in hooks:
        hook.remove()
    assert state_hash(model) == before
    coverage = training_coverage(parent, rows, cfg)
    summary = {name: summarize_opportunity(ledger, name, cfg) for name in ['T', 'U', 'R']}
    chosen = decision(summary, cfg)
    chosen.update(GT_free_T_minus_native_pp=metrics['T']['mask_ap']-metrics['native']['mask_ap'],
                  five_pp_numerical_target_met=metrics['T']['mask_ap'] >= cfg['expected_native_mask_ap']+5,
                  goal_achieved=False)
    dump(run/'OBJECT_LEDGER.json', ledger)
    dump(run/'TRAINING_INSTANCE_COVERAGE.json', coverage)
    dump(run/'OPPORTUNITY_SUMMARY.json', summary)
    dump(run/'DECISION.json', chosen)
    dump(run/'COST.json', dict(median_full_T_U_R_and_wrong_readout_ms=float(np.median(timings)),
                               scope='native_forward_plus_all_diagnostic_controls_excludes_disk_and_COCO_not_single_deployment_cost'))
    meta.update(images_GT_free=432, images_GT_oracle=len([r for r in dev if pairs[r['image_id']]]),
                official_Box75_matched_objects=len(ledger), frozen_model_unchanged=True,
                engineering_frames=2, native_prediction_exact=True)
    from readout_figures import figures_and_report
    figures_and_report(root, run, metrics, summary, coverage, ledger, chosen)
    dump(run/'SUMMARY.json', dict(metrics=metrics, opportunity=summary, decision=chosen,
                                  training_optimizer_steps=0, elapsed_seconds=time.monotonic()-start))
    print('DECISION', chosen, flush=True)


def main(root):
    with context(root) as (run, meta, cfg):
        execute(root, run, meta, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    main(Path(args.root))
