"""Read-only original detector feature cache for the cross-task bridge screen.

Only the Linux CUDA server may execute preparation. Existing official candidate
identities and exact uint8 inputs are reused; no TAL or labels are recomputed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch


IDENTITY_FIELDS = ('split', 'image_id', 'annotation_id', 'branch', 'raw_id',
                   'pyramid_level', 'target_gt_idx')
SCHEMA = 'cross-task-native-bridge-features-v1'
_CHECKED_FILES = set()


def identity(row):
    return tuple(row[k] for k in IDENTITY_FIELDS)


def _runtime(cfg):
    old = str(Path(cfg['screen_root']) / 'scripts')
    if old not in sys.path:
        sys.path.append(old)
    import online_runtime
    return online_runtime


def load_feature_cache(cfg):
    rt = _runtime(cfg)
    path = Path(cfg['feature_cache'])
    meta_path = path.with_name(path.name + '.meta.json')
    meta = rt.load_json(meta_path)
    if meta.get('completed') is not True or meta.get('schema') != SCHEMA:
        raise AssertionError('Feature preparation is not complete')
    stat = path.stat()
    token = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, meta['cache_sha256'])
    if token not in _CHECKED_FILES:
        if rt.sha256(path) != meta['cache_sha256']:
            raise AssertionError('Feature cache hash differs')
        _CHECKED_FILES.add(token)
    bank = torch.load(path, map_location='cpu', weights_only=False)
    if bank.get('schema') != SCHEMA:
        raise AssertionError('Wrong feature cache schema')
    if rt.sha256(Path(cfg['cache'])/'INDEX.json') != bank['source_index_sha256']:
        raise AssertionError('Official source population changed')
    if rt.sha256(Path(cfg['cache'])/'CACHE_IDENTITY.json') != bank['cache_identity_sha256']:
        raise AssertionError('Official source cache identity changed')
    if rt.sha256(cfg['weights']) != bank['weights_sha256']:
        raise AssertionError('Official source checkpoint changed')
    if rt.sha256(cfg['split']) != bank['split_sha256']:
        raise AssertionError('Frozen split file changed')
    bank['_feature_cache_sha256'] = meta['cache_sha256']
    return bank


def selection_for(image, bank, mode, device=None):
    if mode not in ('N', 'T', 'R', 'M'):
        raise ValueError('Unknown bridge source mode')
    iid = int(image['image_id'])
    row = bank['images'][iid]
    if (row['split'] != image['split']
            or not torch.equal(row['raw_ids'], image['raw_ids'].cpu())
            or not torch.equal(row['levels'], image['levels'].cpu())
            or [identity(r) for r in row['rows']] != [identity(r) for r in image['rows']]):
        raise AssertionError('Feature source candidate identities changed')
    result = dict(raw_ids=row['raw_ids'], levels=row['levels'])
    if mode in ('T', 'R'):
        result['source_features'] = row['cls' if mode == 'T' else 'box']
    elif mode == 'M':
        wrong = bank['wrong'][iid]
        pieces = []
        for k, (donor_id, donor_pos) in enumerate(zip(wrong['source_image_ids'].tolist(),
                                                    wrong['source_row_indices'].tolist())):
            donor = bank['images'][donor_id]
            if (donor_id == iid or donor['split'] != row['split']
                    or int(donor['levels'][donor_pos]) != int(row['levels'][k])
                    or int(donor['raw_ids'][donor_pos]) != int(wrong['source_raw_ids'][k])
                    or int(donor['pred_class'][donor_pos]) != int(wrong['donor_pred_class'][k])):
                raise AssertionError('Wrong-source donor identity violated')
            same_class = int(donor['pred_class'][donor_pos]) == int(row['pred_class'][k])
            if same_class != bool(wrong['same_predicted_class'][k]):
                raise AssertionError('Wrong-source class label violated')
            pieces.append(donor['cls'][donor_pos])
        result['source_features'] = torch.stack(pieces) if pieces else row['cls'].new_empty((0, 256))
    return {k: v.to(device) if device is not None else v for k, v in result.items()}


def _normalization(images):
    stats = {}
    for kind, dim in (('cls', 256), ('box', 64)):
        stats[kind] = []
        for level in range(3):
            tensors = [r[kind][r['levels'] == level] for r in images.values() if r['split'] == 'fit']
            values = torch.cat(tensors).double()
            if values.ndim != 2 or values.shape[1] != dim or not len(values):
                raise AssertionError('No fit support for fixed source normalization')
            mean = values.mean(0)
            raw_std = values.std(0, unbiased=False)
            stats[kind].append(dict(mean=mean.float(), std=raw_std.clamp_min(1e-6).float(),
                                    count=len(values), clamped_channels=int((raw_std < 1e-6).sum()),
                                    definition='fit-candidate population std, per scale, ddof=0, min=1e-6'))
    return stats


def _wrong_sources(images):
    """Cyclic first different-image donor, within split and predicted class/level."""
    exact, fallback = defaultdict(list), defaultdict(list)
    for iid, row in sorted(images.items()):
        for k, (level, cls) in enumerate(zip(row['levels'].tolist(), row['pred_class'].tolist())):
            item = (iid, k)
            exact[(row['split'], level, cls)].append(item)
            fallback[(row['split'], level)].append(item)

    def choose(pool, query):
        start = bisect_right(pool, query)
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if candidate[0] != query[0]:
                return candidate
        return None

    result, audit = {}, defaultdict(lambda: dict(candidates=0, same_predicted_class=0,
                                                fallback=0, distinct_donors=set()))
    for iid, row in sorted(images.items()):
        donors, positions, raw_ids, classes, fallbacks = [], [], [], [], []
        for k, (level, cls) in enumerate(zip(row['levels'].tolist(), row['pred_class'].tolist())):
            donor = choose(exact[(row['split'], level, cls)], (iid, k))
            is_fallback = donor is None
            if donor is None:
                donor = choose(fallback[(row['split'], level)], (iid, k))
            if donor is None:
                raise AssertionError(f'No other-image same-level donor: {row["split"]}/{iid}/{level}')
            source = images[donor[0]]
            source_class = int(source['pred_class'][donor[1]])
            if is_fallback != (source_class != cls):
                raise AssertionError('Fallback classification differs from actual donor class')
            donors.append(donor[0]); positions.append(donor[1])
            raw_ids.append(int(source['raw_ids'][donor[1]])); classes.append(source_class)
            fallbacks.append(is_fallback)
            summary = audit[row['split']]
            summary['candidates'] += 1
            summary['same_predicted_class'] += int(not is_fallback)
            summary['fallback'] += int(is_fallback)
            summary['distinct_donors'].add(donor)
        result[iid] = dict(source_image_ids=torch.tensor(donors, dtype=torch.long),
                           source_row_indices=torch.tensor(positions, dtype=torch.long),
                           source_raw_ids=torch.tensor(raw_ids, dtype=torch.long),
                           donor_pred_class=torch.tensor(classes, dtype=torch.long),
                           pred_class=row['pred_class'].clone(), levels=row['levels'].clone(),
                           fallback=torch.tensor(fallbacks, dtype=torch.bool),
                           same_predicted_class=~torch.tensor(fallbacks, dtype=torch.bool))
    for summary in audit.values():
        summary['distinct_donors'] = len(summary['distinct_donors'])
    return result, dict(audit)


def main(args):
    cfg_path, out = Path(args.config), Path(args.out)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    rt = _runtime(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(cfg['feature_cache'])
    meta_path = path.with_name(path.name + '.meta.json')
    temp = path.with_name(path.name + '.tmp')
    if any(p.exists() for p in (path, meta_path, temp, out/'COMPLETE.json', out/'FEATURE_AUDIT.json')):
        raise FileExistsError('Existing preparation output retained; choose an independent target')
    started = time.monotonic()
    limit = int(cfg.get('prepare_max_seconds', 900))
    if limit != 900:
        raise ValueError('This feature-preparation budget is fixed at 900 seconds')

    def check_time():
        if time.monotonic() - started > limit:
            raise TimeoutError('Feature preparation budget exhausted; no prefix conclusion/cache accepted')

    cfg = rt.resolve_runtime_config(cfg)
    index = rt.load_index(cfg)
    declared = rt.load_json(cfg['split'])
    splits = tuple(cfg.get('prepare_splits', ('fit', 'dev')))
    if splits != ('fit', 'dev'):
        raise ValueError('Only the fixed fit/dev screen may be prepared')
    for split in splits:
        if [int(x['image_id']) for x in index[split]] != [int(i) for i in declared[split]]:
            raise AssertionError('Original cache population differs from frozen split')
    replay = rt.FrozenReplay(cfg)
    head = replay.head
    if not head.end2end or head.nm != 32 or head.reg_max != 1 or head.nc != 80:
        raise AssertionError('Unexpected official original head')
    source_before = {k: rt.tensor_sha(v) for k, v in replay.source.state_dict().items()}
    maxima = defaultdict(float)

    def compare(name, actual, expected):
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        if actual.shape != expected.shape:
            raise AssertionError(name + ': shape mismatch')
        if actual.numel():
            maxima[name] = max(maxima[name], float((actual-expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    images, counts = {}, {}
    with torch.no_grad():
        for split in splits:
            counts[split] = dict(planned_images=len(index[split]), effective_images=0,
                                 candidates=0, no_positive_images=[])
            for step, item in enumerate(index[split], 1):
                check_time()
                iid = int(item['image_id'])
                if iid in images:
                    raise AssertionError('Image occurs in more than one frozen split')
                image = replay.read(iid)
                if image['split'] != split or len(image['raw_ids']) != int(item['n']):
                    raise AssertionError('Source image/index identity mismatch')
                raw = image['raw_ids'].to(replay.device).long()
                if len(raw):
                    features = replay.replay([image])
                    hidden_box, hidden_cls, rebuilt_box, rebuilt_cls = [], [], [], []
                    for level, f in enumerate(features):
                        box_branch, cls_branch = head.one2one_cv2[level], head.one2one_cv3[level]
                        b, c = box_branch[:-1](f), cls_branch[:-1](f)
                        if b.shape[1] != 64 or c.shape[1] != 256:
                            raise AssertionError('Actual source channel shape differs')
                        hidden_box.append(b.flatten(2)); hidden_cls.append(c.flatten(2))
                        rebuilt_box.append(box_branch[-1](b).flatten(2))
                        rebuilt_cls.append(cls_branch[-1](c).flatten(2))
                    b_all = torch.cat(hidden_box, 2)[0].T
                    c_all = torch.cat(hidden_cls, 2)[0].T
                    boxes_raw = torch.cat(rebuilt_box, 2)
                    class_logits = torch.cat(rebuilt_cls, 2)
                    direct = head.forward_head(features, **head.one2one)
                    compare('class_final_reconstruction', class_logits, direct['scores'])
                    compare('box_final_reconstruction', boxes_raw, direct['boxes'])
                    compare('original_c0', direct['mask_coefficient'][0].T[raw], image['c0'])
                    compare('original_decoded_boxes', head._get_decode_boxes(direct)[0].T[raw], image['boxes'])
                    score, pred_class = class_logits[0].T[raw].sigmoid().max(1)
                    compare('original_scores', score, image['predicted_scores'])
                    if not torch.equal(pred_class.cpu(), image['predicted_classes']):
                        raise AssertionError('Predicted class changed at same raw locations')
                    live = replay.native_outputs(features, [image])[0]
                    compare('original_h0', live['h'], image['operator']['h0'])
                    compare('original_c0_second_path', live['c'], image['c0'])
                    level_all = torch.cat([torch.full((f.shape[-2]*f.shape[-1],), l,
                                              device=replay.device, dtype=torch.long)
                                           for l, f in enumerate(features)])
                    if not torch.equal(level_all[raw].cpu(), image['levels']):
                        raise AssertionError('Raw/scale alignment changed')
                    cls = c_all[raw].detach().cpu().contiguous()
                    box = b_all[raw].detach().cpu().contiguous()
                    selected_logits = class_logits[0].T[raw].cpu().contiguous()
                    selected_raw_box = boxes_raw[0].T[raw].cpu().contiguous()
                    counts[split]['effective_images'] += 1
                else:
                    cls, box = torch.empty((0, 256)), torch.empty((0, 64))
                    pred_class = torch.empty(0, dtype=torch.long)
                    selected_logits, selected_raw_box = torch.empty((0, 80)), torch.empty((0, 4))
                    counts[split]['no_positive_images'].append(iid)
                if not all(bool(torch.isfinite(v).all()) for v in (cls, box, selected_logits, selected_raw_box)):
                    raise FloatingPointError('Nonfinite frozen feature')
                images[iid] = dict(image_id=iid, split=split, cls=cls, box=box,
                                   pred_class=pred_class.cpu().long(), class_logits=selected_logits,
                                   raw_box=selected_raw_box, levels=image['levels'].cpu().clone(),
                                   raw_ids=image['raw_ids'].cpu().clone(), rows=copy.deepcopy(image['rows']),
                                   source_image_cache_sha256=image['_asset_integrity']['compressed_sha256'],
                                   input_uint8_sha256=image['_asset_integrity']['input_uint8_sha256'])
                counts[split]['candidates'] += len(raw)
                check_time()
                if step % 50 == 0 or step == len(index[split]):
                    progress = dict(split=split, images=step, total=len(index[split]),
                                    counts=counts[split], elapsed_seconds=time.monotonic()-started)
                    rt.dump(out/'PROGRESS.json', progress)
                    print(json.dumps(progress), flush=True)
    check_time()
    replay.assert_unchanged()
    source_after = {k: rt.tensor_sha(v) for k, v in replay.source.state_dict().items()}
    if source_before != source_after:
        raise AssertionError('Frozen source parameter or buffer changed')
    stats = _normalization(images)
    wrong, wrong_audit = _wrong_sources(images)
    feature_hashes = {iid: {key:rt.tensor_sha(row[key])
                           for key in ('cls', 'box', 'class_logits', 'raw_box', 'pred_class', 'raw_ids', 'levels')}
                      for iid, row in images.items()}
    feature_hash = hashlib.sha256(json.dumps(feature_hashes, sort_keys=True).encode()).hexdigest()
    bank = dict(schema=SCHEMA, images=images, stats=stats, wrong=wrong,
                source_feature_hashes=feature_hashes, source_feature_hash=feature_hash,
                source_index_sha256=rt.sha256(Path(cfg['cache'])/'INDEX.json'),
                cache_identity_sha256=rt.sha256(Path(cfg['cache'])/'CACHE_IDENTITY.json'),
                split_sha256=rt.sha256(cfg['split']), weights_sha256=replay.weights_sha256,
                config_sha256=rt.sha256(cfg_path))
    for iid, row in images.items():
        check_time()
        if len(row['raw_ids']):
            # Full donor integrity validation before publishing the cache.
            selection_for(row, bank, 'M')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, temp)
    check_time()
    temp.rename(path)
    audit = dict(passed=True, schema=SCHEMA, counts=counts,
                 max_absolute_differences=dict(maxima), atol=3e-5, rtol=3e-5,
                 source_shapes=dict(cls=256, box=64, native_mask=64),
                 wrong_source=wrong_audit,
                 wrong_rule='same split; cyclic first other image of same level and predicted class; '
                            'only if absent use same-level other-image; never GT labels',
                 normalization='fit candidates only; per-scale mean and population std; std clamp 1e-6; FP64 accumulation',
                 normalization_counts={k:[v['count'] for v in s] for k,s in stats.items()},
                 normalization_clamped={k:[v['clamped_channels'] for v in s] for k,s in stats.items()},
                 original_source_state_unchanged=True, detector_and_upstream_gradients=False,
                 assignment_recomputed=False, source_state_sha256=source_before,
                 source_index_sha256=bank['source_index_sha256'],
                 cache_identity_sha256=bank['cache_identity_sha256'],
                 weights_sha256=replay.weights_sha256, config_sha256=bank['config_sha256'],
                 source_feature_hash=feature_hash,
                 code_sha256={str(Path(__file__).name):rt.sha256(__file__),
                              'online_runtime.py':rt.sha256(rt.__file__)},
                 feature_cache=str(path), feature_cache_sha256=rt.sha256(path),
                 elapsed_seconds=time.monotonic()-started, environment=replay.import_info)
    check_time()
    rt.dump(out/'FEATURE_AUDIT.json', audit)
    receipt = dict(completed=True, passed=True, schema=SCHEMA, cache_sha256=audit['feature_cache_sha256'],
                   source_index_sha256=bank['source_index_sha256'],
                   cache_identity_sha256=bank['cache_identity_sha256'],
                   config_sha256=bank['config_sha256'], counts=counts,
                   audit_path=str(out/'FEATURE_AUDIT.json'),
                   audit_sha256=rt.sha256(out/'FEATURE_AUDIT.json'),
                   elapsed_seconds=time.monotonic()-started)
    rt.dump(meta_path, receipt)
    rt.dump(out/'COMPLETE.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        failure = out/'FAILURE.json'
        if not failure.exists():
            failure.write_text(json.dumps(dict(completed=False, error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc()), indent=2), encoding='utf-8')
        raise
