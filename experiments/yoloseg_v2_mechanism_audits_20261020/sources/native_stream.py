"""Frozen native source batches and final-row extraction for joint mask adaptation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT / 'shared/evaluation',
                  ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts',
                  ROOT / 'experiments/official_gt_failure_rebaseline_20261013/scripts'):
    sys.path.insert(0, str(directory))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), f'refusing to replace {path}')
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def choose_training_ids(all_ids, original_ids, count=20000, seed=20261018):
    all_ids = set(all_ids)
    require(len(original_ids) == 2000 and len(set(original_ids)) == 2000, 'original 2000 manifest required')
    require(set(original_ids) <= all_ids, 'original fit image missing from train2017')
    remaining = sorted(all_ids - set(original_ids))
    extra = random.Random(seed).sample(remaining, count - len(original_ids))
    return sorted(list(original_ids) + extra)


def update_orders(batches, epochs=3, seed=20261018):
    """Shuffle source batch order, then shuffle each batch's positions into 4-image updates."""
    require(all(len(batch['image_ids']) == 16 for batch in batches), 'training source batches must all contain 16 images')
    rng = random.Random(seed)
    result = []
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        groups = []
        for index in order:
            positions = list(range(16))
            rng.shuffle(positions)
            groups.append({'source_batch_index': index, 'microgroups': [positions[s:s+4] for s in range(0, 16, 4)]})
        result.append(groups)
    return result


def make_validator(config, image_list, output, device='cpu'):
    import torch
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.data.converter import coco80_to_coco91_class
    from ultralytics.utils import ops
    profile = read(config['profile'])
    args = {**profile['native_args'], 'task': 'segment', 'mode': 'val'}
    validator = SegmentationValidator(save_dir=Path(output), args=args)
    annotations = read(config['val_annotations'])
    categories = sorted(annotations['categories'], key=lambda x: x['id'])
    validator.data = {'names': {i: row['name'] for i, row in enumerate(categories)}, 'nc': 80,
                      'channels': 3, 'val': str(image_list)}
    validator.device = torch.device(device)
    validator.stride = 32
    validator.nc = 80
    validator.end2end = True
    validator.training = False
    validator.class_map = coco80_to_coco91_class()
    validator.process = ops.process_mask_native
    return validator


def build_dataset(config, image_list, output, device='cpu'):
    validator = make_validator(config, image_list, output, device)
    # Labels do not determine our COCO training join. Any newly built Ultralytics
    # shape/label cache belongs to this Run, never to shared historical data.
    from ultralytics.data.dataset import YOLODataset
    original_cache_labels = YOLODataset.cache_labels
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    def own_cache(instance, path=None):
        return original_cache_labels(instance, path=out/'labels.cache')
    YOLODataset.cache_labels = own_cache
    try:
        dataset = validator.build_dataset(str(image_list), mode='val', batch=16)
    finally:
        YOLODataset.cache_labels = original_cache_labels
    require(not dataset.augment and dataset.rect and dataset.batch_size == 16, 'native dataset geometry policy changed')
    return validator, dataset


def dataset_plan(dataset):
    result = []
    for start in range(0, len(dataset), 16):
        paths = dataset.im_files[start:start+16]
        result.append({'source_batch_index': start // 16, 'indices': list(range(start, start+len(paths))),
                       'image_ids': [int(Path(p).stem) for p in paths],
                       'rect_shape': [int(v) for v in dataset.batch_shapes[start // 16]]})
    return result


def check_dataset_plan(dataset, expected):
    require(dataset_plan(dataset) == expected, 'source batches/images/rect geometry changed')


def historical_val_plan(trace):
    """Use recorded membership and geometry, never a new aspect-ratio tie sort."""
    images = sorted(trace['images'], key=lambda r: (r['batch_index'], r['batch_position']))
    require(len(images) == 5000 and len({r['image_id'] for r in images}) == 5000,
            'historical replay requires all 5000 unique images')
    batches = []
    for start in range(0, len(images), 16):
        group = images[start:start+16]
        bi = start // 16
        require(all(r['batch_index'] == bi and r['batch_position'] == j for j, r in enumerate(group)),
                'historical batch indices/positions are not contiguous')
        rectangle = group[0]['input_hw']
        require(all(r['input_hw'] == rectangle for r in group), 'inconsistent historical rectangle')
        require(len(rectangle) == 2 and all(int(v) > 0 and int(v) % 32 == 0 for v in rectangle),
                'invalid historical rectangle')
        batches.append({'source_batch_index': bi, 'indices': list(range(start, start+len(group))),
                        'image_ids': [r['image_id'] for r in group], 'rect_shape': rectangle})
    return batches


def apply_recorded_plan(dataset, expected):
    """Reindex all image-aligned state together before any sample is consumed."""
    import numpy as np
    require(not dataset.augment and not dataset.buffer, 'replay must precede sample consumption')
    by_id = {int(Path(p).stem): i for i, p in enumerate(dataset.im_files)}
    ids = [iid for b in expected for iid in b['image_ids']]
    require(len(by_id) == len(dataset) == len(ids) == len(set(ids)) and set(ids) == set(by_id),
            'recorded replay image membership mismatch')
    for bi, b in enumerate(expected):
        require(b['source_batch_index'] == bi and b['indices'] == list(range(bi*16, bi*16+len(b['image_ids']))),
                'recorded replay batch layout invalid')
        require(len(b['image_ids']) == 16 or bi == len(expected)-1, 'partial batch before final batch')
    order = [by_id[iid] for iid in ids]
    for name in ('im_files', 'labels', 'ims', 'im_hw0', 'im_hw', 'npy_files'):
        values = getattr(dataset, name)
        require(len(values) == len(ids), 'dataset aligned state size mismatch: '+name)
        setattr(dataset, name, [values[i] for i in order])
    require(all(int(Path(label['im_file']).stem) == iid for iid, label in zip(ids, dataset.labels)),
            'replay label/image alignment mismatch')
    dataset.batch = np.floor(np.arange(len(ids)) / 16).astype(int)
    dataset.batch_shapes = np.asarray([b['rect_shape'] for b in expected], dtype=int)
    check_dataset_plan(dataset, expected)


def prepare(config, out):
    """No model invocation. Record exact selection, native rect groups, file hashes and orders."""
    import numpy as np
    require(np.__version__ == '2.4.3', 'dataset aspect ordering requires frozen NumPy 2.4.3')
    for split in ('train', 'val'):
        require(sha(config[split+'_annotations']) == config['expected_annotations'][split], f'official {split} annotation SHA mismatch')
    parent = read(config['recovery_parent_contract'])
    previous_orders = read(config['recovery_parent_orders'])
    historical = read(config['historical_native_trace'])
    historical_by_id = {r['image_id']: r for r in historical['images']}
    val_plan = historical_val_plan(historical)
    train = read(config['train_annotations'])
    val = read(config['val_annotations'])
    original = read(config['original_split'])['fit_ids']
    train_ids = choose_training_ids([r['id'] for r in train['images']], original,
                                    config.get('fit_images', 20000), config.get('seed', 20261018))
    val_ids = sorted(r['id'] for r in val['images'])
    require(len(train_ids) == 20000 and len(val_ids) == 5000, 'frozen dataset counts changed')
    require(train_ids == parent['train_ids'] and val_ids == parent['val_ids'], 'recovery changed parent selection')
    require(all([r['height'], r['width']] == historical_by_id[r['id']]['original_hw'] for r in val['images']),
            'historical original dimensions differ from official annotations')
    contract = {'train_ids': train_ids, 'val_ids': val_ids, 'original_fit_ids': original,
                'selection_seed': config.get('seed', 20261018), 'selection': 'original 2000 plus Python Random sample 18000 from sorted remaining train IDs',
                'inputs': {key: {'path': config[key], 'sha256': sha(config[key])} for key in
                           ('train_annotations', 'val_annotations', 'original_split', 'profile', 'weights', 'adapter_initial', 'existing_G',
                            'historical_native_trace', 'recovery_parent_contract', 'recovery_parent_orders')}}
    for split, ids in (('train', train_ids), ('val', val_ids)):
        folder = Path(config[split + '_images'])
        files = [folder / f'{image_id:012d}.jpg' for image_id in ids]
        require(all(p.is_file() for p in files), f'{split} image files missing')
        image_list = out / (split + '.txt')
        image_list.write_text(''.join(str(p.resolve()) + '\n' for p in files), encoding='utf-8')
        _, dataset = build_dataset(config, image_list, out / ('dataset_' + split))
        apply_recorded_plan(dataset, parent['train_batches'] if split == 'train' else val_plan)
        contract[split + '_batches'] = dataset_plan(dataset)
        contract[split + '_image_hashes'] = {str(i): sha(p) for i, p in zip(ids, files)}
        require(contract[split + '_image_hashes'] == parent[split + '_image_hashes'], 'recovery source images changed')
        del dataset
    orders = update_orders(contract['train_batches'], config.get('epochs', 3), config.get('seed', 20261018))
    require(orders == previous_orders, 'recovery changed training sequence or update groups')
    require(all(contract['val_image_hashes'][str(i)] == historical_by_id[i]['image_sha256'] for i in val_ids),
            'historical validation image SHA mismatch')
    write(out / 'HISTORICAL_BATCH_PREFLIGHT.json', {'status': 'passed', 'images': 5000, 'batches': 313,
          'all_batch_positions_and_rectangles_exact': True, 'all_image_hashes_exact': True,
          'all_original_dimensions_exact': True, 'train_membership_batches_orders_unchanged': True,
          'historical_trace_sha256': sha(config['historical_native_trace']),
          'scope': 'all metadata checked before GPU; tensor SHA and RLE remain mandatory in native stage'})
    write(out / 'DATA_CONTRACT.json', contract)
    write(out / 'IMAGE_ORDERS.json', orders)
    return {'train_images': len(train_ids), 'val_images': len(val_ids),
            'source_batch': 16, 'effective_batch': 4, 'epochs': len(orders),
            'optimizer_steps_per_arm': sum(len(epoch) * 4 for epoch in orders),
            'data_contract_sha256': sha(out / 'DATA_CONTRACT.json'), 'image_orders_sha256': sha(out / 'IMAGE_ORDERS.json')}


class NativeBatches:
    """Pinned official preprocessing/selection/serialization, with in-memory native F."""
    def __init__(self, config, prepared, split, output):
        import torch
        from ultralytics import YOLO
        from ultralytics.nn.backends.pytorch import PyTorchBackend
        from native_output_bridge import apply_runtime_policy, native_environment
        self.config, self.prepared, self.split = config, Path(prepared), split
        self.contract = read(self.prepared / 'DATA_CONTRACT.json')
        for key, record in self.contract['inputs'].items():
            require(sha(config[key]) == record['sha256'], f'frozen data input changed: {key}')
        profile = read(config['profile'])
        apply_runtime_policy(profile['runtime_policy'])
        self.environment = native_environment()
        require(sha(config['weights']) == profile['model_sha256'], 'base weights changed')
        self.validator, self.dataset = build_dataset(config, self.prepared / (split + '.txt'), output, 'cuda:0')
        for key, expected in profile['native_args'].items():
            require(getattr(self.validator.args,key) == expected, 'effective native argument changed: '+key)
        write(Path(output)/'NATIVE_ARGS.json', vars(self.validator.args))
        self.plan = self.contract[split + '_batches']
        apply_recorded_plan(self.dataset, self.plan)
        check_dataset_plan(self.dataset, self.plan)
        model = YOLO(config['weights']).model
        self.backend = PyTorchBackend(model, device=torch.device('cuda:0'), fp16=False, fuse=True, verbose=False)
        self.model = self.backend.model.eval()
        self.head = self.model.model[-1]
        self.head.max_det = 300
        self.head.agnostic_nms = False
        require(self.head.end2end and self.model.is_fused() and not any(p.requires_grad for p in self.model.parameters()), 'native model policy drift')
        self.topk = None
        original = self.head.get_topk_index
        def observer(scores, max_det):
            require(self.topk is None, 'unconsumed top-k identity')
            value = original(scores, max_det)
            self.topk = tuple(x.detach() for x in value)
            return value
        self.head.get_topk_index = observer
        after = native_environment()
        require(all(after[key] == value for key,value in profile['runtime_policy'].items()), 'runtime policy drift during native initialization')

    def batch(self, index, *, masks=False):
        import torch
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from native_output_bridge import raw_location, decode_native, encode_rle
        plan = self.plan[index]
        paths = [self.dataset.im_files[j] for j in plan['indices']]
        for path in paths:
            require(sha(path) == self.contract[self.split + '_image_hashes'][str(int(Path(path).stem))], 'source image changed')
        batch = self.dataset.collate_fn([self.dataset[j] for j in plan['indices']])
        batch = self.validator.preprocess(batch)
        require(list(batch['img'].shape[2:]) == plan['rect_shape'], 'effective input rectangle changed')
        with torch.no_grad():
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            outputs = self.model(batch['img'])
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            decoding_started = time.perf_counter()
            (final, proto), raw = outputs
            raw = raw['one2one']
            scores, labels, indices = self.topk
            self.topk = None
            require(torch.equal(final[..., 4:5], scores) and torch.equal(final[..., 5:6], labels), 'top-k output identity changed')
            gathered = raw['mask_coefficient'].transpose(1, 2).gather(1, indices.expand(-1, -1, 32))
            require(torch.equal(gathered, final[..., 6:]) and torch.equal(proto, raw['proto']), 'native P/c0 identity failed')
            torch.cuda.synchronize()
            postprocess_started = time.perf_counter()
            predictions = DetectionValidator.postprocess(self.validator, outputs[0])
            torch.cuda.synchronize()
            postprocess_seconds = time.perf_counter() - postprocess_started
            serializer_seconds = 0.
            features = [t.detach() for t in raw['feats']]
            caches, rows = [], []
            shapes = [list(f.shape[1:]) for f in features]
            for position, pred in enumerate(predictions):
                keep = (final[position, :, 4] > self.validator.args.conf).nonzero().flatten()[:300]
                selected = final[position, keep]
                require(torch.equal(selected[:, :4], pred['bboxes']) and torch.equal(selected[:, 6:], pred['extra'])
                        and torch.equal(selected[:, 4], pred['conf']) and torch.equal(selected[:, 5], pred['cls']), 'postprocess identity failed')
                raw_ids = indices[position, keep, 0].tolist()
                locations = [raw_location(i, shapes) for i in raw_ids]
                iid = int(Path(batch['im_file'][position]).stem)
                cache = {'image_id': iid, 'proto': proto[position].detach(), 'c0': selected[:, 6:].detach(),
                         'boxes_input': selected[:, :4].detach(), 'scores': selected[:, 4].detach(),
                         'classes': selected[:, 5].detach(), 'raw_ids': raw_ids, 'raw_locations': locations,
                         'features': [features[l['level']][position, :, l['row'], l['column']].detach() for l in locations],
                         'head_feature_shapes': shapes, 'input_hw': list(batch['img'].shape[2:]),
                         'native_decode_hw': list(batch['img'].shape[2:]), 'device': 'cuda:0',
                         'original_hw': list(batch['ori_shape'][position]), 'ratio_pad': batch['ratio_pad'][position],
                         'batch_index': index, 'batch_position': position,
                         'image_sha256': self.contract[self.split + '_image_hashes'][str(iid)],
                         'input_tensor_sha256': tensor_sha(batch['img'][position]),
                         'output_ranks': list(range(len(selected))),
                         'feature_sources': {'head_module': 'model.23', 'head_layer_index': 23, 'source_layer_indices': [16, 19, 22]}}
                pbatch = {'imgsz': cache['input_hw'], 'ori_shape': cache['original_hw'], 'ratio_pad': cache['ratio_pad'], 'im_file': batch['im_file'][position]}
                self.validator.jdict = []
                if len(selected):
                    if masks:
                        torch.cuda.synchronize()
                        serializer_started = time.perf_counter()
                        pred['masks'] = self.validator.process(cache['proto'], cache['c0'], cache['boxes_input'], shape=cache['input_hw'])
                        self.validator.pred_to_json(self.validator.scale_preds(pred, pbatch), pbatch)
                        torch.cuda.synchronize()
                        serializer_seconds += time.perf_counter() - serializer_started
                        decoded = decode_native(cache)
                        require(all(encode_rle(decoded[j]) == row['segmentation'] for j, row in enumerate(self.validator.jdict)), 'zero-delta RLE mismatch')
                        del decoded
                    else:
                        DetectionValidator.pred_to_json(self.validator, DetectionValidator.scale_preds(self.validator, pred, pbatch), pbatch)
                caches.append(cache)
                rows.append(self.validator.jdict)
            require([cache['image_id'] for cache in caches] == plan['image_ids'], 'source batch membership changed')
            torch.cuda.synchronize()
            self.last_timing = {'base_forward_seconds': forward_seconds,
                                'native_postprocess_seconds': postprocess_seconds,
                                'native_decode_serialize_seconds': serializer_seconds,
                                'native_only_seconds': forward_seconds+postprocess_seconds+serializer_seconds,
                                'native_extraction_decode_serialize_seconds': time.perf_counter()-decoding_started,
                                'source_batch_index': index, 'images': len(caches),
                                'timing_scope': 'CUDA-synchronized; excludes input disk/decode/preprocess; extraction includes identity checks and zero-RLE replay'}
        del batch, outputs, final, proto, raw, predictions
        return caches, rows, features


def batch_identity(caches, rows, features):
    return {'images': [{'image_id': c['image_id'], 'input_tensor_sha256': c['input_tensor_sha256'],
                        'raw_ids': c['raw_ids'], 'proto_sha256': tensor_sha(c['proto']),
                        'c0_sha256': tensor_sha(c['c0']), 'boxes_sha256': tensor_sha(c['boxes_input']),
                        'scores_sha256': tensor_sha(c['scores']), 'classes_sha256': tensor_sha(c['classes']),
                        'geometry': {k: c[k] for k in ('input_hw', 'original_hw', 'ratio_pad')}} for c in caches],
            'feature_sha256': [tensor_sha(f) for f in features],
            'rows_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
