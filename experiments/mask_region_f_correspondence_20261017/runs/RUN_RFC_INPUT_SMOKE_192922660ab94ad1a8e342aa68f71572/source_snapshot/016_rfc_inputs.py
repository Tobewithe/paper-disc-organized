"""In-memory regional-F sampling and identity checks; no model runner or file exporter."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping

import torch
from torch.nn import functional as F

CHANNELS = (256, 512, 512)
GRID = 8
SIDECAR_SCHEMA = 'rfc-region-f-v1'
IDENTITY_FIELDS = ('image_id', 'input_hw', 'original_hw', 'ratio_pad', 'batch_index',
                   'batch_position', 'image_sha256', 'input_tensor_sha256',
                   'head_feature_shapes', 'feature_sources', 'raw_ids',
                   'raw_locations', 'output_ranks', 'prediction_indices')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(value):
    if isinstance(value, Mapping):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def fingerprint(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'A full lowercase SHA256 is required')
    return value


def finite_fp32(tensor, shape, label):
    require(isinstance(tensor, torch.Tensor), f'{label}: tensor required')
    require(tuple(tensor.shape) == tuple(shape), f'{label}: shape mismatch')
    require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
            f'{label}: finite FP32 required')


def raw_location(raw_id, shapes):
    require(type(raw_id) is int and raw_id >= 0, 'invalid raw_id')
    offset = 0
    for level, (channels, height, width) in enumerate(shapes):
        if raw_id < offset + height * width:
            return dict(level=level, row=(raw_id-offset)//width,
                        column=(raw_id-offset)%width, level_offset=offset, channels=channels)
        offset += height * width
    raise ValueError('raw_id outside head feature maps')


def validate_native(cache):
    n = len(cache['c0'])
    shapes = cache['head_feature_shapes']
    require(len(shapes) == 3 and tuple(s[0] for s in shapes) == CHANNELS,
            'native head channels changed')
    require(all(len(s) == 3 and all(type(x) is int and x > 0 for x in s) for s in shapes),
            'invalid feature shapes')
    require(cache['feature_sources']['source_layer_indices'] == [16, 19, 22],
            'head feature sources changed')
    require(cache['feature_sources']['head_module'] == 'model.23' and
            cache['feature_sources']['head_layer_index'] == 23, 'native head identity changed')
    require(type(cache['image_id']) is int and cache['image_id'] >= 0, 'invalid image ID')
    require(list(cache['output_ranks']) == list(range(n)), 'output ranks reordered')
    require(len(cache['prediction_indices']) == n and
            len(set(cache['prediction_indices'])) == n and
            all(type(i) is int and i >= 0 for i in cache['prediction_indices']),
            'missing or duplicate source prediction rows')
    require(len(cache['features']) == len(cache['raw_ids']) == len(cache['raw_locations']) == n,
            'native row count mismatch')
    height, width = cache['input_hw']
    require(type(height) is int and type(width) is int and min(height, width) > 0,
            'invalid real input geometry')
    require(height % 32 == 0 and width % 32 == 0, 'native stride-aligned input geometry required')
    finite_fp32(cache['c0'], (n, 32), 'c0')
    finite_fp32(cache['proto'], (32, height//4, width//4), 'proto')
    finite_fp32(cache['boxes_input'], (n, 4), 'boxes')
    finite_fp32(cache['scores'], (n,), 'scores')
    require(bool((cache['boxes_input'][:, 2:] > cache['boxes_input'][:, :2]).all()),
            'nonpositive predicted ROI')
    classes = cache['classes']
    finite_fp32(classes, (n,), 'classes')
    require(classes.shape == (n,) and bool(torch.isfinite(classes).all()) and
            bool(((classes >= 0) & (classes < 80) & (classes == classes.long())).all()),
            'invalid class labels')
    for rank, raw_id in enumerate(cache['raw_ids']):
        actual = raw_location(raw_id, shapes)
        require(actual == cache['raw_locations'][rank], 'raw location does not identify its row')
        finite_fp32(cache['features'][rank], (CHANNELS[actual['level']],), 'point F')
    for tensor in (cache['proto'], cache['c0'], cache['boxes_input'], cache['scores'],
                   cache['classes'], *cache['features']):
        require(not tensor.requires_grad, 'native cache must be detached')
    return n


def validate_replay(original, replay):
    """Reject drift before new features can be associated with an old detection."""
    validate_native(original)
    validate_native(replay)
    for key in IDENTITY_FIELDS:
        require(normalized(original[key]) == normalized(replay[key]), f'replay metadata changed: {key}')
    for key in ('proto', 'c0', 'boxes_input', 'scores', 'classes'):
        a, b = original[key], replay[key]
        require(a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()), f'replay tensor changed: {key}')
    for a, b in zip(original['features'], replay['features']):
        require(a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()), 'replay point F changed')


def roi_grid(boxes, input_hw):
    """Same physical box-cell centers as original sample_spatial_roi."""
    n = len(boxes)
    b = boxes.detach().float()
    t = (torch.arange(GRID, device=b.device, dtype=b.dtype) + .5) / GRID
    xs = b[:, 0, None] + (b[:, 2]-b[:, 0])[:, None]*t
    ys = b[:, 1, None] + (b[:, 3]-b[:, 1])[:, None]*t
    gx = (2*xs/input_hw[1]-1)[:, None, :].expand(n, GRID, GRID)
    gy = (2*ys/input_hw[0]-1)[:, :, None].expand(n, GRID, GRID)
    return torch.stack((gx, gy), -1)


def sample_region_features(cache, dense_features, selected_ranks=None):
    """Sample supplied head tensors only, without calling a base model or writing a cache.

    dense_features contains the three unbatched head inputs for this image. Their
    raw positions must reproduce *every* cached point F, including duplicate raws.
    """
    n = validate_native(cache)
    require(len(dense_features) == 3, 'three head maps required')
    for level, tensor in enumerate(dense_features):
        finite_fp32(tensor, cache['head_feature_shapes'][level], 'dense F')
        require(not tensor.requires_grad, 'head map must be detached')
    selected = list(range(n)) if selected_ranks is None else list(selected_ranks)
    require(len(set(selected)) == len(selected) and
            all(type(i) is int and 0 <= i < n for i in selected), 'invalid selected rows')
    result = [None]*n
    for rank, loc in enumerate(cache['raw_locations']):
        actual = dense_features[loc['level']][:, loc['row'], loc['column']]
        require(torch.equal(actual.cpu(), cache['features'][rank].cpu()), 'dense raw point != original F')
    with torch.no_grad():
        for level, tensor in enumerate(dense_features):
            ranks = [r for r in selected if cache['raw_locations'][r]['level'] == level]
            # Bound temporary grid-sample batches, not the scientific input grid.
            for start in range(0, len(ranks), 16):
                subset = ranks[start:start+16]
                boxes = cache['boxes_input'][subset].to(tensor.device)
                grid = roi_grid(boxes, cache['input_hw'])
                regions = F.grid_sample(tensor[None].expand(len(subset), -1, -1, -1), grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                for rank, values in zip(subset, regions):
                    result[rank] = values.detach().cpu().contiguous().clone()
    return result


def make_sidecar(cache, regions, cache_sha256, prediction_sha256, scope):
    """Build an in-memory payload. Persistence must be handled by a future registered exporter."""
    n = validate_native(cache)
    require(scope in ('train_all_rows', 'eval_gated_rows'), 'invalid F sidecar scope')
    require(len(regions) == n, 'region row count mismatch')
    fingerprint(cache_sha256)
    fingerprint(prediction_sha256)
    return {'schema': SIDECAR_SCHEMA, 'scope': scope,
            'source_cache_sha256': cache_sha256, 'source_predictions_sha256': prediction_sha256,
            'identity': {k: normalized(cache[k]) for k in IDENTITY_FIELDS},
            'output_detection_ids': [f'{prediction_sha256}:{i}' for i in cache['prediction_indices']],
            'regions': list(regions)}


@dataclass(frozen=True)
class RegionInputs:
    scope: str
    image_id: int
    proto: torch.Tensor
    c0: torch.Tensor
    boxes_input: torch.Tensor
    input_hw: tuple
    features: tuple
    levels: tuple
    scores: torch.Tensor
    classes: torch.Tensor
    regions: tuple


def prepare_inputs(cache, sidecar, cache_sha256, prediction_sha256, phase, gate_mask=None):
    """Only explicitly listed native fields enter forward; GT/labels cannot be sidecar fields."""
    n = validate_native(cache)
    require(phase in ('train:0', 'train:1', 'train:2', 'eval:0'), 'unknown phase')
    fingerprint(cache_sha256)
    fingerprint(prediction_sha256)
    require(set(sidecar) == {'schema', 'scope', 'source_cache_sha256', 'source_predictions_sha256',
                            'identity', 'output_detection_ids', 'regions'}, 'unexpected sidecar fields')
    require(sidecar['schema'] == SIDECAR_SCHEMA and
            sidecar['source_cache_sha256'] == cache_sha256 and
            sidecar['source_predictions_sha256'] == prediction_sha256, 'sidecar provenance mismatch')
    require(sidecar['identity'] == {k: normalized(cache[k]) for k in IDENTITY_FIELDS}, 'sidecar row/geometry mismatch')
    require(sidecar['output_detection_ids'] == [f'{prediction_sha256}:{i}' for i in cache['prediction_indices']],
            'source output_detection_id mismatch')
    training = phase.startswith('train:')
    require(sidecar['scope'] == ('train_all_rows' if training else 'eval_gated_rows'), 'sidecar scope mismatch')
    if training:
        require(gate_mask is None, 'training must include all rows without gating')
        needed = [True]*n
    else:
        require(isinstance(gate_mask, torch.Tensor) and gate_mask.dtype == torch.bool and
                gate_mask.shape == (n,), 'explicit native area gate mask required')
        needed = gate_mask.detach().cpu().tolist()
    regions = sidecar['regions']
    require(len(regions) == n, 'F region count mismatch')
    result = []
    for rank, (values, active) in enumerate(zip(regions, needed)):
        channels = CHANNELS[cache['raw_locations'][rank]['level']]
        if active:
            finite_fp32(values, (channels, GRID, GRID), 'F region')
            require(not values.requires_grad, 'regional F must be detached')
            result.append(values)
        else:
            require(values is None, 'gate-off F must be explicitly omitted; use native point broadcast')
            result.append(cache['features'][rank][:, None, None].expand(-1, GRID, GRID))
    return RegionInputs(sidecar['scope'], cache['image_id'], cache['proto'], cache['c0'], cache['boxes_input'],
                        tuple(cache['input_hw']), tuple(cache['features']),
                        tuple(x['level'] for x in cache['raw_locations']), cache['scores'],
                        cache['classes'], tuple(result))


def spatial_indices(image_id, output_rank, preserve=False):
    require(type(image_id) is int and image_id >= 0 and type(output_rank) is int and output_rank >= 0,
            'invalid permutation identity')
    if preserve:
        return tuple(range(GRID*GRID))
    prefix = f'rfc-f-order-v1:314159:{image_id}:{output_rank}:'
    return tuple(sorted(range(GRID*GRID),
                        key=lambda j: (hashlib.sha256(f'{prefix}{j}'.encode('utf-8')).digest(), j)))
