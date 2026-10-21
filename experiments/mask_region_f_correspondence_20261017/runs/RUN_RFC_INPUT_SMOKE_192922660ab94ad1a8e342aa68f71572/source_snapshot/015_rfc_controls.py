"""Fixed-area deployment and prespecified functional-RMS controls; no GT or optimization."""
import copy
import torch
from rfc_inputs import require, finite_fp32


def native_area_gate(native_areas):
    require(isinstance(native_areas, torch.Tensor) and native_areas.ndim == 1 and
            not native_areas.dtype.is_floating_point and native_areas.dtype != torch.bool and
            bool((native_areas >= 0).all()), 'nonnegative integer original-mask areas required')
    return native_areas >= 9216


def gated_delta(delta, gate):
    finite_fp32(delta, (len(gate), 32), 'delta')
    require(gate.dtype == torch.bool and gate.shape == (len(delta),), 'invalid gate')
    return torch.where(gate.to(delta.device)[:, None], delta.detach(), torch.zeros_like(delta))


def mix_native_rows(native_rows, corrected_masks, gate):
    """Replace segmentation only; native gate-off RLE is preserved verbatim."""
    require(gate.dtype == torch.bool and gate.shape == (len(native_rows),), 'invalid gate')
    require(len(corrected_masks) == len(native_rows), 'mask row count mismatch')
    result = []
    for row, mask, active in zip(native_rows, corrected_masks, gate.cpu().tolist()):
        output = copy.deepcopy(row)
        if active:
            require(isinstance(mask, dict) and set(mask) == {'size', 'counts'}, 'invalid corrected RLE')
            require(mask['size'] == row['segmentation']['size'], 'corrected original-image size changed')
            output['segmentation'] = copy.deepcopy(mask)
        result.append(output)
    return result


def proto_roi(cache, device):
    height, width = cache['proto'].shape[-2:]
    ih, iw = cache['input_hw']
    boxes = cache['boxes_input'].detach().to(device).float()
    boxes = boxes * boxes.new_tensor([width/iw, height/ih, width/iw, height/ih])
    x = torch.arange(width, device=device)[None, None, :]
    y = torch.arange(height, device=device)[None, :, None]
    return ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) &
            (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))


def functional_rms(cache, delta):
    finite_fp32(delta, cache['c0'].shape, 'delta')
    p = cache['proto'].detach().to(device=delta.device, dtype=torch.float64)
    response = (delta.detach().double() @ p.flatten(1)).reshape(len(delta), *p.shape[-2:])
    roi = proto_roi(cache, delta.device)
    counts = roi.sum((1, 2))
    squared = torch.where(roi, response.square(), torch.zeros_like(response))
    q = (squared.sum((1, 2))/counts.clamp_min(1)).sqrt()
    require(bool(torch.isfinite(q).all()), 'non-finite functional norm')
    return q, counts


def match_functional_amplitude(cache, first, second, gate):
    require(first.device == second.device, 'amplitude-control devices differ')
    first, second = gated_delta(first, gate), gated_delta(second, gate)
    q1, counts = functional_rms(cache, first)
    q2, counts2 = functional_rms(cache, second)
    require(torch.equal(counts, counts2), 'ROI changed between controls')
    active = gate.to(first.device) & (counts > 0) & (q1 > 1e-12) & (q2 > 1e-12)
    common = torch.where(active, torch.minimum(q1, q2), torch.zeros_like(q1))
    scale1 = torch.where(active, common/q1.clamp_min(1e-12), torch.zeros_like(q1))
    scale2 = torch.where(active, common/q2.clamp_min(1e-12), torch.zeros_like(q2))
    require(bool(((scale1 <= 1) & (scale2 <= 1)).all()), 'control amplified a direction')
    matched1 = (first.double()*scale1[:, None]).float()
    matched2 = (second.double()*scale2[:, None]).float()
    after1, _ = functional_rms(cache, matched1)
    after2, _ = functional_rms(cache, matched2)
    require(torch.allclose(after1, after2, atol=1e-7, rtol=1e-5),
            'FP32 control failed frozen functional-amplitude tolerance')
    reasons = []
    for enabled, count, a, b in zip(gate.cpu().tolist(), counts.cpu().tolist(), q1.cpu().tolist(), q2.cpu().tolist()):
        reasons.append('gate_off' if not enabled else 'empty_roi' if count == 0 else
                       'near_zero_direction' if min(a, b) <= 1e-12 else 'matched')
    receipt = {'q_first': q1.cpu().tolist(), 'q_second': q2.cpu().tolist(),
               'q_matched_first': after1.cpu().tolist(), 'q_matched_second': after2.cpu().tolist(),
               'scale_first': scale1.cpu().tolist(), 'scale_second': scale2.cpu().tolist(),
               'roi_pixels': counts.cpu().tolist(), 'reason': reasons,
               'all_rows_retained': len(reasons) == len(first), 'uses_GT': False}
    return matched1, matched2, receipt
