"""One GT-free, fixed-threshold binary gate over original-image native mask area."""
AREA_MIN = 96**2


def mix_row(native, corrected):
    from pycocotools import mask as masks
    if set(native) != set(corrected):
        raise ValueError('prediction field set differs')
    if any(native[k] != corrected[k] for k in native if k != 'segmentation'):
        raise ValueError('non-mask field changed')
    if native['segmentation']['size'] != corrected['segmentation']['size']:
        raise ValueError('original image shape differs')
    area = int(masks.area(native['segmentation']))
    apply = area >= AREA_MIN
    output = {**native, 'segmentation': corrected['segmentation'] if apply else native['segmentation']}
    return output, {'native_mask_area': area, 'apply_delta': apply,
                    'mask_changed': output['segmentation'] != native['segmentation']}


def adjudicate(target, overall, damage):
    gates = {'target_effect': target['mean'] >= .005 and target['ci95'][0] > 0,
             'all_bbox75_guardrail': overall['mean'] >= -.001,
             'success_damage': damage['gate'] == 'damage_guardrail_passed'}
    status = 'passed' if all(gates.values()) else ('blocked' if gates['target_effect'] and gates['all_bbox75_guardrail'] and
              damage['gate'] == 'blocked_insufficient_precision' else 'failed')
    return status, gates
