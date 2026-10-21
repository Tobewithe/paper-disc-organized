"""Small synthetic tests of the scientific gate, bounds and hard composition."""
from lea_common import *
import itertools
import numpy as np
from lea_ops import bbox_support, compose, iou, random_support, random_seed, encode, decode
from lea_stats import gate_decision


def main():
    passed = []
    n = np.array([[1, 0], [0, 1]], dtype=bool)
    y = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [1, 0]], dtype=bool)
    scores = []
    for bit in itertools.product([False, True], repeat=int(b.sum())):
        v = n.copy()
        v[b] = bit
        scores.append(iou(v, y))
    assert iou(compose(n, y, b), y) == max(scores)
    assert iou(compose(n, ~y, b), y) == min(scores)
    passed.append('restricted_binary_oracle_and_worst_case_are_exact_by_exhaustion')
    for b2 in (b, np.zeros_like(b), np.ones_like(b)):
        v = compose(n, y, b2)
        assert np.array_equal(v[~b2], n[~b2])
        assert np.array_equal(decode(encode(v)), v)
    passed.append('final_support_identity_and_RLE_roundtrip')
    roi = bbox_support((4, 5), [1.001, .5, 2., 2.])
    assert np.array_equal(np.argwhere(roi), np.array([[1,2], [1,3], [2,2], [2,3]]))
    assert not bbox_support((4, 5), [1, 1, 0, 2]).any()
    passed.append('serialized_fractional_bbox_semantics_and_empty_box')
    for number in (0, 2, int(roi.sum())):
        r = random_support(roi, number, random_seed(1, 2, 'abc:3', 0))
        assert int(r.sum()) == number and not np.any(r & ~roi)
        assert np.array_equal(r, random_support(roi, number, random_seed(1, 2, 'abc:3', 0)))
    passed.append('equal_budget_random_support_deterministic')
    cfg = read_json(STUDY / 'CONFIG.json')
    base = dict(support=dict(C_witnesses=159, C_witness_retained_by_O_B=152, editable_fraction_image_macro=.2),
                task_damage=dict(G_B=dict(rate=.005, damaged=1), G=dict(damaged=1)),
                comparisons={'O_B-G': dict(mean=.01, ci95=[.001,.02]), 'O_B-O_R': dict(mean=.01, ci95=[.001,.02])})
    assert gate_decision(base, cfg)[1] == 'NECESSARY_CONDITIONS_PASSED'
    base['support']['C_witness_retained_by_O_B'] = 151
    assert gate_decision(base, cfg)[1] == 'NEGATIVE_CURRENT_REGION'
    base['support']['C_witness_retained_by_O_B'] = 152
    base['comparisons']['O_B-O_R'] = dict(mean=.004, ci95=[-.001,.006])
    assert gate_decision(base, cfg)[1] == 'INCONCLUSIVE_NOT_READY'
    base['comparisons']['O_B-O_R'] = dict(mean=.001, ci95=[-.001,.004])
    assert gate_decision(base, cfg)[1] == 'NEGATIVE_CURRENT_REGION'
    passed.append('frozen_positive_negative_inconclusive_and_ceil_retention')
    # Moving a local edit before bilinear interpolation does not give this identity.
    import torch
    import torch.nn.functional as F
    x = torch.zeros(1,1,2,2)
    x[0,0,0,0] = 1
    up = F.interpolate(x, (4,4), mode='bilinear', align_corners=False)[0,0].numpy()
    nearest = F.interpolate((x>0).float(), (4,4), mode='nearest')[0,0].numpy().astype(bool)
    assert np.any((up>0) & ~nearest)
    passed.append('preinterpolation_edit_can_leak_outside_nearest_support')
    # A confident wrong pixel outside B remains wrong: protection is not correctness.
    assert not np.array_equal(compose(n, y, b), y)
    passed.append('locked_errors_remain_and_are_not_hidden')
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((STUDY/'scripts').glob('*.py'))}
    write_new(STUDY/'TESTS.json', dict(status='passed', at=now(), tests=passed, count=len(passed),
                                      tested_sources=sources, new_scientific_outcomes=0))
    print(json.dumps(dict(status='passed', tests=passed)))


if __name__ == '__main__':
    main()
