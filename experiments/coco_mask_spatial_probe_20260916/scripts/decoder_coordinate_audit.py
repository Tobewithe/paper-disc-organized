"""Analytic coordinate audit of ROI training vs image local-field decoding.

No model or GT is fitted. A planar 4x4 field has an exact interior interpolant,
so sampling its image decode back at the training ROI coordinates should agree.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import component_seed_probe as probe
import learn_refinement as original


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    shape = (192, 192)
    v = torch.linspace(-.5, .5, 4)
    yy, xx = torch.meshgrid(v, v, indexing='ij')
    correction = ((xx+yy)/2).reshape(1, 16)
    rows = []
    for width in (8, 16, 32, 80):
        boxes = torch.tensor([[40., 40., 40.+width, 40.+width]])
        base = torch.zeros(1, *shape)
        current = probe.image_correct(base, None, None, boxes, shape, correction, 'local4', 'legacy_integer')
        grid = original.roi_grid(boxes, shape)
        roi = original.roi_correct(torch.zeros(1, 32, 32), None, correction, 'local4')
        sampled = F.grid_sample(current[:, None], grid, align_corners=False)[:, 0]
        # Pixel centres in align_corners=False are (index+0.5)/image_size.
        aligned = probe.image_correct(base, None, None, boxes, shape, correction, 'local4', 'pixel_center')[:,None]
        assert torch.equal(aligned[:,0], probe.image_correct(base, None, None, boxes, shape, correction, 'local4'))
        aligned_sampled = F.grid_sample(aligned, grid, align_corners=False)[:, 0]
        region = (slice(None), slice(8,24), slice(8,24))
        error = sampled[region]-roi[region]
        fixed_error = aligned_sampled[region]-roi[region]
        analytic = -2./width
        assert float((error-analytic).abs().max()) < 3e-6
        assert float(fixed_error.abs().max()) < 3e-6
        rows.append(dict(box_width=width, current_signed_logit_error=float(error.mean()),
                         current_max_abs_error=float(error.abs().max()),
                         expected_error=analytic, pixel_center_aligned_max_error=float(fixed_error.abs().max())))
    result = dict(rows=rows, conclusion='Current full-image local field uses integer coordinates; ROI sampling uses pixel centres. A half-pixel offset exists for local4 and the local portion of coeff_local4.',
                  scope='Synthetic analytic coordinate check; no claim about COCO performance or the share of real errors explained. Active experiments remain unchanged.')
    (a.out/'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(passed=True, cases=len(rows))), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
