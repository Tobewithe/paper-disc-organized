"""Clarify an FP32 synthetic-coordinate tolerance failure; no training updates."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.ops import roi_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Preserve the original FP32 1e-6 outcome. Independently test the same
    # continuous-coordinate identity in FP64 at an absolute tolerance of 1e-10.
    for side in (80, 40, 20):
        yy, xx = torch.meshgrid(torch.arange(side, device='cuda'),
                                torch.arange(side, device='cuda'), indexing='ij')
        expected = [(128 + 384) / 2 * side / 640 - .5,
                    (160 + 448) / 2 * side / 640 - .5]
        for dtype in (torch.float32, torch.float64):
            feature = torch.stack((xx, yy)).to(dtype)[None]
            box = torch.tensor([[0., 128., 160., 384., 448.]],
                               device='cuda', dtype=dtype)
            got = roi_align(feature, box, output_size=3,
                            spatial_scale=side / 640,
                            sampling_ratio=2, aligned=True).mean((-2, -1))[0]
            ref = torch.tensor(expected, device='cuda', dtype=dtype)
            error = (got - ref).abs()
            ulp = torch.nextafter(ref, torch.full_like(ref, float('inf'))) - ref
            tol = 1e-6 if dtype == torch.float32 else 1e-10
            rows.append(dict(side=side, dtype=str(dtype), got=got.tolist(),
                             expected=expected, absolute_error=error.tolist(),
                             error_in_ulps=(error / ulp).tolist(),
                             absolute_tolerance=tol,
                             passed=bool((error <= tol).all())))
    result = dict(optimizer_steps=0, original_audit_preserved=True, rows=rows,
                  fp64_identity_passed=all(r['passed'] for r in rows
                                          if r['dtype'] == 'torch.float64'))
    (out / 'ROI_PRECISION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(completed=True,
        optimizer_steps=0, fp64_identity_passed=result['fp64_identity_passed'])), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
