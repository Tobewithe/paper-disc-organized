# Crop sensitivity crossed with paired COCO failure states (S078)

This retrospective analysis uses the S077 frozen train2017 readout cache. The stored bbox-matched candidate is retained; no new forward, training, or inference rule is introduced.

- Targets: 9626 across 1483 images; skipped: 0.
- State thresholds: candidate Box IoU >= 0.50; mask IoU@native crop >= 0.75; GT pixels inside candidate box >= 0.95.
- `gt_box_support` is computed from the candidate box and GT mask, independently of predicted mask positives.

## State table

| State | N | Fraction | Mean oracle crop gain | Mean fixed 1.2 delta | Mean Box IoU | Mean GT-box support |
|---|---:|---:|---:|---:|---:|---:|
| box_bad_mask_bad | 0 | 0.000 | nan | nan | nan | nan |
| box_bad_mask_good | 0 | 0.000 | nan | nan | nan | nan |
| box_good_support_low_mask_bad | 970 | 0.101 | 0.0320 | 0.0001 | 0.653 | 0.839 |
| box_good_support_sufficient_mask_bad | 2559 | 0.266 | 0.0275 | -0.0406 | 0.784 | 0.992 |
| box_good_mask_good | 6097 | 0.633 | 0.0046 | -0.0247 | 0.916 | 0.992 |

## Interpretation guardrails

- Oracle crop gain is a diagnostic upper bound for changing only the crop scale; it is not an end-to-end method result.
- A concentration of oracle gain in `box_good_support_sufficient_mask_bad` would implicate crop sensitivity after adequate geometric support. A concentration in `box_good_support_low_mask_bad` would instead implicate box coverage.
- A large `box_good_mask_good` fraction among high-ICI targets means high-I CI does not automatically imply a mask failure for this paired candidate.
- These labels are retrospective and use GT; they cannot by themselves establish a deployable no-GT selector.
