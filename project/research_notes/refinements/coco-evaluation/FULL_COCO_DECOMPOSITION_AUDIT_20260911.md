# Full COCO failure decomposition audit

Date: 2026-09-11. Scope: `decompose_coco_failures.py` and its full val2017 output only.

## Verdict

The full decomposition is internally consistent for the stated diagnostic scope. Independent checks found 36,335 unique noncrowd COCO annotation IDs, 36,186 with at least one crowd-valid GT pixel, 32,543 official bbox-IoU>=0.5 matched instances, and 3,643 unmatched instances. Summary counts and means recompute from `per_instance.csv` with floating differences below 2e-15.

## Required claim limits

The diagnostic uses COCO GT only after predictions are generated. It assigns ownership with fixed official bbox matching (IoU >= 0.5), then measures original-resolution integer masks. This is a fixed attribution diagnostic, not official COCO mask assignment/AP. The rescue columns are oracle opportunities conditioned on the 11,657 bbox-matched instances whose baseline attribution mask IoU is below 0.75. They exclude the 3,643 bbox-unmatched instances and should not be reported as the fraction of all failures without adding unmatched instances explicitly. Rescue opportunities overlap and cannot be summed.

The `crop_blocked75` count identifies failures where the predicted box support cannot cover 75% of the GT mask. It is not a causal estimate of box error because the same fixed bbox assignment is used for attribution. The all-COCO table is descriptive and does not prove that an intervention will obtain any rescue count.

## Interpretation supported by the data

High ICI instances have lower attributed mask IoU and more same-class-neighbor error in the current frozen model. A category x COCO-area descriptive standardization retains only cells with at least ten high and ten low instances, uses common pooled cell weights, and reports a high-minus-low IoU contrast of -6.25 percentage points (95% paired image-cluster interval [-7.03,-5.47]) and same-neighbor error contrast of +14.18 points [12.61,15.89]. This supports a dense-instance association after basic composition control; category, scale, box quality and occlusion are not fully causal-adjusted.

## Disposition

Keep this report as mechanism localization. Do not call it a complete decomposition of every segmentation failure, a proof that coefficient cosine causes leakage, or evidence of deployable CCL efficacy. The learned pilots require their own task-level audit and must retain frozen-candidate and reused-validation limitations.
