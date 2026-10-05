# Faro Oracle Box Crop Counterfactual: Feature Collapse Confirmation

**Date:** 2026-09-08

## Context & Question
In the aro_mask_quality_20260907 diagnostic, we identified 270 Ground Truths (GTs) that failed the strict mask quality requirement (coverage >= 0.75, purity >= 0.75) despite the network proposing bounding boxes large enough to cover the GTs. 

The open question was: Did the masks fail because the predicted bounding boxes, while "large enough", were geometrically misaligned and cropped out the target? Or did they fail because the YOLO Mask Head (Prototypes + Coefficients) suffered Representation Collapse due to neighbor interference, fundamentally failing to express the instance?

## Method
We intercepted the raw dense prototypes and candidate mask coefficients for these 270 cases. We completely bypassed the network's predicted bounding boxes and directly injected the Ground Truth bounding box (gt_bbox) into the crop_mask operation. We then checked if *any* of the candidate coefficients could now form a strict quality mask.

## Results
- **Total Processed GTs**: 270
- **Recovered by Oracle Box**: 43 (15.9%)
- **Still Failing (Feature/Prototype Collapse)**: 227 (84.1%)

## Conclusion & Interpretation
1. **Geometric Truncation is a Minor Factor**: Providing perfect bounding box localization only rescued 16% of these specific failures.
2. **Representation Collapse is the Dominant Bottleneck**: For the vast majority (84%) of these cases, the mask features (Prototypes) are inherently polluted or collapsed. The network literally cannot express the correct mask boundary, regardless of how perfectly it is bounded. This proves that merely improving candidate selection, sorting, or bounding box regression will NOT solve the dense instance segmentation problem for these cases.
3. **Implication for Method Design**: Any intervention must address the mask representation itself. Methods that only rescore candidates or refine boxes are mathematically bounded to fail on these 227 cases. We need interventions that enforce spatial discriminability in the feature maps or prototypes (e.g., contrastive learning, occlusion-aware mask heads, or higher-resolution prototype generation).
