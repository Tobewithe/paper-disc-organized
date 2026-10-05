# Experiment 7C: coefficient-space gradient geometry

Use the existing 200 validation instances selected before this experiment (100 original Mask75 failures and 100 successes), their fixed official one-to-one positive assignments, cached prototypes, and saved GT-optimized coefficients. No model is trained.

At the original coefficient, split the reproduced official BCE within the **GT box** into positive and negative pixel terms, each divided by the same GT-box area as the official loss. Measure the two gradients, their cosine and cancellation, and each directional derivative toward the saved oracle. Validate the analytic gradient against autograd. Compare failures with successes and the good-box failure subset. A negative positive/negative cosine describes antagonism, not the absence of a simultaneous descent direction.

For every selected instance, interpolate from its original coefficient to its saved oracle coefficient. Separately, for a deterministic subset of 40 failures (20 with Box IoU >= .75 and 20 below), rerun LBFGS **from the original coefficient** and save each accepted optimizer step. The original oracle may have been selected from a zero initialization, so the optimizer trajectory and direct interpolation are different tests. Record official GT-box BCE and, within the fixed **predicted box**, mask IoU, GT coverage, false-positive rate, and ROC AUC. Keep the training-support and evaluation-support distinctions explicit.

Use image-clustered bootstrap intervals for aggregate comparisons. This is a fixed-candidate, GT-dependent diagnostic, not a deployable method or COCO AP measurement. Convexity of BCE in the coefficient under fixed prototypes rules out a nonconvex coefficient-loss barrier; thresholded mask metrics may nevertheless move non-monotonically.
