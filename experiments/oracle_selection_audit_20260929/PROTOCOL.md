# Perfect error-location selection audit

This is a no-training supplement to 7Q/7R. It keeps the candidate identity, predicted box, prototype, frozen correction outputs, and official mask decoding fixed. The original COCO `annToMask` is the only pixel ground truth.

For each candidate, let `M0` be the original decoded mask, `M1` the frozen p-spatial correction mask, and `Y` the COCO mask. The GT-assisted selector is exactly `Msel[u] = M1[u] if M0[u] != Y[u] else M0[u]`; it does not write GT values into the output. We report this only as an upper-bound diagnostic, never as a deployable method.

For every candidate we record FN→TP (`a`), FP→TN (`b`), TP→FN (`c`), TN→FP (`d`), and the two wrong-to-wrong switches FN→FP and FP→FN. We also define the finite-oracle opportunity `U* = {M0 != Y and Mstar == Y}` and the hit rate of the frozen correction inside `U*`. An empty `U*` is undefined, not zero.

Primary evaluation is original-image coordinates with the normal predicted-box crop and two existing p-spatial seeds. The primary cohort is the pre-existing 3,059 candidates with original Mask IoU < 0.75 and Box IoU >= 0.75; all 14,911 candidates are a secondary cohort. Image-cluster bootstrap intervals are descriptive because this test set has already been used by 7Q/7R. Size strata and regional summaries are exploratory.

Decision rule: a large selector gain with low oracle-hit rate would support learning selective application; a small selector gain or low available frozen-correction benefit means an error-location gate cannot be the main fix for this frozen correction. This experiment does not establish that GT-free error location is predictable.
