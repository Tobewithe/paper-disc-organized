# Position × objective complement (Stage 1)

## Goal

Disentangle whether the opposite transfer of the frozen shared affine coefficient readout is caused mainly by candidate **position** or by the **mask objective**. The two historical anchor cells are reused without rerunning: official TAL positions with the official reference loss (O-OFF), and geometry-selected positions with the original-label attainability loss (G-GB). This run fills only the missing cells.

## Fixed objects

- YOLO26m-seg, Ultralytics 8.4.100, frozen original checkpoint, FP32, 640 input, no augmentation.
- Official one-to-one TAL identities: 796/197/196 images and 6058/1402/1346 candidates, from `coefficient_official_tal_affine_20260930/OFFICIAL_MANIFEST.json`.
- Geometry identities: the matching fixed cache from `coefficient_shared_affine_attainability_20260930` (5872/1375/1301). It must be verified by identity and SHA before use; if absent, generate with the frozen `native_bank.py` mapping on exactly the same image split.
- Shared per-level affine readout, double precision L-BFGS, fit only, lambda=0.003; no training, no new input or head.
- Full prototype decode, original predicted box, real letterbox inverse transform, original COCO masks; 5,000 image bootstrap for the primary validation comparison.

## Missing cells

1. `O-GB`: official TAL positions, original-label GT-box BCE and area normalization used by the geometry attainability solver.
2. `G-OFF`: geometry positions, Gate-A official reference ROI BCE, overlap labels, official gain and normalization.
3. `G-BIAS` is secondary: geometry positions and the attainability objective, but the correction is one constant 32-vector per pyramid level. It measures how much of the geometry effect is a shared level bias.

Primary metric is validation image-macro normal-image Mask IoU. Required costs: coverage, continuous-logit AUC, Mask75 repair/damage, and damage on baseline-success instances. Candidate averages are secondary. Fit is for solving only; dev/val are frozen evaluations.

## Decisions fixed before reading results

- O-GB non-positive while G-OFF retains a clear positive transfer: support the position hypothesis and only then consider the preregistered Stage 2 donor readout.
- O-GB positive: objective/label mismatch contributes; stop this branch and register a separate target intervention.
- Both cells shrink or both are positive: report mixed evidence and stop; do not add weights, gates, thresholds, or extra modules.
- If identity, loss-equivalence, or full-decode audits fail, stop interpretation and repair the audit only.

Stage 2 is not started by this protocol. It may be registered only after the position decision: same-image donor raw position selected by highest predicted-box IoU with the retained output box, donor h plus the frozen geometry affine, decoded with the original detection box and prototype. No K sweep, fusion, selector training, or threshold tuning.

## Scope limits

This is a fixed-candidate diagnostic, not COCO AP or an end-to-end deployment claim. It must not relabel the geometry gain as an official TAL result. Existing anchors and prior runs remain immutable.
