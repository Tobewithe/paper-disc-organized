# COCO clean supervised training — prelaunch protocol, 2026-09-11

## Purpose

Rebuild the original Baseline versus CCL comparison after finding polygon-level
instance splitting in the historical training conversion. This is supervised
fine-tuning from the same official YOLO26m-seg COCO-pretrained checkpoint, not
self-supervision, pseudo-label training, or training from random initialization.
No previous experimental checkpoint is reused.

## Data and supervision

- Full-data census completed before formal training. Historical training/val
  image IDs reproduce exactly from original JSON. After numerical boundary repair,
  dense train covers 120,032 of 120,921 same-class high-ICI instances (99.26%), but only 24.55% of its valid
  instances are high-ICI. See `census/COCO2017_CROWDING_REPORT.md` and source CSVs.
  The training set is an image subset containing crowded instances, not a set
  in which every supervised instance is crowded. Full val5000 is the evaluation
  base; same-class and all-category instance stratification are kept distinct.
- Before prediction-based evaluation, snap ICI within absolute 1e-10 of 0.5 or 1
  to that exact boundary; retain raw values. This fixes floating-point artifacts
  in the ICI>1 subgroup (val 1,708→1,523); the primary val ICI>0.5 group stays
  5,245. Historical image IDs stay frozen under the historical rule, not reselected.
- Keep the historical, GT-only dense training image selection: 37,433 train2017
  images and 488,877 non-crowd annotations. This isolates the data-identity repair
  without selecting new training examples according to model performance.
- Use the unmodified Ultralytics 8.4.143 `convert_coco(use_segments=True)` and
  official YOLODataset/augmentations/trainer. The converter merges all polygons
  of one annotation into one label using `merge_multi_segment`, rather than
  making each polygon a new instance. The audit maintains a source-ID sidecar.
- The official converter connects disjoint polygons with thin bridges and
  deduplicates equal class+bbox entries. The official loader resamples polygons
  and derives boxes from segments. These are recorded standard preprocessing
  behaviors, not claimed pixel-exact COCO raster supervision. Quantify affected
  annotations and audit emitted label rows before training; do not silently
  assert that the official loader retains every original annotation.
- `iscrowd=1` is excluded from ordinary training targets, consistently for both
  arms. Official evaluation retains original crowd/ignore semantics.
- Both arms use identical standard 640px preprocessing, `mask_ratio=4`,
  `overlap_mask=True`, HSV, horizontal flips and default geometric augmentation.
  Mosaic is enabled with `close_mosaic=5`; mixup and copy-paste are disabled.
  No custom mask loader is used in the formal training path.
- `exact_coco_dataset.py` is a superseded, unlaunched prototype. Official COCO
  `COCO.annToMask` remains the independent reference for fidelity diagnostics
  and original-JSON evaluation.

## Matched training budget

- Ultralytics 8.4.143, PyTorch 2.8.0+cu128, conda `pytorch`.
- Official checkpoint SHA256:
  `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`.
- Baseline: all stock supervised losses. CCL: identical losses plus the original
  mean `relu(cosine(mean_coeff_i, mean_coeff_j) - 0.1)` for different assigned GT
  instances with bbox IoU > 0.05, weighted by lambda 0.1.
- CCL uses all-category eligible pairs, as in the original method. Both YOLO26
  assignment branches retain the stock weighting schedule.
- Seeds 0, 1, 2, paired per seed, run sequentially on the single RTX 4090.
- Fixed 15 epochs per arm; no metric-triggered early stopping. Image size 640,
  batch 16 subject to preflight memory verification, effective batch 64 through
  stock accumulation (`nbs=64`), workers 8.
- AdamW, explicit `lr0=0.0001`, `lrf=0.01`, cosine decay, weight decay 0.0005,
  momentum 0.9, warmup 1 epoch with `warmup_bias_lr=0.0`, mixed precision BF16,
  deterministic mode. Bias warmup does not start at the stock 0.1, which would
  be 1000 times this fine-tuning learning rate. Both arms share this choice.
  These are prespecified fine-tuning settings, not chosen from this run's AP.
- Primary checkpoint is the EMA at the fixed last epoch, not best official-val
  AP. Preserve resumable checkpoints and raw train logs. Any intervention after
  launch is logged and applied symmetrically to affected comparisons.
- At the user's request, preserve every epoch (`save_period=1`) for all six
  runs, plus final FP32 model/EMA/optimizer/scaler/RNG recovery state. Official
  `epoch0.pt` denotes completed epoch 1, through `epoch14.pt` for epoch 15.

## Evaluation

Report all 5,000 val2017 images using untouched original COCO JSON and official
COCOeval. Dense1576 remains a secondary image subset. Full-image matching precedes
GT instance stratification; never delete low-density GT before matching.
Use `census/COCO_EVAL_INSTANCE_MANIFEST.csv` as the boundary-stable 36,335-instance
metadata input to the existing evaluator; do not reuse the raw-boundary old CSV.

Primary scientific outcome: high-crowding GT Mask Recall@0.75 and the difference
between high- and low-crowding treatment effects. Retain official Mask AP/AP50/
AP75/size breakdown, Box AP, matched-precision recall, pair recovery and GT-defined
spatial errors as prespecified in the existing COCO evaluation plan. Publish all
seeds and image-paired uncertainty, not only a favorable seed or metric.

The official validation set has been analyzed previously. This is a controlled
retraining replication with a frozen protocol, not a newly blinded test set.

## Gates before formal training

1. Environment version checks and seeded real-model CUDA forward/backward.
2. Fresh-agent execution of the documented environment witness.
3. Original annotation/image counts; official multi-polygon conversion fixture;
   emitted rows mapped to original annotation IDs, with explicit duplicate audit.
4. Actual official dataloader instance counts, class/box/mask alignment and
   independently decoded COCO mask fidelity on fragmented examples. Do not
   require bitwise fidelity from the standard polygon approximation.
5. Lambda-zero stock-loss/gradient equivalence; positive CCL gradients and no
   same-transformed-GT-index pairs in both assignment branches. Official mosaic
   may reuse a source image; the patch follows the original method's augmented
   instance indices, not source annotation IDs carried through augmentation.
6. Real 640px training batch memory/loss/gradient checks and a short trainer run.

If any gate fails, fix and rerun that gate before launching the formal queue.
Training completion and scientific success are separate outcomes.
