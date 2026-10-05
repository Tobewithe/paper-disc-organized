# Reproduction inventory and current limitations

This is an inventory of the executed research pipeline, not yet a self-contained public release. Run snapshots and original outputs are authoritative. The paper reuses experiments without copying or renaming their identities.

## Environment and common inputs

- Ultralytics 8.4.100; official COCO-pretrained YOLO26m-seg; actual one-to-one segmentation output.
- Current remote execution uses Python 3.12, PyTorch 2.8.0+cu128, NumPy 2.3.2 and an NVIDIA RTX 4090. Exact run metadata should be consulted for older source runs.
- COCO train2017 and all 5,000 val2017 images with original instance annotations. A multipart annotation remains one instance; `iscrowd` retains its official evaluation semantics.
- Training bank: `coco_mask_spatial_probe_20260916/runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt`.
- 800/200 fitting/selection split: the same study, `runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json`.
- Native features: `coco_spatial_calibration_mechanism_20260922/runs/RUN_098ba724d6034a1eab6ae058362d2ba0/coefficient_features.pt`.
- Residual-template geometry: `coco_template_source_20260921/runs/RUN_143f4861823342c8afdca4251ee764d8/geometry.pt`.

The remote execution roots of reused studies differ (`/root` versus `/root/autodl-tmp`). The queue scripts record these paths explicitly. A portable release must replace them with a user-provided configuration and include the source bank-preparation workflow. Merely distributing a trained correction checkpoint is not a complete reproduction package.

## Executed steps and evidence

1. **Template and reference head.** See `experiments/coco_template_source_20260921/REPORT.md`. The residual template uses fitting labels and original responses, not test labels or a previous head's output. The selected reference checkpoint is `RUN_64d6c0aeacc44d28a4fc13e1832e6a03/epoch8.pt`.
2. **Mechanism panels.** See `experiments/coco_spatial_calibration_mechanism_20260922/REPORT.md` and its decomposition/within-instance protocols. Representative selection is fixed by boxes and class, not by best mask. Changes in this panel are paired fixed-candidate diagnostics, not COCO AP.
3. **Native readout control.** Training `RUN_8ad0b76e6b9f456fac99b476f6cb35bf`; full evaluation `RUN_984bb9ff1cd2432bae2c2c08c3042c66`. Only existing final coefficient mappings change. Checkpoint saves initial states to reconstruct interpolated strength without affecting boxes or scores.
4. **Native feature correction.** See `experiments/coco_native_spatial_readout_20260922/PROTOCOL.json`, `scripts/native_spatial_readout.py`, and `execution_ids.json`. Three independent training runs, a bounded identity check, and one shared-forward five-arm evaluation have distinct Run IDs. Final epoch and training-selection strength are preserved.
5. **Timing.** `scripts/benchmark_readout.py` uses the same 64 images and three orders across all six modes, 12 warm-up images per mode, CUDA synchronization, and batch size one. It includes forward computation, correction, assembly, crop and resize; excludes disk reads, RLE and COCO evaluation. Confidence 0.001 is an evaluation operating point, not a generic deployment-speed claim.
6. **Boundary evaluation.** `scripts/boundary_evaluation.py` applies the published author implementation to unchanged saved predictions. The source revision and two compatibility edits are in `vendor/boundary_iou_api/PROVENANCE.json`. The expected plain-mask AP is checked before using Boundary AP. Results are complete in `RUN_0425bd7208e149c38352d079248fcfee`; the three-seed and subgroup summary is `RUN_5c3523cfdb1f4328ab0901483037e5d8`.

## Inference identity

Keep candidate selection, category scores, boxes and the mask resize rule fixed. The new module changes only selected candidates' mask logits. The default correction output is zero at initialization. During the bounded new-head check, the alpha-zero crop/threshold result agrees pixel-for-pixel with `ops.process_mask`; pooled low-resolution response features agree with the earlier ROI path within 1.2e-6. This is a bounded implementation check, not a claim that every possible image and backend has been tested.

Masks are cropped and thresholded at input resolution, scaled to original image size with the frozen reference rule, and encoded as RLE. COCO result records omit bbox so mask areas are used. Save predictions, all twelve official metrics, Mask75 matched GT IDs, and the run's exact source snapshot. Native-control alpha 0.25 and 0.5 are explicitly sensitivity measurements; alpha 1 remains its training-selected configuration.

## Before public release

Parameterize dataset/weight/output paths; supply one documented preparation and inference entry point; include data split identifiers, seed and strength selection; add an appropriately licensed source package; verify a clean-environment reproduction. Three-seed outputs are linked in the evidence map; transfer outputs are still pending. No unrun command or planned evaluation should be described as completed.
