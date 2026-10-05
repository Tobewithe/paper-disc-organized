# YOLO26 All-Split External Inference

Date: 2026-09-07. R012 completed. This is a fixed-model external baseline, not a trained intervention or a method-acceptance result.

## Scope and Training Provenance

The user confirmed that the active checkpoint was fine-tuned only on PigLife. Its archived training_config.json, config.json and piglife_task05.yaml agree: the training image root and manifest are PigLife. Consequently all FaroPigSeg train/val/test and BamaPig2D train/eval images are eligible for external inference by this checkpoint. Original target-dataset split names do not imply exposure during checkpoint fine-tuning.

All 4,858 requested image records and 27,745 GT instances were processed, yielding 45,664 predictions. External evaluation expanded from the previous 492 images / 2,900 GT. Including the previously evaluated PigLife test set, baseline coverage is now 5,284 image records / 32,219 GT. Full raw-candidate diagnosis still covers only the earlier PigLife/Faro 586 images / 6,226 GT; this run exports final predictions, not new raw tensors.

## Fixed Protocol

- Checkpoint: `C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt`.
- Runtime: Conda pytorch, Python 3.10.19, torch 2.9.1+cu128, local Ultralytics 8.4.100, RTX 5060 Ti. The checkpoint's original runtime metadata is 8.4.27; that is not the runtime used here.
- Inference: 1024 square, rect=false, confidence .05, max_det=300, batch=1, retina_masks=true, seed=20260809. No target-data training or parameter selection occurred.
- Existing entrypoint: `C:/Dpan/codexproject/pigcv_research/workflows/inference/run_yolo26seg.py`. The installed runtime includes previously documented local mask crop/chunking changes. Post-run hashes of ops.py, nms.py, segmentation predictor and head.py match the existing `experiments/yolo26_runtime_source_snapshot_20260905_v1/provenance.json`; this is not a claim of pristine upstream code.
- GT: original Faro YOLO polygons and Bama COCO segmentation polygons. Positive unique image/annotation IDs per dataset prevent pooled-split collisions and zero-ID matching ambiguity. Original IDs and source split are retained. Polygons are unchanged; area and bbox are recomputed from raster masks, with source area retained. Size-stratified metrics therefore require this area definition when compared with older exports.
- Metrics: pycocotools segmentation AP at IoU .50:.95 and maxDets100; AP50/AP75/AR100 also retained. Pooled AP is recomputed across all records, not averaged from split APs. AP is measured on the fixed confidence-.05 output set.
- GT failure taxonomy: existing C/I/L/S/O/M/X/MISS definition, unchanged. C requires a one-GT/one-prediction graph component with coverage and purity both at least .75. Failure rate is non-C GT / all GT, and is not 1 minus AP.

## Results

AP columns use the 0-100 scale. Counts include all original image records, including byte-identical duplicates.

| Dataset | Source Split | Images | GT | Predictions | Mask AP | AP50 | AP75 | Failed GT | Failure Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FaroPigSeg | train | 1,039 | 11,046 | 22,037 | 39.89 | 66.39 | 41.52 | 5,180 | 46.89% |
| FaroPigSeg | val | 319 | 3,443 | 7,015 | 39.87 | 68.05 | 40.55 | 1,629 | 47.31% |
| FaroPigSeg | test | 160 | 1,752 | 3,484 | 39.88 | 66.79 | 40.42 | 797 | 45.49% |
| FaroPigSeg | all | 1,518 | 16,241 | 32,536 | 39.81 | 66.48 | 41.21 | 7,606 | 46.83% |
| BamaPig2D | train | 3,008 | 10,356 | 11,827 | 64.16 | 88.27 | 75.10 | 2,413 | 23.30% |
| BamaPig2D | eval | 332 | 1,148 | 1,301 | 64.43 | 88.06 | 76.04 | 272 | 23.69% |
| BamaPig2D | all | 3,340 | 11,504 | 13,128 | 64.15 | 88.32 | 75.08 | 2,685 | 23.34% |

| Dataset, All Splits | C | I | L | S | O | M | X | MISS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FaroPigSeg | 8,635 | 1,755 | 741 | 154 | 2,801 | 80 | 424 | 1,651 |
| BamaPig2D | 8,819 | 634 | 91 | 28 | 841 | 216 | 84 | 791 |

O/M/X are graph relation categories; O is not automatically a semantic proof of fragmentation. Their counts alone do not attribute failures to ranking, NMS, occlusion or a mask head.

## Verification and Data Quality

- Seeded CUDA kernel and real checkpoint forwards passed, including Bama's Unicode path. A fresh agent followed the environment ledger invocation and reported no documentation/runtime divergence.
- All five inference runs completed with exact image coverage, cache/export accounting and zero missing/unexpected checkpoint keys. Total logged inference-loop duration was 312.016 seconds; this excludes preparation, model startup and evaluation, and is not an end-to-end latency benchmark.
- Input verification reproduced all 2,900 historical GT masks with zero XOR pixels and all 2,900 classifier labels without mismatch when using historical predictions.
- Fresh outputs on all 492 historical Faro-test/Bama-eval images exactly match the old prediction lists, including masks, scores and boxes. Recomputed GT classes also have zero changes.
- Output audit verified exact cache/export scores and mask RLEs, and rechecked 6,397 frozen input/source hashes. Audit status is PASS for these deterministic checks; this coordinator-written audit is not an independent method review.
- Exact file hashing found no duplicate Faro files and 77 duplicate pairs in Bama, including 11 pairs crossing train/eval. Thus Bama has 3,263 unique file contents and the combined external scope has 4,781. This check does not establish video/pen independence or exclude visually similar frames.
- All 77 Bama pairs have some raster-mask annotation differences. Ten pairs differ in instance count; two pairs contain a matched-mask IoU below .75. The median of pairwise mean matched IoU is .9521. This supports documenting both small boundary variation and a smaller set of substantial differences; it does not justify declaring all 77 pairs erroneous.
- All requested records and source annotations remain included. No annotation was selected using model performance. A later grouped or deduplicated robustness evaluation must state its grouping and label policy explicitly.

## Duplicate Sensitivity After User Continuation

Three prediction-independent policies were fixed before evaluation. All use the same cached predictions and original annotations; the original full-data baseline remains primary.

| Bama Policy | Images | GT | Mask AP | AP Change, Points | Failure Rate |
|---|---:|---:|---:|---:|---:|
| All original records | 3,340 | 11,504 | 64.1492 | 0 | 23.3397% |
| Keep minimum image ID per duplicate group | 3,263 | 11,260 | 64.4483 | +0.2990 | 23.1794% |
| Keep maximum image ID per duplicate group | 3,263 | 11,260 | 64.4492 | +0.3000 | 23.1439% |
| Exclude all duplicate-group members | 3,186 | 11,016 | 64.7532 | +0.6040 | 22.9757% |

The observed changes are modest relative to the Faro/Bama baseline gap, and these policies preserve the current qualitative judgment. They do not bound all possible annotation selections or demonstrate video/scene independence. Excluding duplicate groups changes the evaluated image population, so its AP increase is not a model improvement.

Completed results: `experiments/yolo26_external_duplicate_sensitivity_20260907_v2/`; protocol: `experiments/yolo26_external_duplicate_sensitivity_20260907_protocol.md`; implementation: `tools/analyze_external_duplicate_sensitivity.py`. Selected IDs, exact metrics and eight unchanged input/source hashes are retained. The first attempt stopped after one policy because COCO.loadRes mutated shared annotation dictionaries. The script now isolates segmentation-only dictionaries per policy; v2 completed all policies. The partial v1 directory and failure.json are retained, and the original full-inference run was not changed.

## Current Judgments

1. Faro's low AP is also present in its previously unprocessed train and val splits, whose AP point estimates are both about 39.9. The broader baseline supports persistence of the dataset-level performance gap beyond the original 160-image test subset. Split consistency is not independent-farm replication.
2. Bama's full-scope AP and failure rate remain close to its eval subset. The two target datasets differ substantially in observed performance, supporting dataset-specific diagnosis. These numbers cannot separate scene difficulty, annotation conventions and domain shift as causes.
3. Completing this authorized expansion takes priority over collecting a fourth dataset. Current data are sufficient to continue the agreed diagnosis; an additional independent source remains conditional on the eventual paper claim and validation design.
4. This strengthens baseline evidence but does not establish an improvement. The next scientific gap remains the agreed Faro mask-quality investigation, annotation semantics and a deployable intervention with controlled gains. The candidate-scorer Gate and method-selection status remain unchanged.

## Artifacts

- Frozen protocol: `experiments/yolo26_external_full_20260907_protocol.md`.
- Run: `experiments/yolo26_external_full_20260907_v1/`.
- Metrics: `summary.json`, `summary.csv`, and `metrics/` within the run.
- Predictions: `inference/<dataset>_<split>/inference_cache.jsonl` and `predictions.json`; pooled segmentation-only exports `<dataset>_all_predictions.json`.
- Evidence: `input_sha256.json`, `output_sha256.json`, `historical_input_verification.json`, `historical_subset_comparison.csv`, `output_audit.json`, `duplicate_annotation_audit.json`, and `image_inventory.csv`.
- Implementation: `tools/run_yolo26_external_full.py`, `tools/verify_yolo26_external_inputs.py`, `tools/audit_yolo26_external_full.py`.
- Environment: `.aris/compute/local.md` and `.aris/compute/local-pytorch-spec.json`.

Prior test/eval results have informed exploratory diagnosis. External to checkpoint fine-tuning does not mean untouched by the research process. This run changes neither that history nor the original dataset annotations.
