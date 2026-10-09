# Existing-checkpoint evaluation — 2026-10-06

The current task finishes evaluation of the already trained Stage-I checkpoints. It does not retrain the heads, change the model inputs, alter the FIT-derived radius, change the two registered steps, select another checkpoint, or register Stage II. Training implementation deviations are recorded in `AUDIT.md` and remain deviations after evaluation.

## Inputs and versions

- Authorized host: `root@connect.nmb2.seetacloud.com:24084`; execution root `/root/autodl-tmp/qcr_run`; Python `/root/miniconda3/bin/python`; PyTorch 2.8.0+cu128, Ultralytics 8.4.100, official YOLO26m-seg checkpoint SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`.
- Direct32: `runs/RUN_stage1_direct_seed0_retry5/final.pt`, SHA `25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8`.
- Quality: `runs/RUN_stage1_quality_seed0/final.pt`, SHA `07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065`.
- Fixed epoch 3. Both nominal seed labels are 0; actual head-training seed is 20261005. `rho=12.0853271484375`, lambda 0.003, eta=(rho/2,rho/4), total displacement capped at rho. Evaluation bootstrap/probe bookkeeping cannot change these settings.
- `RUN_CONFIG.remote.json` preserves the historical remote configuration. `EVALUATION_CONFIG_20261006.json` changes only the JPEG extraction location to `/root/autodl-tmp/qcr_run/evaluation_inputs/images` on the data filesystem. The old image symlink and failed extraction files on the full system filesystem remain preserved. `TMPDIR` is the existing execution root's `tmp/` on the data filesystem.

## Scope and definitions

`qcr_evaluate_complete.py` produces A (original), B (Direct32), D1 (fixed first Q step), D (registered two Q steps) for each official one-to-one TAL matched candidate. It writes all six permanent identity fields, both classes, COCO area, predicted box quality, per-arm normal original-image metrics, local quality diagnostics and correction timing. GT enters the evaluation identity and labels only; it never enters the trained heads or coefficient-step selection.

Normal mask decoding uses complete P@c, bilinear upsampling to 640, the frozen predicted-box crop, threshold at zero, then inverse letterboxing of the binary mask and threshold at 0.5. IoU, Mask75 and coverage use original COCO `annToMask`. The original manual-logit replay is checked against `ops.process_mask` for every evaluated candidate.

`qcr-original-pixels-v2` reports primary AUC on continuous cropped logits after bilinear inverse letterboxing, against original COCO GT within inverse-letterboxed predicted-box support. FPR uses the actual normally decoded binary mask among original non-GT pixels in that support. Other instances count as non-target pixels. AUC is exact Mann–Whitney with average ties; missing positives/negatives remain undefined. Auxiliary 640-letterbox AUC/FPR are retained separately. `RUN_stage1_dev_complete_seed0` is the earlier complete diagnostic with primary 640-letterbox AUC/FPR; its IoU/Mask75/coverage are original-image metrics. The original-pixel DEV Run is `RUN_stage1_dev_original_metrics_seed0`.

TAL MaskFail means class argmax correct, BoxIoU≥0.75, original MaskIoU<0.75. Strict MaskFail additionally requires that the entire class-argmax-correct raw pool has a Box75 witness and no Mask75. Streaming FINAL checks this full pool, with conservative crop-support pruning and smoke verification against exhaustive decoding. Reported strict effects are on that set's intersection with the fixed TAL class-correct Box75 target. They do not cover unassigned GT or all raw locations. Historical DEV cache has no complete raw pool, so its strict labels remain unknown rather than being replaced with TAL failure. The older 6,459-instance raw classification used a different input geometry and is not silently joined.

Quality diagnostic states reuse the existing GT-free success perturbation rule at each held-out image/raw identity: c0 and ±rho/8*u1, ±rho/4*u1, ±rho/4*u2, +rho/8*u2. No DEV/FINAL oracle is computed. This is an added post-training diagnostic, not the original frozen oracle-direction state test. Pairwise true hard-IoU ties are excluded, predicted ties receive 0.5, and Spearman uses average ranks. First-step improvement includes exact IoU ties in the denominator; Q/IoU sign agreement reports zero exclusions separately.

Image macro averages candidates within eligible images before averaging images. Each paired bootstrap draw resamples entire images, preserving all candidate pairing. Empty subgroup images are not given zero IoU. Output is fixed-candidate mechanism evaluation; no confidence/top-k/NMS/COCO AP reranking is evaluated. Q-Rank retains c0 on this population and is not a separate AP result.

Timing covers evidence encoding and Direct32 or the two Q coefficient-gradient steps, with CUDA synchronization. It excludes frozen YOLO forward, mask decoding, GT labels, pixel metrics, raw-pool diagnostics and statistical aggregation. It is serial batch-one measured overhead, not end-to-end production latency.

## Runs and preservation

The original training, cache-preparation, oracle and failed evaluations remain unchanged. `runner.py` snapshots each actual evaluation's code and config, records checkpoint inputs, interpreter, execution status and expected artifact completeness. The saved source snapshots, not a later working copy, identify each Run's implementation.

The first full FINAL `RUN_stage1_final_complete_seed0` failed during JPEG extraction with ENOSPC on the linked system filesystem. It has no method results. The data-filesystem retry is `RUN_stage1_final_complete_seed0_retry1`. A separate supervisor launch failure caused by CRLF shell line endings is retained as `RUN_eval_supervisor_launch_failure_20261006`; no model execution occurred in that launch. Successful smoke Runs are retained independently. `qcr_eval_chain_retry1_20261006.sh` schedules only the FINAL retry, original-pixel DEV and ten-image FP32 DEV witness, and refuses duplicate Run directories.

Local transfer restores files into the same historical Run directory and retains existing `run_id` values. Each transferred Run contains `manifest.sha256` and `transfer.json`, explicitly separating original execution state, original artifact completeness, transfer scope and local availability. The full FIT/DEV tensor cache and epoch checkpoints remain remote. Local final checkpoints are sufficient to inspect or rerun the saved method; dataset/cache availability and server environment are separately required. `REPORT.md` is the completed result and decision record.
