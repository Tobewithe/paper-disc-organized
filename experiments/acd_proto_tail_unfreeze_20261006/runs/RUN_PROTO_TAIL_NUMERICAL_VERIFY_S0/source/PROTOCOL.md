# ACD prototype-tail unfreezing protocol

Locked before execution on 2026-10-06. The user explicitly requested trying ACD with more layers unfrozen. Local root: `C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006`. Execution uses the previously authorized laptop `ssh 28358lan`, remote root `D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006`, interpreter `C:/Users/28358/anaconda3/envs/pytorch/python.exe`. This is a new experiment; the completed coefficient-only screen retains its original records and conclusion.

## Question and competing explanations

The completed coefficient-only ACD screen trained both complete native coefficient branches, not just the last layers. Its ACD-minus-baseline full-val Mask AP difference was +0.037176 AP points. This experiment tests one new condition: whether allowing the last two prototype-generation layers to change makes the fixed ACD objective more useful. Possible outcomes include an ACD-specific improvement, a benefit of ordinary broader fine-tuning, no measurable benefit, or additional mask damage. A short negative screen does not establish that prototype learning or ACD is universally ineffective.

## Paired arms, frozen state and initialization

Both baseline and ACD independently restart from the original official COCO-pretrained `yolo26m-seg.pt`, SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`. Runtime is Ultralytics 8.4.100 from `D:/coco_wire/vendor_8.4.100`, with the same laptop PyTorch environment as the parent study; actual versions/device are saved in each Run.

| Module | Update policy |
|---|---|
| `model.23.cv4` / `model.23.one2one_cv4` | All native coefficient convolution/linear weight and bias parameters update. |
| `model.23.proto.cv2` / `model.23.proto.cv3` | Non-BN parameters update in both arms; these are prototype layers, not the detector's `model.23.cv2/cv3`. |
| Prototype `cv1`, upsample, feature refinement/fusion and semantic branch | Frozen. |
| Backbone, neck, box/class heads and all other original modules | Frozen. |
| Every BatchNorm affine parameter and running buffer | Frozen; modules remain in eval mode during training. |

Expected trainable counts on the original m checkpoint are 1,708,224 coefficient parameters plus 598,016 prototype-tail parameters, total 2,306,240. These are expectations until checked against actual parameter names/shapes in TRAINING_SETUP; a mismatch stops execution. Shared upstream features remain fixed, so exact box/class/confidence/detection identity parity is required at inference.

Preserve the official forward detach and E2E branch schedule. The one2many branch receives differentiable prototype tensors; the one2one branch receives detached prototypes. Therefore the prototype tail receives one2many mask/ACD gradients, while one2one ACD directly updates its coefficient head. The semantic path is frozen and does not pass through these tail blocks. The three-epoch one2many/one2one weights are 0.8/0.2, 0.45/0.55 and 0.1/0.9. This supervision asymmetry is a declared limitation, not silently removed to improve results.

## Data and budget

Reuse exactly the parent study's 796-image train2017 fit list and 196-image converted-label internal validation list. The 197-image dev list is not used to fit or select settings. Image/list/annotation hashes and counts are checked before execution and saved in DATA_RECEIPT. The historical cohort and val2017 have already informed prior research; full validation is not a fresh blind test.

Run a separate gradient/optimizer diagnostic, then paired one-epoch smoke Runs on the first 32 fit and 8 validation images. If valid, restart both formal arms from official weights for exactly 3 epochs, batch 2, workers 0, imgsz 640, seed 0. This preserves the parent's budget to isolate the unfreezing variable; do not also extend to eight epochs or change the optimizer. It remains a short feasibility screen.

Retain the official checkpoint's supported training hyperparameters: MuSGD, AMP, lr0=0.00038, lrf=0.88219, momentum=0.94751, weight_decay=0.00027, warmup_epochs=0.98745, nbs=64, official augmentation, overlap_mask=true, mask_ratio=1 and semseg_loss=true. The official MuSGD parameter-name rule assigns prototype `cv3` the 3× group LR, while `cv2` receives the ordinary group LR. Save exact parameter groups, learning rates, accumulation and optimizer-attempt identities; do not override this rule. 1,194 input batches are not 1,194 optimizer updates.

## ACD objective: unchanged

Only the ACD arm adds the same action-gated false-positive penalty as the parent: weight 0.05, tau 0.50, minimum IoU gain 0.01, minimum target coverage 0.80, first at most 12 assigned candidates per image per branch in stable raw-index order. Preserve integer overlap instance IDs; interpolate prototypes to the training-mask grid; use the assigned GT support box for action comparison and background terms. Average accepted candidate softplus terms and add `0.05 * batch_size * aux` only to segmentation loss component 1.

This penalty uses all positive-logit background pixels in an accepted candidate, not only pixels removed by the trial action; it is not full tightened-mask/logit distillation. GT selects the training term and does not enter inference. No ACD threshold, weight, candidate policy, foreground term or loss substitution changes after results are observed.

## Execution and gradient validity

Require finite live loss, every gradient actually applied by the optimizer to be finite, recorded nonzero applied gradients and parameter changes for both coefficient branches and the prototype tail, and unchanged frozen live/EMA tensors including all BN state. Record raw AMP observations separately and prove any nonfinite attempt was skipped by GradScaler. Applied-update counts may differ naturally; do not change AMP/scale to force equal counts. Unexplained skips, unsafe updates, absent prototype gradients or frozen drift invalidate the screen.

Both arms must have identical complete transformed-input hash streams and optimizer-attempt epoch/batch/LR/accumulation/AMP schedules. Preserve optimizer-attempt identities, actual application/skip counts, failure Runs and partial outputs. EMA updates at every official optimizer attempt, including GradScaler skips, with frozen tensors restored to their original values. Diagnostic checks must establish the one2many prototype-tail gradient route and the official one2one detach.

Save source/protocol snapshots before every Run. Preserve FP32 live and third-epoch EMA mask state. Final evaluation inserts both coefficient and prototype-tail state into the original official FP32 model. The payload `mask_final_ema.pt` has kind `mask_coefficient_proto_tail_ema_final`, epoch 3 and audit_passed=true. It contains exactly both coefficient branches and prototype cv2/cv3 state; included BN tensors must match official originals. Do not use the older coefficient-only loader or choose best.pt.

## Evaluation, scope and gate

Independently evaluate baseline, ACD and original official reference on all 5,000 original COCO val2017 images with original annotations, native one2one, FP32, tf32=false, square 640 LetterBox scaleup=false, conf=0.001 and max_det=300. Keep the pinned official JSON mask decoder (`process_mask_native`, original-image scale_preds.byte()) and COCO80-to91 categories unchanged. Use pycocotools with maxDets=[1,10,100], separate bbox/segm input fields and mask-area semantics. Report Box/Mask AP, AP50/AP75 and size AP; exact frozen upstream and box/class/confidence/detection parity are validity requirements.

The paired damage diagnostic retains the parent's output-layer definition: every baseline detection selects the same-class noncrowd GT with highest Box IoU >=0.5; multiple detections may match one GT. Baseline mask IoU >=0.75 is success; crossing below is damage and the reverse is repair. Report matched counts, unique GT, success/failure denominators, repair/damage rates and continuous IoU changes, with image-cluster bootstrap 5,000 resamples seed 0. This is not raw score-independent geometry, one-to-one maximum matching, COCO recall or AP uncertainty. Full raw five-state taxonomy, pixel AUC/FPR and crop-support ceilings are not measured by this screen and remain unknown; do not infer them from filtered outputs.

Primary continuation gate: valid execution/evidence and ACD minus the same-scope baseline Mask AP >= 0.003 (+0.3 AP points), AP75/APsmall not both falling, baseline-success damage <=1%, and baseline-success mean IoU change >= -0.005. ACD minus the untouched official model is also reported so ordinary fine-tuning degradation is visible. Compare the new ACD-minus-baseline effect with the parent's effect only after confirming identical initialization/data/inference scope; this interaction is descriptive for one seed, not a significance claim.

Stop after this fixed screen. No automatic seed expansion, backbone/neck unfreezing, detach changes, loss tuning or longer training. Missing or invalid evidence leaves the outcome unknown; failing a valid gate is a negative screen for this configuration, not proof that the broader mechanism cannot work. Complete collection into the new Run directories retains transfer.json and manifest.sha256.
