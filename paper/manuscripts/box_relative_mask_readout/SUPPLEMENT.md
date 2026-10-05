# Supplementary material: box-relative mask readout

Research version, 22 September 2026. This document expands completed experiments. It is not evidence for unrun transfer experiments or for a uniquely identified pretraining defect.

## A. Experimental units and data use

There is one fixed official COCO-pretrained YOLO26m-seg backbone. The three reported seeds concern additional readout training only. The same 800 COCO train2017 images fit the additional heads; the same separate 200 images select strength. These images were already eligible for the original COCO pretraining. They are therefore held out from the new head when appropriate, not unseen by the original pretrained network.

The fitting and selection records are fixed by the stored image IDs and candidate identities. Each record has an image ID, raw-candidate ID and one annotation ID. Same-class box IoU must be at least 0.5. Ambiguous raw-candidate ownership is excluded; duplicate raw identities are not treated as independent training examples. Multipart COCO polygons are decoded as one annotation mask. This produces 14,204 fitting and 3,723 selection records, covering 984 effective images. Some images supply no eligible record. These records are post-detection training examples, not a replay of the original Task-Aligned Assigner.

The diagnostic panels use a different representative selection: among eligible candidates for an object, the candidate with the best box IoU is retained and held fixed across interventions. No best-mask selection is performed. Final COCO AP uses all retained model outputs and official matching, rather than those diagnostic representatives.

## B. Correction and training

Let the original selected-candidate mask logit be z_i(x)=c_i^T P(x). A two-output head predicts v_i=(v_i0,v_i1), each bounded by tanh. The evaluated correction is

z'_i(x)=z_i(x)+4 alpha [v_i0+v_i1 T_i(x)].

The effective b_i and a_i in the main paper are 4v_i0 and 4v_i1. Alpha is a global inference strength chosen separately for each trained head from {0,0.25,0.5,1}, using mean hard-mask IoU on the selection records. Ties favor the smaller alpha. Epoch eight is always used; val2017 is not used to choose a checkpoint or alpha.

The native input contains the 64-dimensional input of the original coefficient output layer and a three-dimensional pyramid-level one-hot vector. Fitting-set channel means and standard deviations normalize this 67-vector, with standard deviations clamped below at 0.05. A 67→64 linear layer, SiLU, 64→2 linear layer and tanh produce v. The output layer is initialized to zero. The scalar control uses the same architecture but multiplies the second output by zero. The response variant adds 16 values obtained by adaptive 4×4 average pooling of the original 32×32 ROI logits, division by eight and clipping to [−4,4]; it uses 83 inputs. The native spatial model has 4,482 trained parameters; the response variant has 5,506.

For the original 32 prototypes, the existing ROI feature cache normalizes each prototype by RMS=sqrt(mean(P²)), clamped below at 0.1. This is not a centered standard deviation. Original GT masks are resized by the saved bilinear letterbox/ROI procedure into soft 32×32 training targets. Hard original-resolution GT is separately used for the diagnostic spatial statistics and official validation.

The per-instance training loss is the mean pixelwise binary cross-entropy plus 0.5 times the implemented soft Dice loss. All compared heads use AdamW, learning rate 0.0003, weight decay 0.0001, batch size 64, eight epochs, and global gradient clipping at 10. The original backbone, prototypes, predicted boxes and category scores are frozen. The native coefficient control instead changes only the three existing final coefficient output layers; it has 6,240 trainable parameters and uses the same labeled records and budget.

The fixed 4×4 residual template is extracted from fitting data only: pool the residual Y−sigmoid(z) into 4×4 ROI bins, average records within an image and then average the effective fitting images. The code maps the field into each predicted box and normalizes its mean and standard deviation over the actual discrete predicted crop. The normalization is part of the method; it cannot be silently replaced by a continuous-box integral or by statistics from a different mask resolution. A degenerate crop requires explicit handling in a deployment implementation. The executed training-bank preparation verified nonempty crops and positive template standard deviations.

## C. All twelve COCO metrics

Values are on the 0–100 scale. SD is the sample standard deviation over seeds 0, 1 and 2. The pretrained baseline is fixed and has no additional-training SD.

|Metric|Pretrained baseline|Native scalar|Native spatial|Native coefficient fine-tuning|
|---|---:|---:|---:|---:|
|AP|43.5185|44.0257 ± 0.0152|44.2684 ± 0.0048|44.1848 ± 0.0060|
|AP50|66.2963|66.5572 ± 0.0071|66.6801 ± 0.0157|66.3852 ± 0.0488|
|AP75|47.0263|47.7080 ± 0.0434|48.1385 ± 0.0594|48.0635 ± 0.0185|
|APS|22.4873|23.1529 ± 0.0172|23.3651 ± 0.0191|23.0518 ± 0.0267|
|APM|47.5486|47.8506 ± 0.0057|48.0756 ± 0.0353|48.1177 ± 0.0175|
|APL|63.4386|63.5487 ± 0.0671|63.8558 ± 0.0332|63.6971 ± 0.0286|
|AR1|34.1040|34.3787 ± 0.0036|34.5028 ± 0.0061|34.5676 ± 0.0031|
|AR10|55.3652|55.9714 ± 0.0156|56.2784 ± 0.0031|56.0246 ± 0.0234|
|AR100|59.6866|60.3404 ± 0.0132|60.7639 ± 0.0193|60.2591 ± 0.0295|
|ARS|40.3070|41.6334 ± 0.0335|42.2191 ± 0.0542|41.3394 ± 0.0688|
|ARM|65.5640|65.6696 ± 0.0253|66.0075 ± 0.0265|65.3833 ± 0.0343|
|ARL|76.3734|76.4665 ± 0.1049|76.8020 ± 0.0300|76.8579 ± 0.0268|

## D. Individual seeds and selection

|Method|Seed|Selected alpha|Mask AP|AP75|AP small|Repaired GT75|Lost GT75|
|---|---:|---:|---:|---:|---:|---:|---:|
|native_scalar|0|0.5|44.0431|47.6771|23.1345|803|454|
|native_scalar|1|0.5|44.0157|47.6893|23.1556|768|416|
|native_scalar|2|0.5|44.0182|47.7575|23.1686|796|434|
|native_spatial|0|0.5|44.2720|48.1513|23.3690|1151|466|
|native_spatial|1|0.5|44.2629|48.0738|23.3819|1112|447|
|native_spatial|2|0.5|44.2702|48.1906|23.3444|1139|458|
|native_finetune|0|1.0|44.1783|48.0696|23.0270|1706|1303|
|native_finetune|1|1.0|44.1901|48.0427|23.0801|1684|1259|
|native_finetune|2|1.0|44.1860|48.0782|23.0484|1681|1254|

These seeds do not sample original pretraining randomness, dataset splits, or model scales. Small SD should not be read as proof of generalization beyond this setup. Post-hoc native coefficient alpha-0.25/0.5 evaluations are sensitivity analyses only; they do not change the train-selected alpha-1 control.

## E. GT matching and uncertainty

Repairs are GT annotation IDs unmatched by the original COCO evaluator at mask IoU 0.75 and matched after correction. Harms are IDs matched before but unmatched after correction. Crowd and ignored GT are excluded from this count. The matched sets come from the all-area official evaluation; predictions may still be false positives, and standard COCO category/rank averaging remains intact for AP. Consequently, repairs minus harms is not an AP decomposition.

The R75 analysis includes 36,335 non-ignored non-crowd GT objects. For each method and GT, success is averaged over its three trained models. The 2,000 bootstrap draws resample 5,000 images with replacement, preserving dependence between objects and methods within an image. Ratios are recomputed from the summed GT successes and denominators. These intervals condition on the three available trained models. They are not intervals for AP and do not include unobserved training-seed uncertainty.

Size groups use provided annotation area: small below 32², medium from 32² to below 96², large at least 96². These nonoverlapping diagnostic groups are stated explicitly rather than assumed identical to every area-boundary detail inside COCOeval. Fill ratio is provided annotation area divided by provided bbox area. It is not an occlusion measure. Two provided annotations have ratios above one; they are retained in the highest group rather than silently clipped. There are no nonpositive bbox areas. Group results are exploratory and not multiplicity adjusted.

## F. Boundary evaluation

The authors’ Boundary IoU implementation is vendored at revision 37d25586a677b043ed585f10e5c42d4e80176ea9. Boundary width is 0.02 times the image diagonal; non-crowd matching uses the implementation's minimum of mask and boundary IoU. The same predictions, scores and strength choices are reused. Compatibility edits replace obsolete np.float aliases with float and limit the worker pool to four. They do not alter dilation or matching.

Before Boundary AP, ordinary mask AP is checked against the reference result. A failed preliminary run is retained: COCO.loadRes mutated the input dictionaries with numpy bbox entries, leading a subsequent load to enter an incompatible old bbox branch. The retry removes those internally added bbox fields before loading segmentation-only predictions. No model output or segmentation mask was changed to fix the interface.

Boundary AP is currently seed 0 only. The native spatial head gains 0.611 points over the original model, but is 0.012 points below native coefficient fine-tuning. This is a limit to a universal boundary-superiority claim.

## G. Timing scope

Timing uses 64 fixed COCO validation images, batch one, 12 warm-up calls per method and three repeated randomized image orders. CUDA is synchronized around wall-clock measurements. Confidence is 0.001 with at most 300 selected candidates, averaging approximately 125 per image. Included operations are network forward, raw decode, correction, mask assembly, crop and resizing to original dimensions. Disk loading, CPU RLE encoding and COCO evaluation are excluded.

Every arm uses the research evaluator's all-raw-candidate prototype matrix multiplication. This is neither a selected-candidate-optimized deployment decoder nor a benchmark at a conventional display threshold. The native spatial head remains approximately 53% slower than the baseline by median image time in this measurement, despite using only 4,482 trained parameters. A small parameter count is not evidence of negligible latency.

## H. Evidence not yet established

Cross-scale and cross-dataset robustness, performance against a full reimplementation of SipMask or another external method, and a unique historical cause in original pretraining remain unestablished. Current quadrant coefficient experiments are structural controls inspired by SipMask, not full SipMask reproduction. Controlled projection demonstrates that much of the observed correction direction is available in the prototype span; it does not upper-bound every GT mask's expressibility. Thin/low-fill objects show no current average recall gain, and the correction can reduce target coverage.

Raw runs, selection files, model states and code snapshots are linked through EVIDENCE_MAP.md and REPRODUCIBILITY.md. Failed or superseded attempts remain attached to their original Run IDs.
