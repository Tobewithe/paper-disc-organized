# Diagnosing and Correcting Box-Relative Decision Bias in Prototype Instance Segmentation

**Research draft, updated 23 September 2026.** This manuscript reports completed mechanism experiments, three-seed native-readout comparisons, seed-0 boundary evaluation, and fixed-prior/coefficient structural controls. The new controls substantially narrow the method claim: ordinary coefficient correction nearly matches overall AP, while a small-object Mask75 recall difference remains. Transfer experiments are incomplete. It is not a submission-ready or state-of-the-art claim. Evidence locations are listed in `EVIDENCE_MAP.md`; unresolved requirements are explicit in Section 7.

## Abstract

Prototype-based instance segmentation separates shared mask features from instance-specific coefficients, but this decomposition alone does not explain why an accurately detected object can still receive a poor mask. We investigate whether some errors reflect insufficient prototype expressivity or a decision rule that fails to exploit information already represented by the model. In a fixed COCO-pretrained YOLO26m-seg, foreground reliability differs between central and peripheral locations even within the same instance and logit interval. A paired analysis of 1,713 instances finds a 6.83 percentage-point difference in the foreground-minus-prediction residual, with a 95% image-cluster bootstrap interval of [5.08, 8.58]. Guided by this observation, we study an additive mask correction consisting of an instance bias and an instance-scaled, box-relative spatial field. Controlled removal, reversal, decoder changes, and projection onto the existing prototype span show that the spatial direction matters and that most of its measured benefit does not require new prototype directions. On the complete COCO val2017 set, a correction trained using 800 training images and selected using another 200 raises mask AP from 43.52 to 44.27 ± 0.005 over three additional-training seeds. The matched native-feature scalar head reaches 44.03 ± 0.015, while fine-tuning only the native coefficient output layers reaches 44.18 ± 0.006. A stronger, seed-0 coefficient-residual MLP reaches 44.26 AP, only 0.016 points below the spatial head. Nevertheless, the spatial head gives 1.03 percentage points higher small-object GT recall at mask IoU 0.75 than this control [0.70, 1.39], under paired image resampling of fixed predictions. The result narrows the contribution to a possible small-object repair–damage advantage rather than overall AP superiority. The evidence identifies a correctable decision-level limitation; it does not uniquely identify its historical training cause or establish generality across architectures.

## 1. Introduction

An instance segmentation failure can occur before or after a suitable object candidate is produced. A missing or misclassified candidate cannot be repaired by changing its mask alone. Conversely, a candidate with an accurate bounding box may still include substantial background or omit object pixels. Understanding the latter case requires following the same candidate through its mask computation, rather than changing the associated prediction whenever a different metric becomes favorable.

Prototype-based segmenters provide a tractable setting for this investigation. A shared set of spatial maps is linearly combined using coefficients predicted for each candidate. Poor masks could arise because the maps cannot express the required shape, because the coefficient readout selects an unsuitable combination, or because thresholding converts a useful but biased response into the wrong pixel set. These explanations imply different modifications. Enlarging the prototype representation is unnecessary for a correction already available in its span; simply predicting more expressive coefficients may also be insufficient if their learning remains poorly constrained.

We first ask a narrower empirical question: does the original mask response carry the same foreground reliability at different relative positions in an instance box? A pooled comparison suggests that it does not. We then pair locations within the same instance and logit interval to reduce confounding by category, object size, and between-instance shape differences. Central and peripheral locations still have different foreground frequencies despite nearly equal predicted probabilities. This observation does not imply that object shape priors are undesirable. It indicates that relative position retains potentially useful decision information after conditioning approximately on the original response.

We investigate that information using a low-dimensional correction. A fixed spatial field is defined in coordinates relative to the predicted box, while two instance-dependent values control its amplitude and a uniform bias. The field can be obtained from training residuals without relying on an earlier refinement network. The resulting mask is generated using a position-dependent threshold while retaining the original object boxes and category scores.

The main contribution of the present evidence is the diagnosis and its controlled decomposition, rather than the generic use of spatial weights or an additional mask head. Spatially varying coefficients, attention, and dynamic mask networks are established ideas [2–4]. We test competing explanations by comparing independent scalar training, reversing or removing the spatial term, changing a binary resizing detail, projecting the correction into the original prototype span, altering training support, fine-tuning native coefficient layers, and training fixed-spatial and more flexible coefficient readouts. Several controls limit the strength of the conclusion: a coefficient-residual MLP nearly matches overall AP, the complete spatial correction loses some target coverage, and a small nonzero benefit remains outside the original prototype span.

The current study therefore supports three concrete findings: within-instance position contains residual foreground information; a directed spatial correction can exploit part of it; and most of the useful correction direction is expressible by existing prototypes for the examined objects. Establishing a broadly useful and computationally efficient method requires the additional evaluations identified in Section 7.

## 2. Related work

**Prototype and spatially conditioned masks.** YOLACT [1] popularized efficient mask assembly from shared prototypes and instance coefficients. BlendMask [2] combines instance attention with spatial bases. SipMask [3] predicts separate coefficients for subregions of a detected box, making it a particularly relevant comparison. CondInst [4] predicts compact instance-specific convolutional parameters. Our formulation does not introduce spatial conditioning or dynamic parameter prediction as general concepts. It studies a constrained correction of an existing mask decision and tests whether its useful directions require an enlarged basis.

**Alignment and refinement.** STMask [5] addresses spatial feature alignment and temporal fusion. PointRend [8] and Mask Transfiner [9] refine masks at selected locations with additional feature processing. B2Inst [10] introduces boundary representation for basis-based segmentation. These methods motivate considering feature alignment, fine details, and boundary representation as alternatives to a low-dimensional decision correction. PatchDCT [15] uses local compressed-mask refinement to limit the global effect of changing a representation coefficient. Boundary Patch Refinement [17] processes image patches around predicted mask boundaries and is a relevant external post-processing comparator. Our experiments do not establish that those mechanisms are unnecessary; their scope is the particular errors repaired by the measured spatial field.

**Scores, probabilities, and decisions.** Mask Scoring R-CNN [7] estimates mask quality to adjust instance scores. We preserve those scores and change mask pixels. Local Temperature Scaling [6] performs spatial probability calibration. A positive temperature preserves the sign of a binary logit, whereas an additive correction can change its foreground decision. We use “decision bias” to describe the observed conditional response discrepancy and resulting mask repair. Kandinsky conformal calibration [16] also exploits spatial structure, but addresses prediction coverage rather than the same mask-AP objective. We have not established improved expected calibration error, Brier score, or conformal coverage, and therefore do not equate higher IoU with better probability calibration.

**Evaluation.** Standard COCO mask AP measures performance over confidence rankings and multiple IoU thresholds. It is not the fraction of GT objects for which an oracle can find a good raw candidate. Boundary IoU [11] offers a complementary view of object boundaries. We evaluate fixed seed-0 predictions with the authors’ Boundary AP implementation, using its minimum of mask and boundary IoU and a boundary width of 2% of the image diagonal. Ordinary mask AP is checked before running the boundary metric; no strength is selected using boundary results.

## 3. Diagnostic design

### 3.1 Fixed model and candidate identity

We use official COCO-pretrained YOLO26m-seg weights [13] with Ultralytics 8.4.100 [14]. The experiments operate on the actual one-to-one output branch of these weights. They do not infer the active branch from a generic configuration or from claims about all YOLO models. This is an explicitly fixed branch, not a claim that every software version's default predictor uses it. Image preparation uses a 640-pixel target size with stride-compatible automatic letterboxing. Full evaluation preserves the original top-300 selection and score cutoff of 0.001.

The controlled diagnostic panel differs from final COCO evaluation. Within each image, a candidate must have a same-category box IoU of at least 0.5 with a GT object. Ambiguous raw-candidate ownership is excluded, and the best box-quality candidate is retained for each eligible GT. The same candidate is used across interventions. Mask quality is not used to select that representative. This panel permits paired measurements but excludes GT objects without an eligible detection and cannot estimate the contribution of all detector failures.

Final reported AP instead evaluates every retained output using official COCO segmentation matching. Prediction records contain masks and scores without a supplied bbox, so segmentation area is not replaced by box area when loading detections. Repairs and harms at Mask75 are changes in the sets of non-ignored GT objects matched by that evaluator.

### 3.2 Data separation

The correction learns from 800 COCO train2017 images [12], with another 200 used to choose correction strength. After the established removal of ambiguous and duplicate candidate records, these provide 14,204 fitting and 3,723 selection records; 984 images have effective records. The template is extracted only from the fitting split.

Two additional, mutually disjoint 1,000-image training panels are used for diagnosis. Each is divided into 500 exploratory and 500 confirmation images. They contain 6,917 and 6,705 fixed representative instances, respectively; the second confirmation set has 3,537 instances. These images are held out from the additional correction training, not from the original COCO pretraining. The within-instance analysis is a follow-up on the second panel, not a third independently selected confirmation set. General performance is evaluated on all 5,000 val2017 images.

### 3.3 Position-conditioned reliability

Let z be an original mask logit, Y the hard binary GT value, and u the location relative to the predicted box. We compare the central quarter of the box with its remaining region, restricting samples to locations inside the GT box. This restriction tests whether the discrepancy can be explained solely by pixels outside the supervised GT support. Hard labels are sampled from the original annotations, avoiding a soft-label interpolation explanation for this analysis.

In the second confirmation set, locations with z in [0, 0.5] have average predicted foreground probabilities of 56.397% centrally and 56.296% peripherally, but foreground frequencies of 60.336% and 48.367%. Since pooled statistics may mix different objects, we additionally require central and peripheral observations from the same instance and logit interval, with at least three samples on each side. Across 1,713 eligible pairs, the foreground-frequency difference is 6.985 percentage points, whereas the predicted-probability difference is 0.153 points. The residual difference is 6.832 points, with a 95% image-cluster bootstrap interval of [5.082, 8.577].

This establishes residual position information at the resolution of the chosen bins. It is not exact conditioning on a continuous logit, and it does not remove differences in object shape within each box. Bootstrap resampling is by image rather than treating sampled pixels as independent observations.

## 4. Box-relative decision correction

### 4.1 Formulation

For candidate i, the prototype mask response is

\[
z_i(x)=c_i^\top P(x).
\]

Let T be a shared spatial field evaluated at box-relative coordinates u_i(x). We study

\[
z_i'(x)=z_i(x)+\alpha\{b_i+a_iT(u_i(x))\}.
\]

The two values b_i and a_i are predicted separately for each candidate. They are not constants shared by all test instances. In the implementation, the network produces two tanh outputs v_i, and the effective logit coefficients used here are b_i=4v_i,0 and a_i=4v_i,1. T is fixed after extraction from training data. The strength α is selected on the separate training-selection split. Within the predicted crop, the binary decision is equivalent to

\[
M_i'(x)=\mathbf1\{z_i(x)>-\alpha[b_i+a_iT(u_i(x))]\}.
\]

A scalar b_i shifts all pixels together. The spatial component permits different shifts at different relative positions. Neither the predicted box nor the category score changes. This design does not repair missing detections by construction.

### 4.2 Template and trainable readout

The training-residual template pools Y−σ(z) into a 4×4 field in each predicted ROI. We average candidate fields within an image and then average across effective fitting images, so heavily annotated images do not dominate merely through instance count. The field is interpolated into each predicted box and normalized using the mean and standard deviation over its actual discrete crop support. The source protocol records the precise sampling and normalization rules.

Our completed reference implementation predicts the two values from a 67-channel, 32×32 ROI representation containing normalized prototypes, coefficient-weighted prototypes, the original response, and relative coordinates. Two strided convolutions and a small fully connected readout produce tanh-bounded values; the final layer is zero-initialized. This implementation has 94,498 trainable parameters. Its two-dimensional output does not imply negligible computational overhead.

An implementation experiment reuses the 64-dimensional input to the native coefficient output layer and a scale-level code. It compares a scalar readout, a two-value spatial readout, and a readout additionally observing 16 pooled original-response values. These have 4,482 or 5,506 parameters. The first two use the same input and MLP, disabling the second output only in the scalar arm. They attain 44.0431 and 44.2720 AP, respectively, while the response-augmented arm attains 44.3032. The native-feature spatial head retains approximately 89% of the reference head's baseline AP gain. Its small output and parameter count still do not imply zero runtime overhead.

### 4.3 Objective and training

The base model is frozen. For each selected training candidate, its corrected ROI response is supervised by the corresponding GT mask. The objective is a mean binary cross-entropy term plus 0.5 times a soft Dice loss:

\[
\mathcal L=\mathcal L_{\mathrm{BCE}}(z',Y)+0.5\mathcal L_{\mathrm{Dice}}(\sigma(z'),Y).
\]

BCE penalizes individual foreground/background errors. Dice couples pixels through overlap and foreground mass. They need not produce identical gradients, and their combination is not an exact optimization of COCO AP. Training uses AdamW, learning rate 3×10⁻⁴, weight decay 10⁻⁴, batch size 64, gradient norm clipping at 10, and eight epochs. The final epoch is retained. Strength is selected from {0, 0.25, 0.5, 1} using only selection-set mean ROI IoU. The reference spatial head selects 0.5.

The independent scalar control uses a uniform per-instance shift. A native-readout control changes only the final 64→32 coefficient mappings at the three one-to-one scales, totaling 6,240 existing parameters. It uses the same candidate records, labels, objective, epoch budget, and separate strength-selection procedure, which selects 1.0. The spatial head and native control are not parameter- or information-matched; those differences are reported rather than hidden.

### 4.4 Connection to conditional BCE

For a fixed pair (z,u), unrestricted additive correction under population BCE has optimum

\[
\delta^*(z,u)=\operatorname{logit}\Pr(Y=1\mid z,u)-z.
\]

This follows by setting the conditional BCE derivative σ(z+δ)−Pr(Y=1|z,u) to zero. If foreground reliability depends on u after conditioning on z, a correction that depends only on z cannot generally represent all conditional optima. The proposed field is a constrained approximation; this argument proves neither its optimality nor improved IoU or AP.

The derivative of BCE with respect to z is σ(z)−Y. Consequently, the residual template summarizes the spatial pattern of a negative BCE gradient. Pooling, image balancing, interpolation, finite samples, and the additional Dice objective prevent interpreting the deployed method as an exact gradient step on final evaluation AP.

## 5. Controlled mechanism experiments

### 5.1 Spatial direction and decoder sensitivity

On the first confirmation panel with the original decoder, the complete spatial method improves fixed-instance mean IoU by 1.060 percentage points, with interval [0.868, 1.254]. Its advantage over a separately trained scalar correction is 0.387 points [0.314, 0.457]. Reversing the spatial direction worsens performance relative to retaining its learned direction.

We also test a specific decoder alternative: after scaling the binary input-resolution mask to original image size, threshold at 0.5 instead of the reference byte conversion. Changing this detail alone changes baseline mean IoU by −0.024 points, with an interval containing zero. Spatial gains persist. This alternative is a diagnostic control, not a replacement of the official evaluation baseline.

The correction has a coverage–purity tradeoff. On the first confirmation panel with the original decoder, target coverage changes from 90.334% to 88.818%, while mask purity changes from 83.738% to 86.194%. Thus the gain cannot be described as preserving every target pixel while eliminating all leakage.

### 5.2 Projection into the existing prototype span

For every fixed predicted crop, we project the spatial addition d=aT onto the span of the existing 32 prototypes using FP64 least squares. The uniform bias remains unchanged, and projection does not use GT. Tiny eigenvalues are truncated according to the recorded protocol. We then compare the complete spatial addition, its projection, and its orthogonal residual using the same predictions.

|Addition on top of the same bias|Mean IoU gain, percentage points|95% image-cluster interval|
|---|---:|---:|
|Complete spatial term|0.4949|[0.4236, 0.5723]|
|Projection into prototype span|0.4574|[0.3923, 0.5305]|
|Orthogonal residual alone|0.0293|[0.0145, 0.0452]|

These second-panel confirmation results use the stated alternative binary resizing rule. The original decoder gives the same qualitative conclusion. The median retained spatial energy is 91.0%, and the projected component retains 92.4% of the point-estimated IoU gain. The complete correction remains 0.0375 points better than its projection [0.0216, 0.0536]. IoU is nonlinear, so component gains are not an additive attribution.

This result supports the availability of useful correction directions in the current representation. It does not show that all desired masks are representable, that the uniform bias is in the prototype span, or that no additional high-frequency representation would help.

### 5.3 Why energy retention is not sufficient

Let s=z+b+d be the complete corrected logit on a fixed pixel grid, let p be the projected spatial term, and let e=d−p. The projected correction has logit s−e. Their binary decisions can differ only where |s|≤|e|. Therefore, for any τ>0,

\[
|M_s\triangle M_{s-e}|
\leq |\{x:|s(x)|\leq\tau\}|+\|e\|_2^2/\tau^2.
\]

The second term follows from counting pixels with |e|>τ. For nonempty GT G, changing one predicted pixel alters IoU by at most 1/|G|, giving

\[
|\operatorname{IoU}(M_s,G)-\operatorname{IoU}(M_{s-e},G)|
\leq |M_s\triangle M_{s-e}|/|G|.
\]

These elementary bounds clarify why a high projection-energy ratio alone is not a mask-quality guarantee: errors near the decision boundary can matter disproportionately. The statements apply on the same pixel grid; they are not direct bounds on COCO AP or on a different resizing pipeline.

### 5.4 Does supervision outside the GT box explain the effect?

We train the correction with a predefined analytic template under three support conditions: the full predicted ROI; its intersection with the GT box; and a random subset that deletes the same numbers of foreground and background pixels as the intersection control. All use the same initialization, eight-epoch budget, and fixed strength 0.5.

Full COCO mask AP is 44.3327, 44.2598, and 44.2992, respectively. The GT-intersection condition is only 0.0394 points below its count-matched random control in this single seed. This does not establish missing outside-box supervision as the dominant historical cause. The experiment changes the new head's supervision support; it does not replay the original model's assignment, area weighting, or complete pretraining objective.

![Controlled mechanism observations](../../../experiments/coco_spatial_calibration_mechanism_20260922/runs/RUN_ba588d1a88f141e9a423ff3b51c0e630/MECHANISM.png)

**Figure 1.** Mechanism and reference implementation results. Left: pooled foreground frequencies in central and peripheral locations at similar original logits; this panel is a descriptive pooled analysis, with the stricter within-instance test reported separately in Section 3. Middle: the spatial term's gain after the same bias, its projection into the prototype span, and the orthogonal residual, on the second confirmation panel using the stated alternative resize rule. Error bars are image-cluster bootstrap intervals. Right: seed-0 full COCO reference results; the truncated horizontal axis is labeled, and these values are distinct from the repeated-seed native-head comparison.

## 6. COCO validation results

All results below use the complete 5,000-image val2017 set and unchanged boxes, category scores, candidate selection, and mask decoding. Values are AP points on a 0–100 scale. The pretrained backbone is identical across runs; seeds vary the additional head initialization and/or training order, not original pretraining.

### 6.1 Three-seed controlled comparison

|Method|Mask AP, mean ± SD|AP75, mean ± SD|AP small, mean ± SD|Mean repairs / harms at Mask75|Mean net matches|
|---|---:|---:|---:|---:|---:|
|Original pretrained model|43.5185|47.0263|22.4873|0 / 0|0|
|Native-feature scalar|44.0257 ± 0.0152|47.7080 ± 0.0434|23.1529 ± 0.0172|789.0 / 434.7|354.3|
|Native-feature spatial|44.2684 ± 0.0048|48.1385 ± 0.0594|23.3651 ± 0.0191|1,134.0 / 457.0|677.0|
|Native coefficient fine-tuning|44.1848 ± 0.0060|48.0635 ± 0.0185|23.0518 ± 0.0267|1,690.3 / 1,272.0|418.3|

The spatial-over-scalar gain is 0.243 AP and repeats in all three seeds. Its advantage over native coefficient fine-tuning is much smaller, 0.084 AP. The spatial head changes fewer successful GT matches into failures, but also repairs fewer previously failed matches than native fine-tuning. Thus, its benefit is a different repair–damage tradeoff, not uniformly greater repair capacity. Strength selection independently chooses 0.5 for both new heads and 1.0 for native fine-tuning in all seeds.

The mean micro-averaged GT recall at IoU 0.75 rises from 62.659% to 64.522%. Paired image-cluster resampling gives a spatial-minus-scalar difference of 0.888 percentage points [0.751, 1.029] and a spatial-minus-native-fine-tuning difference of 0.712 [0.470, 0.955]. These intervals condition on the three trained models and average their per-GT match indicators before resampling 5,000 images 2,000 times. They are not AP confidence intervals and do not represent uncertainty over all possible training seeds.

Exploratory size groups show the largest baseline-to-spatial recall gain among small objects: 38.555% to 42.333%, a gain of 3.778 points [3.343, 4.222]. Its advantage over native fine-tuning is 1.033 points for small objects, whereas the large-object difference is −0.311 [−0.694, 0.076]. For objects with COCO mask-area-to-box-area ratio at most 0.25, recall is effectively unchanged: 25.930% to 25.890%. This is a meaningful limit for thin or low-fill shapes, not evidence of a universal boundary repair. These groups were not used to select the method, are exploratory, and have no multiplicity correction; fill ratio is not an occlusion annotation.

![Three-seed and boundary evaluation](../../../experiments/coco_native_spatial_readout_20260922/runs/RUN_5c3523cfdb1f4328ab0901483037e5d8/REPLICATION_AND_BOUNDARY.png)

**Figure 2.** Repeated-seed performance and scope. Left: mean AP gain over the fixed original model, with sample SD across three additional-training seeds. Middle: repaired and lost GT matches at Mask75, averaged over seeds; these are not an AP decomposition. Right: Boundary AP gain from unchanged seed-0 predictions under the authors’ metric. The native spatial method improves the original model but does not exceed native coefficient fine-tuning on Boundary AP.

### 6.2 Seed-0 implementation and template ablations

The following table keeps all seed-0 implementation experiments separate from the repeated-seed summary. It cannot establish small differences between unrepeated variants as statistically significant.

|Method|Mask AP|AP75|AP small|Mask75 repairs|Mask75 harms|Net matched GT change|
|---|---:|---:|---:|---:|---:|---:|
|Original pretrained model|43.5185|47.0263|22.4873|0|0|0|
|Independent scalar correction|44.0893|47.9296|23.2853|890|467|423|
|Native coefficient output fine-tuning|44.1783|48.0696|23.0270|1,706|1,303|403|
|Free 16-cell local correction|44.3119|48.3099|23.4287|1,338|617|721|
|Analytic-template spatial correction|44.3327|48.2339|23.4990|1,202|548|654|
|Training-residual spatial correction|44.3654|48.3274|23.5733|1,298|573|725|
|Native-feature scalar correction|44.0431|47.6771|23.1345|803|454|349|
|Native-feature spatial correction|44.2720|48.1513|23.3690|1,151|466|685|
|Native-feature and response spatial correction|44.3032|48.2096|23.4085|1,159|482|677|

The training-residual ROI method gains 0.847 AP over the original model, 0.276 over ROI scalar correction, and 0.187 over native output fine-tuning. The native control itself gains 0.660 AP and must be regarded as a meaningful competing explanation and baseline. In the matched native-feature MLP comparison, adding the spatial component improves AP by 0.229 points. Additional original-response features add only 0.031 points over the native-feature spatial version, providing limited current evidence for their necessity.

The native output fine-tuning control reaches 43.9780 and 44.1976 AP at strengths 0.25 and 0.5. These are post-hoc sensitivity results; the training-selected strength remains 1.0. At strength 0.5 it repairs 1,084 and harms 544 Mask75 GT matches, compared with 1,151 and 466 for the native-feature spatial head. The latter's AP advantage is only 0.074 points, so the current experiment does not establish a substantial or statistically significant superiority over tuned native adaptation. It does establish that the spatial-over-scalar effect persists after removing the ROI convolutional feature extractor.

**Stronger readout controls (seed 0).** Additional controls use the same frozen model, 800/200 training/selection images, optimization budget, and train-selected strengths. A shared scalar selects zero correction. A shared two-parameter spatial field attains 44.0120 AP, an instance-conditioned 32-coefficient residual MLP attains 44.2564, and a 2×2 region-coefficient residual MLP attains 44.2292. The last control adapts only the regional-coefficient idea; it is not a full SipMask reproduction. Coefficient controls have more parameters than the spatial head and are not parameter matched. All four cached prediction sets were scored locally on the complete 5,000-image validation set, preserving the incomplete remote scoring runs.

The instance spatial head exceeds the ordinary coefficient MLP by only 0.0155 AP and is lower by 0.0804 AP75. Thus, these controls remove a convincing claim of overall AP superiority, without establishing formal equivalence. A narrower difference remains: spatial correction repairs 1,151 and loses 466 baseline Mask75 GT matches, versus 1,109 and 572 for coefficient correction. The net difference is 148 GT matches. Exploratory size analysis attributes +157 to small objects, +1 to medium, and −10 to large. Small-object GT recall at Mask75 is 42.433% versus 41.405%, a difference of 1.029 percentage points [0.698, 1.389] from 2,000 paired image-cluster resamples. These intervals condition on fixed seed-0 predictions and are neither AP intervals nor uncertainty over training seeds. The result motivates a focused small-object investigation; it does not yet identify the original training mechanism or establish a generally superior spatial architecture.

The template does not require outputs from an older trained local head. Replacing that source with training residuals gives 44.3654 AP compared with 44.3413 for the older source. The difference is too small to support a superiority claim; it supports simplifying the method's dependency. A shuffled-template control achieves 44.1859 AP, but retains nontrivial correlation with the original field, so it cannot be described as removing all spatial structure.

The reference correction repairs 1,298 previously unmatched GT objects at Mask75 while losing 573 previously matched objects, a net gain of 725. These counts describe evaluator matches at one threshold. They are not an AP decomposition: AP also depends on rank, false positives, category averaging, and the other IoU thresholds. Likewise, fixed-instance diagnostic gains do not predict the same numeric increase in full-set AP.

### 6.3 Boundary-aware evaluation

|Method, seed 0|Boundary AP|Change from pretrained model|
|---|---:|---:|
|Original pretrained model|29.3680|0|
|ROI scalar|29.9519|+0.5838|
|ROI residual-template spatial|30.1458|+0.7778|
|Native coefficient fine-tuning|29.9911|+0.6230|
|Native-feature spatial|29.9792|+0.6112|
|Native-feature and response spatial|30.0693|+0.7013|

Boundary evaluation supports a gain over the original model, but it does not favor the retained native spatial head over native coefficient fine-tuning: their difference is −0.0118 Boundary AP. The higher-cost ROI spatial reference performs better on this metric in seed 0. We therefore do not describe the native spatial head as uniformly superior in boundary quality. The author-provided implementation is fixed at commit `37d25586a677b043ed585f10e5c42d4e80176ea9`; compatibility changes replace obsolete NumPy scalar aliases and cap the worker pool, without changing the metric.

### 6.4 Measured computation

 A paired-image timing comparison on an RTX 4090 uses 64 fixed images, batch size one, 12 warm-up calls per mode, and three repeated image orders. Median times are 11.067 ms for the original model, 11.012 ms for native output fine-tuning, 19.662 ms for the ROI spatial head, 13.399 ms for the native-feature scalar head, 16.953 ms for its spatial counterpart, and 18.439 ms with pooled response features. Timing includes forward inference, correction, assembly, crop, and resizing, but excludes loading and RLE. The operating point retains approximately 125 predictions per image at confidence 0.001 and top-300 selection. Removing the ROI feature extractor reduces median time by approximately 13.8% relative to the ROI spatial head, while the remaining overhead over baseline is approximately 53.2%. These results do not justify a negligible-cost claim or predict latency at a different confidence threshold.

## 7. Limitations and remaining validation

The study currently establishes a mechanism of the correction more convincingly than a unique cause of the original training failure. Within-object reliability differences may reflect exploitable shape priors. Prototype projection identifies available directions but not why the original coefficient head did not select them. Supervision-support ablations do not yet isolate a dominant source, and ordinary readout fine-tuning remains competitive.

The current headline method uses one base architecture and a small additional training set. Three additional-training seeds support reproducibility within this setup, but are not separate pretrained backbones or independently sampled training subsets. Completed fixed-prior and coefficient controls narrow the remaining benefit to a particular repair–damage tradeoff, especially small-object Mask75 recall. The new controls have only one training seed. Before model or dataset transfer, a focused investigation should establish whether this difference reflects a reproducible error mechanism rather than the current small training sample or threshold sensitivity. Transfer remains untested, with no assumed positive result.

Both correction implementations add measurable inference work. Although the native-feature head is cheaper than the ROI reference, further implementation improvement is needed before claiming an attractive performance–cost advantage over native fine-tuning. All timing arms retain the research evaluator's all-raw-candidate prototype multiplication; a deployment-optimized selected-candidate decoder was not benchmarked. Parameter count and the fact that only two values are predicted do not establish efficiency. The seed-0 boundary result also limits the claim: the native spatial head improves the original model but is essentially tied with native coefficient fine-tuning. Static spatial priors and richer coefficient residuals are additional competing mechanisms now being evaluated under the same data and budget.

Finally, the diagnostic representative panel excludes objects without eligible same-class boxes. Claims about upstream detection, confidence suppression, all small objects, or all crowded scenes do not follow from it. Subgroup analysis should measure where the correction helps and harms without converting post-hoc findings on val2017 into claims of independent validation.

## 8. Conclusion

For the evaluated pretrained model, mask response reliability retains structure in box-relative position, and a directed spatial correction can exploit that structure. Controlled projection and native-readout fine-tuning show that much of the beneficial change is accessible through existing features and prototypes. These findings motivate studying constrained readout and decision correction alongside representation expansion. They also impose clear limits: the mechanism is not a proof of a unique pretraining defect, the gains involve coverage tradeoffs, and broader method claims require the remaining comparison, transfer, and cost experiments.

## References

1. Bolya et al. **YOLACT: Real-time Instance Segmentation.** ICCV 2019. [Paper](https://arxiv.org/abs/1904.02689).
2. Chen et al. **BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation.** CVPR 2020. [Paper](https://arxiv.org/abs/2001.00309).
3. Cao et al. **SipMask: Spatial Information Preservation for Fast Image and Video Instance Segmentation.** ECCV 2020. [Paper](https://arxiv.org/abs/2007.14772).
4. Tian et al. **Conditional Convolutions for Instance Segmentation.** ECCV 2020. [Paper](https://arxiv.org/abs/2003.05664).
5. Li et al. **Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation.** CVPR 2021. [Paper](https://arxiv.org/abs/2104.05606).
6. Ding et al. **Local Temperature Scaling for Probability Calibration.** ICCV 2021. [Paper](https://arxiv.org/abs/2008.05105).
7. Huang et al. **Mask Scoring R-CNN.** CVPR 2019. [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html).
8. Kirillov et al. **PointRend: Image Segmentation as Rendering.** CVPR 2020. [Paper](https://arxiv.org/abs/1912.08193).
9. Ke et al. **Mask Transfiner for High-Quality Instance Segmentation.** CVPR 2022. [Paper](https://arxiv.org/abs/2111.13673).
10. Kim et al. **The Devil is in the Boundary: Exploiting Boundary Representation for Basis-based Instance Segmentation.** WACV 2021. [Paper](https://arxiv.org/abs/2011.13241).
11. Cheng et al. **Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.** CVPR 2021. [Paper](https://arxiv.org/abs/2103.16562).
12. Lin et al. **Microsoft COCO: Common Objects in Context.** Revised arXiv preprint, 2015. [Verified v3 and its author list](https://arxiv.org/abs/1405.0312v3).
13. Jocher et al. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.** arXiv, 2026. [Paper](https://arxiv.org/abs/2606.03748).
14. Ultralytics. **Software release v8.4.100.** 2026. [Versioned release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.100).

15. Wen et al. **PatchDCT: Patch Refinement for High Quality Instance Segmentation.** ICLR 2023. [Paper and acceptance record](https://arxiv.org/abs/2302.02693).
16. Brunekreef et al. **Kandinsky Conformal Prediction: Efficient Calibration of Image Segmentation Algorithms.** CVPR 2024. [Proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Brunekreef_Kandinsky_Conformal_Prediction_Efficient_Calibration_of_Image_Segmentation_Algorithms_CVPR_2024_paper.html).
17. Tang et al. **Look Closer To Segment Better: Boundary Patch Refinement for Instance Segmentation.** CVPR 2021. [Proceedings](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html).
