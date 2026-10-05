# From Localization Failures to Selective Supervision: Intervention-Guided P3 Learning for Small Objects

**Working manuscript · 15 September 2026 · Authors and venue to be supplied**

This draft reports completed diagnostics and a three-seed training pilot. The paired full-data experiment is not yet complete at this evidence cutoff. Its results are not included in the claims below. Numerical tables and figures are generated from the recorded experiment outputs; their provenance is documented in [EVIDENCE_MAP.md](EVIDENCE_MAP.md).

## Abstract

An instance segmenter can miss a small object before mask decoding begins: none of its raw boxes may provide sufficient geometric support. We investigate this failure in a COCO-pretrained YOLO26m-seg model by tracing candidate availability, perturbing object context, and transplanting intermediate features. On the full COCO validation set, 1,459 of 3,699 ground-truth instances unmatched by the examined one-to-many prediction pipeline lack any raw box with intersection-over-union (IoU) of at least 0.5; 1,429 of these are small objects. In a matched diagnostic sample, weakening an exposed nearby object improves raw P3 box IoU by 13.864 points relative to an identical-shape background edit. A separate feature-transplant experiment reproduces most of the recovery from a joint context-and-contrast intervention by replacing only the local stride-8 head input. Motivated by these observations, we introduce intervention-guided P3 supervision: a frozen teacher processes a training-only edited view, nominates a promising P3 location, and activates additional ground-truth supervision when the original-view student localizes poorly at that location. The inference graph is unchanged. In a three-seed, 1,000-image pilot, the diagnosed failure cohort improves by 1.250 raw P3 IoU points, with a 95% instance-bootstrap interval of [0.705, 1.829], and by 1.959 points in the IoU of its selected final masks. Official COCO mask AP on the 1,576-image pilot evaluation subset is approximately unchanged (−0.011 points on average). These results support targeted localization repair and motivate a larger evaluation; they do not establish an overall AP improvement or a universal explanation of segmentation failures.

## 1. Introduction

Instance segmentation failures are usually observed at the output: a small object is missing, a mask is incomplete, or a prediction includes a neighboring object. These appearances do not uniquely identify the responsible computation. A poor mask may arise because the model never produced a usable box, because a good candidate was discarded, or because mask decoding failed despite adequate localization. A method designed for the wrong stage can improve an auxiliary property while leaving the original failure unchanged.

This distinction matters for real-time prototype-based segmenters. Their final masks depend on both an instance-specific readout and spatial support supplied by detection. Altering mask coefficients cannot directly recover an object for which the detection branch provides no usable candidate. Conversely, changing suppression cannot create geometry that was absent before suppression. We therefore begin with an observable failure and ask where a necessary condition for successful prediction first becomes unavailable.

Our investigation identifies a concrete object of study: small COCO instances for which the one-to-many branch of a strong pretrained segmenter produces no raw box reaching IoU 0.5. An audit of all 36,335 ordinary validation instances finds 1,459 such cases. Of these, 97.94% are small under the COCO area definition. This is a subset of the model's errors, rather than a proposed explanation of all difficult or crowded scenes.

We next ask what makes these instances difficult. A same-category, nearest-area matched comparison associates failure with weak local localization evidence and low target-to-background contrast. Controlled edits go further: weakening a nearby annotated object can recover geometry while leaving the target's original-resolution pixels unchanged. Identical-shape background edits and equal-pixel distant-object edits yield much smaller improvements. Context is not uniformly harmful, however: removing neighboring evidence also damages some initially successful controls. These observations motivate selective use of edited views rather than an unconditional rule to suppress neighbors.

To connect this input-level response to the network, we transplant head-input features from an edited image into the original-image forward pass. Local P3 replacement reproduces most of the improvement produced by a joint neighbor-weakening and contrast-enhancing edit, while P4 and P5 replacement produces smaller responses. This localizes a useful route for supervision at the stride-8 head input. It does not establish that the original defect was created there, because the transplanted representation already includes upstream processing.

We turn this diagnosis into **intervention-guided P3 supervision (IG-P3)**. During training, a frozen teacher receives an edited view constructed with training annotations. It nominates a local P3 location using box quality and class evidence. Additional supervision is activated only when the teacher's box is adequate and the student's box at the same position remains poor. The student sees the original training image, and the box target remains the ground-truth box. The teacher supplies a choice of supervision location and activation condition, not a replacement annotation. At inference, the teacher, editing operation, and gate are removed.

The contributions are threefold:

1. A candidate-availability audit identifies a small-object localization failure that precedes class filtering, suppression, and mask construction in the examined branch.
2. Controlled context edits and scale-specific feature transplants characterize an operationally neighbor-sensitive subset and identify P3 as an effective interface for transferring localization recovery.
3. A training-only auxiliary objective translates this evidence into selective location supervision. A three-seed pilot shows improved raw localization and downstream mask quality on the diagnosed cohort, while explicitly separating those effects from approximately neutral aggregate AP.

The paper's central proposition is that a failure-specific diagnostic can guide where and when to add supervision. It does not require density to be the defining variable, nor does it assume that every nearby object is detrimental.

## 2. Related Work

**Error analysis for detection and segmentation.** COCO evaluates instance-level localization and recognition across overlap thresholds and object sizes [1]. TIDE decomposes detection and instance-segmentation errors and estimates their effects on AP through output-level corrections [4]. Our analysis is complementary: we retain intermediate candidate sets to locate the first stage at which a usable box becomes unavailable. A count of unavailable candidates is not an AP decomposition. We use the counts to select a diagnostic cohort and retain official COCO evaluation for aggregate performance.

**Context and object-level perturbations.** Shetty et al. show how object removal can reveal and reduce context dependence in classification and semantic segmentation [5]. MetaOD inserts objects to expose failures of object detectors under metamorphic tests [9]. These studies establish that manipulating scene content is a useful diagnostic and training tool. Our contribution is not the use of object edits itself. We connect a measured raw-localization failure to matched context controls, a particular head-input scale, and a supervised training objective at selected candidate positions. We call the edits controlled interventions; we do not posit a structural causal model or claim that an edited image is the unique natural counterfactual of a scene.

**Crowded detection and spatial separation.** Repulsion Loss addresses crowd-related localization errors through target attraction and repulsion from other objects [6]. Our cohort is selected by failed raw geometry, not by a crowding threshold. Both same-category and different-category neighbors can produce recoverable responses in our sample, and some successful objects benefit from their context. Thus we use observed teacher–student localization conditions to select auxiliary supervision rather than applying a universal separation interpretation.

**Multiscale prediction and instance masks.** Feature pyramids provide scale-dependent representations for detection [2]. YOLACT demonstrates efficient instance-mask construction using shared prototypes and instance-specific coefficients [3]. These architectures make localization and mask decoding distinct points of investigation. Our P3 transplant evidence concerns box prediction at the input to the segmentation head; it should not be interpreted as evidence of coefficient orthogonality, prototype insufficiency, or an identified cross-scale fusion defect.

**Teacher-guided supervision.** Label Assignment Distillation transfers a teacher's assignment information to a student [7], while Localization Distillation transfers localization information in dense detectors [8]. IG-P3 shares the idea that the teacher can guide localization learning. Its specific design uses a training-only edited view to nominate P3 positions, checks a teacher–student quality condition at each nominated position, and applies a ground-truth box objective. The main distinction is the diagnostic motivation and the intervention-conditioned selection rule. Establishing the incremental value of each component requires comparisons with an unedited teacher and equal-budget hard-example supervision; these component-isolating comparisons are not yet completed.

## 3. Diagnosing Recoverable Localization Failure

### 3.1 Model, candidate sets, and endpoints

We use the official COCO-pretrained YOLO26m-seg checkpoint with Ultralytics 8.4.100. The diagnostics explicitly inspect the one-to-many branch with 640 × 640 letterboxed input, confidence threshold 0.001, class-aware NMS at IoU 0.7, and at most 300 retained predictions per image. This is a chosen diagnostic branch, not a claim about every default prediction path. The checkpoint also supports one-to-one prediction, which is used by the pilot's standard post-training evaluation. We report the branches separately.

For image \(x\), instance box \(b_g\), and raw boxes \(\{b_k(x)\}\), define

\[
q_{\mathrm{all}}(x,g)=\max_k\operatorname{IoU}(b_k(x),b_g),\qquad
q_3(x,g)=\max_{k\in\mathcal P_3}\operatorname{IoU}(b_k(x),b_g).
\]

These are **class-agnostic geometric availability** endpoints. They use ground truth for diagnosis and do not describe a test-time selection policy. At 640 × 640, P3, P4, and P5 contribute 6,400, 1,600, and 400 locations, respectively. A geometry failure satisfies \(q_{\mathrm{all}}<0.5\). A raw Box50 recovery means crossing this threshold after an intervention; it does not establish correct classification, final retention, or successful mask prediction.

### 3.2 Where does a candidate first become unavailable?

For each ordinary, non-crowd COCO validation instance, we examine successive sets: all raw boxes; boxes with the correct top-ranked class; those passing confidence 0.001; NMS survivors; the top 300 predictions; nonempty masks; the per-category evaluation limit of 100; and final COCO bbox matching at IoU 0.5. The first set without a qualifying box assigns the instance's stage label. Final matching competition is recorded separately.

Of 36,335 GT instances, 32,636 receive a final match and 3,699 do not. Table 1 partitions the latter. Availability at intermediate stages is assessed independently for each GT; a single prediction may be geometrically available to multiple GTs. Therefore the table is a first-unavailability taxonomy, not a mutually assignable recall ceiling or a causal attribution of AP loss.

<!-- generated:lineage:start -->
| First unavailable stage | GT count | % of unmatched |
| --- | --- | --- |
| Raw geometry | 1459 | 39.44% |
| Correct class | 1033 | 27.93% |
| Confidence 0.001 | 745 | 20.14% |
| NMS | 184 | 4.97% |
| Top 300 | 93 | 2.51% |
| Evaluation limit 100/category | 135 | 3.65% |
| Final matching competition | 50 | 1.35% |
<!-- generated:lineage:end -->

**Table 1.** First unavailable stage among 3,699 unmatched GTs in the specified one-to-many pipeline on all 5,000 COCO validation images. The nonempty-mask stage contains zero first losses. The first three stages account for 87.51% of this unmatched set. Of the 1,459 raw-geometry failures, 1,429 are COCO-small instances.

### 3.3 Matched failure and success cohorts

We construct 384 small-object failure/control pairs. Failure order is shuffled with seed 0; each control is selected from successfully matched small objects by exact category and nearest log area, with images kept distinct throughout selection. All 768 targets are in different images. This controls category and approximately controls area, but it does not balance every scene variable or provide a representative sample of all COCO errors.

At the highest true-class-score P3 position within a half-stride-expanded GT box, mean box IoU is 0.2105 for failures and 0.5760 for controls; normalized center error is 0.4602 versus 0.1719, and true-class score is 0.1267 versus 0.2367. These local score-selected endpoints differ from the oracle maxima in Section 3.1. Failures also have lower target/ring color contrast in the recorded comparison. The observations motivate tests of local evidence and context, but do not alone identify a cause.

### 3.4 Controlled input edits

An initial experiment modifies target contrast and nearby annotated content. Increasing target lightness contrast produces a mean \(q_{\mathrm{all}}\) gain of 7.002 IoU points and recovers 102/384 failures at Box50. Weakening proximal neighbors with inpainting produces a gain of 3.690 points and 82 recoveries. Their joint edit gains 10.219 points and recovers 146/384. Contrast reduction also has a smaller positive mean effect of 1.051 points, so the evidence does not support a universal monotonic contrast rule. The joint edit is useful as a diagnostic donor, not as a uniquely isolated neighbor manipulation.

We subsequently isolate a single exposed neighbor and introduce spatial controls. For target mask \(S_g\), define an influence region by elliptical dilation with radius \(\max(5,2r)\), where \(r=\max(3,\operatorname{round}(0.2\sqrt{|S_g|}))\). Among other ordinary instance masks, select the neighbor contributing the most pixels inside this region but outside \(S_g\), breaking ties by annotation ID. Only that exposed portion is edited. This selection does not use model outcomes. A target is eligible if at least eight exposed pixels can be edited; 302 failures meet the criterion.

We compare the following operations:

- **Neighbor flattening:** fill the exposed neighbor pixels with a background-derived median color.
- **Background flattening:** translate exactly the same binary edit shape to a location outside the union of ordinary instance masks and apply the same color.
- **Distant-object flattening:** edit the same number of pixels in a distant annotated object, using the same color.
- **Blur controls:** replace the selected neighbor or translated background pixels by values from a Gaussian-blurred image, avoiding a constant-fill-specific interpretation.

“Background” here means outside the non-crowd annotation union; it is not a guarantee that every edited pixel is semantically empty. The translated control matches shape and area but not distance to the target. All edits preserve target pixels before resizing; image interpolation and later receptive-field mixing can still affect target-adjacent model inputs. These controls address arbitrary image simplification and selected edit artifacts, while leaving proximity and boundary-texture mechanisms open.

<!-- generated:intervention:start -->
| Comparison | n | Δ P3 IoU [95% CI] |
| --- | --- | --- |
| Neighbor fill − matched-shape background fill | 302 | +13.864 [+11.674, +16.141] |
| Neighbor blur − matched-shape background blur | 302 | +10.438 [+8.460, +12.465] |
| Neighbor fill − distant-object fill | 299 | +13.493 [+11.300, +15.736] |
<!-- generated:intervention:end -->

**Table 2.** Within-image controlled differences in raw P3 box IoU, multiplied by 100. Intervals are percentile bootstrap intervals over eligible images. The distant-object comparison has 299 eligible failures. The effects hold under both fill and blur operations.

Neighbor flattening yields 126 upward and zero downward P3 Box50 transitions among the 302 failures, compared with seven upward and zero downward transitions for the background control. An operationally stricter definition additionally requires an absolute neighbor-edit gain of at least 0.1, continued failure under the background edit, and a neighbor-over-background margin of at least 0.05. It identifies 105/302 eligible failures (34.77%; 105/384 of the complete sampled failure cohort). This is an edit-dependent recovery count, not the prevalence of a causally unique failure type in COCO.

The neighbor-minus-background P3 IoU contrast is larger in failures than in their matched successes: the interaction over 214 complete eligible pairs is +14.604 points, with 95% CI [11.591, 17.662]. Exploratory splits show positive effects with same-category and different-category neighbors. Successful controls also reveal useful context: their net P3 Box50 rate drops by 7.605 points under neighbor rather than background flattening. Thus removing nearby content can either help or harm, depending on the target.

### 3.5 Which network representation carries the recovery?

Let \((F_3,F_4,F_5)\) be the original-image tensors entering the segmentation head, and \((F'_3,F'_4,F'_5)\) the tensors from the initial joint neighbor-and-contrast edit. We replay the same one-to-many head while replacing one tensor, or a local part of it, by its edited-view counterpart. For local replacement, a dilated target-region support is projected to the feature map by adaptive max pooling. The unchanged replay matches the original output within numerical tolerance (maximum difference below \(10^{-6}\)).

<!-- generated:transplant:start -->
| Donor / replacement | Δ raw IoU [95% CI] | Recovered / 384 |
| --- | --- | --- |
| Full joint edited view | +10.219 [+8.615, +11.880] | 146 |
| P3, full tensor | +10.087 [+8.514, +11.763] | 138 |
| P3, local support | +10.049 [+8.502, +11.696] | 135 |
| P4, full tensor | +1.969 [+1.331, +2.642] | 39 |
| P5, full tensor | +0.445 [+0.133, +0.821] | 7 |
<!-- generated:transplant:end -->

**Table 3.** Changes in all-level raw geometric availability on 384 failures. Donor features come from the joint edit, whereas Table 2 concerns isolated single-neighbor edits. The two protocols should not be treated as one factorial experiment. “Recovered” counts crossings of all-level raw Box50.

Local P3 replacement preserves most of the donor-view improvement, making it a useful supervision interface. This is consistent with small objects being decoded at high spatial resolution. It does not show that the initiating disturbance arises inside P3, that cross-scale fusion is defective, or that P3 is the only valid intervention site. It also does not locate a mask-prototype or coefficient failure.

![Candidate audit, controlled edits, and head-input transplants](figures/diagnostic_evidence.png)

**Figure 1.** Three complementary views of the diagnosis. Counts refer to the unmatched full-validation cohort; edit and transplant panels use their stated diagnostic samples. Error bars show recorded 95% bootstrap intervals. The edit panel uses P3 IoU, while the transplant panel uses all-level raw IoU.

## 4. Intervention-Guided P3 Supervision

### 4.1 Training-only privileged views

The student receives the ordinary augmented training image \(x\). We construct a teacher image \(x'=T(x,G)\) from the same augmented image and its training annotations \(G\). No validation/test annotation enters training gradients or inference.

We select at most one target per image. Eligible targets have normalized bounding-box area at most 0.01, at least four mask pixels, and a nonempty local background ring. This training rule is not identical to the original-resolution COCO-small definition. A square dilation radius is

\[
r_g=\operatorname{clip}\left(\operatorname{round}(0.2\sqrt{|S_g|}),4,16\right).
\]

Within the dilated region, let \(N_g\) contain other labeled instance pixels and \(R_g\) contain background-labeled pixels. With mean RGB vectors \(\mu_g\) and \(\mu_R\), we rank eligible objects by

\[
v_g=\frac{|N_g|}{|S_g|}+\frac{1}{\operatorname{mean}_c|\mu_g-\mu_R|+0.05}.
\]

For the highest-ranked target, the teacher view replaces \(N_g\) by \(\mu_R\) and shifts all target RGB channels by +0.11 or −0.11, clipped to [0,1]. The sign increases mean target/ring contrast; near equal means use the target's position relative to 0.5 to choose the sign. The training view therefore combines neighbor weakening with target contrast enhancement. It can select a low-contrast target without an exposed neighbor and is broader than the isolated-neighbor diagnostic.

### 4.2 Teacher nomination and location-level gating

A teacher initialized from the student's starting weights is frozen during each uninterrupted training segment, including frozen batch-normalization statistics. Its edited-view predictions nominate a P3 location. Let \(\mathcal K_g\) be P3 anchor centers within the GT box expanded by one P3 stride (eight input pixels). For true category \(y_g\), define

\[
k_g^*=\arg\max_{k\in\mathcal K_g}\left[
\operatorname{IoU}(b^T_k(x'),b_g)+0.02\,\sigma(s^T_{k,y_g}(x'))\right].
\]

The class term is a bounded secondary contribution; it is not an exact tie-breaker. At this same position, compute detached teacher and student qualities

\[
q_g^T=\operatorname{IoU}(b^T_{k_g^*}(x'),b_g),\quad
q_g^S=\operatorname{IoU}(b^S_{k_g^*}(x),b_g).
\]

The activation condition is

\[
a_g=\mathbb 1[q_g^S<0.5]\,
\mathbb 1[q_g^T\geq0.5]\,
\mathbb 1[q_g^T-q_g^S\geq0.1].
\]

This is a **location-level** condition: the student may have another good box elsewhere. Nor does it compare the frozen teacher on original and edited images, so its quality difference combines view differences with teacher–student differences. The current method uses a diagnostically motivated selection heuristic; it does not estimate an isolated causal effect during training.

### 4.3 Additional ground-truth supervision

For activated targets, normalize coordinate errors by \(d_g=(\max(w_g,8),\max(h_g,8),\max(w_g,8),\max(h_g,8))\). The auxiliary objective is

\[
\ell_g=\frac14\sum_{j=1}^{4}\operatorname{SmoothL1}_{\beta=0.1}
\left(\frac{b^S_{k_g^*,j}-b_{g,j}}{d_{g,j}}\right)
+0.05\operatorname{BCEWithLogits}(s^S_{k_g^*,y_g},1),
\]

\[
\mathcal L=\mathcal L_{\mathrm{official}}+
\lambda B\,\frac{\sum_g a_g\ell_g}{\max(1,\sum_g a_g)},\qquad\lambda=0.5,
\]

where \(B\) is batch size, matching the implementation's batch scaling. If no gate activates, the auxiliary term is zero. The official training objective and assignment remain active. The added term acts on one-to-many box coordinates and the true-class logit; it has no direct one-to-one or mask-specific term. Its gradients can update the student's shared features.

The box supervision target is \(b_g\), not the teacher's box. Consequently, IG-P3 is most precisely described as edited-view teacher-guided selective supervision. We retain the repository's historical “distillation” names only for artifact traceability. At inference, the student uses its original architecture and ordinary prediction path. We have not yet measured training overhead or inference timing across devices.

![Training-only intervention-guided selective supervision](figures/method_overview.png)

**Figure 2.** The student sees the ordinary training view. Training GT constructs the teacher view, defines location-quality checks, and supplies the auxiliary regression target. Only the nominated student location receives the extra loss. The teacher and editing branch are absent at inference.

## 5. Experimental Evaluation

### 5.1 Data, training budget, and statistical units

We distinguish four populations: full-validation candidate tracing (5,000 images, 36,335 ordinary GTs); the matched diagnostic sample (384 failures and 384 successes); the training pilot (1,000 train2017 images); and its aggregate evaluation subset (1,576 val2017 images, 20,827 ordinary GTs). The latter is a previously constructed crowded-image subset, not the full COCO validation set and not an instance-level density-controlled sample. It is used consistently across pilot arms.

The pilot uses three epochs and seeds 0, 1, and 2, with matched baseline/method settings and batch size two. Executable training hyperparameters are inherited from the official checkpoint: the run uses 640-pixel input, mask ratio 1, overlapping mask labels, MuSGD, initial learning rate 0.00038, and the same augmentation configuration in both arms. This is short fine-tuning of pretrained weights, not training from scratch or a convergence study. One seed required restart; the current implementation reconstructs the teacher during setup, which can refresh its weights from the resumed student. We therefore treat the three runs as a pilot and retain this detail in the reproducibility record.

For target-level learned-model evaluation, 438 of the 768 diagnostic images were available in the evaluation environment: 239 failures and 199 controls, with 135 complete matched pairs. This availability subset is not the isolated-neighbor subset of Table 2. We freeze its identities across models and seeds. The diagnostic cohort was also used during development, so these are training-image-disjoint observations rather than an independent, untouched confirmation set.

Intervention differences use paired image-level bootstrap resampling; feature-transplant intervals use 3,000 resamples. The single-neighbor contrasts use 5,000 resamples and their matched interactions use 10,000. For the learned-model cohort, we first average the paired model difference for each instance across three seeds, then bootstrap instances or complete failure/control pairs 10,000 times. These intervals quantify sampling uncertainty conditional on the trained models. They are not confidence intervals over all possible training seeds. Secondary metrics are exploratory and are not corrected for multiple comparisons.

### 5.2 Mechanism and downstream endpoints

The mechanism endpoint is raw P3 geometric availability \(q_3\). We additionally report all-level raw IoU, the normalized center error of the best all-level raw box, and raw Box50 transitions. Center error is Euclidean center distance divided by GT-box diagonal; “points” for this metric mean 100 times that normalized ratio.

For downstream diagnostics, select the highest-box-IoU final prediction among predictions of the GT category. Measure its box IoU and its associated mask. If no same-category prediction exists, box IoU and mask overlap are zero. This GT-assisted association is independently performed per target, so a prediction can be associated with more than one GT. It measures available final-candidate quality and is not official one-to-one recall or AP. Masks use the original predicted boxes with native construction; no box expansion or removal is applied.

For GT mask \(S\) and associated prediction \(M\), coverage is \(|M\cap S|/|S|\), purity is \(|M\cap S|/|M|\), and mask IoU is \(|M\cap S|/|M\cup S|\). Boundary F1 uses eroded-mask contours and an image-diagonal tolerance \(\max(1,\operatorname{round}(0.0075\sqrt{H^2+W^2}))\); it is a tolerance-dependent diagnostic, not a COCO metric. Same-category-neighbor leakage counts predicted pixels inside another same-category mask but outside the target, normalized by predicted area.

<!-- generated:pilot:start -->
| Metric | Δ [95% CI] |
| --- | --- |
| Raw P3 box IoU ↑ | +1.250 [+0.705, +1.829] |
| Raw all-level box IoU ↑ | +1.259 [+0.711, +1.826] |
| Raw all-level center error ↓ | -1.709 [-2.877, -0.623] |
| Raw all-level Box50 rate ↑ | +2.510 [+0.000, +5.021] |
| Selected final box IoU ↑ | +1.597 [+0.819, +2.406] |
| Associated final mask IoU ↑ | +1.959 [+1.237, +2.722] |
| Target coverage ↑ | +3.966 [+1.991, +5.953] |
| Prediction purity ↑ | +1.983 [+1.044, +2.933] |
| Boundary F1 ↑ | +3.869 [+2.490, +5.320] |
| Same-category neighbor leakage ↓ | -0.379 [-1.776, +1.046] |
| Background leakage ↓ | -0.558 [-2.083, +0.925] |
| Associated Mask75 rate ↑ | +0.000 [+0.000, +0.000] |
<!-- generated:pilot:end -->

**Table 4.** Learned-method minus matched-baseline effects, multiplied by 100, on the frozen 239-failure cohort. “Final” denotes GT-assisted association within retained predictions. Intervals average each instance over the three seeds before bootstrap. The normalized center-error endpoint uses the best raw box across all levels, not necessarily P3.

Raw P3 IoU improves in all three seeds (+2.171, +1.024, and +0.555 points). The failure-minus-control interaction over 135 complete pairs is +1.496 points [0.490, 2.562], supporting preferential improvement of the diagnosed representation. The corresponding all-level raw center-error interaction is −2.450 points [−4.068, −1.012]. These contrasts are differences of paired improvements; they are not obtained by subtracting two differently sized cohort means.

Selected final-box IoU, mask IoU, coverage, purity, and boundary F1 also improve on average. This shows that the repair can propagate to retained-prediction quality under ordinary cropping. However, no selected failure mask crosses IoU 0.75 in the pilot, the raw Box50 interval touches zero, and neighbor/background leakage reductions do not have intervals excluding zero. Final mask-IoU improvement is also present in controls: its matched interaction is +0.080 points [−1.479, 1.626]. The strongest evidence for failure-specific improvement therefore lies in raw localization, not a uniquely selective downstream mask effect.

### 5.3 Standard evaluation and checkpoint sensitivity

We evaluate saved post-training predictions with official COCOeval against original COCO annotations on the 1,576-image subset. These predictions use the model's one-to-one path and the saved `best.pt` checkpoints selected by the trainer. They should not be pooled with the one-to-many target diagnostics. Table 5 reports means across the three paired seeds and the sample standard deviation of their differences.

<!-- generated:official:start -->
| Metric | Baseline mean | IG-P3 mean | Paired Δ ± seed SD |
| --- | --- | --- | --- |
| Box AP | 44.116 | 44.172 | +0.055 ± 0.104 |
| Mask AP | 36.482 | 36.470 | -0.011 ± 0.196 |
| Mask AP50 | 59.128 | 59.176 | +0.048 ± 0.228 |
| Mask AP75 | 38.442 | 38.337 | -0.106 ± 0.282 |
| Mask AP small | 23.495 | 23.457 | -0.038 ± 0.164 |
| Box AR100 | 64.624 | 65.033 | +0.409 ± 0.479 |
| Mask AR100 | 52.942 | 53.132 | +0.190 ± 0.246 |
<!-- generated:official:end -->

**Table 5.** Official COCOeval on the pilot evaluation subset, using saved post-training best-checkpoint predictions. AP and AR are on the 0–100 scale. Δ SD is descriptive across three seeds, not a significance interval. No aggregate AP improvement is established.

The official mask AP differences are +0.205, −0.063, and −0.176 points. Their mean is −0.011, whereas mask AR100 rises by 0.190 points on average. These outcomes support a narrow conclusion: the pilot improves selected localization and mask-quality endpoints while aggregate mask AP remains approximately neutral. Approximate neutrality is not a formal non-inferiority result.

The checkpoint rule changes the apparent summary. Native training-validation metrics at the fixed final epoch report mean Box AP and Mask AP gains of +0.391 and +0.383 points, respectively. Those use the native validator and fixed epoch three; Table 5 uses official original-annotation evaluation and post-training best checkpoints. We retain both observations without selecting the more favorable one as the headline AP claim. A confirmatory experiment must declare its checkpoint rule and evaluate both branches consistently.

### 5.4 Preliminary recipe comparison

In the initial one-epoch pilot, broad teacher-guided supervision with auxiliary weight 2.0 reduced native Box/Mask AP by 0.485/0.484 points. The selective recipe with weight 0.5 reduced them by 0.069/0.043 points relative to the same baseline. This comparison motivated the current recipe, but simultaneously changed selection and weight. It is not an isolated estimate of the gate's contribution. We reserve claims about the gate itself for an equal-weight comparison.

![Three-seed targeted effects and aggregate evaluation](figures/pilot_effects.png)

**Figure 3.** Target-level effects conditional on the three pilot models, with instance-bootstrap intervals. Positive bars improve IoU, coverage, purity, or boundary F1. The separate official-AP table measures score-ranked, one-to-one task performance; it is not inferred from these diagnostic gains.

## 6. Discussion and Limitations

**What has been identified?** The study identifies a reproducible response pattern: a subset of small-object raw-localization failures improves under local context weakening, and recovery from a combined edit can be carried through P3 head-input features. This provides a practical locus for extra supervision. The experiments do not isolate the original upstream computation responsible for the failure, and P3's role may partly reflect the expected scale specialization of the architecture.

**What remains of the context explanation?** Shape-matched background and distant-instance controls make arbitrary global simplification an insufficient explanation for the measured recovery. However, their spatial relation to the target is different. A distance-matched local background control, boundary-only edits, and appearance-preserving neighbor manipulations would distinguish neighbor identity, local texture, and edge effects more sharply. The successful-control regressions also show that context should not simply be discarded.

**What does the current training result establish?** The student improves on the original view without access to evaluation GT at inference. This demonstrates a usable transfer from an annotation-assisted training procedure to predicted localization. It does not establish that edited teacher views outperform ordinary teacher assignment, that the gate is essential, or that the method fixes precisely the 105 isolated-neighbor recoveries. Equal-budget original-view teacher, ungated same-weight, and GT hard-example controls are the most direct next comparisons.

**How broad is the performance claim?** The pilot uses three short runs on 1,000 training images, an availability-limited diagnostic subset, and a repeatedly inspected validation cohort. The full-data seed-0 baseline/method screen was still incomplete at the writing cutoff; its one-epoch budget will itself be a screen rather than a convergence experiment. Independent-cohort evaluation, a declared adequate fine-tuning budget, stronger seed replication without teacher-reset ambiguity, and architecture transfer are needed before claiming a general benchmark improvement. Timing and memory costs of the teacher should also be measured.

These limitations do not negate the measured repair. They determine its current scope: controlled failure analysis with a promising training translation, rather than a completed state-of-the-art segmentation benchmark claim.

## 7. Conclusion

Starting from missing raw geometry exposes a failure that cannot be explained solely by suppression or mask decoding. On the studied COCO segmenter, this failure concentrates in small objects, exhibits a recoverable dependence on nearby visual content, and admits a useful intervention at the P3 head input. Intervention-guided P3 supervision converts that diagnosis into extra ground-truth learning at selected locations while preserving the ordinary inference graph. The completed pilot improves the diagnosed localization endpoint and the quality of associated final masks; aggregate mask AP remains approximately unchanged. The evidence supports further evaluation of selective localization repair and a general research strategy of choosing supervision from the structure of observed failures.

## References

1. Tsung-Yi Lin et al. **Microsoft COCO: Common Objects in Context.** ECCV, 2014. [Primary manuscript](https://arxiv.org/abs/1405.0312).
2. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. **Feature Pyramid Networks for Object Detection.** CVPR, 2017. [Primary manuscript](https://arxiv.org/abs/1612.03144).
3. Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. **YOLACT: Real-time Instance Segmentation.** ICCV, 2019. [Primary manuscript](https://arxiv.org/abs/1904.02689).
4. Daniel Bolya, Sean Foley, James Hays, and Judy Hoffman. **TIDE: A General Toolbox for Identifying Object Detection Errors.** ECCV, 2020. [Primary manuscript](https://arxiv.org/abs/2008.08115).
5. Rakshith Shetty, Bernt Schiele, and Mario Fritz. **Not Using the Car to See the Sidewalk: Quantifying and Controlling the Effects of Context in Classification and Segmentation.** CVPR, 2019. [Primary manuscript](https://arxiv.org/abs/1812.06707).
6. Xinlong Wang, Tete Xiao, Yuning Jiang, Shuai Shao, Jian Sun, and Chunhua Shen. **Repulsion Loss: Detecting Pedestrians in a Crowd.** CVPR, 2018. [Primary manuscript](https://arxiv.org/abs/1711.07752).
7. Chuong H. Nguyen, Thuy C. Nguyen, Tuan N. Tang, and Nam L. H. Phan. **Improving Object Detection by Label Assignment Distillation.** WACV, 2022. [Primary manuscript](https://arxiv.org/abs/2108.10520).
8. Zhaohui Zheng et al. **Localization Distillation for Dense Object Detection.** CVPR, 2022. [Primary manuscript](https://arxiv.org/abs/2102.12252).
9. Shuai Wang and Zhendong Su. **Metamorphic Object Insertion for Testing Object Detection Systems.** ASE, 2020, pp. 1053–1065. [Publisher DOI](https://doi.org/10.1145/3324884.3416584).

## Appendix A. Artifact and Protocol Notes

The corresponding source files, cohorts, checkpoint conventions, bootstrap units, and known record inconsistencies are listed in [EVIDENCE_MAP.md](EVIDENCE_MAP.md). [references.bib](references.bib) contains the citation records. [build_paper_assets.py](build_paper_assets.py) regenerates all five numerical tables and three figures from existing recorded results without rerunning model inference or training. The historical repository names containing `counterfactual` are provenance identifiers; this paper does not make a formal counterfactual-causal claim.
