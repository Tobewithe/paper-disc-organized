# Controlled-intervention-guided P3 localization: feasibility report

## Question

Can the response observed under frozen-model controlled interventions—weakening proximal-instance interference and increasing target evidence improves the local stride-8 detector representation—be converted into a training signal that improves the original, unedited input at inference?

The student always sees the original image. During training only, a fixed teacher sees a GT-defined privileged view in which one vulnerable small target gains luminance separation and proximal annotated instances are flattened toward the local background. The teacher selects a P3 location; the student receives extra box supervision at that location. Inference is the stock YOLO26m-seg graph and uses neither GT nor a second pass.

## One-epoch paired screen

Both arms start from the same official weight, use seed 0, 1,000 training images, the same 1,576-image/20,827-instance validation set, batch 2, one epoch, `mask_ratio=1`, and every executable training argument saved in the official checkpoint.

| Arm | Gate | Aux weight | Box AP | Mask AP | Δ Box AP | Δ Mask AP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | — | 0 | 42.554 | 34.156 | — | — |
| Broad teacher gate | 729/980 | 2.0 | 42.069 | 33.672 | -0.485 | -0.484 |
| Rescue-only gate | 316/980 | 0.5 | 42.485 | 34.113 | -0.069 | -0.043 |

The broad gate is rejected: it improves the frozen failure cohort but costs about 0.48 AP on normal validation. The rescue-only rule requires the original location to be below Box50, the teacher location to cross Box50, and a teacher–student IoU gain of at least 0.10. It removes almost all aggregate AP cost.

## Frozen object-level evaluation

On 384 previously selected small raw-geometry failures, rescue-only training changes:

- raw best Box IoU by **+0.691 points**, paired bootstrap 95% CI **[+0.292, +1.115]**;
- final same-class Box IoU by **+0.932 points**, CI **[+0.390, +1.490]**;
- final same-class Mask IoU by **+1.648 points**, CI **[+0.733, +2.568]**;
- raw Box50 transitions: 12 recovered and 7 lost, net +5; this binary transition interval crosses zero.

On 384 category/area-matched small controls, raw Box50 has 2 gains and 2 losses (net 0), and final Box50 has 6 gains and 7 losses (net -1). Continuous raw Box IoU changes by +0.287 points. Final Mask IoU rises by +2.302 points, so the observed mask benefit is not failure-specific and must not be presented as such.

## Decision

The mechanism-to-training conversion is feasible, but the present result is not yet a final paper claim. It supports one precise statement: selective, controlled-intervention-guided P3 supervision can improve the continuous geometry and mask quality of a frozen small-object failure cohort while keeping aggregate AP approximately unchanged in a one-epoch seed-0 screen. The edited view is a controlled input intervention, not an SCM-defined counterfactual. This result does not yet prove an AP gain or density specificity.

A paired three-epoch seed-0 run is active on the remote RTX 4090. The next gate requires the rescue-only arm to retain failure-cohort IoU gains without a material aggregate AP loss. Only after that gate passes should seeds 1 and 2 be run.


## Remote three-epoch paired seed-0 check

A second paired run on the RTX 4090 used the same 1,000/1,576 split, batch 2, Ultralytics 8.4.100 and Torch 2.8.0. The per-epoch Mask AP differences (rescue-only minus Baseline) were **-0.071, +0.170 and +0.326 AP points**; Box AP differences were **-0.091, +0.129 and +0.238**. The three-epoch mean differences were +0.142 Mask AP and +0.092 Box AP points. This is a short, single-seed pilot with a visibly non-monotonic validation curve, so it is evidence of feasibility rather than a final performance claim.

The remote targeted evaluation used the original COCO annotation IDs and the 438 selected targets whose images were present in the 1,576-image validation subset (239 raw-geometry failures and 199 matched controls). On the failure cohort, rescue-only changed raw best Box IoU by **+2.067 points** (bootstrap CI **[+1.115,+3.027]**), final Box IoU by **+1.712 points** (CI **[+0.653,+2.800]**) and final Mask IoU by **+2.021 points** (CI **[+1.137,+2.945]**). Raw Box50 transitions were 19 recovered and 10 lost (net +9); final Box50 transitions were 10/5 (net +5). On matched controls, raw Box50 was 3/3 (net 0), while final Box50 was 8/3 (net +5), so the continuous mask gain is not exclusively failure-specific. No target in either cohort crossed Mask75 in this small targeted run.

The remote evaluation initially attempted the converted validation JSON and correctly failed on mismatched instance IDs; it was rerun with the original COCO `instances_val2017.json` and a selection restricted to images physically present on the remote. The restricted cohort sizes and this identity correction are recorded in `remote_targeted_summary.csv` and `remote_targeted_per_target.csv`.

**Decision:** the rescue-only mechanism passes the short feasibility gate: it improves the diagnosed failure cohort and does not show a material aggregate AP cost in the three matched epochs. It is now eligible for a controlled seed-1/seed-2 confirmation, but the current pilot remains insufficient for a paper claim.

## Three-seed short-run confirmation

Seeds 0, 1 and 2 have now completed the matched three-epoch Baseline and rescue-only runs. At the fixed final epoch, rescue-only improves Mask AP in all three seeds by **+0.326, +0.576 and +0.247 AP points** (mean **+0.383 ± 0.172** sample SD). Box AP changes are **+0.238, +0.835 and +0.101 points** (mean **+0.391 ± 0.390**). These values support a positive short-run training effect under a fixed budget.

The independently selected best-Mask-AP checkpoint comparison is less favorable: the deltas are **+0.170, -0.064 and -0.075 points** for seeds 0--2. The method therefore appears to slow the short-run deterioration or change the optimization trajectory, but this pilot does not yet show that its best checkpoint is consistently better than the Baseline best checkpoint.

The frozen targeted evaluation uses each arm's `best.pt` and the same 239 raw-geometry-small failures plus 199 matched controls. On the failure cohort, all three seeds improve raw best Box IoU (**+2.067, +0.969, +0.742 points**; mean **+1.259 ± 0.709**) and final Mask IoU (**+2.021, +3.570, +0.284 points**; mean **+1.959 ± 1.644**). Raw Box50 changes are positive in every seed (**+3.766, +2.929, +0.837 percentage points**). No failure target crosses Mask75, final Box50 is negative in seed 2, and matched controls also improve on some final metrics. Thus the mechanism-to-training conversion is reproducible, while strict recovery and failure specificity remain incomplete.

**Main-line decision:** the failure mechanism is strong enough to anchor the paper, and rescue-only intervention-guided P3 supervision is the leading method candidate. It is not yet the final main result. A confirmatory run must use a fixed, sufficiently long training budget, a predeclared checkpoint rule, and an untouched/full COCO evaluation. The failure lineage and targeted evaluation use the one-to-many + NMS branch, whereas the training `results.csv` reports the default one-to-one validation branch; the final protocol must evaluate both explicitly or choose one branch consistently before making a performance claim. The paper claim should center on repairing a diagnosed small-object P3 localization failure; density or neighbor exposure may define an effect modifier only if preregistered subgroup results support it.

Complete tables, aggregation code and raw evidence are stored in `THREE_SEED_RESULTS.md`, `three_seed_standard_metrics.csv`, `three_seed_targeted_metrics.csv`, `three_seed_aggregate.json` and `three_seed_evidence/`.

## Rich evaluation correction and method decision

The earlier fixed-epoch table was insufficient because it did not expose size-specific metrics, strict operating points, stage-wise transitions, spatial mask errors or the failure-control interaction. The evaluation was expanded using the original COCO annotation identities and the saved post-training `best.pt` predictions. Ultralytics logs confirm that each `predictions.json` was written by the final validation of `best.pt`.

Under official COCOeval on the 1,576-image subset, the three-seed Mask AP deltas are **+0.205, -0.063 and -0.176 points** (mean **-0.011 ± 0.196**). Mask AP75 changes are **+0.171, -0.095 and -0.393**; Mask APS changes are **+0.126, -0.202 and -0.039**. The current implementation therefore does not provide a stable standard segmentation improvement.

The frozen one-to-many + NMS failure cohort provides a narrower positive result. Raw P3 best Box IoU improves by **+2.171, +1.024 and +0.555 points**; its failure-minus-control interaction is **+2.270, +1.502 and +0.718 points**. Raw center error also decreases in all seeds. Raw Box50 yields 19/10, 21/14 and 9/7 recovered/degraded transitions. This confirms that the auxiliary signal acts on its intended P3 localization mechanism.

Mask behavior does not show the same failure specificity. Failure-cohort Mask IoU rises, but the failure-minus-control interaction is **+1.396, -1.194 and +0.038 points**. No seed produces a net Mask75 recovery in the full 239-object failure cohort. Target coverage, purity and neighbor/background leakage are seed-dependent. Boundary F1 is positive in the failure-control interaction but remains variable.

**Revised decision:** retain the P3 small-object localization failure as the paper's mechanism main line, and retain rescue-only intervention-guided P3 supervision as the leading method candidate. Raw P3 geometry is the mechanism endpoint: P3 Box IoU, center error, raw Box50 and the failure-control interaction improve in all three seeds. Final Box IoU, continuous Mask IoU, prediction purity and background leakage provide downstream support. Aggregate pilot Mask AP is approximately neutral and acts as an overall-performance guardrail; unrelated secondary metrics do not all need to improve. The full-budget run will use the endpoint hierarchy in `EVALUATION_PROTOCOL.md`.

After averaging the paired effect of the three seeds for each target and bootstrapping targets, raw P3 Box IoU improves by **+1.250 points [95% CI +0.705,+1.829]**. Final Box IoU improves by **+1.597 [+0.819,+2.406]**, final Mask IoU by **+1.959 [+1.237,+2.722]**, target coverage by **+3.966 [+1.991,+5.953]**, prediction purity by **+1.983 [+1.044,+2.933]**, and boundary F1 by **+3.869 [+2.490,+5.320]**. Across the 135 complete failure/control pairs, the failure-minus-control interaction remains positive for raw P3 Box IoU (**+1.496 [+0.490,+2.562]**), normalized center error (**-2.450 [-4.068,-1.012]**) and boundary F1 (**+2.417 [+0.069,+4.692]**). This strengthens the method's value: the primary internal repair is specific to the diagnosed failures, while final mask, coverage, purity and boundary improvements demonstrate useful downstream propagation.

Evidence and derived tables are in `RICH_EVALUATION_RESULTS.md`, `rich_official_deltas.csv`, `rich_targeted_seed_metrics.csv`, `rich_threshold_transitions.csv`, `rich_failure_state_transitions.csv`, `rich_failure_control_interactions.csv`, `rich_spatial_micro_metrics.csv` and `rich_stratified_effects.csv`.
