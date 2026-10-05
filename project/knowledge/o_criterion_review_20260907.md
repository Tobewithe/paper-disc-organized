# Review of the Current O Criterion

Date: 2026-09-07. User requested a review, not a change to evaluation labels. Verdict: internally coherent as a strong one-to-many component label; insufficient as a comprehensive fragmentation criterion. Baseline labels and method Gates remain unchanged.

## Findings, Ordered by Importance

### 1. The 50-percent edge gate excludes unequal and small-fragment partitions

The active full-scope classifier is `tools/run_yolo26_external_full.py:222`, following `C:/Dpan/codexproject/pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py:290`.

For GT mask G and prediction mask P:

- coverage = |G intersect P| / |G|;
- purity = |G intersect P| / |P|;
- core edge = IoU >= .50 OR coverage >= .50;
- O = a connected component with exactly one GT and at least two predictions.

Since |G union P| >= |G|, IoU <= coverage. The core rule therefore reduces mathematically to coverage >= .50. The audit found zero edges contributed exclusively by the IoU disjunct across the full 27,745-GT scope.

If predictions are disjoint parts wholly inside one GT and exactly partition it, the current rule identifies a two-part split only at exactly 50/50. An unequal two-part split has only one core edge. Three or more nonempty disjoint parts cannot have two parts each covering at least half of the GT. This is a structural blind spot, not merely an empirical threshold preference.

The actual production classifier was exercised on explicit binary masks:

| Synthetic Configuration | Current GT Class | Implication |
|---|---|---|
| Two identical complete masks | O | Full duplication is included |
| Disjoint 50% / 50% partition | O | Equal two-way split is included |
| Disjoint 60% / 40% partition | I | The smaller fragment is absent from the core graph |
| Disjoint 40% / 35% / 25% partition | MISS | Complete union coverage can coexist with zero core edges |
| One complete mask plus a 30% internal part | C | GT class C does not certify a clean prediction set |
| Two masks each covering 60% of G but purity .2308 | O | Very impure masks can form a one-to-many component |
| Two GT, one complete prediction plus a union prediction | X for both GT | Local one-to-many structure can coexist with a multi-GT component |

These are specification counterexamples, not real dataset samples or model results.

The real-data screen required at least two predictions with purity >= .75 and .10 <= coverage < .50, whose union also had coverage and purity >= .75. It found **245 Faro and 29 Bama GT currently labeled MISS**. Thus small high-purity output parts that collectively cover a GT occur in the actual cache, not only in hypothetical examples. These are geometric screening positives, not 274 manually verified fragmentation cases or replacement O labels.

Other screen-positive current classes were Faro C=4, I=27, S=1, O=6, M=1, X=1, and Bama C=1, I=1. Counts can involve multiple GT in one image. This screening does not establish prediction ownership, independent scenes, or an absence of annotation issues.

### 2. Current O combines duplication, partial competition and fragmentation

The component cardinality rule does not inspect prediction-prediction overlap, union improvement or whether one prediction already covers the GT well. That is reasonable for a broad relation category, but it cannot define pure fragmentation or identify its cause.

| Descriptor Within Current O | Faro, n=2,801 | Bama, n=841 |
|---|---:|---:|
| Has at least one prediction pair with mask IoU >= .80 | 1,498 (53.48%) | 731 (86.92%) |
| Has at least one single mask with coverage/purity >= .75 | 2,031 (72.51%) | 749 (89.06%) |
| Has at least one core edge with purity < .50 | 244 (8.71%) | 5 (0.59%) |

These descriptors overlap and are not mutually exclusive subtypes. A high-overlap pair does not prove every prediction in that component is redundant; a good single mask does not imply a deployable rule can select it. O count alone must not be presented as fragmentation prevalence, a ranking defect or a mask-head defect.

### 3. Purity and component context need separate treatment

The core graph ignores purity. Large impure predictions can form O, M or X, while weak partial predictions do not participate. Moreover O requires the entire connected component to have one GT, not merely that a particular GT has two associated predictions. There are 338 Faro and 62 Bama GT labeled X that individually have core degree >= 2. Their exclusion from O is correct under the current mutually exclusive component definition, but local one-to-many structure still exists.

A blanket purity >= .50 edge guard is not a valid universal repair. Applied globally to the fixed masks, it changes Faro M/X counts from 80/424 to 0/0, and Bama from 216/84 to 4/2. A prediction covering several GT often has low purity relative to each individual GT. Deleting these edges can erase the very merge or mixed structure the graph is supposed to record.

Under this guard, 203 existing Faro O and four existing Bama O leave O, but 105 Faro X and 26 Bama X enter O. Net O count alone would conceal these changes. Per-GT transitions are retained.

### 4. Threshold and rule-version sensitivity are material

All rows below reuse exactly the same masks and scores; no model improvement occurs. Deltas are relative to the baseline O count. Alternative rules are audit probes, not accepted definitions.

| Association Rule | Faro O | Change | Bama O | Change |
|---|---:|---:|---:|---:|
| Core threshold .40 | 3,438 | +22.74% | 914 | +8.68% |
| Current core threshold .50 | 2,801 | 0 | 841 | 0 |
| Core threshold .60 | 2,331 | -16.78% | 801 | -4.76% |
| Current core AND purity >= .50 | 2,703 | -3.50% | 863 | +2.62% |
| Core OR legacy weak association | 4,891 | +74.62% | 1,136 | +35.08% |

The older helper `C:/Dpan/codexproject/pigcv_research/scripts/classify_public_test_instance_failures.py:119` includes purity >= .50 AND coverage >= .10 in its association function. The active diagnostic classifier computes those weak edges separately and does not use them to build primary components. These are two different definitions, not interchangeable implementations of one graph. The review is of the active core-only classification; this code difference alone does not establish that an earlier change was unauthorized or accidental.

## What Is Reasonable to Retain

GT-prediction bipartite relations avoid forcing all errors into one-to-one matches. The existing O/M/X component partition is reproducible and cleanly separates strong one-to-many, many-to-one and mixed components. Counting affected GT rather than edges gives a clear denominator. The baseline can remain a stable reference if its name, edge rule, confidence threshold and interpretation are explicit.

The issue is construct validity: a strong-component O label and semantic fragmentation are different objects. The current C/I/L/S/MISS labels must also be read relative to the core association graph, not as exhaustive descriptions of all predictions near a GT.

## Recommended Revision for Discussion

1. Retain the current primary label as a versioned strong-relation baseline and preserve its old results.
2. Add a separate part-association analysis using prediction purity and meaningful GT coverage, with cross-GT ambiguity checked explicitly. Do not put every weak edge directly into the primary graph or impose a universal purity guard.
3. Within one-to-many evidence, distinguish redundant/high-overlap predictions, complementary partial predictions, and coexistence of complete and partial masks. Report union gain, prediction overlap, single-mask quality and distinct contribution; avoid choosing subtype thresholds after looking for favorable counts.
4. Keep geometric quality and relation structure as separate descriptors so impure predictions and mixed components remain visible. In particular, do not force every local split within an X component into an exclusive O category.
5. Validate candidate subtype rules on a documented, manually reviewed sample stratified across current O, I, MISS and X, with threshold sensitivity and disagreement records. The 274 screened MISS are a useful review pool, not a substitute for that validation.

No alternative is adopted in this review. In particular, neither .40 nor legacy weak association has been selected because it produces more O cases.

## Literature Alignment

The existing project Wiki was consulted before the audit. Official sources were rechecked on 2026-09-07:

- [TIDE, official paper](https://arxiv.org/abs/2008.08115) and [official DuplicateError implementation](https://github.com/dbolya/tide/blob/master/tidecv/errors/main_errors.py): duplicate detections are a separately defined output-error category tied to an already-used GT. That definition is not equivalent to this project's component O or to semantic fragmentation. No TIDE AP oracle is substituted for this review.
- [Redefining Instance Matching](https://arxiv.org/abs/2605.31094): the authors explicitly distinguish matching families and show strategy/threshold-dependent behavior in segmentation evaluation. This supports making the matching definition and its sensitivity explicit; it does not validate the current pig-specific thresholds or provide a deployable model intervention.

## Evidence and Verification

- Protocol: `experiments/o_criterion_review_20260907_protocol.md`.
- Audit: `tools/review_o_criterion.py`.
- Results: `experiments/o_criterion_review_20260907_v1/summary.json`, `rule_summary.csv`, `transitions.csv`, `audit_rows.csv`, `synthetic_cases.json`, `input_sha256.json`.
- All 27,745 saved baseline GT classes reproduced exactly; 12 input/source hashes remained unchanged. Seven synthetic configurations were run through the actual current classifier.
- Scope: coordinator deterministic criteria review, not an independent reviewer judgment, newly annotated GT, method acceptance or a change in research direction.
