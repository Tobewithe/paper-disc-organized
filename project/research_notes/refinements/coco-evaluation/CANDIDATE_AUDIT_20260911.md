# Initial Mechanism-Candidate Experiment Audit

**Date:** 2026-09-11  
**Auditor:** fresh GPT-5.6-Sol ultra reviewer plus deterministic read-only checks  
**Review class:** same-family, provisional  
**Scope:** only the frozen mechanism/capacity experiments and the three initial prediction-derived candidate probes listed below. The later `supervised_direction_probe.py` and its dev/holdout runs are outside this audit.

The fresh reviewer returned substantive findings on pair selection and the two exact candidate-direction identities, but did not finish its requested A–F response before the delivery cutoff. The A–F checklist below therefore combines those preserved reviewer findings with deterministic executor checks; every positive status is limited to locally available artifacts, and this remains a provisional same-family audit.

## Overall verdict: FAIL

The frozen mechanism experiment is a traceable, honestly bounded `real_gt` diagnostic and its main narrative is supportable with the stated conditioning. The initial candidate bundle is not sound enough to support a candidate-efficacy claim: the purported GT-free workflow uses a GT-selected and GT-matched cohort; two box-core variants use the wrong coordinate frame; all five distillation weights are mathematically and empirically identical; and two output-overlap variants are the same normalized direction. These findings warrant archiving the initial candidates as failed diagnostics, not treating them as independent method evidence.

No fake GT or prediction-statistic score normalization was found. All locally available CSV values inspected were finite, candidate summary means recomputed from per-target rows, and the three candidate `COMPLETE.json` script hashes match the current source files.

## Critical findings

### 1. `core06_excl` and `core08_excl` use the wrong coordinate frame

`Capture` stores raw post-NMS boxes before the official predictor scales its result boxes to the original image (`frozen_mechanism_probe.py:82-90`). Those cached boxes are therefore in the letterboxed inference grid. `box_spatial_probe.py:63-64` correctly creates an original-resolution support mask by cropping on the inference grid and then applying `scale_masks`, but `box_spatial_probe.py:39` passes the same cached inference-grid box directly to `mask_box(..., capture['shape'])`. `mask_box` interprets those numbers as original-image pixel coordinates (`box_spatial_probe.py:27-32`). On non-square images, padding and scale are therefore ignored and the core region is shifted/scaled.

Impact: results named `core06_excl` and `core08_excl` are invalid as core-of-predicted-box interventions. Do not cite or compare their numerical outcomes. `box_excl` and `box_overlap` derive regions from the correctly transformed support masks, so this specific coordinate defect does not apply to them.

### 2. “GT-free” describes only direction construction, not pair selection

The original cohort is chosen entirely from COCO GT: same-category annotation pairs, GT box IoU above 0.05, GT-derived ICI strata, and GT annotation IDs (`frozen_mechanism_probe.py:55-79`). Predictions are then assigned to those GT instances through COCOeval bbox matching (`frozen_mechanism_probe.py:93-109`). All three candidate probes reuse only `status == 'ok'` cached pairs (`box_spatial_probe.py:58`, `output_overlap_probe.py:24`, `overlap_distill_probe.py:15`). The frozen protocol itself correctly says `sampling: GT only` (`protocol.json:18`).

Impact: the candidate objectives/directions are prediction-only **conditional on a GT-defined cohort and GT ownership mapping**. They are not a GT-free pair-selection or deployable inference pipeline. Their outcomes may be reported only as conditional diagnostic results unless pair discovery and ownership are replaced with an inference-available rule and evaluated independently.

### 3. All five `overlap_distill` weights are identical interventions

The preservation anchor is `lambda * mean((z - z0)^2)` with `z0` detached at the starting coefficient (`overlap_distill_probe.py:19-26`). At the evaluated point `x = c`, `z == z0`, so this term has exactly zero first derivative for every lambda. The only first-order direction comes from the overlap/background terms; tangent normalization then fixes the step to 5 degrees (`overlap_distill_probe.py:27`).

The local result confirms the identity: every output value for all 524 targets is exactly equal across `distill01`, `distill03`, `distill1`, `distill3`, and `distill10`; the high-group mask-IoU summaries are identical at `-0.1412828498` (`overlap_distill_probe/summary.csv:11`, `:26`, `:41`, `:56`, `:71`).

Impact: this is one intervention repeated five times, not a weight sweep. It cannot support a lambda sensitivity, preservation, robustness, or efficacy claim.

### 4. `q_overlap_strong` duplicates `q_overlap_excl`

The weights for `q_overlap_strong` are exactly twice those for `q_overlap_excl` (`output_overlap_probe.py:13-17`). Multiplying the full objective by a positive scalar multiplies its gradient, while line 46 normalizes the tangent before taking the fixed 5-degree step. Both configurations must therefore produce the same direction. All 524 target outcomes are exactly identical; the high-group rows match field-for-field (`output_overlap_probe/summary.csv:22-26` and `:52-56`).

Impact: treat these as one arm. The “strong” arm supplies no independent evidence.

### 5. Zero-direction and finite-value guards are incomplete in the candidate scripts

The frozen oracle explicitly detects zero tangent norms and keeps the coefficient unchanged while recording `zero_directions` (`frozen_mechanism_probe.py:169-175`); the saved run contains 786 oracle-direction checks with zero zero-directions. The three candidate scripts instead apply `F.normalize` directly to a tangent without an active/finite guard (`box_spatial_probe.py:48-50`, `output_overlap_probe.py:46`, `overlap_distill_probe.py:27`). `q_overlap` has no objective terms when predicted box supports do not overlap and its saved result contains six pair-level zero objectives (12 target rows), so the no-op depends on framework zero-normalization behavior rather than an explicit protocol.

All downloaded candidate CSVs are finite, so this did not create a nonfinite saved value in this run. It remains an implementation/provenance defect: future runs should record tangent norm, active status, actual angular change, and finite checks per pair instead of labeling every row as a 5-degree intervention.

## A-F integrity checklist

### A. Ground Truth Provenance: WARN

- Evaluation GT comes from dataset-provided COCO annotations loaded through `pycocotools.COCO`, not from model outputs (`frozen_mechanism_probe.py:272-277`; `frozen_capacity_check.py:25-30`; candidate scripts load the same COCO source at `box_spatial_probe.py:55-57`, `output_overlap_probe.py:22-23`, and `overlap_distill_probe.py:13-14`).
- Official COCOeval bbox matching is used for ownership (`frozen_mechanism_probe.py:93-109`). Pixel metrics use COCO `annToMask` instance masks (`frozen_mechanism_probe.py:184-191`).
- Oracle directions and capacity fits explicitly use GT (`frozen_mechanism_probe.py:159-175`; `frozen_capacity_check.py:41-68`) and are labeled diagnostic/oracle rather than deployable (`MECHANISM_LOCALIZATION.md:37`, `:80`).
- Warning: initial candidate pair selection is not GT-free, as detailed above. The local copy does not contain `data/annotations/instances_val2017.json`, so this audit could not independently decode source GT or repeat COCOeval matching; the saved protocol records GT SHA-256 `e8c7...0b6f` (`protocol.json:3`).

### B. Score Normalization: PASS

- `coverage`, neighbor errors, and background error divide pixel counts by dataset GT area; IoU uses `tp/(GT area + neighbor + background)` (`frozen_mechanism_probe.py:147-157`). This is not normalization by prediction maxima/minima/means.
- Raw intervention metrics are retained in `interventions.csv` and candidate `per_target.csv` files, with paired deltas in summaries.
- The image-cluster bootstrap weights resample image IDs and aggregate target deltas (`frozen_mechanism_probe.py:236-265`; candidate aggregation at `box_spatial_probe.py:73-83`, `output_overlap_probe.py:52-60`, `overlap_distill_probe.py:33-41`). No score rescaling toward 1 was found.
- All checked CSV numeric fields were finite. The IoU identity held for every row in all three candidate result files.

### C. Result Existence and Claim Traceability: WARN

- The frozen run is marked complete with 300 selected images, 262 valid pairs, 38 unmatched pairs, and 16,768 intervention rows (`COMPLETE.json:2-9`; `audit/AUDIT.json:5-13`). These counts match local `selection.json`, `statuses.json`, and CSV row counts.
- The main narrative numbers match the local summaries: coefficient pushing reduces raw pair IoU by `0.052355` (`audit/representation_summary.csv:4`) without a significant high-group mask-IoU gain (`summary.csv:126`); the bounded ridge direction improves high-group IoU by `0.005988` (`audit/bounded_summary.csv:16`); full GT fit improves it by `0.128772` (`capacity_check/summary.csv:11`).
- Candidate outputs now exist: 2,096 rows each for box-spatial and output-overlap, 2,620 rows for overlap-distill; each `COMPLETE.json` hash equals its current script hash.
- Warning: `LOCAL_ARTIFACT_SHA256.json` predates the candidate downloads and covers none of their nine `COMPLETE.json`/`per_target.csv`/`summary.csv` files (`LOCAL_ARTIFACT_SHA256.json:12-20`). Their hashes are recorded in the companion audit JSON, but the original manifest is not a receipt for them.
- `MECHANISM_LOCALIZATION.md` predates these candidate runs and correctly says no candidate had yet been selected or trained (`:98-100`); therefore it contains no candidate efficacy claim to validate.

### D. Dead Code Detection: WARN

- Core frozen functions are called: selection, ownership, perturbation construction, decoding, pixel metrics, summarization, and the capacity/audit pipelines all feed saved outputs (`frozen_mechanism_probe.py:277-312`; `frozen_capacity_check.py:37-88`; `audit_frozen_mechanism.py:92-204`).
- Candidate loops and summaries execute and have matching result receipts.
- In `output_overlap_probe.py:29-37`, an initial `z0`, objective, gradient, tangent, and `baseang` are computed before the per-variant loop. Only `baseang` is used later; the first objective/gradient/tangent and `z0` are dead calculations. This does not change saved variants, but it obscures which objective produced the intervention.
- The candidate probes were written after the earlier `audit_frozen_mechanism.py` run and are not covered by that audit script. There is no independent replay audit of their directions, coordinate masks, actual angles, or bootstrap summaries in the supplied artifacts.

### E. Scope Assessment: WARN

- Scope is one official pretrained model, one COCO val2017 GT-selected enriched sample, one deterministic seed, 300 selected images, 262 matched pairs, 524 evaluated targets, and pointwise image-cluster bootstrap intervals. It is conditional on both instances matching at bbox IoU at least 0.5 and is not population-weighted COCO AP (`protocol.json:7-19`; `MECHANISM_LOCALIZATION.md:11-17`).
- The narrative states these limits, distinguishes oracle from deployable evidence, and avoids claiming a new loss works (`MECHANISM_LOCALIZATION.md:7`, `:37`, `:72`, `:84`, `:94`, `:98-100`). Its scope language is appropriate.
- The initial candidates use the same development cohort, no independent holdout, no training, no seeds, and no official task-level AP. Their multiple nominal variants are further reduced by the identities above. They can localize failure modes only; they cannot support robustness, generalization, deployability, or learned-method efficacy.

### F. Evaluation Type: real_gt

- Primary and candidate outcome metrics are `real_gt`: dataset COCO masks provide the targets.
- `spatial_oracle`, full ridge fit, and bounded ridge direction are `real_gt` post-hoc oracle diagnostics.
- Push/pull/random and the candidate directions are prediction-derived interventions evaluated with `real_gt`; they are not `self_supervised_proxy` because the reported outcome is against COCO GT.
- The coefficient/logit cosine, positive-product, and raw pair-IoU representation measures are GT-free proxy descriptors only. The narrative correctly says raw pair IoU is not a GT leakage metric (`MECHANISM_LOCALIZATION.md:23`).

## Mechanism-specific checks

| Check | Status | Evidence and implication |
|---|---|---|
| Frozen decoder replay | PASS for saved audit | Zero replay XOR is 0 (`COMPLETE.json:6`); independent official replay covered 16 images and reports maximum metric error 0 (`audit/AUDIT.json:9-13`). Local tensors are absent, so this audit did not replay them again. |
| Coordinate gauge identity | PASS | Code applies `c' = cD`, `P' = D^-1 P` and asserts logit/binary parity (`frozen_mechanism_probe.py:206-215`). All 524 saved transformations changed coefficient cosine, had raw XOR 0, and maximum saved logit difference 0. |
| Candidate spatial coordinates | FAIL | `core06_excl`/`core08_excl` rasterize input-grid boxes directly on original-image grids (`box_spatial_probe.py:27-39`). |
| Pair selection GT-free | FAIL as stated workflow | Selection and matching use COCO GT (`frozen_mechanism_probe.py:55-79`, `:93-109`); candidates inherit only matched cached pairs. |
| Region definitions | WARN | Main metrics partition predicted positive pixels into own GT, other ordinary GT outside own, and background, excluding crowd (`frozen_mechanism_probe.py:147-156`). Candidate box/output regions use prediction boxes, but box-core coordinates are wrong and none records region sizes/empty flags. |
| Control distinctness | WARN | Frozen zero and random± arms are distinct in code (`frozen_mechanism_probe.py:116-130`, `:248-264`); 521/524 sampled target outcomes differed between random+ and random−. Candidate variants are not all distinct: `q_overlap_strong == q_overlap_excl`; all five distill weights are identical. |
| Zero/finite directions | WARN | Frozen oracle has an explicit zero-direction path and saved count 0 (`frozen_mechanism_probe.py:169-175`). Candidate scripts have no equivalent active/finite guard. Saved CSVs are finite, but six `q_overlap` pairs have zero objectives/no-op outcomes. |
| Weight identity | FAIL | Positive scalar objective scaling vanishes under tangent normalization for `q_overlap_strong`; distillation-anchor first derivative is zero at the starting point. Saved outputs confirm exact identity. |

## Claim impact and archive disposition

- **Supported with qualifiers:** The fixed decoder admits GT-informed coefficient changes that improve conditional per-instance pixel metrics on this enriched matched cohort. This is an oracle feasibility result, not learned performance, AP, an optimum, or a bound.
- **Supported with qualifiers:** Euclidean coefficient separation reduces representation overlap but does not establish improved GT ownership; coefficient cosine is coordinate-dependent under an exact decoder gauge transformation.
- **Unsupported:** Any claim that the initial candidate pair-selection/evaluation workflow is GT-free or deployable.
- **Unsupported:** Any efficacy/sensitivity claim from `core06_excl` or `core08_excl`, any lambda effect from `overlap_distill`, or any independent “strong” effect from `q_overlap_strong`.
- **Archive label:** `FAILED_INITIAL_DIAGNOSTIC_CANDIDATES` — retain code/results for negative evidence and provenance, exclude them from positive method selection.

## Required corrections before reuse

1. Transform cached boxes into the exact raster coordinate frame before constructing core regions, and save coordinate parity/overlay receipts.
2. Use prediction-available pair discovery if claiming GT-free selection; describe COCO GT only as evaluation. Otherwise call the current setup “prediction-only direction on a GT-selected matched cohort.”
3. Replace the zero-gradient distillation anchor or evaluate it away from the anchor; verify per-weight gradient/direction distinctness before calling it a sweep.
4. Deduplicate globally proportional objectives under normalized steps; record cosine between candidate directions and actual output-space RMS/angular displacement.
5. Add explicit empty-region, tangent-norm, finite-value, and actual-angle fields and a candidate-specific replay audit.
6. Add the candidate outputs to a new immutable artifact manifest. Preserve the current files and hashes as the failed-run receipt rather than overwriting them.

## Audit limits

- The local workspace lacks the frozen run’s `tensors/` and `per_image/` directories and the source `data/annotations/instances_val2017.json`. The Linux path `/root/autodl-tmp/coco_clean_20260911/diagnostics/mechanism_official_300_20260911/` is not mounted locally. Consequently, this audit verified code, downloaded tables, hashes, algebraic identities, row counts, finiteness, metric identities, and summary means, but did not rerun inference, decode GT, rebuild selection, recompute bootstrap quantiles from tensors, or inspect remote-only files.
- The previous `audit/AUDIT.json` predates the three candidate scripts, was written by the same implementation lineage, and cannot be treated as an independent candidate audit (`MECHANISM_LOCALIZATION.md:94`).
- The fresh reviewer response is partial: it independently confirmed the GT-conditioned selection and the exact output-overlap/distillation identities, but did not return a complete A–F verdict. The remaining A–F entries are deterministic executor findings and are not represented as an independent reviewer judgment.
- Same-family fresh-agent review is provisional under the ARIS reviewer-independence contract. No cross-family acceptance is claimed.
