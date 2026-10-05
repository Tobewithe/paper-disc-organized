# YOLO26 Strict Full-Trace Integrity Audit

**Date:** 2026-09-05  
**Review type:** independent CPU-only, read-only acceptance audit  
**Independence:** `same-family`  
**Acceptance status:** `provisional`

## Verdict

**Experimental integrity: PASS.** The completed full-trace strict-quality and strict-matching artifacts reconcile with their provenance records and with independent recomputation. The full scope is 586 images, 6,226 GT instances, and 193,413 pairwise quality edges. Final-source decoding parity passed with zero total and maximum XOR pixels.

**Method acceptance: PENDING.** These artifacts establish pairwise strict-quality availability and maximum-cardinality simultaneous allocation under the recorded frozen traces. They do not establish a deployable intervention, an output-set recovery result, a causal score/ranking/NMS mechanism, overall mask AP/AP50/AP75 improvement, or satisfaction of the plan's Route-A Gate.

No GPU work, model inference, full rerun, private/locked/guard/target-data access, or modification of existing scripts, dependencies, or experiment artifacts was performed in this audit.

## Evidence and Independent Checks

- **Scope and completion.** `strict_quality` reports `full_coverage: true`, 586 images, 6,226 GT, 193,413 edges, and zero final-source XOR. `strict_matching` reports the same scope, 74,712 matching rows (`6,226 x 3 quality criteria x 4 stages`), and 1,191 historical IoU50 failed-GT comparisons.
- **Hash chain.** The current SHA256 values of both reviewed scripts, the four strict-quality evidence files consumed by matching, the prior stage table, and all three matching outputs match the recorded validation hashes. The quality run metadata also matches all recorded frozen-trace and runtime-source hashes.
- **Edge and availability reconstruction.** Every one of the 193,413 edges has a unique `(dataset, image, annotation, source)` identity; its IoU/coverage/purity algebra, strict quality flags, and raw/Top-K/conf/final membership nesting recompute. All per-GT edge counts and all 204 strict-quality summary rows recompute exactly.
- **Historical IoU50 reproduction.** Every one of the 1,191 rows in `yolo26_stage_availability_full_20260905_v1/stage_gt.csv` matches the new strict-quality per-GT IoU50 counts at raw, Top-K, confidence, and final stages. This reproduces the prior failed-GT availability measurement, not a causal stage attribution.
- **Independent exact matching.** SciPy `sparse.csgraph.maximum_bipartite_matching` independently reproduced the maximum cardinality for all 7,032 image/quality/stage graphs (`586 x 3 x 4`). The recorded selections are admissible, do not reuse a source within a graph, and have the same cardinalities.
- **Component accounting.** An independent bipartite connected-component reconstruction, including isolated GT vertices, matched all 7,032 component partitions and all component statistics. The 72 matching-summary point estimates recompute directly from `per_gt_matching.csv`.
- **Bootstrap.** All 216 reported confidence interval endpoints recompute exactly using 3,000 image-cluster draws, seed `20260905`, and paired per-draw numerator/denominator aggregation. No selected scope has a null bootstrap denominator.

## Material Results With Interpretation Limits

- Exact allocation exposes 20 genuine assignment deficits across the full component table. Each deficit is one GT; there are 20 conflict-marked GT rows. These are not errors: they are the measured difference between pairwise availability and simultaneous source allocation.
- For failed GT at final `IoU50`, PigLife has 373 pairwise-available but 365 simultaneously matched of 394; Faro has 553 available and 553 matched of 797. For final `coverage >= 0.75 AND purity >= 0.75`, PigLife has 312 available and 310 matched; Faro has 224 available and 224 matched. For final `IoU75`, PigLife has 296 available/matched and Faro has 196 available/matched.
- The results remain pairwise mask-quality eligibility plus a source-uniqueness matching construction. An admissible edge does not exclude contamination with other GT instances, and a maximum-cardinality assignment does not demonstrate final output-set recovery or preserve all baseline-C outputs under an intervention.
- The companion all-trace Top-K audit still limits the claim: every raw source with `score > 0.05` was already Top-300. A raw-to-Top-K availability drop therefore cannot by itself be called the effect of relaxing only Top-K; such sources may fail the fixed confidence gate.

## Residual Boundary

The full artifacts are acceptance-quality for their declared diagnostic measurements. They do not yet supply the plan's required deployable candidate-generation/scoring or deletion control, causal isolation of retention/ranking/NMS effects, AP evaluation, or independent cross-family review. Keep `same-family` and `provisional` labels until those distinct requirements are met.
