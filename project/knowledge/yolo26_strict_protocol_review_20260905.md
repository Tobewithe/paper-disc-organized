# YOLO26 Strict Protocol Independent Review

**Date:** 2026-09-05  
**Review type:** independent CPU-only, read-only audit  
**Independence:** `same-family`  
**Acceptance status:** `provisional`

## Verdict

**PASS for implementation consistency and the recorded smoke scope.** The strict-quality edge construction, final-source mapping guards, all-GT matching layout, smoke artifact hashes, matching cardinalities, component statistics, and Top-K single-gate accounting reconcile.

**Gate status: PENDING, not passed.** The strict-quality and strict-matching artifacts cover only two images (one per dataset), 24 GT instances, and 640 pairwise IoU-50 edges. They are smoke validation, not a 586-image / 6,226-GT result. They cannot close the R006 route gate or support a method-selection claim.

**Top-K single-gate conclusion: PASS within its stated frozen-trace condition.** Across all 586 cached images, every raw source with `score > 0.05` is already in the recorded Top-300 set. Relaxing only that Top-K retention gate therefore adds no strict-confidence source in these traces. This is a negative single-gate result; it does not establish a score/ranking cause, NMS cause, mask-match cause, or the plan's 30% ranking/competition criterion.

No GPU work, model inference, full rerun, private/locked/guard/target-data access, or source/dependency modification was performed.

## Checks That Passed

- **Threshold definitions and raw filtering.** `analyze_yolo26_strict_quality.py` defines exactly `IoU >= 0.50`, `coverage >= 0.75 AND purity >= 0.75`, and `IoU >= 0.75` ([source](/C:/Dpan/codexproject/paper-disc/tools/analyze_yolo26_strict_quality.py:57)). It only decodes sources with box-envelope GT coverage at least 0.50, with no quota, purity screen, or downsampling ([source](/C:/Dpan/codexproject/paper-disc/tools/analyze_yolo26_strict_quality.py:195)). This is the stated lossless screen conditional on the shared decoder's box-cropped mask semantics; that external decoder implementation was not part of the permitted review inputs.
- **Source and final mapping integrity.** The script requires one-to-one final prediction IDs and source IDs, exact prediction-to-mapping coverage, final-source subset of strict-confidence sources, native decoded-mask versus final-RLE XOR equality, and nested raw/Top-K/conf/final counts ([source](/C:/Dpan/codexproject/paper-disc/tools/analyze_yolo26_strict_quality.py:75), [source](/C:/Dpan/codexproject/paper-disc/tools/analyze_yolo26_strict_quality.py:190), [source](/C:/Dpan/codexproject/paper-disc/tools/analyze_yolo26_strict_quality.py:259)). The smoke has total/max final-source XOR of zero.
- **All-GT simultaneous matching.** Matching consumes every row in `per_gt.csv`, including baseline-C neighbors, not only failed GT. It checks each edge count against the quality table, assigns sources once, and emits component totals including `baseline_correct_gt`, absent GT, and assignment deficit ([source](/C:/Dpan/codexproject/paper-disc/tools/audit_yolo26_strict_matching.py:25)). The smoke output has 288 rows, exactly `24 GT x 3 quality criteria x 4 stages`.
- **Independent reconstruction.** A separate CPU implementation reconstructed all 24 image/quality/stage bipartite graphs from `edge_metrics.csv`. Every maximum-cardinality total and every component statistic matched `per_gt_matching.csv` and `components.csv`. All 72 summary point estimates also recomputed exactly from matching rows.
- **Original IoU-50 reproduction on the smoke scope.** The matching artifact reports six comparisons to the pre-existing stage table and no mismatch. These are the six prior failed-GT rows present in the two sampled images; this is not a full-dataset reproduction.
- **Bootstrap boundary.** The matching artifact declares 3,000 image-cluster draws, seed `20260905`, with scope conditional on images containing the selected GT subset. This boundary is appropriate for its named rates. With one image per dataset, all smoke confidence intervals necessarily collapse to the point estimate; interval behavior requires the full run to be evidence-bearing.
- **Top-K accounting.** `audit_yolo26_gate_binding.py` checks source-ID domains, 300 unique Top-K members per image, cached confidence and Top-K flags, and strict-confidence membership ([source](/C:/Dpan/codexproject/paper-disc/tools/audit_yolo26_gate_binding.py:36)). Aggregates reproduce from all 586 per-image rows: PigLife has 4,843 strict-confidence raw sources and Faro has 3,515; both have zero strict-confidence sources outside Top-300, zero framework sources outside Top-300, and zero framework sources failing `score > 0.05`.
- **Maximum-cardinality test coverage.** The shared matcher has an existing augmenting-reassignment test in `test_oracle_assignment_is_maximum_cardinality` ([source](/C:/Dpan/codexproject/paper-disc/tools/test_yolo26_candidate_protocol.py:36)). Its options are `{10: {0}, 11: {1, 2}, 12: {0, 1}}` and it asserts `{10: 0, 11: 2, 12: 1}`. This covers the required reallocation from an initially occupied source to reach cardinality three; no duplicate matching test is needed.
- **Provenance and tests.** Current SHA256 values for the three reviewed scripts and all reviewed artifact outputs match their embedded validation manifests. `test_yolo26_strict_matching.py` and the existing candidate-protocol maximum-cardinality test passed in the project `pytorch` environment.

## Required Full-Run Acceptance Checks

1. Let the already-running full `strict_quality` job finish without changing its script or dependencies; require `full_coverage: true`, 586 images, 6,226 GT instances, and zero final-source XOR.
2. Run strict matching against that completed quality directory without `--allow-smoke`; require all prior IoU-50 rows to be compared, all-GT/C-neighbor totals to reconcile, and image-cluster intervals computed from the full scoped-image sets.
3. Preserve the current hash/provenance checks and report strict quality as pairwise eligibility plus simultaneous assignment only. Do not translate either artifact into output-set recovery, deployable behavior, AP gain, causal attribution, or Gate satisfaction.
