# YOLO26 Evidence Protocol Audit

审计日期：2026-09-05。目的：在进入方法选择前，核对已有失败诊断的推理口径、统计分母和候选分析可比性。

## Findings

| Item | Audited evidence | Status | Boundary |
|---|---|---|---|
| Formal relation diagnostic | YOLO26-seg, Ultralytics 8.4.100, `imgsz=1280`, `rect=True`, `conf=0.05`, `max_det=300`, `end2end=True`; 19,320 raw positions/image; 54/54 Top-300 parity | Pass for the audited PigLife sample | Exact runtime differs from the older 8.4.27 baseline; internal-candidate claims remain runtime-scoped |
| Failure denominator | 7,374 GT over the fixed YOLO26 diagnostic cache; baseline failure 19.84% | Pass as a cache-level descriptive statistic | Not a target-domain population estimate; confidence intervals are still missing |
| Dense/Touching onset sample | 43 relation components from 59 failure GT rows, with 30 unique C controls | Partial | Component denominator differs from GT denominator; report both explicitly |
| SPLIT_TYPE candidate competition | 50 GT; complete candidate proxy and local candidates; 11 absent, 8 dropped, 31 jointly surviving | Partial / exploratory | This analysis used the separate 1024 square-input cache with 21,504 raw positions and must not be merged with the formal 1280-rect diagnostic |
| Matching semantics | Candidate rules include IoU, coverage and purity thresholds; `SPLIT_TYPE` is a project label | Open | Modal/visible/amodal annotation semantics and threshold sensitivity are not yet fully audited across all three datasets |

## Safe claims now

1. The formal PigLife relation sample contains descriptive evidence of duplicate survival, mask/relationship failure, and a small score-threshold component.
2. The separate exploratory cache suggests complete/local candidate competition is heterogeneous: candidate absence, candidate dropping, and joint survival all occur.
3. Ordinary Mask-NMS is a bounded control result; it does not establish a fix for low-overlap complementary candidates.

## Claims not yet allowed

- Do not combine 19,320 and 21,504 raw-position counts in one table.
- Do not call `SPLIT_TYPE` a literature-established failure class or a causal network-stage diagnosis.
- Do not report the 50-case candidate competition result as formal cross-dataset evidence.
- Do not interpret outside-GT pixels as background errors before annotation semantics are verified.

## Required next audit

1. Re-run the candidate competition analysis under the formal `1280/rect=True` protocol on the same PigLife split.
2. Repeat the failure taxonomy and candidate slices on FaroPigSeg and BamaPig2D with frozen matching rules.
3. Add bootstrap intervals and a threshold-sensitivity table for the GT-level failure rates and `SPLIT_TYPE` counts.

## 2026-09-05 progress

- Completed an image-cluster bootstrap audit over the existing 1024-square single-forward cache. The all-dataset total failure rate is 19.84% (95% CI 18.50-21.14%), relation failure is 11.05% (10.19-11.85%), and miss rate is 3.63% (3.19-4.11%). Dataset-level and scene-stratified intervals are in `yolo26_gate_uncertainty_20260905.md`.
- Reconfirmed the cache-only NMS sensitivity table from the archived comparison. `MASK_NMS` reduces total failure to 13.59-15.31% across thresholds 0.70-0.95, while the relation-failure rate rises from 3.72% to 6.33%; this is a control result, not a causal mechanism proof.
- The formal 1280/rect candidate trace now exists for the reconstructed sample. Complete/local candidate-competition numbers remain unavailable as a formal result because the archived `SPLIT_TYPE`/component labels and original sample list are missing.
- The mechanism Gate remains open: uncertainty is quantified, but candidate formation versus selection versus mask construction has not yet been isolated by a formal-protocol counterfactual.

## Reconstructed-sample replay (2026-09-05)

- A geometry-aligned replay was completed under `imgsz=1280`, `rect=True` with Ultralytics `8.4.100`.
- Because the archived 118-row CSV was missing, the replay used a reconstructed 118-row sample: 25 O, 9 M, 25 X failures plus 59 nearest C controls. The reconstruction yielded 49 unique controls and 72 images, not the archived 30 controls and 54 images.
- All 72 images had 19,320 raw positions and exact Top-300 count parity.
- The replay produced 24 O, 4 M, and 13 X components; 40 components were assigned `MASK_RELATION_FAILURE` and one X component `LOW_SCORE`. No duplicate-survival component was observed.
- This discrepancy cannot be interpreted as a mechanism change. It is confounded by sample reconstruction, retained prediction provenance, and runtime difference (`8.4.100` versus archived `8.4.27`). Results are stored separately at `research-wiki/yolo26_formal_replay_20260905.md` and must not be merged with the archived report or 1024-square cache.

## Exact-runtime replay (2026-09-05)

- Isolated Ultralytics `8.4.27` wheel import succeeded and was used for a 4-image smoke plus the full 72-image reconstructed-sample replay; the active Conda environment was unchanged.
- Exact-runtime aggregate invariants matched the 8.4.100 replay: 19,320 raw positions/image; Top-300 parity 72/72; 53 failure GT rows; 49 controls; components O/M/X = 24/4/13; onset counts `MASK_RELATION_FAILURE=40`, `LOW_SCORE=1`, `DUPLICATE_TOPK_SURVIVAL=0`.
- Per-row candidate CSVs differ at the hash level, so this is runtime-consistency evidence for aggregate taxonomy only, not byte-level prediction equivalence. Reconstructed sampling remains the dominant provenance limitation and the Mechanism Gate remains open.

## Sample-list recovery search (2026-09-05)

- Re-scanned the legacy project, imported experiment assets, and current project for the archived dense-failure CSV, `SPLIT_TYPE` labels, and 54-image/30-control manifest.
- Only derived analysis reports and caches were found; no original 118-row sample list or equivalent provenance-complete manifest was recovered.
- The reconstructed replay must therefore remain audit-only. Candidate-stage statistics can be reported for the recovered O/M/X trace, but the formal complete/local `SPLIT_TYPE` comparison remains unavailable.

## Exact-trace matching sensitivity (2026-09-05)

- Reclassified the retained exact-runtime trace at association thresholds 0.30, 0.50, and 0.70 without rerunning inference.
- All O (24/24) and M (8/8) failure GT rows retained an associated, conf-qualified Top-300 candidate at every threshold; X was 21/21 through 0.50 and 20/21 at 0.70.
- This is evidence against candidate absence or Top-K loss as prevalent explanations in this reconstructed trace. It does not identify a causal stage or replace the missing formal `SPLIT_TYPE` comparison.

## Exact-trace single-candidate oracle (2026-09-05)

- Among failed GT rows, at least one raw candidate with mask IoU >= 0.50 existed for O `24/24`, M `8/8`, and X `21/21`.
- The same candidate-level threshold survived conf/Top-300 for O `24/24`, M `8/8`, and X `19/21`.
- This separates a small X selection-sensitive slice from the dominant O/M relation failure, but it is not a causal proof and does not test multi-candidate mask unions.

## Exact-trace multi-candidate union oracle (2026-09-05)

- The retained exact-runtime candidate trace was evaluated without a new forward pass. Among 59 reconstructed failure rows, the Top-300 best single candidate reached mask IoU >= 0.50 for 57/59; a best pair reached the threshold for 32/59; unioning all associated Top-300 candidates reached it for 57/59.
- By reconstructed state, best-pair success was O/M/X = 5/25, 8/9, 19/25. The result supports a complementary-candidate hypothesis for a substantial reconstructed M/X slice, but not for most O rows.
- Pair/all-union values are oracle upper bounds and can over-segment or use unavailable association decisions. They must not be reported as an implemented method or causal diagnosis. Sample reconstruction and missing archived `SPLIT_TYPE` provenance remain the dominant limitations.

## Provenance

- `C:\Dpan\document\pigcv_reserch\03-成果与资产\2026-08-31-YOLO26-Dense-Touching-关系失败内部诊断归档.md`
- `C:\Dpan\document\pigcv_reserch\03-成果与资产\2026-09-02-YOLO26-SPLIT_TYPE完整候选与局部候选竞争分析.md`
- `C:\Dpan\document\pigcv_reserch\03-成果与资产\2026-09-02-YOLO26-SPLIT_TYPE原始候选多峰与正确候选存在性分析.md`
