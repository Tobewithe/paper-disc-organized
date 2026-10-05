# Ownership Ranking Experiment Audit

**Verdict:** `WARN` — no fatal execution or evaluation defect found. The recorded `NO_GO` is supported by the available artifacts.
**Review class:** fresh Codex reviewer, `same-family`, `provisional`.
**Reason:** exploratory reused validation data and a bounded frozen-candidate experiment limit claim scope.

### A. Ground-Truth Provenance — PASS

Training and evaluation use original COCO instance annotations. The protocol requires `pycocotools.COCO.annToMask`, multipart union, crowd exclusion, and disclosed misses (`ownership_ranking_protocol_20260912.json:6`). Cache construction loads COCO annotations and excludes crowds/padding from valid ownership regions (`cache_ownership_ranking.py:27`, `cache_ownership_ranking.py:75`). Target-exclusive and directed neighbor-exclusive regions are formed from real masks within both candidate supports (`cache_ownership_ranking.py:88`). Fixed bbox-IoU matching uses COCOeval assignments and asserts one-to-one mapping (`frozen_mechanism_probe.py:93`). Task evaluation loads GT separately after inference and asserts every ordinary GT appears in the denominator (`eval_ownership_ranking.py:60`, `eval_ownership_ranking.py:68`).

### B. Score Normalization — PASS

Reported AP, R75, recovery, and spatial metrics are not normalized against the method’s own output. Feature standardization is fitted from training-cache features only (`train_ownership_ranking.py:17`). Losses operate on raw predicted mask probabilities/logits (`ownership_ranking.py:15`). The coverage term compares against the frozen model response, but the protocol explicitly labels it a soft preservation surrogate rather than a recall guarantee (`ownership_ranking_protocol_20260912.json:22`). It must remain described as a regularizer, not evaluation recall.

### C. Result-File Existence — PASS

The main archive contains 1,200 cache files with matching declared hashes, 18 training histories, 270 epoch checkpoints, complete evaluation tables, 19 prediction gzip files, and completion receipts. Both deterministic local archive verifiers passed. The result narrative identifies the concrete archives and verifier outputs (`OWNERSHIP_RANKING_RESULTS_20260912.md:91`). The tracker records the completed 18-arm/three-seed experiment and `NO_GO` outcome (`EXPERIMENT_TRACKER.md:24`). Smoke and main artifacts were distinguished; only main results support the reported conclusion.

### D. Dead-Code Detection — PASS

All material metric paths are active: task COCOeval and denominator checks run in `eval_ownership_ranking.py:68`; GT/pair recovery is computed in `summarize_relative_ownership.py:17`; paired cohorts, bootstrap intervals, family means, and the locked gate are consumed in `summarize_ownership_ranking.py:22`. No unused metric implementation was found that could substitute for the reported path. An unused auxiliary `gates()` helper in `local_coefficient_head.py` is outside the ownership-ranking evaluation path and does not affect results.

### E. Scope Assessment — WARN

Evidence supports only this frozen-head, fixed-candidate pilot: 1,200 hashed train images, 500 hashed validation images, three seeds, and matched spatial diagnostics. The validation set was previously explored and is explicitly not pristine (`ownership_ranking_protocol_20260912.json:9`; `MAINLINE_PROPOSAL_20260912.md:57`). Candidate training/spatial analyses are restricted to nonempty post-NMS detections with fixed bbox50 matches; task AP/R75 correctly evaluates all ordinary GT. Confidence intervals condition on this subset and three heads, with no multiple-comparison correction (`OWNERSHIP_RANKING_RESULTS_20260912.md:77`). The failed locked gate correctly stops expansion to 5,000 images (`OWNERSHIP_RANKING_RESULTS_20260912.md:3`).

Fairness checks pass: equal architecture, initialization policy, data ordering, observations, epochs, and update budgets are frozen (`ownership_ranking_protocol_20260912.json:18`; `train_ownership_ranking.py:83`). Inference accepts predicted features only (`eval_ownership_ranking.py:19`). Sampling replay and interpolation equivalence are checked before training (`train_ownership_ranking.py:42`). Hash selection precedes outcomes; the remaining selection concern is reuse of explored validation data.

### F. Evaluation Type — `real_gt`

Primary task AP/R75 and recovery use `real_gt`. Matched spatial and pair diagnostics also compare against real COCO masks, but on a fixed matched subset. The coverage loss uses a model-derived reference solely as a preservation proxy; it is not GT and does not establish actual recall.

**Claim impact:** The exact implementation’s `NO_GO` is supported. BCE+Dice improvements may be reported only within the frozen 500-image pilot. Ranking efficacy, causal ownership mechanism, full-COCO generalization, and independent confirmation are unsupported.

**Not independently verified:** The original COCO annotation JSON was unavailable in this local audit, so annotation contents and official COCOeval values were not recomputed from scratch. Training and inference were not rerun. Archive hashes, receipts, code paths, raw outputs, and deterministic local verifiers were checked.

## Executor clarification of reviewer wording

The reviewer response above is preserved. The task-level GT-pair recovery uses all ordinary GT pairs that satisfy the geometry rule; it is not limited to the bbox-matched spatial subset. Only the spatial diagnostics and training use fixed matched targets. Sampling interpolation replay occurs in cache_ownership_ranking.py, while the referenced training witness checks gradients and finite updates. There are six loss arms with three seeds each, totaling eighteen runs. Some reviewer line anchors identify the surrounding function rather than the exact check. The original annotation JSON and a fresh official AP rerun were not independently verified by this local reviewer.
