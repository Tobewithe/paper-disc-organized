# R006 v2 Final Integrity Audit

**Audit date:** 2026-09-05  
**Scope:** CPU/read-only review of the frozen R006 v2 artifacts for PigLife public test and FaroPigSeg test. No inference, GPU work, experiment rerun, private/locked-test data, guard-band data, or target data was accessed.

## Verdict

**Artifact integrity: PASS.** Provenance, launch-time source identity, decoder parity, denominator identities, matching uniqueness, and prediction accounting agree across the frozen artifacts.

**Scientific interpretation: WARN.** R006 measures an achieved **GT-guided output-set replacement recovery** under one specified maximum-cardinality matching policy. It is not evidence of deployable inference, an optimal output-set recovery policy, AP improvement, a candidate-scorer Gate, or a causal ranking/NMS attribution.

The review label remains `review_independence: same-family` and `acceptance_status: provisional`. This is not cross-family acceptance.

## Evidence Checked

### Provenance and runtime boundary

- The current SHA256 values of `run_summary.json` and `counterfactual_summary.csv` are respectively `4ca8bf28f9a189a1294910af51190dbd001e256a967687545f2864a32766fa88` and `84caa52346c242f21f13d5d555f22416791c700843fa720adfe78aed756c2ffe`; both match `validation.json`.
- `validation.json` records SHA256 values for all v2 analysis inputs and the derived condition, bootstrap, accounting, and statistics outputs.
- The on-disk `r006_v2_script.py` was edited after process launch. The applicable main-module record is `r006_v2_at_launch.py`, whose SHA256 is `fdc393636537cf38a358ab56e221f86e63ff1e2a63f8c1f0c524bdca7a7cade7` and matches the pre-launch observed full-file hash. The legacy classifier snapshot predates process creation; the later disk edits were not part of the already-loaded v2 process.
- Batch-2 final-source parity passed with Ultralytics `8.4.100`: 586 images and 8,327 final-source masks were exact, with total and maximum XOR pixels both zero.

### Denominators and matching

- The baseline contains 6,226 GT instances: PigLife 4,474 across 426 images and Faro 1,752 across 160 images. Failed-GT denominators are 394 and 797, totaling 1,191; baseline class-C counts are 4,080 and 955, totaling 5,035.
- There are 1,136 raw-good GT instances and 1,136 selected GT instances. The selected `(dataset, image, source)` count is 1,136, with zero duplicate groups and zero assignment conflicts; 55 failed GT instances have no raw-good candidate.
- Thus the realized maximum-cardinality matching selected every raw-good GT uniquely on this frozen sample. This establishes matching feasibility for this policy only; it does not establish a globally optimal recovery policy.
- The v1/v2 comparison reports zero per-GT class changes and zero changes to the checked mapping fields; both versions report `raw_good_gt = assigned_gt = 1,136`.

### Transition and output identities

For `ADDITION_REMOVAL`, the GT transitions reconcile exactly:

| Dataset | Recovered to C | C to failure | Final failures | Identity |
| --- | ---: | ---: | ---: | --- |
| PigLife | 322 / 394 | 0 | 72 | `394 - 322 = 72` |
| Faro | 477 / 797 | 1 | 321 | `797 - 477 + 1 = 321` |
| All | 799 / 1,191 | 1 | 393 | `1,191 - 799 + 1 = 393` |

The final class-C count is 5,833, an increase of 798 from 5,035. Prediction counts also balance: baseline 8,327; addition `+1,035 = 9,362`; removal `-1,566 = 6,761`; addition/removal `8,327 - 1,566 + 1,035 = 7,796`.

The uncertainty table uses image-cluster bootstrap with 3,000 replicates, seed `20260905`, paired numerator/denominator draws, manifest images included, and recorded null denominators; all reported draws are valid.

## Interpretation Boundary

- Describe final-stage tables as **per-GT threshold existence/availability**. They do not show simultaneous matching feasibility or a causal decomposition by stage.
- Do not describe R006 as an upper bound beyond the stated GT-guided matching construction, or as a deployable intervention.
- Do not infer mask AP, AP50, AP75, score calibration, ranking behavior, NMS causality, or production benefit from these traces.

## Remaining Evidence Needed

1. A deployable candidate-generation and scoring/deletion intervention evaluated without GT guidance.
2. Overall mask AP, AP50, and AP75 on an evaluation protocol appropriate to the claim.
3. Controlled causal experiments that isolate scorer, addition, deletion, ranking, and NMS effects.
4. Independent validation of semantic and annotation limits, plus replication beyond these two frozen traces.
