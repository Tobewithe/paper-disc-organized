# Asset Recovery and Gate Boundary

Date: 2026-09-05

## Recovered sources

The executable data/model root is `C:\Dpan\document\model_datasets` (the previously supplied `C:\Dpan\document\model\_datasets` path does not exist). The prior YOLO26 research project remains the authoritative source for the completed diagnostic artifacts:

`C:\Dpan\codexproject\pigcv_research`

The legacy report archive is:

`C:\Dpan\document\pigcv_reserch\03-成果与资产`

## Provenance-checked tables

| Artifact | Scope | SHA256 |
|---|---|---|
| `artifacts/analysis/yolo26seg_diagnostic_full_single_forward_20260901_final02/gt_analysis.csv` | 918 images, 7,374 GT | `890E8D7E2FFDD62D797D85D79672C24A43EFD3CA7F652DFC770E76715CF9D657` |
| `artifacts/analysis/yolo26seg_diagnostic_full_single_forward_20260901_final02/o_subtype_analysis.csv` | residual-O subtype analysis | `A9607BBB701B55249DB2AD8B930CE74E92DF53D719382CD2BEA023CA117B6C51` |
| `artifacts/analysis/yolo26_split_candidate_competition_20260902_v6/split_o_candidate_competition_gt.csv` | 50 SPLIT_TYPE rows and controls | `B5B8186B4BED43B7E4E67EC6A2EA28D4868B81055A2C3A1DC28F7C1E7C2AE93A` |
| `artifacts/analysis/yolo26_split_candidate_competition_20260902_v6/split_o_candidate_competition_detail.csv` | raw candidate detail for the exploratory comparison | `not recomputed in this record` |

The source files are read-only inputs; no legacy code, checkpoint, GT, or cache was modified.

## Evidence boundary

- The full single-forward cache is a valid descriptive 1024-square YOLO26 cache: 19.84% GT-level total failure, 11.05% relation failure, and 3.63% miss.
- Mask-NMS@0.80 changes 427/675 original O cases to C; residual O is heterogeneous (`LOCAL_REPEAT=65`, `SPLIT_TYPE=50`, `MULTI_PRED_COMPLEX=25`, `OTHER_UNRESOLVED=56`).
- The complete/local candidate comparison is an exploratory 1024-square result. The archived 118-row formal sample manifest was not recovered, so the reconstructed 1280/rect replay remains audit-only.
- No result here identifies a causal intervention or selects a model, loss, or paper contribution.

## Gate resolution

The former blocker was incorrectly phrased as “missing data.” Data, models, and derived GT/candidate tables are available. The remaining issue is provenance parity for the archived formal sample, which is a limitation to record, not a reason to restart the same cache-only analyses.

The next ARIS action is therefore a bounded `research-refine` pass using the existing evidence and literature, with the formal-sample limitation carried into the proposal and experiment plan. Any new formal candidate comparison must use a newly generated, provenance-complete sample manifest under the fixed YOLO26 protocol.
