# Deterministic Asset Provenance Check

Date: 2026-09-05
Verifier: deterministic PowerShell existence, row-count, SHA256, and JSON-summary check

## Result

All required diagnostic tables, the diagnostic summary, and the available YOLO26-seg checkpoints were found. The table row counts and summary counts agree for the recovered 1024-square diagnostic cache. This verifies asset availability and integrity only; it does not accept the Mechanism Gate or establish causality.

## Recovered diagnostic artifacts

| Artifact | Exists | Data rows | SHA256 |
|---|---:|---:|---|
| `C:\Dpan\codexproject\pigcv_research\artifacts\analysis\yolo26seg_diagnostic_full_single_forward_20260901_final02\gt_analysis.csv` | yes | 7,374 | `890E8D7E2FFDD62D797D85D79672C24A43EFD3CA7F652DFC770E76715CF9D657` |
| `C:\Dpan\codexproject\pigcv_research\artifacts\analysis\yolo26seg_diagnostic_full_single_forward_20260901_final02\o_subtype_analysis.csv` | yes | 675 | `A9607BBB701B55249DB2AD8B930CE74E92DF53D719382CD2BEA023CA117B6C51` |
| `C:\Dpan\codexproject\pigcv_research\artifacts\analysis\yolo26_split_candidate_competition_20260902_v6\split_o_candidate_competition_gt.csv` | yes | 100 | `B5B8186B4BED43B7E4E67EC6A2EA28D4868B81055A2C3A1DC28F7C1E7C2AE93A` |
| `C:\Dpan\codexproject\pigcv_research\artifacts\analysis\yolo26seg_diagnostic_full_single_forward_20260901_final02\summary.json` | yes | JSON | `7AD90B7A1BF7105264680FB9B638FAAA2E7983A9B504DE561F5FDB65CEA1B14B` |

## Summary invariants

- Images: 918
- GT instances: 7,374
- Predictions: 9,628
- Total failure rate: 0.1983997830
- Relation failure rate: 0.1105234608
- Miss rate: 0.0363439110
- Primary counts: C=5,911, I=253, L=113, S=14, O=675, M=50, X=90, MISS=268
- `cache_integrity_passed`: true

## Available YOLO26-seg checkpoints

| Checkpoint | Exists | Bytes | SHA256 |
|---|---:|---:|---|
| `C:\Dpan\document\model_datasets\models\source-assets\pigvit\yolo\yolo26l-seg.pt` | yes | 63,700,037 | `636024306410AFA1732692322FBA57D22EA2B1C2F07613FCEE131A93D7DD380C` |
| `C:\Dpan\document\model_datasets\models\source-assets\pigcv-task05\yolo26m-seg.pt` | yes | 54,750,385 | `16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5` |

## Boundary retained

The archived 118-row formal sample manifest was not recovered. The 1280/rect replay therefore remains audit-only and must not be merged with the valid 1024-square descriptive cache. A new formal candidate comparison must generate and retain its own provenance-complete sample manifest.
