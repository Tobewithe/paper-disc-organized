# YOLO26-seg baseline replay and inherited failure evidence

Date: 2026-09-05  
ARIS runs: R002 (PigLife), R004 (BamaPig2D), with FaroPigSeg replication

## Provenance

- Formal manifest: `research-wiki/formal_manifest_20260905.csv` (SHA256 `EC018E5135C7FE5D09D6C6AB8E6BBAA45F885CD2D3711C229DFF88B8CC6A4ADB`).
- PigLife GT: `C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json` (426 images, 4,474 instances).
- Faro prediction: `C:\Dpan\document\model_datasets\artifacts\pigcv-task05\remote-runs\external-faropigseg-yolo-20260827-02\predictions.json` (3,989 predictions, validation valid=true).
- Bama prediction: `C:\Dpan\document\model_datasets\artifacts\pigcv-task05\remote-runs\external-bamapig2d-yolo-20260827-03\predictions.json` (1,340 predictions, validation valid=true).
- PigLife diagnostic cache: `C:\Dpan\codexproject\pigcv_research\artifacts\inference_cache\yolo26seg_diagnostic_20260901_imgsz1024_rectfalse` (918 images across three datasets, 9,628 final predictions, 19,740,672 raw candidates).

## Fixed inference protocol

YOLO26-seg checkpoint `yolo26_task05_final.pt`, score export threshold 0.05, IoU/NMS parameter 0.7, `imgsz=1024`, seed 20260809. External prediction validation confirms exact manifest coverage and prediction counts for Faro and Bama. PigLife replay uses the retained diagnostic cache; fresh forward requires extracting `Image/test.zip` and is intentionally not rerun here.

## Baseline result inherited from audited analyses

| Dataset | GT | Any non-C (YOLO) | Relation O/M/X | Geometry I/L/S | A1+A2 |
|---|---:|---:|---:|---:|---:|
| PigLife public_test | 4,474 | 5.34% | 1.90% | 1.14% | 2.30% |
| FaroPigSeg test | 1,752 | 42.29% | 12.61% | 17.24% | 12.44% |
| BamaPig2D eval | 1,148 | 23.43% | 5.57% | 4.79% | 13.07% |

These are descriptive baseline rates from the existing three-model failure analyses, not a new causal estimate. The external datasets show substantially higher YOLO failure than PigLife, while the failure-family mix differs by dataset.

## Mechanism evidence available for M2

- PigLife 1024 diagnostic cache: total failure 19.84%; relation failure 11.05%; miss 3.63%.
- O residual after Mask-NMS control: `LOCAL_REPEAT=65`, `SPLIT_TYPE=50`, `MULTI_PRED_COMPLEX=25`, `OTHER_UNRESOLVED=56`.
- SPLIT_TYPE candidate audit: 11/50 had no complete raw candidate, 8/50 had a complete candidate dropped before final output, and 31/50 had complete and local candidates co-surviving. The local-minus-complete score gap median was 0.185 in the co-survival group.
- Exact-runtime reconstructed union oracle is an upper bound only; it is not deployable evidence and does not close the mechanism gate.

## ARIS status

- R001: DONE (manifest/protocol).
- R002: DONE as cache replay; fresh PigLife forward remains a reproducibility follow-up because the test image archive is not extracted.
- R004: DONE (Bama COCO polygon GT and validated YOLO prediction asset).
- R003: RESOLVED from BLOCKED to replay-eligible after locating the PigLife derived COCO test view.
- R005-R007: evidence available, but R005/R006 remain diagnostic/audit-scoped until provenance-complete PigLife candidate traces are regenerated from the corrected manifest.

No training, model selection, loss selection, or contribution claim is made here.
