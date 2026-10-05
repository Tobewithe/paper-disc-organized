# Formal Manifest Audit

Date: 2026-09-05
Manifest: `research-wiki/formal_manifest_20260905.csv`
SHA256: `EC018E5135C7FE5D09D6C6AB8E6BBAA45F885CD2D3711C229DFF88B8CC6A4ADB`

## Inventory

| Dataset | Split | Image rows | Paired segmentation labels | Status |
|---|---|---:|---:|---|
| FaroPigSeg | test | 160 | 160 | eligible for protocol audit |
| BamaPig2D | eval | 332 | 1,148 instances in `annotations/eval_pig_cocostyle.json` | eligible via COCO polygon annotations; YOLO pose labels excluded |
| PigLife | public_test | 426 | 4,474 instances in `derived/task05_v1/pig_coco_test_task05_v1.json` | eligible for GT/prediction replay; image archive remains unextracted |

The CSV records every image discovered under the three supplied roots, its expected label path, existence flags, and eligibility status. PigLife rows are retained as an explicit unavailable-source record. BamaPig2D uses the verified COCO polygon annotation source; its pose-label files are not used for segmentation.

## Gate impact

This closes the manifest inventory step for all three datasets: FaroPigSeg (160 images/1,752 instances), BamaPig2D (332 images/1,148 instances via COCO polygons), and PigLife public_test (426 images/4,474 instances via the derived COCO test view). PigLife image files are still inside `Image/test.zip`, so existing prediction/cache replay is formal for GT alignment but fresh forward reproducibility requires extracting that archive.

## Correction log

The previous manifest scan treated the PigLife root as label-unavailable because it did not descend into `derived/task05_v1`. The corrected v2 manifest replaces those 1,705 inventory rows with the 426-image COCO test view and preserves the earlier Faro/Bama rows. No prediction, annotation, or model file was changed.
