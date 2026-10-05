# YOLO26 FaroPigSeg Full Same-Forward Trace Audit (2026-09-05)

## Scope

Audited the completed trace at `C:\Dpan\codexproject\pigcv_research\artifacts\inference_cache\yolo26seg_trace_20260905_1945_faro_full` without changing its code or artifacts. The run used the 160-image FaroPigSeg test manifest, `yolo26_task05_final.pt`, `imgsz=1024`, `conf=0.05`, `max_det=300`, `retina_masks=true`, and local Ultralytics 8.4.100 on `cuda:0`.

## Deterministic Checks

| Check | Result |
| --- | --- |
| `run_status.json` and `summary.json` | completed |
| Manifest IDs / JSONL rows / raw NPZ files | 160 / 160 / 160; exact one-to-one coverage |
| Final prediction records | summary 3,484; independently counted 3,484 |
| Final mask parity records | 3,484 / 3,484 have `xor_pixels=0` and `parity=true` |
| Parity construction | all use `same_forward_process_mask_native_output` on `cuda:0` |
| Framework raw-input match | 160 / 160 calls have `match=true`, coefficient delta 0.0, box delta 0.0 |
| Framework-to-source trace | every `framework_mask_row` resolves to its recorded source candidate ID |
| Required NPZ audit fields | present in every file; framework coefficient/box/source lengths and JSONL references agree |
| Checkpoint key audit | 0 missing keys, 0 unexpected keys |
| Checkpoint and manifest provenance | current SHA-256 values match the run's recorded hashes |
| Artifact manifest | 164 expected files, 164 actual files, 0 missing, 0 extra, 0 SHA-256 mismatches |

Required NPZ fields checked: raw candidate IDs, boxes, scores, coefficients, prototype, Top-K fields, candidate-stage fields, feature/grid/stride fields, and `framework_coefficients`, `framework_boxes_xyxy`, `framework_source_candidate_id`, and `framework_shape`.

## Interpretation Boundary

This audit establishes internal trace consistency: the final COCO RLE masks equal the masks returned by the framework's captured native-mask call in the same forward pass, and the captured framework inputs resolve back to the recorded raw candidates. It does not measure AP, recall, failure rates, or any improvement over another model or run.

The audit does not inspect or adjudicate FaroPigSeg annotation semantics. It must not be read as a claim that a prediction is correct relative to real GT, that a failure mechanism is causal, or that any method improves performance. The provenance check verifies the recorded local checkpoint, manifest, selected image hashes as recorded by the run, and artifact integrity; it does not reproduce the original remote runtime, which the run configuration records as Ultralytics 8.4.27 while this trace used local 8.4.100.
