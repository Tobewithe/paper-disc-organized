---
type: experiment
node_id: exp:yolo26-faropigseg-raw-trace-20260905
slug: yolo26-faropigseg-raw-trace-20260905
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "797 failed GT; raw mask-IoU>=0.50: 466/797=58.47%; Top-300: 388/797=48.68%; 78 raw-good rows dropped by Top-300"
reasoning: "Existing provenance-complete FaroPigSeg same-forward cache replicates candidate availability above the 40% oracle threshold, but Top-300 loss among raw-good cases is 16.7%, below the preregistered 30% competition gate. Class and dataset differences remain descriptive until perturbation controls are run."
provenance: "C:\\Dpan\\codexproject\\pigcv_research\\artifacts\\inference_cache\\yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse\\faropigseg_test; experiments/yolo26_candidate_gate_20260905/faropigseg_raw_trace_candidate_coverage.csv"
added: 2026-09-05T20:20:00+08:00
---

# YOLO26 FaroPigSeg raw-candidate trace

## Result

The fixed YOLO26 runtime cache covers all 160 FaroPigSeg test images. Of 797 failed GT rows, 466 have a raw candidate with mask IoU >= 0.50 and 388 retain one in Top-300. The 78 raw-good rows removed by Top-300 are a measurable but sub-threshold selection-loss slice.

| Primary class | Failed GT | Raw good | Top-300 good |
|---|---:|---:|---:|
| I | 186 | 120 | 101 |
| L | 75 | 46 | 34 |
| M | 12 | 7 | 2 |
| MISS | 170 | 57 | 35 |
| O | 291 | 208 | 196 |
| S | 12 | 3 | 3 |
| X | 51 | 25 | 17 |

## Boundary

This is a same-forward candidate availability upper bound, not a deployable intervention. Both PigLife and Faro exceed the 40% oracle threshold, but the 30% ranking/competition gate is not met by the observed Top-300 loss; candidate removal/addition perturbations are still required before any scorer or postprocessor is considered.

## Artifacts

- `experiments/yolo26_candidate_gate_20260905/faropigseg_raw_trace_candidate_coverage.csv`
- `experiments/yolo26_candidate_gate_20260905/faropigseg_raw_trace_candidate_coverage.json`
