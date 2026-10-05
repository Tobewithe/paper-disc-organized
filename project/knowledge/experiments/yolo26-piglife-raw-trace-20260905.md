---
type: experiment
node_id: exp:yolo26-piglife-raw-trace-20260905
slug: yolo26-piglife-raw-trace-20260905
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: high
metrics: "394 failed GT; raw mask-IoU>=0.50: 388/394=98.48%; Top-300: 387/394=98.22%; I/L/M/X raw coverage 100%; MISS raw 8/13"
reasoning: "A provenance-complete same-forward trace over all 426 PigLife test images shows candidate availability is high across non-MISS failure classes and Top-300 loss is small. This rules against candidate absence as the dominant PigLife explanation, but oracle recovery remains an upper bound and does not establish ranking causality."
provenance: "experiments/yolo26_raw_trace_piglife_full_20260905_v1/summary.json; experiments/yolo26_candidate_gate_20260905/piglife_raw_trace_candidate_coverage.csv; C:\\Dpan\\document\\model_datasets\\datasets\\piglife\\derived\\task05_v1\\pig_coco_test_task05_v1.json"
added: 2026-09-05T20:05:00+08:00
---

# YOLO26 PigLife full raw-candidate trace

## Result

The fixed YOLO26 runtime processed all 426 PigLife test images (4,843 final predictions; zero checkpoint key mismatches). Among 394 failed GT rows from the aligned baseline, 388 had a raw candidate with mask IoU >= 0.50 and 387 retained such a candidate in Top-300.

| Primary class | Failed GT | Raw good | Top-300 good |
|---|---:|---:|---:|
| I | 15 | 15 | 15 |
| L | 29 | 28 | 28 |
| M | 20 | 20 | 20 |
| MISS | 13 | 8 | 7 |
| O | 289 | 289 | 289 |
| X | 28 | 28 | 28 |

## Boundary

This is a same-forward candidate availability/oracle upper bound using GT for audit only. It does not prove a ranking intervention will improve AP, and it does not close the two-dataset mechanism gate. FaroPigSeg replication and candidate removal/addition perturbations remain required before R008.

## Artifacts

- `experiments/yolo26_raw_trace_piglife_full_20260905_v1/summary.json`
- `experiments/yolo26_candidate_gate_20260905/piglife_raw_trace_candidate_coverage.csv`
- `experiments/yolo26_candidate_gate_20260905/piglife_raw_trace_candidate_coverage.json`
