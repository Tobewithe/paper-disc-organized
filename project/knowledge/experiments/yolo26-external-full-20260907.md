---
type: experiment
node_id: exp:yolo26-external-full-20260907
title: "YOLO26 all-split Faro and Bama external baseline"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: high
date: "2026-09-07"
hardware: "RTX 5060 Ti; Conda pytorch; torch 2.9.1+cu128; local Ultralytics 8.4.100"
duration: "312.016 seconds summed inference loops; preparation, startup and evaluation excluded"
provenance: "experiments/yolo26_external_full_20260907_v1/summary.json; output_audit.json; historical_input_verification.json; duplicate_annotation_audit.json; research-wiki/yolo26_external_full_20260907.md. Coordinator deterministic verification, not independent method acceptance."
added: 2026-09-06T18:04:22Z
tags: ["yolo26", "external-baseline", "all-splits", "faropigseg", "bamapig2d", "diagnostic"]
---

# YOLO26 all-split Faro and Bama external baseline

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
4858 image records, 27745 GT, 45664 predictions. All-split mask AP: Faro .3980607253, Bama .6414921771. Non-C failure rates: 46.8321% and 23.3397%. Historical 492-image prediction lists and 2900 GT classes unchanged.

## Reasoning
Supports reproducible external baseline and persistence of the Faro gap across source splits. No deployable method or causal attribution established. Bama has 77 byte-identical pairs with annotation differences, including 11 cross-split pairs. All records retained. Method gates remain unchanged.

## Connections
Tests `idea:diagnosis-to-improvement` by extending the external baseline. Detailed report: `research-wiki/yolo26_external_full_20260907.md`. Original raw-candidate diagnosis remains limited to the previous 586 PigLife/Faro images.

## Duplicate Sensitivity

Under fixed minimum-ID and maximum-ID duplicate representatives, Bama AP is .6444825919 and .6444922376 on 3,263 image records. Excluding all duplicate groups gives .6475318762 on 3,186 records, versus .6414921771 on all 3,340. Failure rates are 23.1794%, 23.1439%, 22.9757% versus 23.3397%. These are alternative evaluation populations, not model improvements or independent replications.

Provenance: `experiments/yolo26_external_duplicate_sensitivity_20260907_v2/summary.json` and `input_sha256.json`; all three policies completed with eight source hashes unchanged. The partial v1 attempt is retained with failure.json after an evaluator input-mutation error; v2 isolates per-policy inputs.

