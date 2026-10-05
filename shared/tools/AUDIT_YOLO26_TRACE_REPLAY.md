# YOLO26 Trace Replay Audit

This audit reads the immutable 20260905 PigLife and FaroPigSeg full traces. It writes only a new output directory containing `run_metadata.json`, `run_status.json`, one compact JSONL per trace, and `summary.json`.

Each image is forwarded once through `models.yolo26seg.diagnostic_inferencer._collect_raw_candidates` with the formal diagnostic configuration. The audit requires exact equality for retained raw tensors, Top-K IDs, final predictions and final-source mappings. It also rebuilds each old final mask from the old raw prototype, coefficient and box using `process_mask_native` in batches of two, then records aggregate XOR pixels, minimum IoU and pass counts at IoU thresholds 0.50 through 1.00.

The legacy NPZ files do not contain `framework_prototype`. The JSONL records this explicitly. The new forward still validates the framework prototype against `one2one.proto` using the diagnostic inferencer's exact-match gate; it cannot establish that missing historical field after the fact.

Run only when the GPU is available and the output directory is new or empty:

```powershell
C:\Dpan\envsfiles\CondaData\envs\pytorch\python.exe tools\audit_yolo26_trace_replay.py --output-dir C:\Dpan\codexproject\pigcv_research\artifacts\audit\yolo26_trace_replay_20260905
```

The command exits nonzero after writing evidence if any image fails a gate. It does not modify either source trace.
