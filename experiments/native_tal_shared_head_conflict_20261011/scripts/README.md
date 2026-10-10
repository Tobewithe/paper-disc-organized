# Script plan: native TAL shared-head conflict audit

本目录的正式脚本只做冻结 tensor 读数，不训练模型，也不重新求 shared affine 参数。实现为 `native_conflict_audit.py`，独立核验脚本为 `verify_conflict.py`，在笔记本 `C:\Users\28358\anaconda3\envs\pytorch\python.exe` 下运行。

## Reuse and do-not-reuse

- 复用 `experiments/coefficient_official_tal_affine_20260930/scripts/official_tal_solve.py` 的 `load_rows` 逻辑，确保 prototype 上采样、`crop_mask` 支持画布、owner 标签和 `seg_gain/area` 与官方 cache 一致。
- 复用 official cache 的 `INDEX.json` 和既有 split 清单；不要从原始模型重新做 TAL assignment。
- 复用 `coefficient_official_tal_affine_20260930/PER_CANDIDATE.jsonl` 的既有 oracle/IoU 记录，只按冻结身份连接。
- 不要复用 `coefficient_gradient_conflict_20260924/scripts/gradient_conflict.py` 作为结果脚本：它计算的是 h-only MLP probe head 梯度，不是官方 `one2one_cv4` 原生末层。

## Planned CLI

```text
native_conflict_audit.py \
  --vendor D:/coco_wire/vendor_8.4.100 \
  --weights D:/coco_wire/models/yolo26m-seg.pt \
  --cache D:/coco_wire/data/official_tal_affine_20260930/runs/official_cache \
  --oracle <immutable PER_CANDIDATE.jsonl> \
  --out D:/coco_wire/experiments/native_tal_shared_head_conflict_20261011/runs/RUN_CONFLICT_AUDIT_S0 \
  --max-images 40
```

The 40-image panel is the frozen first attempt. The actual formal run is 60 images per split because the 40-image `val` panel had only 49 recoverable candidates across 18 images, below the protocol's effective-count gate; the expansion reuses the same deterministic ranking and adds no training or data curve.

Required functions:

1. `select_images(index, split, n, seed_tag)` — SHA256-ranked deterministic image selection, write manifest and hash.
2. `load_official_rows(cache, manifest)` — return `h`, `coeff`, `p`, `y`, `area`, level, and identity without a forward pass.
3. `native_bce_head_gradient(row)` — compute `gc = gain/area * p.T @ (sigmoid(p@c)-y)` in FP64, then concatenate `vec(gc[:,None]*h[None,:])` and `gc`.
4. `match_cross_image_pairs(rows)` — fixed 4×4 quantile cells of `||h||` and `||c||`, same level, greedy different-image matching, save thresholds and pair manifest. Do not use target boxes or GT fields for matching.
5. `aggregate_instance_conflict(pairs)` — derive per-instance same/cross means, negative fraction and negative mass; join existing `iou_A/iou_C/objective_A/objective_C`.
6. `bootstrap_candidate_and_image(...)` — B=2000, seed fixed; candidate bootstrap resamples per-instance summaries, image bootstrap resamples image clusters.
7. `verify(...)` — check official identities, finite gradients, exact pair manifest, autograd smoke parity, and deterministic bootstrap before writing `DECISION.json`.

The script must fail closed if a candidate is missing from either cache or the existing oracle record, if a level/head shape is not `32×64`, if `branch` is not `one2one`, or if the pair matcher falls back to an unregistered rule. It must never silently drop a candidate based on IoU, oracle gap, conflict, score, or class.
