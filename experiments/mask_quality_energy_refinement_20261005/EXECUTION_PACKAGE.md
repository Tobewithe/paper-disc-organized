# QCR Stage I remote execution package

This package is ready for the authorized CUDA server only. The desktop is used only for code, manifest creation, hidden transfer, and result collection.

## Sequence

1. Copy this experiment directory to `/root/mask_quality_energy_refinement_20261005` and verify `CODE_MANIFEST.sha256`.
2. Run the original one-to-one TAL cache preparation with `scripts/prepare_qcr_cache.py --config RUN_CONFIG.json --out runs/<run>/prepare`.
3. Run candidate selection with `scripts/select_candidates.py --config RUN_CONFIG.json --out assets/CANDIDATE_MANIFEST.json`; stop if the locked 20,000 Failure and 20,000 Success minima are not met.
4. Run the finite oracle only for the frozen Failure manifest with `scripts/build_oracle_selected.py`; no oracle is generated for DEV or FINAL.
5. Run a 2-image smoke for both arms before training. The smoke must verify finite losses, identity at zero correction, differentiable `c -> P c`, and immutable cache fingerprints.
6. Run Stage I seed 0 for `direct` and `quality`, each through the workbench `runner.py`, 3 FIT epochs, gradient accumulation 8 images, fixed final epoch checkpoint.
7. Evaluate both checkpoints on DEV, then prepare/evaluate the untouched COCO val2017 FINAL list. No DEV/FINAL quantity can alter `rho`, step sizes, epochs, or checkpoint choice.

## Locked scientific boundary

The original YOLO model is frozen. The only trainable objects are the new EvidenceEncoder and either Direct32 or QualityHead. Stage II is not part of this Run and may only be registered if the Stage I gate in `PROTOCOL.md` is passed.

## Failure handling

A failed preparation, oracle, smoke, training, or evaluation keeps its partial Run and log. It is not a scientific negative. Fix only an explicit execution fault in a separate retry Run.
