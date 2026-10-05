# Fixed-action coverage-cost decomposition

This code implements one prespecified comparison under
`../../PROTOCOL_COVERAGE_V7_20260920.json`. It does not train or change YOLO.

## Inputs and action

The original prediction bank supplies stable `(image_id, candidate_index)` identities,
the unchanged scores/classes/boxes, official-zero RLEs, and the existing smooth
trial RLEs. Trial masks are nested in baseline masks. Empty trials restore the
baseline. The five deployment features are original predicted area, elongation,
extent, input-grid compactness, and the foreground fraction removed by that trial.
No annotations enter `calibrator.py` or its actual prediction adapter.

Original train2017 IDs are split into 1,500 fitting images and 500 selection images.
Fitting uses all available matched success and failure slots, without changing the
historical same-class Box50 matching. Unmatched predictions receive no regression
labels but are retained and processed during normal evaluation. This limitation
is distinct from the complete raw-candidate taxonomy.

## Models

- The capacity-matched direct regressor uses 300 seven-leaf HGB trees.
- Decomposition uses three 100-tree regressors for baseline purity, baseline
  coverage, and the true-target fraction among pixels removed by the fixed trial.
- Every head sees the same five features. All models use the declared seed,
  learning rate and regularization; there is no hyperparameter search.
- The original frozen RCMC is an additional reference with its original 100 trees.

With original predicted area A, estimated purity p, estimated coverage c, observed
removed fraction r, and estimated removed-pixel purity z, removed target area is
uA where `u = clip(z*r, max(0,p+r-1), min(p,r))`. Removed false area is `(r-u)A`.
Estimated GT area is `pA/c`. This gives the predicted before/after IoU and coverage
cost used in `decompose`. Ground-truth versions of these identities are checked
against all fitting rows before fitting. Component predictions are clipped to
physically feasible nested-mask counts; this is not evidence of calibration.

## Reproduction

Run on the recorded laptop environment. `common.setup` verifies actual imported
library versions, including the existing `D:/coco_wire/py` Ultralytics overlay.
The original bank and frozen RCMC paths are recorded in the protocol.

1. Run `fit_calibrator.py --protocol <protocol> --output <fresh Run>` through the
   recorded `runner.py`. Four JSON numeric tree models and matching joblib files
   are saved. Numeric inference must agree with fitted sklearn models to 1e-12.
2. Run `evaluate_calibrator.py --phase selection --protocol <protocol> --models
   <fit Run> --output <fresh Run>`. Decisions are selected only on the 500 train
   images, under the frozen RCMC damage budget and the fixed cutoff grid. Selection
   does not use val AP. `selection.json` freezes the choices.
3. Run the same script with `--phase evaluation --selection <selection.json>`.
   It evaluates all five policies on the original 4,500 val images with original
   full COCO GT. Baseline and frozen RCMC AP must match previous results. GT is
   used to evaluate the already-frozen decisions, not to construct deployment inputs.
4. `check_inference.py` replays the real YOLO adapters, verifies pixel/box/score/class
   correspondence, and measures synchronized batch-one full-pipeline costs.
5. `analyze_results.py` performs 2,000 paired complete-image resamples, reports
   continuous/repair/damage/cost metrics, and applies the prespecified decision rule.

Commands and snapshots are saved in individual Runs. `execution_runs.json` records
the original interrupted selection and its replacement. The fit is not repeated:
`resume_selection.ps1` reuses the same completed model Run after verification that
the old process was absent. Preserve the interrupted Run and its logs.

`launch_final_checks.ps1` waits for the exact evaluation Run to finish successfully,
then runs inference verification and analysis. It does not add another training
seed or select new policies. Shell sessions must remain connected until their
direct remote commands finish; a client-side subprocess timeout is not a valid
way to detach a Windows SSH process.

## Interpretation

The comparison asks whether this decomposition improves the repair/damage trade-off
over both matched direct regression and existing RCMC without lowering normal AP.
It does not promise a positive finding. Additional same-cutoff protection statistics
isolate the protection switch using already selected cutoffs; they are descriptive
and do not select another deployed method. Image-bootstrap intervals concern fixed
slot repair/damage/net recall, not AP. The validation data have prior research use.
No claim of unique upstream cause, new blind testing or general architectural
superiority follows from this one-seed comparison.
