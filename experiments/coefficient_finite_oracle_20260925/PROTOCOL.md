# Experiment 7D: finite regularized coefficient oracle

Reuse the 7A/7C sample of 200 official one-to-one positives, the original frozen COCO-pretrained YOLO26m-seg prototypes and coefficients, and the official GT-box BCE reproduced and gradient-checked in the source experiment. The old unregularized 100-step oracle is a reference only. No model weights are trained.

For each instance and each predeclared positive `lambda` in `{0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003}`, solve

`delta_lambda = argmin_delta [BCE_GTbox(P(c0+delta), y)/area + lambda*||delta||_2^2/2]`.

The quadratic centered at the original coefficient makes this objective strictly convex and coercive, hence the coefficient-space minimizer exists and is unique. `lambda` controls distance by penalty, **not** by a hard radius: record the achieved norm and the stationarity residual for each solve. A common penalty can yield different norms across instances. The solution is GT-dependent and only a diagnostic upper bound for what a network could infer.

First run 20 instances to verify the penalty grid, then run all 200 with unchanged grid. Record official GT-box BCE and, inside the original predicted box on the 640 training grid, mask IoU, target coverage, FPR and pixel ROC AUC. Keep these two support regions separate. Compare each bounded direction with the previous unregularized oracle in coefficient and pixel-response space; a small coefficient cosine need not mean a small pixel-response cosine.

Save the 32-dimensional `delta_lambda` vectors keyed by annotation ID and penalty in `TARGETS.pt`. This is a diagnostic target bank only; validation-image GT targets must not be used to train a predictor evaluated on those same images.

Numerical uniqueness check: for a deterministic subset of 40 original failures, solve `lambda=0.03` again from a different initialization, measuring coefficient and objective disagreement and stationarity. Data stability check: in the same subset, fit two independent stratified 80% pixel samples of each GT box, reweight each class to preserve its full pixel mass, and compare their directions with the full-data solution. This tests pixel-subsample stability, not annotation robustness or prediction from GT-free features. Both checks use a fixed seed and do not select a preferred checkpoint.

The 200 instances were deliberately balanced by original Mask75 failure status and do not represent COCO failure prevalence. No reported value is COCOeval AP or deployable model gain.
