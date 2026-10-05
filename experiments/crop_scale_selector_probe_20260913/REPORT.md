# Inference-visible crop-scale selector probe (S079)

The selector is calibrated on the fit split with GT-derived per-scale IoU labels and evaluated on disjoint transfer targets. At transfer time it uses only raw border/inside response, candidate score, and predicted class.

- Fit targets: 7811; transfer targets: 1815
- Transfer baseline IoU@1.0: 0.771219
- Transfer selector IoU: 0.771343 (delta +0.000124)
- Transfer oracle delta: +0.015349

## Transfer subgroup deltas
- high: n=181, selector -0.001955, oracle +0.017415
- low: n=1634, selector +0.000354, oracle +0.015120
- support_sufficient_mask_bad: n=468, selector +0.007944, oracle +0.034932
- support_low_mask_bad: n=177, selector -0.006056, oracle +0.030809
- mask_good: n=1170, selector -0.002069, oracle +0.005177

Interpretation: a learned inference-visible selector is compared against the unchanged 1.0 crop. This result is not an AP claim and cannot use the GT-derived state at deployment.
