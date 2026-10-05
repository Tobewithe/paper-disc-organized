# Crop support mechanism probe (S077)

Frozen COCO train2017 readout cache; fit threshold is selected only on fit images and evaluated on disjoint transfer images.

- Device: `cuda`; targets: fit 7811, transfer 1815
- Heuristic: choose crop scale 0.9 when raw 20% border positive fraction >= 0.322884, otherwise 1.0.

## Transfer means
- all: IoU@1.0 0.8163; fixed 1.2 0.7936; oracle gain 0.0105 [0.0088, 0.0123]; heuristic gain 0.0012.
- high: IoU@1.0 0.7259; fixed 1.2 0.6943; oracle gain 0.0149 [0.0106, 0.0195]; heuristic gain -0.0008.
- low: IoU@1.0 0.8190; fixed 1.2 0.7968; oracle gain 0.0102 [0.0086, 0.0120]; heuristic gain 0.0014.

Interpretation is restricted to crop sensitivity and an inference-visible border statistic; oracle gain is not a deployable method result.
