# Boundary-aware spatial correction supervision (pilot)

## Question
7R shows that the frozen p-spatial correction misses a large portion of the oracle correction, with the largest recoverable component near the object boundary for small instances. This pilot tests whether the failure is partly caused by uniform pixel-logit supervision averaging boundary errors with interior pixels.

## Frozen object
Use the same official COCO-pretrained YOLO26m-seg / Ultralytics 8.4.100 cached coefficient inputs and finite-oracle targets as 7Q (lambda=0.003). The backbone, neck, prototypes, boxes, scores, and candidate set remain frozen. Only a small spatial correction predictor is trained.

## Arms
- `uniform`: 7Q spatial predictor and uniform normalized pixel-logit loss.
- `boundary_weighted`: same architecture, initialization, optimizer, data, and epoch budget; pixels in a 2-pixel GT boundary band within the predicted ROI receive weight 3 before per-instance normalization. The GT is used only to construct the training loss.

The primary comparison is held-out dev direction cosine and spatial effect error. This is a mechanism pilot, not a COCO AP claim. If boundary weighting helps only small objects and does not damage medium/large objects, proceed to a new independent val subset and full inference evaluation. If it does not improve held-out direction or boundary-region recovery, stop this branch.

## Split and leakage
Fit uses the existing 10,000 train2017 images from 7Q; dev uses the existing image-disjoint 200 train2017 images. No val2017 labels are used for training or arm selection. The reused 7Q 2,000-image val set is exploratory only and cannot be used to choose the method.

## Metrics
Report overall, Mask75-failure, BoxIoU>=.75 & MaskIoU<.75, and COCO small/medium/large strata:
- pixel-logit direction cosine against finite-oracle target;
- normalized spatial MSE;
- boundary/interior target-effect cosine;
- correction-induced original-image Mask IoU change, repair and damage counts when decoded.

The pilot is successful only if the weighted arm improves held-out boundary direction and does not trade it for a material interior or overall regression. No claim is made if only fit metrics improve.
