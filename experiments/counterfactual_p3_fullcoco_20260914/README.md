# Full COCO seed-0 controlled-intervention-guided P3 rescue screen

This experiment tests the leading mechanism-derived method on all COCO2017 train images. The baseline and method start from the same official YOLO26m-seg COCO checkpoint and inherit every training argument that Ultralytics 8.4.100 can execute from that checkpoint. Runtime capacity settings are paired.

The first decision is made after one seed and one full epoch. Direct P3 geometry repair and failure-specific downstream recovery are the mechanism endpoints. Official COCO AP/AR measure the overall benefit or cost. A useful gain does not require every reported metric to rise.

The paired run uses batch 2 and workers 8. A batch-4 launch was stopped at 72/29,572 iterations because the baseline alone reached 19.6 GB on dense batches, leaving insufficient headroom for the method's frozen teacher. No result from that aborted launch is used.

The method uses ground truth only during training to construct a privileged teacher view through controlled input interventions: proximal annotated instances are attenuated and target contrast is increased. Validation and inference use the stock model path without ground truth or a second pass. These edits measure and exploit model responses to controlled perturbations; they do not define a structural causal model or a strict counterfactual causal effect.

For memory safety, the frozen teacher runs before the student's differentiable forward pass. It emits only the selected P3 index and teacher IoU for each chosen target; its feature tensors are released before student activations are created. This is algebraically the same rescue gate and auxiliary target as the pilot implementation.
