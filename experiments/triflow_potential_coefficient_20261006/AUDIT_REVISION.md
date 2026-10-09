# Observation refinement before the selected complete training

The initial formal Run RUN_TRIFLOW_TRAIN_S0 was intentionally interrupted with exact PID/source ownership guards; its partial trace and failure status are retained. Its parent RUN_TRIFLOW_PIPELINE_S0 settled failed and its one-shot scheduler was removed with a receipt. This was not an observed nonfinite-gradient or model instability failure.

The existing mask-task-only autograd probe originally reported whole parameter-group norms. The field-head group combines the two potential outputs with the stiffness output. Those group norms alone do not establish that the potentials received task gradients. The revision records the first two final-field-head output rows separately and requires finite nonzero potential and cross-attention norms in the same actual probe. The same autograd call is used; objective, sampling, optimizer, seed, backward count, method implementation and eight-epoch budget are unchanged.

Actual smoke comparison, both 32 images / 247 candidates / 69 applied updates: RUN_TRIFLOW_SMOKE_TRAIN_S0 and RUN_TRIFLOW_SMOKE_TRAIN_S0_R1 have exactly identical initial and final learned-state hashes. This is observed equivalence for this smoke, not a universal determinism guarantee. Initial hash: 5867b59007e8ff764659449f64473266549c940b7e621c7c3f21f59788071132. Final hash: 5f3c4751cd96132a26eeb586be37bca692b6f7c6f72061f8a32f275e1cd4987f.

The revised actual smoke probe at image_id 1146 recorded phi task-gradient L2=0.34488612876181546 and cross-attention task-gradient L2=0.020166610689935027, both finite and nonzero in the same probe, without warm-up. Sources: the two smoke TRAINING_AUDIT.json and revised TASK_GRADIENT_EVIDENCE.json, plus the interrupted Run's INTERRUPTION.json and parent SCHEDULER_INTERRUPTED.json.

The selected formal training is RUN_TRIFLOW_TRAIN_S0_R1, initialized afresh from seed0, not resumed from the partial Run. The verified 796-image frozen cache RUN_TRIFLOW_FIT_CACHE_S0 is reused unchanged. The new owned one-shot queue is RUN_TRIFLOW_PIPELINE_S0_R1. Full evaluation remains RUN_TRIFLOW_COCO5000_EVAL_S0; completion and AP are unknown until actual results return.
