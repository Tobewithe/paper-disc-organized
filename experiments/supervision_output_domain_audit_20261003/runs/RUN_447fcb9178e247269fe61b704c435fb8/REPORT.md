# Frozen supervision/output domain audit

Same fixed A/N/P/PC epoch-3 outputs; no training, assignment changes or calibration. Current 640 label projection uses nearest, exactly as the prior metric implementation.

| Arm − A | Original macro IoU (pp) | Official BCE Δ | COCO-label BCE Δ | Polygon BCE Δ | Removed TP on conflict (%) [image-bootstrap CI] | Repair/damage |
|---|---:|---:|---:|---:|---:|---:|
| N | +0.2841 | -0.154524 | -0.004554 | -0.017560 | 1.61 [1.61, 1.61] | 0/0 |
| P | -1.3525 | +0.094838 | +0.376973 | +0.204224 | 15.38 [0.00, 15.48] | 0/0 |
| PC | -1.2406 | +0.268436 | +0.248369 | +0.098679 | 22.60 [0.00, 22.92] | 0/0 |

Scope and interpretation:

- T is official overlap ownership; D is the SAME converted polygons rasterized independently before overlap sorting; G is original COCO projected to the fixed 640 grid; U is the GT training box, V the predicted output box.
- Pixel edit counts partition V into shared foreground, missing-overlap, missing-polygon, untrained foreground, extra foreground, shared background and untrained background. GT-box-exterior pixels are unsupervised, not negative training labels.
- BCE comparisons use the identical logits and U/area/gain. Original-image IoU uses the unchanged official decoder and original annToMask, and is checked against prior per-candidate rows.
- Exact identity checked: BCE_off − BCE_COCO = sum(weight * (G−T) * logit); it separates label effects on the scalar objective, not the causal effect of training with different labels.
- The >=50% removed-TP conflict share plus official macro BCE down / COCO macro BCE up is a predeclared triage rule, not a universal scientific threshold. Failing it does not prove labels irrelevant. Always inspect region rates, area and boundary/confidence strata.
- These previously viewed development images provide exploratory explanation only. No new blind test, AP, model, optimizer, gate, or automatic label retraining. Local-view outputs are not included because that intervention did not train on these labels.
- S031–S033 already studied label semantics. This audit only connects the CURRENT official candidate identities and frozen updates to those hypotheses; no old-version numerical results are transferred.

N: label_conflict_not_established_as_primary_direct_damage_explanation
P: label_conflict_not_established_as_primary_direct_damage_explanation
PC: label_conflict_not_established_as_primary_direct_damage_explanation