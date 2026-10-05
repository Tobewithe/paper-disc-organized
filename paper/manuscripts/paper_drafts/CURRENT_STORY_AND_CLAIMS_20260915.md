---
type: manuscript-story
node_id: paper:current-story-20260915
stage: mechanism-supported-method-under-full-evaluation
supersedes_story: coefficient-contrastive-loss
---

# Current paper story: diagnose, localize, and selectively rescue neighbor-sensitive failures

**Full manuscript now available:** [English](../neighbor_sensitive_p3/PAPER_DRAFT.md), [中文](../neighbor_sensitive_p3/PAPER_DRAFT_ZH.md), [evidence and implementation mapping](../neighbor_sensitive_p3/EVIDENCE_MAP.md). The manuscript supplies the precise endpoint and training definitions; this file is its short story index.

## Working title

**From Localization Failures to Selective Supervision: Intervention-Guided P3 Learning for Small Objects**

## One-sentence claim

The examined COCO-pretrained segmenter contains small-object raw-geometry failures that respond to nearby-content edits. Recovery from a combined edit can be transferred through local stride-8 head inputs. A teacher-guided, location-conditional GT supervision objective improves the diagnosed representation in a short pilot without changing the inference graph.

## Story arc

1. **Start from an observable failure, not from density.** On COCO val2017, trace every GT through the one-to-many candidate lifecycle. Among 3,699 GTs with no final slot, 1,459 first fail because no raw candidate reaches Box IoU 0.5; 1,429 of those are small objects. Most losses therefore occur before NMS and before mask decoding.
2. **Describe the failed object with matched controls.** In 384 image-disjoint, exact-class, nearest-area failure/control pairs, failures have much worse local P3 geometry and centre error, weaker class evidence, lower target/background contrast, and greater nearby-object exposure.
3. **Separate association from intervention response.** The preselected exposed-neighbour edit gains +13.864 raw-P3 Box-IoU points over an identical-shape background edit, 95% CI [+11.674,+16.141]. Blur and distant-instance controls also show positive contrasts. These controls weaken a generic global-simplification explanation but do not match distance to the target; local texture and boundary explanations remain to be separated.
4. **Localize a useful response interface.** Feature transplantation using the separate joint-edit donor reproduces most recovery through local P3 (+0.1005 IoU), with smaller full-P4 (+0.0197) and full-P5 (+0.0045) effects. This identifies a head-input interface that carries recovery, not the unique upstream origin of the defect or a mask-coefficient mechanism.
5. **Convert the diagnosis into a train-time signal.** GT annotations construct a joint neighbor-weakening/target-contrast teacher view. The teacher first nominates a P3 location; auxiliary supervision activates when the student at that location is below Box50, the teacher is at least Box50, and the quality gap is at least 0.1. This does not require all student candidates to fail. The regression target remains the GT box. Inference remains the stock model.
6. **Evaluate the promised effect at three levels.** The mechanism endpoints are raw P3 Box IoU, centre error, Box50 transition, and failure-minus-control interaction. Final box/mask quality measures propagation. Official COCO AP is the deployment guardrail. The three-seed short pilot supports the mechanism endpoint; a full-COCO paired run is still in progress and will determine the final performance claim.

## Evidence that can already be stated

- Failure lifecycle and scale concentration are measured on all 36,335 ordinary COCO val2017 GT instances.
- The phenotype analysis uses 384 exact-class, nearest-area, image-disjoint matched pairs.
- The neighbour experiment changes zero original-resolution target pixels and preselects the neighbour without reading model outcomes; resizing and receptive fields can still mix nearby edits into target-adjacent inputs.
- On 302 eligible failures, neighbour flattening minus exact-shape background flattening improves raw P3 Box IoU by +13.864 points [11.674,16.141]; 105/302 satisfy the strict predeclared neighbour-specific recovery definition.
- The effect is present for both same-class and different-class neighbours. “Neighbor-sensitive localization” is supported; a same-class coefficient-assimilation explanation is not established.
- In three short-training seeds, selective supervision improves raw P3 Box IoU by +1.250 points [0.705,1.829] after averaging each instance across seeds; the failure-minus-control interaction is +1.496 [0.490,2.562]. Target metrics are GT-assisted candidate diagnostics, not official recall. Official best-checkpoint Mask AP averages −0.011 points. The manuscript discloses the seed-1 interruption and possible teacher refresh during resume.

## Claims that wait for the full COCO run

- A positive official COCO Mask/Box AP improvement.
- The final size of the recoverable-failure reduction on the full training budget.
- Any efficiency statement beyond an unchanged inference graph; training overhead must be measured separately.
- Generalization beyond YOLO26m-seg or COCO.

## Claims to remove from the old draft

- CCL or coefficient orthogonalization as the paper's method.
- ICI as a standalone contribution or the primary sample definition.
- “Feature assimilation,” “mask leakage cure,” “box-dependency facade,” or any claim that cropping is deceptive.
- Strict causal or counterfactual terminology.
- State-of-the-art, universal dense-scene, and coefficient-to-mask causal claims.

## Result table layout

The paper should present results in this order:

1. all-GT lifecycle taxonomy;
2. failure/control phenotype table;
3. controlled intervention and matched-control table;
4. P3/P4/P5 feature-transplant table;
5. preliminary recipe comparison: broad versus selective supervision also changes loss weight, so it is not an isolated gate ablation;
6. full COCO official metrics plus failure-specific mechanism metrics;
7. limitations and transfer tests.

The diagnostic tables establish why the method exists. The final COCO table determines how strongly the paper can claim practical improvement.
