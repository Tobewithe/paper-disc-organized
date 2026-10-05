---
type: paper
node_id: paper:zhang2026_topologyaware_query_selection
title: "Topology-Aware Query Selection for Surgical Instrument Instance Segmentation"
authors: ["Ze Zhang", "Yang Zhang"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2608.11607"
  doi: null
  s2: null
tags: ["instance-segmentation", "query-selection", "relational-reasoning", "fragmentation", "surgical-instruments"]
added: 2026-09-05T13:36:44Z
---

# Topology-Aware Query Selection for Surgical Instrument Instance Segmentation

## One-line thesis
Relational variable-cardinality selection over fixed Mask2Former mask candidates uses complete-graph geometry and exact subset optimization to construct coherent surgical instrument instance sets.

## Problem / Gap
Foreground overlap can remain favorable while the predicted instance set is wrong through duplicate, fragmented, merged, missed, or empty-frame outputs. The paper treats final mask-query selection as a variable-cardinality set problem.

## Method
Starting from a fixed Mask2Former output, the method retains nonempty candidates, builds a complete graph over candidates, and encodes overlap, containment, scale, score margin, boundary distance, and boundary contact. Relational message passing feeds candidate validity, pair-competition, and cardinality heads. An exact cardinality-constrained subset objective selects the final set. The graph path is compared with a node-feature-matched path that shares candidate inputs and selector machinery but omits edge geometry and graph messages.

## Key Results
On the sealed CholecInstanceSeg source test, all three discovery seeds satisfy the paper's complete criterion; instance F1 increases by 0.0504--0.0612 and positive-frame set-failure decreases by 0.0848--0.1060. Direct ROBUST-MIPS transfer satisfies the complete criterion for all three seeds. Endoscapes satisfies it for one of three seeds, so stable direct transfer is not established. The graph-versus-node comparison is a package comparison, not an isolated component ablation.

## Assumptions
The fixed segmenter supplies scored mask candidates and native instance identifiers are available for supervision/evaluation. The selector uses candidate-derived features rather than image pixels, temporal features, or query embeddings. Candidate count is bounded for the complete graph and exact solver.

## Limitations / Failure Modes
The method is evaluated on surgical-instrument Mask2Former candidates, not YOLO26 or pig masks. It does not establish stable transfer to every domain; Endoscapes is seed-unstable. The complete graph is quadratic and exact subset optimization has combinatorial worst-case cost. Edge geometry, message passing, and extra relational-path capacity are not separately identified.

## Reusable Ingredients
Use a fixed-candidate, set-level comparator when testing output-set construction: preserve candidate-wise inputs, cardinality handling, and evaluation contracts, then separate native-instance set metrics from foreground-union fidelity and safety checks. Treat relational selection as an integrated package unless component controls are added.

## Open Questions
Does a comparable relational selector help frozen YOLO26 candidates under pig-specific label contracts? Which candidate-stage losses are observable before mask matching, and can low-IoU complete/local cases be separated from ordinary duplicate control without importing surgical labels?

## Claims
Literature claim only: fixed-candidate relational set selection is an established near-neighbor for generic candidate-graph selector wording. No project claim or novelty clearance is inferred.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This paper creates high novelty risk for a generic relational candidate-selection contribution. Its fixed Mask2Former/surgical/native-instance setting is mechanistically adjacent but not evidence for pig-domain transfer, YOLO26 stage causality, or deployable AP gain.

## Abstract (original)

> Accurate foreground masks can still form an incorrect surgical-instrument instance set: duplicate, fragmented, merged, missed, or empty-frame predictions may preserve favorable pixel overlap while violating object identity and count. Final query selection is therefore a relational, variable-cardinality problem rather than a collection of independent candidate decisions. We evaluate topology-aware query selection, which represents the nonempty candidates of a fixed Mask2Former as a complete graph, learns relational candidate and pair representations, predicts set cardinality, and solves an exact structured subset problem. The formal comparison is the complete relational path versus a node-feature-matched path; it evaluates the combined effect of pairwise geometry, message passing, and the additional relational-path capacity, not an isolated component. On the sealed 22-case source test, all three discovery seeds supported instance-set performance improvement with segmentation fidelity and predefined technical-safety preservation: instance F1 increased by 0.0504--0.0612 and positive-frame set-failure rate decreased by 0.0848--0.1060. Direct ROBUST-MIPS transfer reproduced the complete result in all three seeds. Endoscapes supported only one of three seeds and therefore did not establish stable direct transfer. Taken together, the results support a bounded conclusion: the evaluated complete path improved coherent instance-set construction from fixed Mask2Former candidates in specified native-instance contracts, while stable cross-domain transfer and component-specific effects remain unestablished.

