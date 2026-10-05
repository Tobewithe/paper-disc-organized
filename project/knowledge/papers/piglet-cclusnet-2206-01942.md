# Occlusion-Resistant Instance Segmentation of Piglets in Farrowing Pens Using Center Clustering Network

- Authors: Endai Huang et al.
- Year / venue: 2022; arXiv preprint
- Identifier: arXiv:2206.01942
- Official source: https://arxiv.org/abs/2206.01942
- Sources: DeepXiv; arXiv-related search
- Verification: arXiv metadata verified; reported metric values require full-text audit before citation.

## Evidence extraction

- Problem: piglets in farrowing pens with occlusion, repeated detections, wrong associations and incomplete detections.
- Mechanism: center prediction followed by DBSCAN clustering and C2M/RC2M conversion to instance masks.
- Training/inference location: center representation and post-prediction clustering, rather than query-level global consistency.
- Data/metrics: abstract reports 4,600 images and mAP 84.1; treat as author-reported pending full-text verification.
- Limitation: domain-specific center clustering; no evidence that it resolves complete/local low-overlap candidate co-survival in YOLO26-seg.
- Relation: strongest pig-domain precedent and a useful contrast class for mechanism comparison.
