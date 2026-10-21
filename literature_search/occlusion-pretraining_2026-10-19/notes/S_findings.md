# Snowball stage: supervision, synthetic-video and temporal-pretraining evidence

## What the citation paths actually establish

This pass follows documented seed-paper related work and official project pages, rather than assuming that any topically similar paper is a citation.

| Seed path | New source verified | What it establishes for this project |
|---|---|---|
| TCOW official page → Kubric | *Kubric* (CVPR 2022) | A physically rendered generator can yield synchronized target, occluder and temporal supervision. TCOW also reports real-world evaluation, but it does not establish detector transfer to YOLO. |
| S-AModal related work → SaVos, CoTracker, TAPIR, TAP-Vid | *SaVos* (NeurIPS 2022), *CoTracker* (ECCV 2024), *TAPIR* and *TAP-Vid* | Visible masks, motion and temporal correspondence have established use as supervision or memory input. They do not remove the need to evaluate identity-level reappearance separately. |
| S-AModal related work → PointOdyssey | *PointOdyssey* (ICCV 2023) | A concrete generation pattern: long clips, motion/appearance/camera randomization, RGB/depth/instance masks/2D+3D trajectories/visibility labels. This is directly reusable as a data-contract template, with animal assets replacing its subjects. |
| SAMBA and Amodal SAM related work → pix2gestalt | *pix2gestalt* (CVPR 2024) | Synthetic pairs of occluded observations and whole objects already train a conditional model for amodal completion. Pair supervision alone cannot be claimed new. |

## SAMBA: exact overlap and gap

**Verified canonical metadata.** Zhaochen Liu, Limeng Qiao, Xiangxiang Chu, Lin Ma and Tingting Jiang, *Towards Efficient Foundation Model for Zero-shot Amodal Segmentation*, CVPR 2025, pp. 20254–20264, DOI `10.1109/CVPR52734.2025.01886`. The official CVF paper calls the method **SAMBA**.

**What SAMBA already does.** It has a separation-to-fusion structure that jointly learns modal and amodal segmentation, uses target/principal focusing, and assigns mixtures of experts according to occlusion rates. Therefore, the following are already occupied as standalone novelty claims:

- splitting visible and complete-object pathways;
- supervising modal and amodal outputs together;
- making architecture or loss depend on occlusion-ratio strata;
- calling the resulting model a SAM-based zero-shot amodal foundation model.

**What it does not establish from the verified source.** The accessible official record is an image amodal-segmentation paper. It does not verify a causal video pretraining objective with identity-preserving reappearance, physically generated animal trajectories, or transfer into a YOLO detector/segmenter under matched training budgets. Those remain hypotheses requiring an experiment.

**Code status.** Targeted searches found the official CVF paper and the CVPR citation/DOI. They did not find an author or organization GitHub repository for SAMBA. This must be phrased as “no official code repository located in this search,” not “SAMBA has no code.”

## Automatable labels: literature-supported design boundary

The user clarified that scenes are self-built and labels are automatically generated. The cited work supports this direction directly:

- **SAIL-VOS** obtains modal/amodal masks by engine-side object-visibility toggling and depth/stencil buffers; its public layout keeps persistent IDs, RGB, visible masks and amodal masks.
- **PointOdyssey** releases RGB, depth, normals, instance masks, 2D/3D trajectories, camera parameters and visibility labels from generated long sequences.
- **TCOW** uses Kubric to obtain accurate target, occluder and container labels in synthetic videos.
- **Amodal Optical Flow** documents a multi-layer flow target covering visible and occluded regions; its project page also documents AmodalSynthDrive's multi-modal synthetic video labels.
- **Amodal SAM** (arXiv only) supplies an alternate, weaker-geometry route: paste a selected occluder on a target and derive complete/visible/occlusion masks automatically. It is useful as a matched 2D-paste control, not as evidence that compositing is physically realistic.

This means no manual annotation is required to create the core synthetic training labels. Manual work is still required if the project claims real-video accuracy, because generated labels cannot validate real-video occlusion or identity recovery.

## Strongest implication for training design

The literature supports generating each target identity over a controlled trajectory and emitting **visible mask, full in-frame mask, occluder/occlusion mask, depth order, 2D motion/trajectory, visibility and persistent ID**. It also supports temporally consistent randomization of lighting, materials, scene, camera and motion. The discriminating experimental question is not whether these labels can be generated; it is whether a representation trained with them improves held-out real occlusion slices over a matched synthetic-augmentation baseline.

For method scope, a defensible separation is:

1. Use automatic geometry labels to train a target-conditioned visual encoder on visible/full masks and calibrated occlusion proportion.
2. Add a distinct causal temporal correspondence/reappearance objective only if the model must preserve identity across a gap.
3. Treat point-track visibility/position as an auxiliary temporal measurement, not proof of a correct object identity after occlusion.
4. Compare with image-only paired supervision and matched synthetic augmentation before claiming the temporal or physically consistent component is causal.

## Important limits and conflicts

- **Amodal SAM** is dated 2026-04-22 and is in range by date, but only an arXiv version was verified. Its TAOS pipeline already produces visible, full and occlusion masks automatically from 2D mask compositing, and its regional-consistency loss aligns visible/occluded features. It is strong prior art against a generic “automatic synthetic occlusion plus paired-region consistency” contribution.
- **SAMEO** already adapts a SAM-style decoder to work behind detector boxes and trains on its 300K-image Amodal-LVIS extension. This is a direct baseline/interface precedent for detector-conditioned amodal masks, not a verified YOLO-transfer result.
- **DiffVAS** is a CVPR 2025 video-amodal published comparison. Diffusion-based video amodal completion is occupied.
- **CoTracker3** provides a useful recipe—synthetic Kubric training plus pseudo-labeling unannotated real video—but its official project notes that it still struggles on occluded-point position. It should not be cited as solved full-occlusion identity recovery.
- No source in this pass verifies that any of these pretraining schemes improves YOLO under an equal-data, equal-compute comparison. That remains the empirical gap.
