# B / Semantic Scholar: implementation findings and limitation

## Retrieval status

Semantic Scholar Graph API search queries and the direct seed request for TCOW (`ARXIV:2305.03052`) both returned **HTTP 429 Too Many Requests** on 2026-10-19.  The Semantic Scholar web search route was unavailable in this environment.  Therefore this bucket is a transparent *primary-source fallback*, not a successful Semantic Scholar corpus or citation snowball.  It contains ten high-relevance entries whose bibliographic and technical fields were checked against CVPR/ICCV Open Access, ML Anthology, or arXiv.  It contains no Semantic Scholar citation count, reference, or cited-by claim.

The fallback directly addresses the user’s operating constraint: scenes and objects are self-built, so all supervision should be exported automatically from the scene assets rather than manually annotated.

## Implementation-oriented evidence

1. **Automatic occlusion labels are already feasible without human annotation.**  SAIL-VOS generates GTA-V video with visible and amodal pixel masks, semantic labels and instance identities (about 1.8M objects).  TCOW further shows why a physical scene generator is useful: it can export full per-object X-ray masks and identify the frontmost occluder and outermost container.  For the project’s own renderer, this supports rendering at least: full-scene visible instance mask, target-alone projection/amodal mask, persistent object ID, depth/order, and frontmost occluder ID/mask.  [SAIL-VOS, CVPR 2019; TCOW, arXiv:2305.03052]

2. **A paired clean/occluded image can be exported as a counterfactual, not hand-composed.**  WALT automatically mines an unoccluded object then creates object/occluder/occluded layers through composition.  In a self-built 3D scene, an analogous but stronger construction is available: retain target pose, camera and illumination, render once with the foreground occluder and once with that occluder excluded.  The resulting pair yields visible and hidden regions from rendering geometry.  The first clause is published evidence; the second is an implementation inference from having scene control, not a result reported by WALT.  [WALT, CVPR 2022]

3. **A full-occlusion target needs an explicit temporal state, rather than only a framewise amodal mask.**  Learning to Track with Object Permanence uses recurrent memory and synthetic invisible-object ground truth before joint synthetic/real training.  TCOW makes the responsible occluder or container an output when a hidden target cannot be localized from image evidence alone.  This supports training a causal state with at least `visible / partially-occluded / fully-occluded / no-supported-recovery` labels and a competing-object or occluder relation.  The final label schema is a recommended design choice, not a claim from either paper.  [Tokmakov et al., ICCV 2021; TCOW, arXiv:2305.03052]

4. **Synthetic visibility supervision and real-video adaptation should be separated.**  CoTracker3 keeps a visibility/confidence head frozen while adapting trajectory coordinates with pseudo-labels on real video, because pseudo visibility labels are unreliable.  This is direct negative evidence against automatically treating detector confidence or teacher tracker output as real-video occlusion ground truth.  A proposed transfer pipeline should pretrain visibility/occlusion on exact rendered labels and only adapt real video with reliable correspondence/consistency objectives unless a verified real visibility set is introduced.  [CoTracker3, arXiv:2410.11831]

5. **A detector interface is documented, but downstream YOLO improvement is not.**  SAMEO accepts conventional detector boxes (including RTMDet-style front ends) to prompt an amodal decoder and trains clean/occluded pairs.  This supports a detachable auxiliary head after a YOLO box/segmentation result.  It does *not* show that occlusion pretraining transfers into YOLO detector weights or rescues detector misses.  The literature retrieved here contains no matched, downstream YOLO-transfer result; that must be validated rather than stated as established.  [SAMEO, arXiv:2503.06261]

6. **The closest overlap rules out several weak novelty claims.**  Same-object clean/occluded pairing is present in WALT/SAMEO; target/occluder masks are present in TCOW; synthetic invisible-object supervision plus real-video joint training is present in Object Permanence; amodal SAM/SAM2 adaptation with target-aware synthesis and temporal prior is claimed by Amodal SAM.  A contribution cannot rest on any one of these pieces alone.  A narrower possible test is whether physically derived *same-versus-competing-individual* relation supervision improves identity recovery after a real occlusion under a matched-capacity downstream control.  This is a proposed hypothesis, not supported yet by an experiment.

7. **Diffusion methods are useful comparison points but not an efficient first transfer baseline.**  DiffVAS needs two video-diffusion stages, and TABE chains visible segmentation, monocular depth, video outpainting and a second segmentation pass.  Both target hidden-region completion, whereas a YOLO-side identity recovery system needs causal, low-latency state and uncertainty.  They are suitable for a high-capacity completion comparator or qualitative oracle, but no evidence here says they are the right pretraining backbone for detector transfer.  [DiffVAS, CVPR 2025; TABE, arXiv:2411.19210]

## Gap map for the project

| Question | Closest evidence | What remains unverified for this project |
|---|---|---|
| Can built scenes export exact labels with no annotation? | SAIL-VOS, TCOW | The project’s renderer must confirm target-alone/full-scene ID consistency and depth-order extraction on its assets. |
| Does occlusion-only training improve post-gap identity? | Object Permanence, TCOW | No retrieved work proves it for visually similar animals or for this detector and camera setting. |
| Can a reusable pretraining model plug into YOLO? | SAMEO box-to-decoder interface | No matched study here proves detector-weight transfer or segmentation AP gain for YOLO. |
| How should real video be used without IDs/occlusion labels? | CoTracker3 | Need a protocol separating exact rendered visibility supervision from pseudo-labelled trajectory adaptation. |
| Is paired target/occluder rendering novel? | WALT, SAMEO, TCOW, Amodal SAM | No. Any novelty must be in a testable relation/state mechanism and a downstream recovery result. |

## Unresolved literature work

- Retry Semantic Scholar with a key or after the rate window resets to obtain actual citation relations and citation counts; no citation-network conclusion can be made from this bucket.
- Verify whether WALT’s reported detector/segmentation gains use a matched backbone/parameter budget before using it as a direct baseline.
- Retrieve full experimental protocols for SAIL-VOS and Object Permanence if they become candidate baselines, including renderer output conventions and split leakage controls.
- Search a dedicated detector-transfer corpus for a matched *YOLO* downstream fine-tuning or distillation study; none was verified in this bucket.
