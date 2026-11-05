# Bucket C — Crossref + OpenAlex findings

## Scope and retrieval status

- Cutoff: 2026-10-19. Sources queried directly: Crossref Works API and OpenAlex Works API.
- OpenAlex returned ten items only for the first broad query before HTTP 429. Crossref returned HTTP 429 for three broad queries, partial results for three, and three successful exact-DOI resolutions. Details are in `raw/C_crossref_openalex.status.json`.
- This bucket is therefore a verified-metadata contribution, not an exhaustive source sweep. Citation counts are source snapshots and should not be compared across Crossref and OpenAlex.

## Strongest verified records by mechanism

| Mechanism | Paper | What the metadata supports | Project relevance |
|---|---|---|---|
| Physically generated multimodal data | Greff et al., *Kubric* (CVPR 2022), DOI 10.1109/cvpr52688.2022.00373 | A published scalable synthetic dataset generator with a CVPR venue and resolvable DOI. | Supports using a renderer to produce synchronized image, mask and motion labels. It does **not** by itself establish that the current low-poly animal generator will transfer to real video. |
| Video amodal representation | Fan et al., *Rethinking Amodal Video Segmentation…* (ICCV 2023), DOI 10.1109/iccv51070.2023.00123 | A peer-reviewed video-amodal paper whose canonical title explicitly uses object-centric representation. | Strong precedent for treating amodal/visible information as object-centric video supervision. Full text is required before copying its supervision or claiming the same contribution. |
| Motion through occlusion | Luz et al., *Amodal Optical Flow* (ICRA 2024), DOI 10.1109/icra57147.2024.10611314 | Published ICRA work with an arXiv full-text route. | Supports an additional training target for a future generated video corpus: object motion can be labelled beyond visible pixels. |
| Diffusion for video amodal segmentation | Chen, Ramanan, Khurana, *Using Diffusion Priors for Video Amodal Segmentation* (CVPR 2025), DOI 10.1109/cvpr52734.2025.02131 | Crossref directly resolves the published title, authors and CVPR 2025 venue. | This is the most relevant modern comparator for amodal video completion. It changes the novelty bar: diffusion-based video amodal segmentation is already published. |
| Earlier amodal baselines | Li & Malik (ECCV 2016); Zhu et al. (CVPR 2017); Mohan & Valada (CVPR 2022) | Canonical DOI/venue metadata for instance, semantic and panoptic amodal segmentation. | The project cannot claim amodal masks, visible-vs-hidden decomposition, or panoptic amodal labels as new by themselves. |

## Evidence for transfer to YOLO or detector training

This bucket found older detection-with-occlusion precedents (Gao et al., CVPR 2011; Brahmbhatt et al., BMVC 2015), but **no verified paper in the captured Crossref/OpenAlex hits demonstrating that an object-conditioned occlusion-pretrained encoder transfers reliably into YOLO detection/segmentation under matched downstream data and compute**. That is a research gap, not evidence that transfer will work.

For a credible project claim, downstream evaluation has to include: ordinary detector training, matched synthetic occlusion augmentation, and the proposed pretraining transfer; report occluded-object benefit and clean-object damage separately. A score only on the synthetic occlusion classifier does not establish detector transfer.

## Video augmentation and reappearance

The captured Crossref temporal-translation hit (Fan et al., IJCNN 2024) establishes only that temporally consistent unpaired translation exists. It is insufficient evidence to choose video translation as the main data strategy. The stronger design premise from this bucket is geometric consistency: render the same identity through controlled occluder trajectories and retain visible mask, full mask, occlusion order, depth and motion. Appearance randomization should preserve those labels and be assessed as a separate domain-gap intervention.

## Metadata conflicts and exclusions

- Do not cite `10.1109/iccv48922.2021.01420` as Crossref-verified: OpenAlex returned it for *Occlusion-Aware Video Object Inpainting*, while this bucket's exact Crossref request returned 404.
- OpenAlex surfaced 2026 arXiv records whose arXiv months fall after the 2026-10-19 cutoff. They are excluded.
- `TAO-Amodal`, `SAMEO`, `Amodal SAM`, and `Track Anything Behind Everything` were not resolved to canonical published metadata here. In particular, retain the known TAO-Amodal 880-vs-833 category/version ambiguity until the arXiv and official-site buckets reconcile it.
- SSRN records (`A2vis`, `Occlusion-Supervised Amodal Instance Segmentation`) remain discovery leads. They are not treated as peer-reviewed evidence.

## Recommendation to parent synthesis

The evidence supports a new, testable direction: use physically consistent synthetic video to train **separate visible/amodal/occlusion-order and motion targets**, then test whether their encoder improves an occlusion-sliced detector over matched augmentation. It does not support presenting generic 3D rendering, amodal mask prediction, diffusion completion, or use of a detector box as a novel contribution. The immediate literature step is to acquire/read Kubric, Fan et al. ICCV 2023, Luz et al. ICRA 2024, and DiffVAS, then resolve TAO-Amodal/SAMEO/TABE from primary pages.
