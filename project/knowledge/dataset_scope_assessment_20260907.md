# Dataset Scope Assessment

Date: 2026-09-07. Updated status: all-split Faro/Bama inference was subsequently authorized and completed as R012. Recommendations for an additional dataset remain conditional.

## Recommendation

Update after user clarification, 2026-09-07: the active checkpoint was fine-tuned only on PigLife, corroborated by its archived training manifest and image root. The user authorized inference on all FaroPigSeg and BamaPig2D splits. Their original training splits are external inference data for this checkpoint. R012 therefore expands evaluation from 492 to 4,858 external images. Prior diagnostic inspection is distinct from training exposure. Prioritize this authorized expansion before spending effort on a fourth dataset; an additional independent confirmation source remains conditional.

Continue method development on the existing datasets. Conduct a bounded search for one additional, independently collected pig instance-segmentation dataset as a candidate for final external validation. Dataset count is not a substitute for method effectiveness, annotation compatibility, or independent evaluation. Adding a fourth dataset is conditional, not a publication requirement.

## Existing Evidence and Roles

| Dataset | Local evaluated subset | Current role | Remaining question |
|---|---|---|---|
| PigLife | 426 images / 4,474 GT | Full candidate and mask diagnosis | Video/group independence, dataset-version identity, and later method-development split |
| FaroPigSeg | All 1,518 images / 16,241 GT; raw diagnosis on 160 / 1,752 | Full-scope external baseline and current single-mask quality investigation | Occlusion annotation policy and pen/video separation |
| BamaPig2D | All 3,340 image records / 11,504 GT | Full-scope external baseline | 77 byte-identical pairs with annotation differences; group independence and relevant dense/contact failure strata |

R012 processed all Faro/Bama original splits with the fixed PigLife-finetuned model. Faro all-split AP is 39.81 and Bama 64.15; failure rates are 46.83% and 23.34%. Original 492-image predictions match exactly. Bama has about 3.44 annotated instances per image record across all splits; that arithmetic is not a crowding or occlusion metric. Density also depends on object size, contact and visibility. See `yolo26_external_full_20260907.md` for protocol, duplicate-label findings and evidence receipts.

The PigLife and Faro test subsets have informed repeated exploratory diagnosis. Bama baseline results have also been inspected. Existing results remain useful, but later work must disclose this history. A method should be trained/tuned on permitted development data, frozen before final evaluation, and tested on an independently held-out group or source where available. Merely shuffling already inspected images cannot restore an untouched test set.

## Primary Source Check

| Source | Verified information | Implication and boundary |
|---|---|---|
| Li et al., Promote computer vision applications in pig farming scenarios: High-quality dataset, fundamental models, and comparable performance, Journal of Integrative Agriculture; publisher record https://www.sciencedirect.com/science/article/pii/S2095311924003058 | Describes an image-random 80/20 train/test split | This is not evidence of farm/video-independent testing. Published counts and the current datasheet differ, so do not silently replace the local manifest with website counts. |
| PigLife official datasheet, https://data.aifarms.org/datasheet/piglife | Images from the same video can be similar; filename encodes contextual/sequential information; human masks checked by a second annotator | Useful basis for checking group splits, not proof that the project's current split separates groups. |
| Compte et al., Housed pig identification and tracking for precision livestock farming, Expert Systems with Applications 293 (2025), 128466; https://doi.org/10.1016/j.eswa.2025.128466 and https://github.com/yudong888/FaroPig-Datasets | FaroPigSeg has YOLO instance polygons; README specifies random image-level 70/20/10 splits and CC BY-NC 4.0 | Do not describe random image splits as unseen-pen validation. Segmentation occlusion policy remains unresolved. ReID masks and their SAM/CoTracker procedure are a separate dataset description. |
| An et al., Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL, Nature Communications (2023); https://doi.org/10.1038/s41467-023-43483-w and https://github.com/anl13/MAMMAL_datasets | Official COCO split: 3,008 training images / 10,356 instances; 332 evaluation images / 1,148 instances. Segmentation usage is documented at https://github.com/anl13/pig_silhouette_det | Bama is already an available source of segmentation annotations, not merely pose labels. Project evaluation uses the COCO polygons. |

## Admission Criteria for an Additional Dataset

1. Instance masks with documented human annotation or sufficiently documented human verification; boxes, keypoints, semantic foreground and unverified pseudo masks are not equivalent GT.
2. Enough dense/contact instances to test the eventual mechanism, with easier scenes retained to measure regressions. Fix inclusion criteria before observing method gains.
3. Independent acquisition and traceable video/session/pen groups; check both duplicate images and shared source footage against existing assets.
4. Annotation semantics compatible with the claim, especially visible versus completed occluded masks and image-border handling.
5. An accessible official release, usable research terms, and an affordable conversion/evaluation cost.
6. Keep any final held-out partition out of model selection; failures on it must be reported rather than used to choose another favorable test source.

If no suitable new dataset is available, first investigate genuinely uninspected groups within permitted existing data and document all prior exposure. This is a conditional fallback, not a claim that such groups currently exist. A general-purpose crowded-object benchmark becomes useful only if a broader cross-category contribution is chosen jointly with the user.

## Bounded Candidate Check

| Candidate and primary sources | Potential value | Unresolved admission conditions |
|---|---|---|
| PigTrace, Tangirala et al., Livestock Monitoring with Transformer, BMVC 2021; https://arxiv.org/abs/2111.00801 and https://github.com/serket-tech/starformer | Pig instance masks from multiple indoor farms; a candidate for visible-light external validation | The inspected author repository does not expose an executable dataset download. Access, license, annotation policy, group split and overlap with existing sources remain unverified. Do not count it as an available experiment asset. |
| INPC, Wang et al., Infrared Physics & Technology 141 (2024), 105491; https://doi.org/10.1016/j.infrared.2024.105491 and https://github.com/HUBUwg96/INPC | Author-released infrared pig instance-segmentation data with a linked download; possible later modality-shift stress test | Download/account requirements, license, actual annotation schema, counts and splits have not been checked. Infrared changes the sensing modality, so it is not the default replacement for a same-modality independent validation dataset. |

These are candidates for a later admission review, not selected benchmarks. Prefer an independently collected dataset compatible with the eventual method's input modality and mask definition. Public repository visibility and a paper's availability claim do not establish a usable data license or current download access. No author was contacted.

## Search Scope and Uncertainty

Sources used: current project Wiki and experiment manifests, official author repositories, primary publisher pages, official PigLife datasheet, web discovery, and the ARIS arXiv metadata helper. Zotero/Obsidian were not requested. Local PDFs currently focus on method neighbors rather than these dataset releases; existing Wiki source records were reused.

The arXiv helper query `all:"pig" AND (all:"segmentation" OR all:"dataset")`, max 5, completed successfully. Results included Panoptic Instance Segmentation on Pigs (2005.10499), already in the Wiki, plus behavior, point-cloud and acronym matches that do not establish a compatible 2D instance-mask dataset. This was a bounded suitability check, not an exhaustive dataset survey or novelty audit. No dataset, model, training run, or final holdout was created or selected in this assessment.
