---
type: paper
node_id: paper:compte2025_housed_pig_identification
title: "Housed pig identification and tracking for precision livestock farming"
authors: ["Albert Compte", "Yudong Yan", "Xavier Cortes", "Sergio Escalera", "Julio C.S. Jacques-Junior"]
year: 2025
venue: "Expert Systems with Applications 293, 128466"
external_ids:
  arxiv: null
  doi: "10.1016/j.eswa.2025.128466"
  s2: null
tags: ["faropigseg", "dataset", "annotation", "provenance"]
added: 2026-09-06T17:31:01Z
---

# Housed pig identification and tracking for precision livestock farming

## One-line thesis
Introduces the FaroPigSeg and FaroPigReID-33 datasets within a pig identification and tracking pipeline; retrieved official documentation leaves segmentation occlusion policy unspecified.

## Problem / Gap
Pig identification and tracking in commercial pens requires a segmentation stage and consistent identity evidence. The publisher describes a collection/annotation/segmentation/ReID/tracking pipeline.

## Method
The journal record introduces FaroPigSeg and FaroPigReID-33. The author repository specifies FaroPigSeg as single-class YOLO instance polygons and an image-random 70/20/10 split. Detailed segmentation annotation tooling and quality assurance are not verified from retrieved primary text.

## Key Results
The author repository reports 1,518 annotated images and over 16,000 instances in the full segmentation dataset. Project counts of 160 test images and 1,752 GT are local-manifest evidence, not exact split counts published in the README. No paper performance number is used as a comparable project baseline.

## Assumptions
The published segmentation and reidentification datasets have different annotation descriptions; SAM/CoTracker and visibility filtering statements for ReID must not be transferred to segmentation.

## Limitations / Failure Modes
Modal/amodal, occlusion-completion, truncation and minimum-visibility policies are unverified. Polygon format does not determine them. Image-random splitting does not establish pen/video independence. The journal and SSRN full-text endpoints were not readable during this verification; only metadata, indexed section descriptions and author repository documentation support this page.

## Reusable Ingredients
Original normalized polygons can be audited against derived COCO masks without choosing an occlusion policy. Record original file hashes and split identity.

## Open Questions
Obtain primary annotation guidance or accessible Section 3.2 to resolve occlusion and image-border handling. Current geometry is insufficient to answer these questions.

## Claims
Metadata/schema verified; annotation semantics unresolved. No method, causal or novelty acceptance.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Supports source provenance for the user-selected Faro quality diagnostic. R005d verifies local source-to-manifest raster parity; it does not validate the authors' semantic policy.

## Verification Sources

- [Author repository](https://github.com/yudong888/FaroPig-Datasets), inspected 2026-09-07; repository commit `273334167a1db562f21d2adc343deea047198491` records the inspected README lineage, not an immutable data release.
- [Published journal record](https://doi.org/10.1016/j.eswa.2025.128466), 2025-12-01, volume 293, article 128466; [institutional record](https://vbn.aau.dk/en/publications/housed-pig-identification-and-tracking-for-precision-livestock-fa/).
- [Earlier SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4929890), DOI `10.2139/ssrn.4929890`, posted 2024-08-19, 27 pages. Its author order differs; the journal record controls this page's citation.
- [Author archive](https://data.chalearnlap.cvc.uab.cat/FaroPig/FaroPigSeg.zip): HTTP metadata read 2026-09-07, 153,611,092 bytes, ETag `67d415a4-927eb54`, Last-Modified 2025-03-14. Archive not downloaded or compared this round; local/upstream archive identity is not claimed.

