# Literature Search Audit

Run date: 2026-09-04 (Asia/Shanghai). Four query axes: dense/crowded occlusion; low-overlap fragmentation and candidate competition; relation/global consistency; pig/animal instance segmentation and evaluation.

| Source | Calls | Availability | Query/result record |
|---|---:|---|---|
| arXiv helper | 4 | available | `raw-search/arxiv_*.txt`, exit 0 |
| Semantic Scholar helper | 4 | available | `raw-search/s2_*.txt`, exit 0, venue/ID metadata |
| DeepXiv helper | 4 | available | `raw-search/deepxiv_*.txt`, exit 0 |
| OpenAlex helper | 4 | partial | `raw-search/openalex_*.txt`; rate limiting/low relevance observed |
| Exa helper | 4 | unavailable | `EXA_API_KEY environment variable is required`; see `raw-search/exa_*.txt` |
| Scholar Feed MCP | 4 | unavailable | transient network error / HTTP 503 |
| Obsidian MCP | 4 | available | Existing notes in `pigcv_reserch` on dense-touching failures and pig literature |
| Zotero MCP | 0 | unavailable | No `mcp__zotero__*` tool exposed |
| Gemini MCP/CLI | 0 | unavailable | No Gemini tool exposed |
| Web Search MCP | 0 | unavailable | No WebSearch tool exposed; official arXiv/S2 URLs retained |

All returned papers, including low-relevance and duplicate candidates, remain in `literature_registry.jsonl`. Dedup keys: arXiv ID, DOI, S2/OpenAlex ID, then normalized title. Duplicate rows retain `status=duplicate`.

## ARIS Stage-1 YOLO-focused replay (2026-09-05)

The configured Scholar Feed source was explicitly replayed for the four YOLO-focused query axes. Calls were made through `mcp__scholar_feed__search_papers`; returned records are retained in `literature_registry.jsonl` with `source=scholar_feed` and `status=needs_verification` unless an existing canonical entry already supplied verification.

| Query axis | Calls | Result | Audit interpretation |
|---|---:|---:|---|
| YOLO dense/occluded candidate selection | 1 | 10 structured hits | Returned metadata; mostly general YOLO/occlusion neighbors |
| low-IoU fragments/partial predictions | 1 | 10 structured hits | Returned metadata; includes incomplete-supervision and occlusion neighbors |
| top-k/query ranking/duplicate suppression | 2 | 0; transient upstream error | Not interpretable as zero literature; replay required |
| pig/animal crowded instance segmentation | 1 | 10 structured hits | Returned metadata; pig/animal and adjacent crowd papers |
| lineage/full-text follow-up | 5 | 1 lineage success, 1 full-text success, 3 transient errors | Scholar Feed graph/content coverage is partial |

The replay does not replace the prior arXiv, Semantic Scholar, DeepXiv, OpenAlex, Gemini-CLI, Obsidian and local-library audit. Missing/failed sources remain explicit; no model-memory candidate is promoted to a verified citation.

Subagent cross-check added local-library audit (no PDFs in this project), Crossref exact-title checks, and Scholar Feed broad-query hits. The subagent found 13 core/adjacent papers and marked keyword-neighbor noise as tangential/excluded; details are retained in the session result and registry additions.

## Failure detail retained for replay

- Semantic Scholar returned initial structured results, then HTTP 429 rate limiting; later empty/failed calls must not be interpreted as zero coverage.
- Scholar Feed had one successful query, three HTTP 503 responses, and subsequent HTTP 429 rate limiting.
- Exa failed for all four queries because `EXA_API_KEY` was not present.
- Zotero MCP and Gemini MCP were not exposed in the active subagent tool surface; no fabricated results were added.
- WebSearch MCP was not exposed in the active subagent tool surface; official arXiv, DOI, Crossref and OpenAlex URLs were used where available.

## Next-run source requirement

- `gemini` is a mandatory source for the next ARIS literature run (the `research-lit` source set must explicitly include `gemini`).
- Gemini MCP is still not exposed in the current Codex tool surface, but the local Gemini CLI is installed at `C:\Users\Administrator\AppData\Roaming\npm\gemini.ps1`; use the ARIS CLI fallback and record authentication, query, exit status, and hit count.
- The previous `Gemini MCP/CLI: unavailable` row describes the 2026-09-04 run only; it must not be treated as the result of the next run.

## Exploratory novelty-check (2026-09-04)

Purpose: compare mechanism families for the broader diagnosis-to-improvement task; no method route selected.

| Source | Query | Result | Interpretation |
|---|---|---:|---|
| arXiv API | `instance segmentation duplicate removal low IoU partial mask candidates candidate reranking` | 10 | Returned Rank & Sort Loss, relational prior work and related noise; no direct confirmed match to the project failure taxonomy |
| arXiv API | `dense instance segmentation candidate graph global consistency selection` | 10 | Returned BCNet, Graph Relation Distillation, Topology-Aware Query Selection and related work |
| arXiv API | `recurrent instance segmentation explained area memory candidate suppression` | 2 | Returned Recurrent Instance Segmentation and referring-expression variant |
| Crossref | exact-title search for Topology-Aware Query Selection | 3 approximate | No exact DOI match; arXiv record retained as a recent preprint requiring venue-status verification |
| Scholar Feed MCP | three targeted candidate queries | 3 failures (503/transient timeout) | Not interpretable as zero coverage; retained as unavailable for this pass |
| OpenAlex API | three targeted queries | 0 usable records | Search endpoint returned zero; not interpreted as absence of literature |

Important near-neighbor: *Topology-Aware Query Selection for Surgical Instrument Instance Segmentation* (arXiv:2608.11607) explicitly frames fixed-query selection as a relational variable-cardinality problem and mentions fragmented predictions. This weakens any claim that a generic candidate-graph selector is novel; it does not settle whether the project's pig-specific failure mechanism or diagnostic finding is novel.

## Targeted primary-source verification (2026-09-05, strict-quality continuation)

This bounded follow-up verifies two named evaluation/scoring anchors, not a new broad landscape pass. Local Wiki search found no existing pages for Mask Scoring R-CNN or TIDE. Scholar Feed exact-title search found TIDE; Mask Scoring keyword results were noisy, so explicit arXiv-ID lookup was used. The ARIS arxiv_fetch helper returned official Mask Scoring metadata and downloaded both official PDFs. Full-text reading used the bundled pypdf runtime. The scope covered Mask Scoring Sections 1-4 (pages 1-6) and TIDE Sections 1-3 (pages 1-9). CVF metadata independently confirmed Mask Scoring's CVPR 2019 venue. No later citation search or novelty-completeness claim is made by this two-paper verification.

- `paper:huang2019_mask_scoring_rcnn`: calibration uses predicted MaskIoU after retained-mask selection; generic quality scoring is established prior art.
- `paper:bolya2020_tide_general_toolbox`: each output-error oracle starts at the same baseline; progressive correction can distort contribution magnitudes.
- Durable sources and SHA256 values are recorded on the two Wiki pages. Their implications concern comparator and metric design, not evidence that a project scorer works.

## Recent-neighbor primary-source verification (2026-09-05)

Bounded verification of only arXiv:2608.11607 and arXiv:2608.29253 used the canonical arXiv helper and official v1 PDFs. Both records matched title, authors, abstract, dates, and categories. Full text confirms fixed-candidate relational selection for the former and in-network MaskDINO query recombination/alignment for the latter. Both remain mechanism-specific near neighbors; neither clears a project novelty claim or establishes transfer/causality. Canonical pages and PDF hashes are recorded in `research-wiki/recent_neighbor_verification_20260905.md`.

## Faro annotation-source verification (2026-09-07)

The user prioritized Faro box/mask/annotation diagnosis. Targeted queries `FaroPigSeg`, `FaroPigSeg annotation dataset farm`, and the exact paper title resolved the author repository, publisher record and SSRN preprint. A read-only literature researcher shard independently inspected those primary sources. The local collection had no Faro dataset paper. Canonical journal DOI is `10.1016/j.eswa.2025.128466`; ingested with the official `research_wiki.py` helper as `paper:compte2025_housed_pig_identification`. Author repository format/split and publication metadata were verified; precise segmentation visibility/occlusion/truncation policy remains unverified. Publisher/SSRN full text returned 403; indexed section summaries do not replace the missing annotation rules. ReID-specific annotation mechanisms are not transferred to FaroPigSeg. No broad literature or novelty claim is made.

Coordinator follow-up queried the exact title with `pdf`, `FaroPigSeg annotation visible occlusion`, `site:ub.edu "Housed pig"`, `site:cvc.uab.es "Housed pig"`, and `"128466" "pdf" "Compte"`. No usable institutional full-text copy was identified; unrelated hits were excluded. The indexed SSRN delivery URL also returned HTTP 403 on direct retrieval. This access outcome leaves the annotation policy unresolved and does not establish that no public copy exists.
