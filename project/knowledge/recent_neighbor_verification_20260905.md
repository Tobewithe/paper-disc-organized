# Recent Neighbor Verification: arXiv 2608.11607 and 2608.29253

Date: 2026-09-05. Scope is limited to the official arXiv records and their v1 PDFs for the two IDs named in `research-restart/literature_landscape.md`. No project result, protected data, or model run was used.

## Sources and record status

| arXiv ID | Official arXiv metadata verification | PDF inspected | PDF SHA256 |
|---|---|---|---|
| [2608.11607](https://arxiv.org/abs/2608.11607) | Yes, API record published/updated 2026-08-12, category `cs.CV, eess.IV` | `papers/2608.11607.pdf`, 13 pages, v1 | `6d31960002e8d344fc601e2c73e37bceb96519edcd1c0be5b5d9f31727f2fc2f` |
| [2608.29253](https://arxiv.org/abs/2608.29253) | Yes, API record published/updated 2026-08-29, category `cs.CV, cs.AI, cs.LG` | `papers/2608.29253.pdf`, 18 pages, v1 | `7590ee6969b5b22d8aca7f21ae5e4501fcb3444aca05c6837ac27c3da3b9c2fe` |

The metadata was fetched with `C:/Dpan/repo/ARIS/tools/arxiv_fetch.py` using the project pytorch Python environment. The PDFs were fetched from the corresponding official `arxiv.org/pdf/<id>.pdf` URLs and read with `pypdf`. This check did not verify a journal/conference acceptance, a later arXiv revision, code availability, or the authors' reported results independently.

## 2608.11607: Topology-Aware Query Selection for Surgical Instrument Instance Segmentation

**Official title and authors:** *Topology-Aware Query Selection for Surgical Instrument Instance Segmentation*, Ze Zhang and Yang Zhang. The official abstract matches the landscape's central characterization: it names duplicate, fragmented, merged, missed, and empty-frame predictions; casts final query selection as a relational variable-cardinality problem; and describes a complete graph over nonempty candidates from a fixed Mask2Former, cardinality prediction, and exact structured subset selection.

**Full-text method verification:** the PDF states that a fixed Mask2Former emits 100 scored mask queries before external score thresholding or mask NMS. The method retains nonempty candidates, builds a complete graph with pair features including overlap, directional containment, relative scale, score margin, and boundary contact, predicts candidate validity/pair competition/cardinality, then solves an exact cardinality-constrained subset. It explicitly says this changes set construction rather than pixel prediction. Therefore it **does involve candidate/query selection** and explicitly frames adjacent candidate fragments as a selection-consistency problem. It is not a query-recombination method.

**Experiment boundary:** source development is 4,246 frames from 18 CholecInstanceSeg cases; the sealed source test is 10,566 frames from 22 cases. Direct transfers are reported separately for ROBUST-MIPS (4,057 frames/30 cases) and Endoscapes (74 frames/10 cases). The paper reports all three discovery seeds satisfying its stated complete criterion for the sealed source test and ROBUST-MIPS; Endoscapes has complete support in only one of three seeds, so the paper itself does not claim stable direct transfer there. Its graph-versus-node comparator keeps shared candidate-wise information and selector machinery, but the reported contrast combines graph edge geometry, message passing, and additional relational-path capacity; it does not isolate these components.

**Project relevance and limit:** this is a direct near-neighbor against a generic post-hoc relational candidate-set selector claim. It is evidence of a surgical Mask2Former-specific method with native-instance label contracts, not evidence that the same mechanism, candidate family, or transfer behavior applies to frozen YOLO26 traces or pig masks. It cannot by itself establish that the project's complete/local cases arise from ranking, Top-K, NMS, or a causal selector effect.

## 2608.29253: QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation

**Official title and authors:** *QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation*, Yaroslav Prytula, Anton Popov, and Dmytro Fishman. The official abstract matches the landscape description: QCell is a query-based microscopy model that combines latent instance-query recombination with contrastive query alignment, introduces an Organoids benchmark, and reports `+2.2 AP` and `+2.7 AJI` on ISBI2014.

**Full-text method verification:** QCell extends a MaskDINO-style model. For each decoder content query it derives amodal, visible, and invisible sub-query representations through MLP heads, predicts their masks, and learns a refined full-instance query by recombining those representations. It adds consistency regularization between visible/invisible and recombined masks. Separately, it uses matched main queries and denoising queries for instance-discriminative contrastive learning and cosine query alignment. Thus it **does involve query recombination** and query separation for overlap. It does **not** select a structured subset from frozen candidates, and it does not describe a post-hoc candidate selector. The PDF does not present fragmentation as its stated problem; its stated problem is semi-transparent overlapping cell structure, weak boundaries, and merged/separated cell instances.

**Experiment boundary:** the paper trains/evaluates amodal masks on ISBI2014 cytoplasm, Revvity-25, and a new Organoids dataset. It reports a three-seed test average after validation checkpoint selection, ResNet-50-FPN backbones, 100 queries on ISBI2014/Revvity-25 and 300 on Organoids. On ISBI2014, Table 1 reports MaskDINO `63.7 AP, 75.9 AJI` and QCell `65.9 AP, 78.6 AJI`; Table 3 is an ISBI2014 ablation. The paper's structural supervision uses amodal, visible, and invisible component masks. This label and imaging setting is materially different from a visible/modal pig-mask setting unless the annotation semantics are separately verified.

**Project relevance and limit:** QCell is a direct near-neighbor against a generic claim to recombine query representations or align queries for dense overlap. It is not evidence for post-hoc YOLO26 raw-candidate recovery, not a candidate-selection result, and not evidence that recombination transfers to pigs without compatible amodal/visible/invisible supervision.

## Bounded conclusion

Both landscape entries are real and their title, author list, and abstract descriptions are accurately grounded in official arXiv metadata and the v1 PDFs. The novelty risk is real but mechanism-specific: 2608.11607 conflicts with broad relational final-candidate selection, while 2608.29253 conflicts with broad query recombination/alignment for overlap. Neither paper alone validates a project intervention, establishes project failure causality, or clears a novelty claim for the project.

Canonical Wiki pages: `research-wiki/papers/zhang2026_topologyaware_query_selection.md` and `research-wiki/papers/prytula2026_qcell_recombining_aligning.md`.
