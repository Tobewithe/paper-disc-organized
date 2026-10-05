---
type: paper
node_id: paper:huang2019_mask_scoring_rcnn
title: "Mask Scoring R-CNN"
authors: ["Zhaojin Huang", "Lichao Huang", "Yongchao Gong", "Chang Huang", "Xinggang Wang"]
year: 2019
venue: "CVPR 2019"
external_ids:
  arxiv: "1903.00241"
  doi: null
  s2: null
tags: ["instance-segmentation", "mask-quality", "calibration"]
added: 2026-09-05T12:29:25Z
---

# Mask Scoring R-CNN

## One-line thesis
Mask Scoring R-CNN predicts MaskIoU from RoI features and masks, then multiplies it by the classification score to calibrate retained-mask confidence.

## Problem / Gap
Classification confidence can be poorly aligned with mask IoU and mask completeness. This can reduce AP even when the mask coordinates remain unchanged. Verified from the official arXiv v1 PDF, Sections 1 and 3.

## Method
The MaskIoU head consumes RoI features concatenated with the predicted class mask and regresses mask IoU using L2 loss. Section 3.2 specifies four convolution layers and three fully connected layers. Final confidence is classification score times predicted MaskIoU. The paper trains the complete network end to end. Inference calibrates the top 100 masks selected after Soft-NMS.

## Key Results
COCO 2017 train/validation/test-dev protocol; mask AP, AP50, AP75 and size-specific AP. The authors report consistent improvements across backbone variants. No numerical improvement is imported into this project or interpreted as pig-domain evidence.

## Assumptions
Training has instance-mask GT and RoI features. Inference has retained predicted masks and learned IoU estimates. The head does not use evaluation GT at inference.

## Limitations / Failure Modes
The reported inference mechanism recalibrates retained masks. It does not restore candidates removed before the mask head, change their pixels, or resolve a relation graph by itself. These are scope limits derived from Section 3.2, not separately measured failures in this paper. No YOLO26 or dense-pig experiment is reported.

## Reusable Ingredients
Quality-aware scoring is a required established comparator if project evidence later supports a scorer. Test scoring-only changes with fixed masks separately from candidate retention, deletion and mask refinement.

## Open Questions
Would quality scoring be identifiable and useful at the raw-candidate stage in this YOLO26 runtime? Would any AP gain reduce the fixed relation taxonomy's failures? Current GT-guided recovery cannot answer either question.

## Claims
Literature mechanism only; no project claim node or method acceptance.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Generic mask-quality calibration is established prior art. Any contribution must explain the specific raw/retained/output-set gap and compare against this mechanism family. Official venue metadata verified at https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html (CVPR 2019, pages 6409-6418). PDF: `papers/1903.00241.pdf`, SHA256 `87fb301a260af6c76f4821a0298085a122289f4d16644dcba30e9a18cd5cfd97`. Read pages 1-6 on 2026-09-05. Code: https://github.com/zjhuang22/maskscoring_rcnn.

## Abstract (original)

> Letting a deep network be aware of the quality of its own predictions is an interesting yet important problem. In the task of instance segmentation, the confidence of instance classification is used as mask quality score in most instance segmentation frameworks. However, the mask quality, quantified as the IoU between the instance mask and its ground truth, is usually not well correlated with classification score. In this paper, we study this problem and propose Mask Scoring R-CNN which contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models, and outperforms the state-of-the-art Mask R-CNN. We hope our simple and effective approach will provide a new direction for improving instance segmentation. The source code of our method is available at \url{https://github.com/zjhuang22/maskscoring_rcnn}.

