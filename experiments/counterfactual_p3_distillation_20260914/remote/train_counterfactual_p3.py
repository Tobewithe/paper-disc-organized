"""Counterfactual-guided P3 localization training pilot.

The method uses GT only during training to construct one privileged view per image.
The edited view reveals target-local box evidence after reducing proximal-instance
interference and increasing target/background luminance contrast. A fixed teacher
selects improved P3 cells; the student receives auxiliary box supervision at the
same cells while seeing the unedited image. Inference remains stock YOLO26m-seg.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel, torch_safe_load
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model
import ultralytics.engine.trainer as trainer_module

ROOT = Path("/root/autodl-tmp")
WEIGHT = Path("/root/autodl-tmp/boundary_code_20260914/yolo26m-seg.pt")
DATA = Path("/root/autodl-tmp/boundary_coco_20260914/boundary_ownership_remote.yaml")
PROJECT = Path("/root/autodl-tmp/counterfactual_p3_20260914/runs")
ORIGINAL_LOSS = SegmentationModel.loss
CF_TEACHER = None
METHOD_ENABLED = False
AUX_WEIGHT = 2.0
RUN_STATS = {"batches": 0, "selected": 0, "gated": 0, "aux_sum": 0.0,
             "teacher_iou_sum": 0.0, "student_iou_sum": 0.0}


def pairwise_iou(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes[:, :2], target[:2])
    rb = torch.minimum(boxes[:, 2:], target[2:])
    inter = (rb - lt).clamp_min(0).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    ta = (target[2:] - target[:2]).clamp_min(0).prod()
    return inter / (area + ta - inter).clamp_min(1e-9)


def branch(preds):
    if isinstance(preds, tuple):
        preds = preds[1]
    return preds["one2many"] if "one2many" in preds else preds


def counterfactual_aux(model, preds, teacher_preds, batch):
    student = branch(preds); teacher = branch(teacher_preds)
    criterion = model.criterion.one2many if hasattr(model.criterion, "one2many") else model.criterion
    anchor_points, stride_tensor = make_anchors(student["feats"], criterion.stride, 0.5)
    student_dist = student["boxes"].permute(0, 2, 1).contiguous()
    teacher_dist = teacher["boxes"].permute(0, 2, 1).contiguous()
    student_boxes = criterion.bbox_decode(anchor_points, student_dist) * stride_tensor
    with torch.no_grad():
        teacher_boxes = criterion.bbox_decode(anchor_points, teacher_dist) * stride_tensor
        teacher_scores = teacher["scores"].permute(0, 2, 1).sigmoid()
    h, w = batch["img"].shape[-2:]
    scale = torch.tensor([w, h, w, h], device=student_boxes.device, dtype=student_boxes.dtype)
    gt_xyxy = ops.xywh2xyxy(batch["bboxes"].to(student_boxes.device)) * scale
    selected = batch.get("_cf_selected", [])
    terms = []; diagnostics = []
    p3_count = student["feats"][0].shape[-2] * student["feats"][0].shape[-1]
    p3_points = anchor_points[:p3_count] * stride_tensor[:p3_count]
    for global_index in selected:
        if global_index < 0:
            continue
        bi = int(batch["batch_idx"][global_index].item())
        cls = int(batch["cls"][global_index].item())
        gt = gt_xyxy[global_index]
        margin = float(criterion.stride[0])
        local = ((p3_points[:, 0] >= gt[0] - margin) & (p3_points[:, 0] <= gt[2] + margin) &
                 (p3_points[:, 1] >= gt[1] - margin) & (p3_points[:, 1] <= gt[3] + margin))
        ids = local.nonzero(as_tuple=False).view(-1)
        if not len(ids):
            continue
        with torch.no_grad():
            tiou = pairwise_iou(teacher_boxes[bi, ids], gt)
            # True-class evidence breaks ties but cannot override geometry.
            quality = tiou + teacher_scores[bi, ids, cls] * .02
            pos = ids[int(quality.argmax())]
            siou = pairwise_iou(student_boxes[bi, pos:pos+1].detach(), gt)[0]
            best_tiou = tiou[int(quality.argmax())]
            gate = bool(best_tiou >= .35 and best_tiou >= siou + .02)
        if not gate:
            continue
        wh = (gt[2:] - gt[:2]).clamp_min(8.0)
        norm = torch.cat((wh, wh))
        normalized_error = (student_boxes[bi, pos] - gt) / norm
        reg = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), beta=.1, reduction="mean")
        # A small class term ensures the newly supervised location remains usable by normal candidate ranking.
        cls_logit = student["scores"][bi, cls, pos]
        cls_term = F.binary_cross_entropy_with_logits(cls_logit, torch.ones_like(cls_logit))
        terms.append(reg + .05 * cls_term)
        diagnostics.append((float(best_tiou), float(siou)))
    zero = student["boxes"].sum() * 0
    aux = torch.stack(terms).mean() if terms else zero
    RUN_STATS["batches"] += 1; RUN_STATS["selected"] += sum(i >= 0 for i in selected); RUN_STATS["gated"] += len(terms)
    RUN_STATS["aux_sum"] += float(aux.detach())
    RUN_STATS["teacher_iou_sum"] += sum(x[0] for x in diagnostics); RUN_STATS["student_iou_sum"] += sum(x[1] for x in diagnostics)
    return aux


def patched_loss(self, batch, preds=None):
    if getattr(self, "criterion", None) is None:
        self.criterion = self.init_criterion()
    if preds is None:
        preds = self.forward(batch["img"])
    base, items = self.criterion(preds, batch)
    if not METHOD_ENABLED or CF_TEACHER is None or "_cf_img" not in batch:
        return base, items
    with torch.no_grad():
        teacher_preds = CF_TEACHER(batch["_cf_img"])
    aux = counterfactual_aux(self, preds, teacher_preds, batch)
    augmented = base.clone()
    augmented[0] = augmented[0] + AUX_WEIGHT * aux * batch["img"].shape[0]
    return augmented, items


SegmentationModel.loss = patched_loss


class CounterfactualTrainer(SegmentationTrainer):
    def _setup_train(self):
        global CF_TEACHER
        super()._setup_train()
        if METHOD_ENABLED:
            CF_TEACHER = copy.deepcopy(unwrap_model(self.model)).to(self.device).requires_grad_(False)
            CF_TEACHER.train()
            for module in CF_TEACHER.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not METHOD_ENABLED:
            return batch
        image = batch["img"]; masks = batch["masks"].to(image.device)
        if masks.ndim == 4:
            masks = masks[:, 0]
        if tuple(masks.shape[-2:]) != tuple(image.shape[-2:]):
            masks = F.interpolate(masks[:, None].float(), image.shape[-2:], mode="nearest")[:, 0].long()
        else:
            masks = masks.long()
        edited = image.detach().clone(); selected = []
        for bi in range(len(image)):
            global_ids = (batch["batch_idx"].view(-1).long() == bi).nonzero(as_tuple=False).view(-1)
            candidates = []
            for local_id, gi in enumerate(global_ids.tolist(), 1):
                box = batch["bboxes"][gi]
                if float(box[2] * box[3]) > .01:
                    continue
                target = masks[bi] == local_id
                area = int(target.sum())
                if area < 4:
                    continue
                # Radius scales mildly with object extent and is bounded for efficiency.
                radius = max(4, min(16, int(round(area ** .5 * .20))))
                dilated = F.max_pool2d(target[None, None].float(), 2 * radius + 1, 1, radius)[0, 0].bool()
                neighbor = dilated & (masks[bi] != 0) & ~target
                ring = dilated & (masks[bi] == 0)
                if not ring.any():
                    continue
                fg_mean = image[bi, :, target].mean(1); bg_mean = image[bi, :, ring].mean(1)
                contrast = float((fg_mean - bg_mean).abs().mean())
                neighbor_fraction = float(neighbor.sum()) / max(area, 1)
                vulnerability = neighbor_fraction + 1.0 / (contrast + .05)
                candidates.append((vulnerability, gi, target, neighbor, ring, fg_mean, bg_mean))
            if not candidates:
                selected.append(-1); continue
            _, gi, target, neighbor, ring, fg_mean, bg_mean = max(candidates, key=lambda x: x[0])
            if neighbor.any():
                edited[bi, :, neighbor] = bg_mean[:, None]
            sign = torch.where(fg_mean.mean() >= bg_mean.mean(), 1.0, -1.0)
            if abs(float(fg_mean.mean() - bg_mean.mean())) < .01:
                sign = torch.where(fg_mean.mean() < .5, 1.0, -1.0)
            edited[bi, :, target] = (edited[bi, :, target] + sign * .11).clamp(0, 1)
            selected.append(gi)
        batch["_cf_img"] = edited
        batch["_cf_selected"] = selected
        return batch


def official_weight_config():
    ckpt, _ = torch_safe_load(str(WEIGHT))
    source = ckpt["train_args"]
    keys = ["imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "nbs", "amp", "deterministic", "rect", "cos_lr",
            "close_mosaic", "multi_scale", "semseg_loss", "overlap_mask", "mask_ratio", "dropout",
            "box", "cls", "dfl", "o2m", "muon_w", "sgd_w", "cls_w", "stride_ratio", "topk",
            "hungarian", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste",
            "copy_paste_mode", "auto_augment", "erasing"]
    # The official checkpoint was produced by 8.3.222 and contains several private
    # search-time keys removed from the public 8.4.100 configuration schema. Use
    # the exact saved value for every key that the pinned runtime can execute.
    return {k: source[k] for k in keys if k in source and k in DEFAULT_CFG_DICT}


def main():
    global METHOD_ENABLED, AUX_WEIGHT
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["baseline", "cfp3"], required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--aux-weight", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    METHOD_ENABLED = args.arm == "cfp3"; AUX_WEIGHT = args.aux_weight
    settings.update({k: False for k in ["wandb", "comet", "mlflow", "clearml", "neptune"] if k in settings})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    trainer_module.check_amp = lambda model: True
    cfg = official_weight_config()
    data = DATA
    if args.smoke:
        smoke_dir = Path(__file__).resolve().parent / "smoke"
        smoke_dir.mkdir(exist_ok=True)
        source = (ROOT / "experiments/coco_clean_20260911/data/pilot_1000/train2017.txt").read_text().splitlines()
        (smoke_dir / "train.txt").write_text("\n".join(source[:16]) + "\n", encoding="utf-8")
        (smoke_dir / "val.txt").write_text("\n".join(source[16:24]) + "\n", encoding="utf-8")
        names = "\n".join(f"  {i}: class_{i}" for i in range(80))
        data = smoke_dir / "data.yaml"
        data.write_text(f"train: {smoke_dir/'train.txt'}\nval: {smoke_dir/'val.txt'}\nnames:\n{names}\n", encoding="utf-8")
    cfg.update(data=str(data), epochs=1 if args.smoke else args.epochs, batch=args.batch, workers=0,
               device=0, project=str(PROJECT), name=f"{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else ""),
               exist_ok=False, seed=args.seed, save=True, save_period=1, val=True, plots=False, cache=False,
               patience=0, verbose=True)
    yolo = YOLO(str(WEIGHT))
    yolo.train(trainer=CounterfactualTrainer, **cfg)
    run = Path(yolo.trainer.save_dir)
    receipt = {"status": "complete", "arm": args.arm, "seed": args.seed, "epochs": cfg["epochs"],
               "batch": args.batch, "aux_weight": AUX_WEIGHT if METHOD_ENABLED else 0,
               "runtime_ultralytics": __import__("ultralytics").__version__,
               "source_weight_train_args_executable": {k: cfg[k] for k in official_weight_config()},
               "source_weight_version": "8.3.222", "runtime_schema_omissions": ["semseg_loss", "o2m", "muon_w", "sgd_w", "cls_w", "stride_ratio", "topk", "hungarian"],
               "method_stats": RUN_STATS}
    (run / "TRAINING_COMPLETE.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINING_COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__": main()
