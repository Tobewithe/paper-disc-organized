"""Paired full-COCO screen for counterfactual-guided P3 rescue.

Both arms start from the same official YOLO26m-seg COCO checkpoint and reuse its
executable training arguments.  The cfp3r arm alone receives a training-only,
GT-constructed privileged view and a rescue-gated P3 auxiliary loss.  Inference
is unchanged.
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
import yaml
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel, torch_safe_load
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model
import ultralytics.engine.trainer as trainer_module

SOURCE_WEIGHT: Path
ORIGINAL_LOSS = SegmentationModel.loss
CF_TEACHER = None
METHOD_ENABLED = False
AUX_WEIGHT = 0.5
EXTRA_CLASS_WEIGHT = 0.05
FOLLOW_O2M = False
SCHEDULE_OBSERVED = {}
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


def make_teacher_guidance(model, teacher_preds, batch, selected):
    teacher = branch(teacher_preds)
    criterion = model.criterion.one2many if hasattr(model.criterion, "one2many") else model.criterion
    anchor_points, stride_tensor = make_anchors(teacher["feats"], criterion.stride, 0.5)
    teacher_dist = teacher["boxes"].permute(0, 2, 1).contiguous()
    teacher_boxes = criterion.bbox_decode(anchor_points, teacher_dist) * stride_tensor
    teacher_scores = teacher["scores"].permute(0, 2, 1).sigmoid()
    h, w = batch["img"].shape[-2:]
    scale = torch.tensor([w, h, w, h], device=teacher_boxes.device, dtype=teacher_boxes.dtype)
    gt_xyxy = ops.xywh2xyxy(batch["bboxes"].to(teacher_boxes.device)) * scale
    guidance = []
    p3_count = teacher["feats"][0].shape[-2] * teacher["feats"][0].shape[-1]
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
        tiou = pairwise_iou(teacher_boxes[bi, ids], gt)
        quality = tiou + teacher_scores[bi, ids, cls] * 0.02
        local_best = int(quality.argmax())
        guidance.append((global_index, int(ids[local_best]), float(tiou[local_best])))
    return guidance


def counterfactual_aux(model, preds, batch):
    student = branch(preds)
    criterion = model.criterion.one2many if hasattr(model.criterion, "one2many") else model.criterion
    anchor_points, stride_tensor = make_anchors(student["feats"], criterion.stride, 0.5)
    student_dist = student["boxes"].permute(0, 2, 1).contiguous()
    student_boxes = criterion.bbox_decode(anchor_points, student_dist) * stride_tensor
    h, w = batch["img"].shape[-2:]
    scale = torch.tensor([w, h, w, h], device=student_boxes.device, dtype=student_boxes.dtype)
    gt_xyxy = ops.xywh2xyxy(batch["bboxes"].to(student_boxes.device)) * scale
    terms, diagnostics = [], []
    for global_index, pos_index, best_tiou_value in batch.get("_cf_teacher_guidance", []):
        bi = int(batch["batch_idx"][global_index].item())
        cls = int(batch["cls"][global_index].item())
        gt = gt_xyxy[global_index]
        pos = int(pos_index)
        best_tiou = torch.as_tensor(best_tiou_value, device=student_boxes.device, dtype=student_boxes.dtype)
        with torch.no_grad():
            siou = pairwise_iou(student_boxes[bi, pos:pos + 1].detach(), gt)[0]
            gate = bool(siou < 0.50 and best_tiou >= 0.50 and best_tiou >= siou + 0.10)
        if not gate:
            continue
        wh = (gt[2:] - gt[:2]).clamp_min(8.0)
        normalized_error = (student_boxes[bi, pos] - gt) / torch.cat((wh, wh))
        reg = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), beta=0.1, reduction="mean")
        cls_logit = student["scores"][bi, cls, pos]
        cls_term = F.binary_cross_entropy_with_logits(cls_logit, torch.ones_like(cls_logit))
        terms.append(reg + EXTRA_CLASS_WEIGHT * cls_term)
        diagnostics.append((float(best_tiou), float(siou)))
    zero = student["boxes"].sum() * 0
    aux = torch.stack(terms).mean() if terms else zero
    if not torch.isfinite(aux.detach()):
        raise FloatingPointError("Nonfinite P3 auxiliary loss")
    RUN_STATS["batches"] += 1
    RUN_STATS["selected"] += sum(i >= 0 for i in batch.get("_cf_selected", []))
    RUN_STATS["gated"] += len(terms)
    RUN_STATS["aux_sum"] += float(aux.detach())
    RUN_STATS["teacher_iou_sum"] += sum(x[0] for x in diagnostics)
    RUN_STATS["student_iou_sum"] += sum(x[1] for x in diagnostics)
    return aux


def effective_aux_weight(criterion):
    if not FOLLOW_O2M:
        return AUX_WEIGHT
    initial = float(criterion.o2m_copy)
    current = float(criterion.o2m)
    if initial <= 0 or not 0 <= current <= initial + 1e-9:
        raise ValueError(f"Unexpected official branch schedule: {current}/{initial}")
    return AUX_WEIGHT * (current / initial)


def patched_loss(self, batch, preds=None):
    if getattr(self, "criterion", None) is None:
        self.criterion = self.init_criterion()
    if preds is None:
        preds = self.forward(batch["img"])
    base, items = self.criterion(preds, batch)
    if not METHOD_ENABLED or CF_TEACHER is None or "_cf_teacher_guidance" not in batch:
        return base, items
    aux = counterfactual_aux(self, preds, batch)
    weight = effective_aux_weight(self.criterion)
    state = (float(self.criterion.o2m), weight)
    SCHEDULE_OBSERVED[state] = SCHEDULE_OBSERVED.get(state, 0) + 1
    augmented = base.clone()
    augmented[0] = augmented[0] + weight * aux * batch["img"].shape[0]
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
        image = batch["img"]
        masks = batch["masks"].to(image.device)
        if masks.ndim == 4:
            masks = masks[:, 0]
        if tuple(masks.shape[-2:]) != tuple(image.shape[-2:]):
            masks = F.interpolate(masks[:, None].float(), image.shape[-2:], mode="nearest")[:, 0].long()
        else:
            masks = masks.long()
        edited = image.detach().clone()
        selected = []
        for bi in range(len(image)):
            global_ids = (batch["batch_idx"].view(-1).long() == bi).nonzero(as_tuple=False).view(-1)
            candidates = []
            for local_id, gi in enumerate(global_ids.tolist(), 1):
                box = batch["bboxes"][gi]
                if float(box[2] * box[3]) > 0.01:
                    continue
                target = masks[bi] == local_id
                area = int(target.sum())
                if area < 4:
                    continue
                radius = max(4, min(16, int(round(area ** 0.5 * 0.20))))
                dilated = F.max_pool2d(target[None, None].float(), 2 * radius + 1, 1, radius)[0, 0].bool()
                neighbor = dilated & (masks[bi] != 0) & ~target
                ring = dilated & (masks[bi] == 0)
                if not ring.any():
                    continue
                fg_mean = image[bi, :, target].mean(1)
                bg_mean = image[bi, :, ring].mean(1)
                contrast = float((fg_mean - bg_mean).abs().mean())
                neighbor_fraction = float(neighbor.sum()) / max(area, 1)
                vulnerability = neighbor_fraction + 1.0 / (contrast + 0.05)
                candidates.append((vulnerability, gi, target, neighbor, ring, fg_mean, bg_mean))
            if not candidates:
                selected.append(-1)
                continue
            _, gi, target, neighbor, ring, fg_mean, bg_mean = max(candidates, key=lambda x: x[0])
            if neighbor.any():
                edited[bi, :, neighbor] = bg_mean[:, None]
            sign = torch.where(fg_mean.mean() >= bg_mean.mean(), 1.0, -1.0)
            if abs(float(fg_mean.mean() - bg_mean.mean())) < 0.01:
                sign = torch.where(fg_mean.mean() < 0.5, 1.0, -1.0)
            edited[bi, :, target] = (edited[bi, :, target] + sign * 0.11).clamp(0, 1)
            selected.append(gi)
        batch["_cf_selected"] = selected
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device.type, enabled=bool(self.amp)
        ):
            teacher_preds = CF_TEACHER(edited)
            batch["_cf_teacher_guidance"] = make_teacher_guidance(model, teacher_preds, batch, selected)
        return batch


def official_weight_config(weight: Path):
    ckpt, _ = torch_safe_load(str(weight))
    source = ckpt["train_args"]
    keys = ["imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "nbs", "amp", "deterministic", "rect", "cos_lr",
            "close_mosaic", "multi_scale", "semseg_loss", "overlap_mask", "mask_ratio", "dropout",
            "box", "cls", "dfl", "o2m", "muon_w", "sgd_w", "cls_w", "stride_ratio", "topk",
            "hungarian", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste",
            "copy_paste_mode", "auto_augment", "erasing"]
    return {k: source[k] for k in keys if k in source and k in DEFAULT_CFG_DICT}


def make_smoke_yaml(data_yaml: Path, out_dir: Path) -> Path:
    source = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(source.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    train_list = dataset_root / str(source["train"])
    val_list = dataset_root / str(source["val"])
    train_lines = train_list.read_text(encoding="utf-8").splitlines()[:16]
    val_lines = val_list.read_text(encoding="utf-8").splitlines()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    smoke = {"path": str(dataset_root), "train": str(out_dir / "train.txt"),
             "val": str(out_dir / "val.txt"), "names": source["names"]}
    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(smoke, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def main():
    global METHOD_ENABLED, AUX_WEIGHT, SOURCE_WEIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "cfp3r"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weight", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    SOURCE_WEIGHT = args.weight.resolve()
    METHOD_ENABLED = args.arm == "cfp3r"
    AUX_WEIGHT = args.aux_weight
    settings.update({k: False for k in ["wandb", "comet", "mlflow", "clearml", "neptune"] if k in settings})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trainer_module.check_amp = lambda model: True

    cfg = official_weight_config(SOURCE_WEIGHT)
    data = args.data.resolve()
    if args.smoke:
        data = make_smoke_yaml(data, args.project.resolve() / "smoke_data")
    name = f"{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else "_full")
    cfg.update(data=str(data), epochs=1 if args.smoke else args.epochs, batch=args.batch,
               workers=0 if args.smoke else args.workers, device=0, project=str(args.project.resolve()),
               name=name, exist_ok=False, seed=args.seed, save=True, save_period=1, val=True,
               plots=False, cache=False, patience=0, verbose=True)
    yolo = YOLO(str(SOURCE_WEIGHT))
    yolo.train(trainer=CounterfactualTrainer, **cfg)
    run = Path(yolo.trainer.save_dir)
    receipt = {
        "status": "complete", "arm": args.arm, "seed": args.seed, "epochs": cfg["epochs"],
        "batch": args.batch, "workers": cfg["workers"],
        "aux_weight": AUX_WEIGHT if METHOD_ENABLED else 0.0,
        "runtime_ultralytics": __import__("ultralytics").__version__,
        "data": str(data), "source_weight": str(SOURCE_WEIGHT),
        "source_weight_train_args_executable": official_weight_config(SOURCE_WEIGHT),
        "method_stats": RUN_STATS,
    }
    (run / "TRAINING_COMPLETE.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINING_COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
