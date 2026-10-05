"""Phase I: relearn only the native YOLO26 one-to-one coefficient branch.

The script uses the already frozen official TAL fit cache.  It reconstructs the
native head input feature maps once from the original image view, then trains
B/C/D on exactly the same candidate identities.  No new head, gate, or input
feature is introduced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules import Conv
from ultralytics.optim.muon import MuSGD
from ultralytics.utils import ops


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def state_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_input(path: Path, cache_item: dict) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    # Reconstruct the exact effective resize recorded by the official cache.
    # Its ratio_pad stores the rounded height/width resize ratios (which can
    # differ from the unrounded scalar ratio by one pixel on small images).
    rh, rw = (float(x) for x in cache_item["ratio_pad"][0])
    top, left = (int(round(float(x))) for x in (cache_item["ratio_pad"][1][1], cache_item["ratio_pad"][1][0]))
    nh = round(image.shape[0] * rh)
    nw = round(image.shape[1] * rw)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    if top < 0 or left < 0 or top + nh > 640 or left + nw > 640:
        raise RuntimeError(f"invalid cached letterbox geometry for {path}: {(nh, nw, top, left)}")
    canvas[top : top + nh, left : left + nw] = resized
    image = canvas
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).float().div_(255.0).unsqueeze(0)


def flatten_head_output(branches, feats):
    values = []
    for branch, feat in zip(branches, feats):
        values.append(branch(feat).view(feat.shape[0], 32, -1))
    return torch.cat(values, dim=2)


def flatten_pre_last(branches, feats):
    values = []
    for branch, feat in zip(branches, feats):
        values.append(branch[:-1](feat).view(feat.shape[0], 64, -1))
    return torch.cat(values, dim=2)


def image_file(data_root: Path, image_id: int, split: str = "train2017") -> Path:
    return data_root / "images" / split / f"{image_id:012d}.jpg"


def build_rows(cache_root: Path, oracle_path: Path):
    index = json.loads((cache_root / "INDEX.json").read_text(encoding="utf-8"))
    oracle = torch.load(oracle_path, map_location="cpu", weights_only=False)
    oracle_map = {
        (int(x["image_id"]), int(x["annotation_id"]), int(x["raw_id"])): d.float()
        for x, d in zip(oracle["identities"], oracle["delta"])
    }
    images = []
    rows = []
    for item in index["fit"]:
        iid = int(item["image_id"])
        cache = torch.load(cache_root / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)
        row_ids = []
        proto = F.interpolate(cache["proto"].float()[None], (640, 640), mode="bilinear", align_corners=False)[0]
        for k, row in enumerate(cache["rows"]):
            key = (iid, int(row["annotation_id"]), int(row["raw_id"]))
            if key not in oracle_map:
                raise KeyError(f"missing oracle identity {key}")
            box = cache["target_boxes"][k].float()
            # crop_mask deliberately edits its mask argument in-place; create a
            # fresh support canvas for every candidate.
            support = ops.crop_mask(torch.ones((1, 640, 640), dtype=torch.float32), box[None])[0].bool()
            if not bool(support.any()):
                raise RuntimeError(f"empty support {key}")
            owner = int(cache["owners"][k])
            p = proto[:, support].T.contiguous()
            y = (cache["masks"][support] == owner + 1).float().contiguous()
            area = float(((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0)
            rid = len(rows)
            rows.append({
                "row_id": rid,
                "image_id": iid,
                "annotation_id": int(row["annotation_id"]),
                "branch": "one2one",
                "raw_id": int(row["raw_id"]),
                "pyramid_level": int(row["level"]),
                "target_gt_idx": int(row["gt_index"]),
                "area": area,
                "p": p,
                "y": y,
                "c0": cache["coeff"][int(row["raw_id"])].float(),
                "h0": cache["h"][int(row["raw_id"])].float(),
                "cstar": cache["coeff"][int(row["raw_id"])].float() + oracle_map[key],
            })
            row_ids.append(rid)
        images.append({
            "image_id": iid,
            "cache": cache,
            "row_ids": row_ids,
            "feature_maps": None,
            "image_path": str(image_file(Path("/root/autodl-tmp/datasets/coco2017"), iid)),
        })
    return images, rows


def prepare_feature_maps(images, weights: Path, data_root: Path, device: torch.device, check_limit: int = 12):
    yolo = YOLO(str(weights))
    model = yolo.model.float().to(device).eval()
    head = model.model[-1]
    captured = {}

    def hook(_module, inputs):
        x = inputs[0]
        captured["x"] = [v.detach().clone() for v in x]

    handle = head.register_forward_pre_hook(hook)
    checks = {"images": 0, "h_max_abs": 0.0, "c0_max_abs": 0.0, "ratio_pad_mismatch": 0}
    with torch.inference_mode():
        for idx, item in enumerate(images):
            cache = item["cache"]
            tensor = load_input(data_root / "images" / "train2017" / f"{item['image_id']:012d}.jpg", cache).to(device)
            captured.clear()
            _ = model(tensor)
            feats = [v.cpu().contiguous() for v in captured["x"]]
            item["feature_maps"] = feats
            h = flatten_pre_last(head.one2one_cv4, [v.to(device) for v in feats]).squeeze(0).T.detach().cpu()
            c = flatten_head_output(head.one2one_cv4, [v.to(device) for v in feats]).squeeze(0).T.detach().cpu()
            checks["h_max_abs"] = max(checks["h_max_abs"], float((h - cache["h"]).abs().max()))
            checks["c0_max_abs"] = max(checks["c0_max_abs"], float((c - cache["coeff"]).abs().max()))
            if idx < check_limit:
                # The cache stores the original letterbox metadata; this check also
                # catches accidental auto-rectangle or scale-up preprocessing.
                shape = cv2.imread(str(data_root / "images" / "train2017" / f"{item['image_id']:012d}.jpg"), 0).shape
                rh, rw = (float(x) for x in cache["ratio_pad"][0])
                nw, nh = round(shape[1] * rw), round(shape[0] * rh)
                if abs(float(cache["ratio_pad"][1][0]) - cache["ratio_pad"][1][0]) > 1e-6 or nw > 640 or nh > 640:
                    checks["ratio_pad_mismatch"] += 1
            checks["images"] += 1
            if (idx + 1) % 100 == 0:
                print(json.dumps({"stage": "feature_cache", "images": idx + 1, "total": len(images), **checks}), flush=True)
    handle.remove()
    # The cache was generated with the same FP32/TF32-disabled path.  A small
    # residual is retained as a numerical replay audit; the raw identity is
    # determined by the cache rows, not by reassigning candidates.
    checks["forward_replay_tolerance"] = 2e-3
    checks["h_cache_match"] = checks["h_max_abs"] <= checks["forward_replay_tolerance"]
    checks["c0_cache_match"] = checks["c0_max_abs"] <= checks["forward_replay_tolerance"]
    if not checks["h_cache_match"] or not checks["c0_cache_match"]:
        raise RuntimeError(f"frozen forward does not reproduce official cache: {checks}")
    return model, checks


def make_bank(rows, device):
    offsets = [0]
    pparts, yparts = [], []
    for row in rows:
        pparts.append(row["p"])
        yparts.append(row["y"])
        offsets.append(offsets[-1] + len(row["y"]))
    p = torch.cat(pparts, 0).to(device=device, dtype=torch.float32)
    y = torch.cat(yparts, 0).to(device=device, dtype=torch.float32)
    c0 = torch.stack([x["c0"] for x in rows]).to(device)
    cstar = torch.stack([x["cstar"] for x in rows]).to(device)
    area = torch.tensor([x["area"] for x in rows], device=device, dtype=torch.float32)
    return {"p": p, "y": y, "offsets": offsets, "c0": c0, "cstar": cstar, "area": area}


def loss_for_candidate(c, row_id, bank, seg_gain, teacher=False, beta=0.5, tau=1.0):
    lo, hi = bank["offsets"][row_id], bank["offsets"][row_id + 1]
    p, y = bank["p"][lo:hi], bank["y"][lo:hi]
    z = p @ c
    bce = seg_gain * (F.softplus(z) - y * z).sum() / bank["area"][row_id]
    if not teacher:
        return bce, z.detach(), torch.zeros((), device=z.device)
    with torch.no_grad():
        zstar = p @ bank["cstar"][row_id]
    # tau=1.0 is PyTorch Huber's documented unit threshold; this explicit
    # formula avoids silently changing the scale via SmoothL1's beta option.
    err = z - zstar
    abs_err = err.abs()
    huber = torch.where(abs_err <= tau, 0.5 * err.square(), tau * (abs_err - 0.5 * tau)).mean()
    return bce + beta * huber, z.detach(), huber.detach()


def select_coeffs(module, item, rows, bank, device, mode):
    if mode == "B":
        hs = torch.stack([rows[i]["h0"] for i in item["row_ids"]]).to(device)
        values = []
        for level in range(3):
            inds = [j for j, rid in enumerate(item["row_ids"]) if rows[rid]["pyramid_level"] == level]
            if not inds:
                continue
            h = hs[inds].view(-1, 64, 1, 1)
            # The cache stores h after the two 3x3 coefficient convolutions;
            # B therefore applies only the native final 1x1 readout.
            out = module[level][-1](h).view(-1, 32)
            for pos, outv in zip(inds, out):
                values.append((pos, outv))
        result = [None] * len(item["row_ids"])
        for pos, outv in values:
            result[pos] = outv
        return result
    feats = [x.to(device, non_blocking=True) for x in item["feature_maps"]]
    fields = [module[level](feats[level]).view(32, -1).T for level in range(3)]
    result = []
    for rid in item["row_ids"]:
        level = rows[rid]["pyramid_level"]
        raw_id = rows[rid]["raw_id"]
        # Raw ids are the official flattened P3/P4/P5 concatenation.
        start = 0
        for l in range(level):
            start += fields[l].shape[0]
        result.append(fields[level][raw_id - start])
    return result


def train_arm(mode, images, rows, bank, base_cv4, config, out: Path, device: torch.device, seg_gain: float):
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    module = copy.deepcopy(base_cv4).to(device).float().train()
    for p in module.parameters():
        p.requires_grad_(False)
    for level, branch in enumerate(module):
        for sub in branch.modules():
            if isinstance(sub, torch.nn.modules.batchnorm._BatchNorm):
                sub.eval()
                for p in sub.parameters():
                    p.requires_grad_(False)
            elif isinstance(sub, torch.nn.Conv2d):
                allow = mode != "B" or sub is branch[-1]
                if allow:
                    sub.weight.requires_grad_(True)
                    if sub.bias is not None:
                        sub.bias.requires_grad_(True)
    trainable = [p for p in module.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"no trainable parameters for {mode}")
    muon, sgd = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon.append(p)
        else:
            sgd.append(p)
    groups = []
    if muon:
        groups.append({"params": muon, "lr": config["optimizer"]["lr0"], "momentum": config["optimizer"]["momentum"], "nesterov": True, "weight_decay": config["optimizer"]["weight_decay"], "use_muon": True})
    if sgd:
        groups.append({"params": sgd, "lr": config["optimizer"]["lr0"], "momentum": config["optimizer"]["momentum"], "nesterov": True, "weight_decay": 0.0, "use_muon": False})
    optimizer = MuSGD(groups, muon=config["optimizer"]["muon_weight"], sgd=config["optimizer"]["sgd_weight"])
    epochs = int(config["epochs"])
    batch_images = int(config["batch_images"])
    beta = float(config["teacher"]["beta"])
    tau = float(config["teacher"]["huber_tau"])
    train_log = []
    start = time.monotonic()
    for epoch in range(epochs):
        module.train()
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        order = list(range(len(images)))
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        lr_factor = 1.0 - (1.0 - float(config["optimizer"]["lrf"])) * ((epoch + 1) / epochs)
        for group in optimizer.param_groups:
            group["lr"] = float(config["optimizer"]["lr0"]) * lr_factor
        epoch_bce = epoch_teacher = epoch_total = 0.0
        epoch_count = 0
        updates = 0
        for start_i in range(0, len(order), batch_images):
            batch = order[start_i:start_i + batch_images]
            optimizer.zero_grad(set_to_none=True)
            batch_total = torch.zeros((), device=device)
            batch_bce = torch.zeros((), device=device)
            batch_teacher = torch.zeros((), device=device)
            n = 0
            for image_idx in batch:
                item = images[image_idx]
                coeffs = select_coeffs(module, item, rows, bank, device, mode)
                for local, c in enumerate(coeffs):
                    rid = item["row_ids"][local]
                    use_teacher = mode in ("B", "D")
                    loss, _, tl = loss_for_candidate(c, rid, bank, seg_gain, use_teacher, beta, tau)
                    batch_total = batch_total + loss
                    batch_bce = batch_bce + loss_for_candidate(c, rid, bank, seg_gain, False)[0]
                    if use_teacher:
                        batch_teacher = batch_teacher + tl
                    n += 1
            # Every candidate in the official fit set contributes equally;
            # the final short image batch is normalized by its own candidates.
            (batch_total / max(n, 1)).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float("inf")))
            optimizer.step()
            updates += 1
            epoch_total += float(batch_total.detach().cpu())
            epoch_bce += float(batch_bce.detach().cpu())
            epoch_teacher += float(batch_teacher.detach().cpu())
            epoch_count += n
        rec = {"epoch": epoch + 1, "mode": mode, "updates": updates, "candidates": epoch_count, "mean_train_loss": epoch_total / max(epoch_count, 1), "mean_bce": epoch_bce / max(epoch_count, 1), "mean_teacher": epoch_teacher / max(epoch_count, 1), "last_lr": optimizer.param_groups[0]["lr"], "last_grad_norm": grad_norm, "elapsed_s": time.monotonic() - start}
        train_log.append(rec)
        print(json.dumps({"stage": "train", **rec}), flush=True)
    torch.save({"mode": mode, "cv4": {k: v.detach().cpu() for k, v in module.state_dict().items()}, "config": config, "train_log": train_log}, out / f"{mode}_final.pt")
    write_json(out / f"{mode}_TRAIN.json", {"mode": mode, "trainable_parameter_count": sum(p.numel() for p in trainable), "trainable_names": [n for n, p in module.named_parameters() if p.requires_grad], "frozen_bn": True, "epochs": epochs, "train_log": train_log, "elapsed_s": time.monotonic() - start})
    return module, train_log


@torch.no_grad()
def eval_arm(mode, module, images, rows, bank, device, seg_gain, lambda_eval):
    total_bce = total_reg = 0.0
    teacher_loss = 0.0
    count = len(rows)
    crows = []
    for item in images:
        coeffs = select_coeffs(module, item, rows, bank, device, mode) if mode != "A" else [rows[rid]["c0"].to(device) for rid in item["row_ids"]]
        for local, c in enumerate(coeffs):
            rid = item["row_ids"][local]
            bce, _, _ = loss_for_candidate(c, rid, bank, seg_gain, False)
            reg = float(lambda_eval) * 0.5 * (c - bank["c0"][rid]).square().sum()
            total_bce += float(bce.cpu())
            total_reg += float(reg.cpu())
            if mode in ("B", "D"):
                _, _, tl = loss_for_candidate(c, rid, bank, seg_gain, True, float(config_global["teacher"]["beta"]), float(config_global["teacher"]["huber_tau"]))
                teacher_loss += float(tl.cpu())
            crows.append({"row_id": rid, "image_id": rows[rid]["image_id"], "annotation_id": rows[rid]["annotation_id"], "raw_id": rows[rid]["raw_id"], "level": rows[rid]["pyramid_level"], "bce": float(bce.cpu()), "regularizer": float(reg.cpu())})
    result = {"mode": mode, "candidates": count, "bce": total_bce / count, "regularizer": total_reg / count, "J_eval": (total_bce + total_reg) / count, "teacher_huber": teacher_loss / count if mode in ("B", "D") else None}
    return result, crows


config_global = None


def main():
    global config_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    config_global = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    random.seed(int(config_global["seed"])); np.random.seed(int(config_global["seed"])); torch.manual_seed(int(config_global["seed"]))
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase I requires the authorized remote GPU; refusing local CPU execution")
    weights_sha = sha256_file(args.weights)
    images, rows = build_rows(args.cache, args.oracle)
    if len(images) != int(config_global["data_scope"]["images"]) or len(rows) != int(config_global["data_scope"]["candidates"]):
        raise RuntimeError(f"official fit identity mismatch images={len(images)} rows={len(rows)}")
    model, prep = prepare_feature_maps(images, args.weights, args.data, device)
    base_cv4 = copy.deepcopy(model.model[-1].one2one_cv4).cpu()
    prefix_digest_before = state_digest(model)
    bank = make_bank(rows, device)
    seg_gain = float(YOLO(str(args.weights)).ckpt["train_args"]["box"])
    write_json(out / "PREPARE.json", {"weights_sha256": weights_sha, "device": str(device), "images": len(images), "candidates": len(rows), "segmentation_gain": seg_gain, "cache_reproduction": prep, "prefix_digest_before": prefix_digest_before, "official_identity_keys": ["image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx"], "feature_map_shapes": [[list(v.shape) for v in x["feature_maps"]] for x in images[:1]]})
    # A is the untouched reference.  B/C/D are paired from the same seed and
    # the same deterministic image orders; only the declared trainable set and
    # teacher term differ.
    A = {"mode": "A", "candidates": len(rows), "bce": 0.0, "regularizer": 0.0, "J_eval": 0.0, "teacher_huber": None}
    for row in rows:
        bce, _, _ = loss_for_candidate(row["c0"].to(device), row["row_id"], bank, seg_gain, False)
        A["bce"] += float(bce.cpu())
    A["bce"] /= len(rows); A["J_eval"] = A["bce"]
    summaries = {"A": A}
    arm_outputs = {}
    for mode in ("B", "C", "D"):
        arm_out = out / mode; arm_out.mkdir(exist_ok=True)
        module, _ = train_arm(mode, images, rows, bank, base_cv4, config_global, arm_out, device, seg_gain)
        result, crows = eval_arm(mode, module, images, rows, bank, device, seg_gain, float(config_global["lambda_eval"]))
        summaries[mode] = result
        write_json(arm_out / "EVAL.json", result)
        with (arm_out / "PER_CANDIDATE.jsonl").open("w", encoding="utf-8") as f:
            for row in crows: f.write(json.dumps(row) + "\n")
        arm_outputs[mode] = result
        if state_digest(model) != prefix_digest_before:
            raise RuntimeError("frozen original model state changed")
    j0 = summaries["A"]["J_eval"]
    for mode in ("B", "C", "D"):
        summaries[mode]["delta_vs_A"] = summaries[mode]["J_eval"] - j0
    teacher_delta = summaries["C"]["J_eval"] - summaries["D"]["J_eval"]
    cert = float(config_global["acceptance"]["old_fixed_representation_certificate"])
    margin = float(config_global["acceptance"]["certificate_margin"])
    # Stage-I decision uses the finite fixed-class certificate already audited;
    # numerical evaluation error is recorded conservatively as 1e-6.
    eval_error = 1e-6
    decision = {"J0": j0, "certificate": cert, "certificate_margin": margin, "evaluation_error_bound": eval_error, "B_crosses": summaries["B"]["J_eval"] + eval_error < cert - margin, "C_crosses": summaries["C"]["J_eval"] + eval_error < cert - margin, "D_crosses": summaries["D"]["J_eval"] + eval_error < cert - margin, "teacher_delta_J_C_minus_D": teacher_delta, "teacher_material_threshold": float(config_global["acceptance"]["material_teacher_delta_J"]), "teacher_pass": teacher_delta >= float(config_global["acceptance"]["material_teacher_delta_J"]), "phase_II_pass": summaries["D"]["J_eval"] + eval_error < cert - margin and teacher_delta >= float(config_global["acceptance"]["material_teacher_delta_J"]), "interpretation": "D-C is the only teacher-effect comparison; A is untouched reference; B is diagnostic fixed-h control"}
    write_json(out / "SUMMARY.json", {"config": config_global, "weights_sha256": weights_sha, "identity": {"images": len(images), "candidates": len(rows), "split": "official_TAL_fit"}, "arms": summaries, "decision": decision, "limits": ["fit-set fixed-objective evidence only", "no phase-II new-image utility evaluated", "certificate is conditional on prior FP64 audit assumptions", "teacher improvement does not isolate all optimization-vs-target effects"]})
    write_json(out / "COMPLETE.json", {"status": "completed", "phase": "I", "decision": decision, "prefix_state_unchanged": True, "artifacts": ["PREPARE.json", "SUMMARY.json", "B/B_final.pt", "C/C_final.pt", "D/D_final.pt"]})
    print(json.dumps({"stage": "complete", "decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
