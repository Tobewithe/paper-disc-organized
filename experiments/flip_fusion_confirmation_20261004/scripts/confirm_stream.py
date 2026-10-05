"""Stream official TAL A/MIX confirmation from new COCO images, without training.

No old tensor cache, operator matrix, optimizer, oracle or projection is used.
The fixed source matcher and normal mask metrics are imported unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

if "--config" in sys.argv:
    _startup = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    sys.path.insert(0, _startup["source_python"])

import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.converter import coco91_to_coco80_class
import ultralytics.data.augment as augmentation
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou

import legacy_prepare_cache as official
from online_runtime import dump, load_json, resolve_runtime_config, sha256, tensor_sha
from flip_source import FlipSource
import evaluation_metrics as em

ARMS = ("A", "MIX")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
KEYS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
em.ARMS = ARMS


def key(row):
    return tuple(row[k] for k in KEYS)


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed confirmation resource limit reached; retain incomplete run, do not shorten cohort")


def ids_from(path):
    value = load_json(path)
    ids = value.get("image_ids", value.get("confirm")) if isinstance(value, dict) else value
    if not isinstance(ids, list):
        raise ValueError("confirm_ids must be a JSON list or an object with image_ids/confirm list")
    ids = list(map(int, ids))
    if len(set(ids)) != len(ids):
        raise AssertionError("Repeated confirmation image ID")
    return ids


def ensure_images(cfg, ids, deadline):
    """Extract only selected original JPEGs, with bytes/hash verification."""
    target = Path(cfg["images"]) / "train2017"
    target.mkdir(parents=True, exist_ok=True)
    receipt = []
    with zipfile.ZipFile(Path(cfg["coco_root"]) / "train2017.zip") as archive:
        for iid in ids:
            check_time(deadline)
            name = f"{iid:012d}.jpg"
            data = archive.read("train2017/" + name)
            digest = hashlib.sha256(data).hexdigest()
            path = target / name
            if path.exists():
                if sha256(path) != digest:
                    raise AssertionError(f"Existing original JPEG has different bytes: {iid}")
            else:
                temporary = path.with_suffix(".jpg.tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
            receipt.append(dict(image_id=iid, source_split="train", sha256=digest, bytes=len(data)))
    return receipt


@torch.no_grad()
def normal_output_metadata(head, raw, output, ids, target_classes, predicted_classes):
    """Track raw->top300 output, preserving possible multi-class duplicates."""
    probabilities = raw["scores"].sigmoid().permute(0, 2, 1)
    scores, classes, indices = head.get_topk_index(probabilities, 300)
    all_boxes = head._get_decode_boxes(raw).permute(0, 2, 1)
    all_coefficients = raw["mask_coefficient"].permute(0, 2, 1)
    reconstructed = torch.cat((all_boxes.gather(1, indices.repeat(1, 1, 4)), scores, classes,
                               all_coefficients.gather(1, indices.repeat(1, 1, 32))), dim=-1)
    if not torch.equal(reconstructed, output):
        raise AssertionError("Raw-index tracking does not reproduce the official top300 output exactly")
    by_raw = defaultdict(list)
    for rank, (rid, score, cls) in enumerate(zip(indices[0, :, 0].tolist(),
                                                scores[0, :, 0].tolist(), classes[0, :, 0].tolist()), 1):
        by_raw[int(rid)].append(dict(output_rank=rank, predicted_class_id=int(cls), score=float(score),
                                    passes_conf_001=float(score) > .001))
    result = []
    for rid, gt_class, raw_class in zip(ids.tolist(), target_classes, predicted_classes.tolist()):
        entries = by_raw[int(rid)]
        kept = [r for r in entries if r["passes_conf_001"]]
        exact = [r for r in kept if r["predicted_class_id"] == int(raw_class)]
        correct = int(raw_class) == int(gt_class)
        result.append(dict(normal_top300_present=bool(entries), normal_output_entries=entries,
                           normal_top300_best_rank=min((r["output_rank"] for r in entries), default=None),
                           retained_anyclass=bool(kept), retained_exactclass=bool(exact),
                           retained_exactclass_best_rank=min((r["output_rank"] for r in exact), default=None),
                           predicted_class_matches_gt=correct,
                           retained_exactclass_correct=bool(exact) and correct,
                           retained_definition="original top300 score > .001 before empty-mask filtering",
                           output_class_semantics="source matcher still uses original raw argmax, not a selected nonargmax output class"))
    return result


@torch.no_grad()
def direct_metrics(logits, image, j, coco):
    if logits.shape != (640, 640) or not bool(torch.isfinite(logits).all()):
        raise AssertionError("Invalid complete continuous source response")
    support = ops.crop_mask(torch.ones((1, 640, 640), device=logits.device), image["boxes"][j:j+1])[0].bool()
    binary = (logits.gt(0) & support).byte()[None]
    cropped = ops.crop_mask(logits[None].clone(), image["boxes"][j:j+1]).gt(0).byte()
    if not torch.equal(binary, cropped):
        raise AssertionError("Direct response crop/threshold mismatch")
    mask = em._scale_binary(binary, tuple(image["original_shape"]), image["ratio_pad"])[0]
    ann = coco.anns[int(image["rows"][j]["annotation_id"])]
    if int(ann["image_id"]) != int(image["image_id"]):
        raise AssertionError("Evaluation annotation belongs to a different image")
    gt = torch.as_tensor(coco.annToMask(ann).astype(bool), device=logits.device)
    intersection = int((mask & gt).sum())
    iou = intersection / max(1, int((mask | gt).sum()))
    padded = em._padded_gt(gt, image["ratio_pad"], (640, 640))
    auc, fpr = em._pixel_auc_fpr(logits, padded, support)
    return dict(iou=iou, mask75=int(iou >= .75), coverage=intersection/max(1, int(gt.sum())),
                auc=auc, fpr=fpr, empty_mask=not bool(mask.any()))


@torch.no_grad()
def baseline_pixel_check(logits, image, j):
    boxes = image["boxes"][j:j+1]
    native = ops.process_mask(image["proto"], image["c0"][j:j+1], boxes, (640, 640), upsample=True)
    direct = ops.crop_mask(logits[None].clone(), boxes).gt(0).byte()
    padded_difference = int((native != direct).sum())
    native_orig = em._scale_binary(native, tuple(image["original_shape"]), image["ratio_pad"])
    direct_orig = em._scale_binary(direct, tuple(image["original_shape"]), image["ratio_pad"])
    original_difference = int((native_orig != direct_orig).sum())
    if padded_difference or original_difference:
        raise AssertionError("A direct full-response decoding differs pixelwise from original process_mask")
    return dict(padded_pixel_difference=padded_difference, original_pixel_difference=original_difference)


def summarize(rows, out, population, seed, smoke):
    spec = {"all": lambda r: True, "box_good_mask_bad": lambda r: bool(r["box_good_mask_bad"]),
            "original_success": lambda r: bool(r["mask75_A"]), "original_failure": lambda r: not r["mask75_A"],
            "retained_anyclass": lambda r: bool(r["retained_anyclass"]),
            "retained_exactclass": lambda r: bool(r["retained_exactclass"]),
            "retained_exactclass_correct": lambda r: bool(r["retained_exactclass_correct"])}
    for level in range(3):
        spec["P" + str(level+3)] = lambda r, l=level: r["pyramid_level"] == l
    for size in ("small", "medium", "large"):
        spec[size] = lambda r, s=size: r["size_group"] == s
    tables, per_image = {}, []
    for name, predicate in spec.items():
        selected = [r for r in rows if predicate(r)]
        groups = em.image_groups(selected)
        table = dict(images=len(groups), candidates=len(selected), candidate={}, image_macro={}, undefined={}, comparisons={})
        for metric in METRICS:
            table["candidate"][metric] = {a: em.avg([r.get(metric+"_"+a) for r in selected]) for a in ARMS}
            table["image_macro"][metric] = {a: em.avg([em.avg([r.get(metric+"_"+a) for r in g]) for g in groups]) for a in ARMS}
            table["undefined"][metric] = {a: sum(not em.finite(r.get(metric+"_"+a)) for r in selected) for a in ARMS}
        comparison = {m: em.paired(groups, "MIX", "A", m, seed, 5000) for m in METRICS}
        repair = sum(r["mask75_MIX"] == 1 and r["mask75_A"] == 0 for r in selected)
        damage = sum(r["mask75_MIX"] == 0 and r["mask75_A"] == 1 for r in selected)
        comparison["crossings"] = dict(repair=repair, damage=damage, net=repair-damage)
        differences = [r["iou_MIX"]-r["iou_A"] for r in selected]
        comparison["continuous_iou_changes"] = dict(
            increased=sum(d > 0 for d in differences), decreased=sum(d < 0 for d in differences),
            unchanged=sum(d == 0 for d in differences), increased_more_than_one_pp=sum(d > .01 for d in differences),
            decreased_more_than_one_pp=sum(d < -.01 for d in differences),
            empty_A=sum(bool(r["empty_mask_A"]) for r in selected),
            empty_MIX=sum(bool(r["empty_mask_MIX"]) for r in selected))
        table["comparisons"]["MIX_minus_A"] = comparison
        tables[name] = table
        for group in groups:
            item = dict(split=group[0]["split"], image_id=group[0]["image_id"], group=name, candidates=len(group))
            for metric in METRICS:
                for arm in ARMS:
                    item[metric+"_"+arm] = em.avg([r.get(metric+"_"+arm) for r in group])
                item["delta_"+metric+"_MIX_minus_A"] = em.avg([r[metric+"_MIX"]-r[metric+"_A"] for r in group
                    if em.finite(r.get(metric+"_MIX")) and em.finite(r.get(metric+"_A"))])
            per_image.append(item)
    em.append_rows(out/"PER_IMAGE.jsonl", per_image)
    target = tables["box_good_mask_bad"]["comparisons"]["MIX_minus_A"]["iou"]["image_macro"]
    overall = tables["all"]["comparisons"]["MIX_minus_A"]["iou"]["image_macro"]
    target_signal = (em.finite(target["delta"]) and target["delta"] >= .005
                     and em.finite(target["ci95"][0]) and target["ci95"][0] > 0)
    overall_guard = em.finite(overall["ci95"][0]) and overall["ci95"][0] > -.001
    decisions = dict(prespecified_target_signal=bool(target_signal),
                     prespecified_overall_noninferiority_guard=bool(overall_guard),
                     primary_and_overall_guard_passed=bool(target_signal and overall_guard),
                     statistical_rule="target estimate>=.005 AND target CI lower>0 AND all IoU CI lower>-.001",
                     all_metrics_and_success_damage_require_tradeoff_review=True,
                     smoke_not_a_confirmation=bool(smoke))
    result = dict(schema="frozen-flip-mix-confirmation-v1", population=population, arms=ARMS, tables=tables,
                  bootstrap=dict(draws=5000, seed=seed, unit="paired image; candidate average also resamples whole images"),
                  no_training=True, alpha=.5, no_projection=True, method_frozen=True, smoke=smoke, decisions=decisions,
                  scope="Method-development holdout under documented historical exclusion; train2017 seen by official pretraining and earlier full-COCO project training. Official TAL GT-conditioned candidate confirmation, not globally unseen data",
                  retained_subset_scope="Only the TAL candidates present in original top300/conf>.001; excludes other outputs, false positives and unmatched GT; not complete deployment evaluation",
                  automatic_followup=False)
    dump(out/"SUMMARY.json", em.clean(result))
    def pp(value):
        return "undefined" if value is None else f"{100*value:+.4f}"
    lines = ["# Frozen A/MIX confirmation", "", f"{population['planned_images']} planned images; {population['effective_images']} effective images; {len(rows)} official TAL candidates.", "",
             "Same frozen original model, full-canvas horizontal flip, source matcher and equal continuous-logit fusion. No training or projection.", "",
             "| Group | Images / candidates | Macro IoU change pp [95% CI] | Candidate IoU change pp | Repair / damage | Coverage pp | AUC pp | FPR pp |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, table in tables.items():
        c = table["comparisons"]["MIX_minus_A"]; m = c["iou"]["image_macro"]; cr = c["crossings"]
        lines.append(f"| {name} | {table['images']} / {table['candidates']} | {pp(m['delta'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}] | {pp(c['iou']['candidate']['delta'])} | {cr['repair']} / {cr['damage']} | {pp(c['coverage']['image_macro']['delta'])} | {pp(c['auc']['image_macro']['delta'])} | {pp(c['fpr']['image_macro']['delta'])} |")
    lines += ["", "All candidates remain in the denominator after correction, including empty or degraded masks. Undefined AUC/FPR does not remove IoU observations.",
              "The original-failure group has structurally zero Mask75 damage versus A; this does not establish absence of continuous degradation.",
              "Retained subsets are secondary same-candidate diagnostics, not all output predictions or COCO AP. Source matching uses predicted classes only; GT class agreement is evaluation metadata.",
              "Predeclared confirmation decisions and historical image exclusion provenance are specified by this Study's frozen protocol. No automatic method changes follow."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def run(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Model execution permitted only on the authorized Linux CUDA server")
    started = time.monotonic()
    cfg = resolve_runtime_config(load_json(args.config))
    limit = 240 if args.smoke else 1800
    deadline = started + limit
    if cfg.get("alpha", .5) != .5 or int(cfg.get("bootstrap", 5000)) != 5000:
        raise AssertionError("Frozen method or statistical configuration changed")
    if cfg.get("target_practical_effect", .005) != .005 or cfg.get("overall_noninferiority_margin", .001) != .001:
        raise AssertionError("Predeclared confirmation effect/guard threshold changed")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"PER_CANDIDATE.jsonl").exists() or (out/"COMPLETE.json").exists():
        raise RuntimeError("Existing Run output must be preserved; use an independent Run")
    new_ids = ids_from(cfg["confirm_ids"])
    if len(new_ids) != int(cfg["planned_images"]):
        raise AssertionError("Confirmation manifest disagrees with planned image count")
    exclusion_path = Path(cfg.get("exclusion_ids", str(Path(cfg["confirm_ids"]).with_name("EXCLUSION_IDS.json"))))
    exclusions = load_json(exclusion_path)
    excluded_ids = set(map(int, exclusions["excluded_train2017_ids"]))
    if set(new_ids) & excluded_ids:
        raise AssertionError("Formal confirmation IDs overlap the frozen historical exclusion set")
    prior_rows = [json.loads(line) for line in Path(cfg["prior_results"]).read_text().splitlines() if line.strip()]
    if len({key(r) for r in prior_rows}) != len(prior_rows):
        raise AssertionError("Historical A/MIX records have duplicate identities")
    prior_images = list(dict.fromkeys(int(r["image_id"]) for r in prior_rows))
    if set(new_ids) & set(prior_images):
        raise AssertionError("Confirmation images overlap this method-selection development set")
    ids = prior_images[:2] if args.smoke else new_ids
    if args.smoke and len(ids) != 2:
        raise AssertionError("Two historical smoke images are required")
    group = "dev" if args.smoke else "confirm"
    previous = {key(r): r for r in prior_rows if int(r["image_id"]) in set(ids)} if args.smoke else {}
    dump(out/"FROZEN_IMAGE_IDS.json", dict(split=group, image_ids=ids, planned_images=len(ids), smoke=bool(args.smoke)))
    inputs = {str(p): sha256(p) for p in (Path(args.config), Path(cfg["confirm_ids"]), Path(cfg["prior_results"]),
                Path(cfg["weights"]), Path(cfg["annotations_train"]), Path(cfg["annotations_val"]), exclusion_path)}
    dump(out/"INPUT_MANIFEST.json", inputs)
    receipts = ensure_images(cfg, ids, deadline)
    dump(out/"SOURCE_IMAGES_EXTRACTED.json", receipts)
    check_time(deadline)
    source = FlipSource(cfg)
    model, head = source.source, source.head
    model.args = SimpleNamespace(**source.replay.train_args)
    if model.args.mask_ratio != 1 or not model.args.overlap_mask:
        raise AssertionError("Official mask_ratio/overlap settings changed")
    criterion = model.init_criterion().one2one
    if int(criterion.assigner.topk) != 7 or int(criterion.assigner.topk2) != 1:
        raise AssertionError("Original official one-to-one TAL configuration differs")
    dump(out/"ENVIRONMENT.json", source.import_info)
    signature = dict(weights_sha256=source.weights_sha256, image_ids=ids,
                     conversion_code_sha256=sha256(official.__file__), annotations_sha256=sha256(cfg["annotations_train"]),
                     script_sha256=sha256(__file__), split=group)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    converted, mapping, source_meta = official.prepare_dataset(cfg, {"fit": [], "dev": ids, "val": []},
                                                               out/"dataset", fingerprint)
    check_time(deadline)
    listing = out/"dataset"/"confirmation_input.txt"
    listing.write_text("\n".join(str(converted/"images"/"train2017"/f"{iid:012d}.jpg") for iid in ids)+"\n", encoding="utf-8")
    dataset = YOLODataset(img_path=str(listing), imgsz=640, batch_size=1, augment=False,
                         hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
                         data={"names": model.names, "nc": 80, "channels": 3}, task="segment")
    positions = {int(Path(p).stem): j for j, p in enumerate(dataset.im_files)}
    if set(positions) != set(ids):
        raise AssertionError("Official dataset changed the planned image cohort")
    coco = COCO(cfg["annotations_train"])
    classmap = coco91_to_coco80_class()
    order = {}
    old_rasterizer = augmentation.polygons2masks_overlap
    def rasterizer(*a, **kw):
        masks, indices = old_rasterizer(*a, **kw)
        order["indices"] = indices.copy()
        return masks, indices
    augmentation.polygons2masks_overlap = rasterizer
    all_rows, source_rows, image_audits, identity_rows, no_positive = [], [], [], [], []
    seen = set(); reference_error = 0.; pixel_difference_max = 0
    try:
        for number, iid in enumerate(ids, 1):
            check_time(deadline)
            labels = dataset.labels[positions[iid]]
            before_order = []
            for cls, polygon in zip(labels["cls"].flatten(), labels["segments"]):
                serial = np.r_[cls, polygon.flatten()].astype(np.float32)
                matches = [aid for arr, aid in mapping[iid] if np.array_equal(serial, arr)]
                if len(matches) != 1:
                    raise AssertionError(f"Ambiguous exact official annotation identity: {iid}")
                before_order.append(matches[0])
            order.clear()
            sample = dataset[positions[iid]]
            ann_ids = np.asarray(before_order)[order["indices"]].tolist() if before_order else []
            if len(ann_ids) != len(sample["cls"]):
                raise AssertionError("Overlap instance sorting and annotation identities differ")
            batch = YOLODataset.collate_fn([sample])
            batch = {k: v.to(source.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if tuple(batch["img"].shape) != (1, 3, 640, 640) or tuple(batch["masks"].shape[-2:]) != (640, 640):
                raise AssertionError("Official full image/mask resolution changed")
            with torch.no_grad():
                output, branches = model(batch["img"].float()/255.)
                retained, returned_proto = output
                pred = branches["one2one"]
                assigned, _, _ = criterion.get_assigned_targets_and_loss(pred, batch)
                foreground, ownership, target_boxes = (value[0] for value in assigned[:3])
                raw_ids = torch.where(foreground)[0]
                owners = ownership[raw_ids]
                if len(set(owners.tolist())) != len(owners):
                    raise AssertionError("Official one-to-one assignment repeated a GT owner")
                if [tuple(f.shape[-2:]) for f in pred["feats"]] != [(80, 80), (40, 40), (20, 20)]:
                    raise AssertionError("Original raw tensor spatial layout changed")
                levels = torch.cat([torch.full((f.shape[-2]*f.shape[-1],), l, device=source.device, dtype=torch.long)
                                    for l, f in enumerate(pred["feats"])])[raw_ids]
                c0 = pred["mask_coefficient"][0].T[raw_ids]
                boxes = head._get_decode_boxes(pred)[0].T[raw_ids]
                probabilities = pred["scores"][0, :, raw_ids].sigmoid()
                scores, classes = probabilities.max(0)
                if not torch.equal(returned_proto, pred["proto"]):
                    raise AssertionError("Returned prototype and official candidate prototype differ")
                original_boxes = ops.scale_boxes((640, 640), boxes.clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"])
            candidate_rows, gt_classes = [], []
            for j, (rid, owner) in enumerate(zip(raw_ids.tolist(), owners.tolist())):
                ann = coco.anns[int(ann_ids[owner])]
                gt_class = int(classmap[ann["category_id"]-1])
                if gt_class != int(sample["cls"][owner].item()):
                    raise AssertionError("Official sorted target class disagrees with its original annotation")
                gt_classes.append(gt_class)
                gt_box = boxes.new_tensor(ann["bbox"]); gt_box[2:] += gt_box[:2]
                candidate_rows.append(dict(split=group, image_id=iid, annotation_id=int(ann["id"]), branch="one2one",
                    raw_id=int(rid), pyramid_level=int(levels[j]), target_gt_idx=int(owner), candidate_index=j,
                    area=float(ann["area"]), box_iou=float(box_iou(gt_box[None], original_boxes[j:j+1])[0, 0]),
                    predicted_class_id=int(classes[j]), predicted_score=float(scores[j])))
            normal = normal_output_metadata(head, pred, retained, raw_ids, gt_classes, classes)
            identity_rows.extend([{**r, **m} for r, m in zip(candidate_rows, normal)])
            if not candidate_rows:
                no_positive.append(iid)
                continue
            image = dict(image_id=iid, split=group, rows=candidate_rows, input_shape=[640, 640],
                         input_uint8=sample["img"].cpu().contiguous(), proto=pred["proto"][0].detach(),
                         c0=c0.detach(), boxes=boxes.detach(), raw_ids=raw_ids.detach(), levels=levels.detach(),
                         original_shape=sample["ori_shape"], ratio_pad=sample["ratio_pad"],
                         target_boxes=target_boxes[raw_ids].detach())
            payload = {k: image[k] for k in ("input_uint8", "boxes", "c0", "proto", "raw_ids", "levels")}
            direct = []
            for record in source.infer(payload, deadline_monotonic=deadline):
                j = int(record["candidate_index"])
                if j != len(direct):
                    raise AssertionError("Frozen source changed compact candidate order")
                check = baseline_pixel_check(record["baseline_logits"], image, j)
                pixel_difference_max = max(pixel_difference_max, check["padded_pixel_difference"], check["original_pixel_difference"])
                direct.append(direct_metrics(.5*(record["baseline_logits"]+record["match_logits"]), image, j, coco))
                meta = {k: v for k, v in record["match"].items() if k != "source_coefficient"}
                source_rows.append(dict(image_id=iid, candidate_index=j, raw_id=int(raw_ids[j]), source_match=meta,
                                        baseline_pixel_audit=check))
            if len(direct) != len(candidate_rows):
                raise AssertionError("Frozen fusion source omitted a candidate")
            current = em.evaluate_image(image, {"A": c0}, coco)
            for j, row in enumerate(current):
                for metric, value in direct[j].items():
                    row[metric+"_MIX"] = value
                row.update(normal[j])
                row["normal_retained_A_nonempty"] = row["retained_anyclass"] and not row["empty_mask_A"]
                row["source_match"] = source_rows[-len(current)+j]["source_match"]
                identity = key(row)
                if identity in seen:
                    raise AssertionError("Repeated permanent candidate identity")
                seen.add(identity)
                if args.smoke:
                    if identity not in previous:
                        raise AssertionError("Historical smoke candidate identity changed")
                    for arm in ARMS:
                        for metric in METRICS:
                            value, old = row.get(metric+"_"+arm), previous[identity].get(metric+"_"+arm)
                            if em.finite(value) != em.finite(old):
                                raise AssertionError("Historical metric definedness changed")
                            if em.finite(value):
                                reference_error = max(reference_error, abs(value-old))
                    if reference_error > 1e-12:
                        raise AssertionError("Historical A/MIX five-metric replay differs")
            all_rows.extend(current)
            em.append_rows(out/"PER_CANDIDATE.jsonl", current)
            image_audits.append(dict(image_id=iid, **source.last_audit, all_annotation_ids=ann_ids,
                                    input_uint8_sha256=tensor_sha(image["input_uint8"]), masks_sha256=tensor_sha(sample["masks"]),
                                    proto_sha256=tensor_sha(image["proto"]), ratio_pad=image["ratio_pad"], original_shape=image["original_shape"]))
            if number % 25 == 0 or number == len(ids):
                progress = dict(stage="confirming_frozen_mix", images_done=number, images_planned=len(ids),
                                candidates_done=len(all_rows), seconds=time.monotonic()-started)
                dump(out/"PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
            del image, payload, current, batch, pred, branches, output, retained, returned_proto
        check_time(deadline)
        source.assert_unchanged()
        if args.smoke and seen != set(previous):
            raise AssertionError("Historical smoke population incomplete")
        if len(image_audits)+len(no_positive) != len(ids):
            raise AssertionError("Fixed confirmation cohort not fully processed")
        population = dict(planned_images=len(ids), effective_images=len(image_audits), candidates=len(all_rows),
                          no_positive_images=no_positive, smoke=bool(args.smoke), source_split="train2017")
        dump(out/"OFFICIAL_MANIFEST.json", dict(fingerprint=fingerprint, population=population, candidates=identity_rows))
        em.append_rows(out/"SOURCE_ROWS.jsonl", source_rows)
        audit = dict(passed=True, population=population, no_training=True, original_model_unchanged=True,
                     historical_smoke_reference_max_error=reference_error if args.smoke else None,
                     baseline_direct_process_mask_pixel_difference_max=pixel_difference_max,
                     identity_unique=True, source_image_audits=image_audits,
                     image_ids_sha256=sha256(out/"FROZEN_IMAGE_IDS.json"), source_tensor_cache_used=False,
                     frozen_historical_exclusion_sha256=sha256(exclusion_path), formal_ids_disjoint_from_exclusions=True,
                     GT_free_source_payload_fields=sorted(payload.keys()) if "payload" in locals() else sorted(("input_uint8", "boxes", "c0", "proto", "raw_ids", "levels")),
                     normal_output_metadata_only=True, no_full_output_matching=True, no_COCO_AP=True,
                     limit_seconds=limit)
        dump(out/"AUDIT.json", em.clean(audit))
        check_time(deadline)
        summarize(all_rows, out, population, int(cfg.get("bootstrap_seed", 20261004)), bool(args.smoke))
        check_time(deadline)
        dump(out/"COMPLETE.json", dict(passed=True, completed=True, smoke=bool(args.smoke), population=population,
                                      seconds=time.monotonic()-started, no_training=True, no_automatic_followup=True))
    finally:
        augmentation.polygons2masks_overlap = old_rasterizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        failure = dict(completed=False, passed=False, error=repr(exc), traceback=traceback.format_exc())
        dump(Path(args.out)/"FAILURE.json", failure)
        dump(Path(args.out)/"COMPLETE.json", failure)
        raise

