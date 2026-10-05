"""Frozen one2one coefficient experiments with official data, TAL and mask objective.

This replays supervision on an unaugmented checkpoint. It does not reconstruct
the original training history. COCO ann IDs are metadata, never label assignment.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import ultralytics
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as augment_module
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.metrics import box_iou

from feature_probes import Probe


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def setup():
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def prepare_dataset(a, split, names):
    """Official conversion; keep source IDs through both official reorderings."""
    inp = a.out / "conversion_input"
    inp.mkdir()
    converted = a.out / "official_data"
    source, identities = {}, {}
    classmap = coco91_to_coco80_class()
    excluded = Counter()
    for domain in ("train", "val"):
        with (a.data / "annotations" / f"instances_{domain}2017.json").open() as f:
            data = json.load(f)
        selected = set(split["val"] if domain == "val" else split["fit"] + split["dev"])
        data["images"] = [r for r in data["images"] if r["id"] in selected]
        data["annotations"] = [r for r in data["annotations"] if r["image_id"] in selected]
        ims = {r["id"]: r for r in data["images"]}
        by_image = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)
        for iid in selected:
            kept, seen = [], []
            for ann in by_image[iid]:
                if ann.get("iscrowd", False):
                    excluded["crowd"] += 1
                    continue
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= ims[iid]["width"]
                box[[1, 3]] /= ims[iid]["height"]
                if box[2] <= 0 or box[3] <= 0:
                    excluded["invalid_box"] += 1
                    continue
                key = [classmap[ann["category_id"] - 1], *box.tolist()]
                if key in seen:
                    excluded["converter_duplicate_box"] += 1
                    continue
                assert isinstance(ann.get("segmentation"), list) and ann["segmentation"], ann["id"]
                seen.append(key)
                kept.append(ann["id"])
            identities[iid] = kept
        write(inp / f"instances_{domain}2017.json", data)
        source[domain] = COCO(str(inp / f"instances_{domain}2017.json"))
    convert_coco(str(inp), str(converted), use_segments=True)
    for domain in ("train", "val"):
        (converted / "images" / f"{domain}2017").symlink_to(a.data / "images" / f"{domain}2017", target_is_directory=True)
    mapping = {}
    for group, ids in split.items():
        domain = "val" if group == "val" else "train"
        paths = [converted / "images" / f"{domain}2017" / source[domain].imgs[i]["file_name"] for i in ids]
        (converted / f"{group}.txt").write_text("\n".join(map(str, paths)) + "\n")
        for iid in ids:
            label = converted / "labels" / f"{domain}2017" / f"{iid:012d}.txt"
            lines = label.read_text().strip().splitlines() if label.exists() else []
            assert len(lines) == len(identities[iid]), (iid, len(lines), len(identities[iid]))
            # Read the exact official serialized polygons; lookup also survives
            # verify_image_label's duplicate removal and row reordering.
            mapping[iid] = [(np.asarray(s.split(), dtype=np.float32), aid) for s, aid in zip(lines, identities[iid])]
    write(a.out / "CONVERSION.json", dict(excluded=excluded, source_annotation_ids=identities,
        scope="Official convert_coco use_segments=True; multi-polygons merged per annotation. Evaluation retains original COCO masks."))
    return converted, source, mapping


def decode(image, coefficients, raw_ids):
    """Official normal-mask decoding followed by recorded letterbox inversion."""
    proto = image["proto"].cuda()
    boxes = image["boxes"][raw_ids].cuda()
    pieces = []
    for lo in range(0, len(coefficients), 16):
        m = ops.process_mask(proto, coefficients[lo:lo+16], boxes[lo:lo+16], (640, 640), upsample=True)
        m = ops.scale_masks(m[None], image["original_shape"], ratio_pad=image["ratio_pad"])[0] > 0.5
        pieces.extend(m.cpu().numpy())
    return pieces


def target_rois(image, indices=None):
    """Only remove the zero-weight outside-GT-box pixels from official BCE.

    Equivalence in values AND coefficient gradients is checked against the
    directly invoked official SegmentationLoss.loss during cache construction.
    """
    up = F.interpolate(image["proto"].cuda()[None], (640, 640), mode="bilinear", align_corners=False)[0]
    mask = image["masks"].cuda()
    selected = range(len(image["rows"])) if indices is None else indices
    result = []
    for k in selected:
        box = image["target_boxes"][k].cuda()
        support = ops.crop_mask(torch.ones(1, 640, 640, device="cuda"), box[None])[0].bool()
        pixels = up[:, support].T.contiguous()
        y = (mask[support] == image["owners"][k].item() + 1).float()
        area = ((box[2:] - box[:2]) / 640).prod() * (640 * 640)
        assert float(area) > 0
        result.append((pixels, y, area))
    return result


def roi_losses(coefficients, rois):
    # coefficients: arms x instances x channels. Sum over instances as official.
    result = coefficients.sum((1, 2)) * 0
    for k, (p, y, area) in enumerate(rois):
        z = coefficients[:, k] @ p.T
        result = result + F.binary_cross_entropy_with_logits(z, y.expand_as(z), reduction="none").sum(1) / area
    return result


def cache(a):
    split = json.loads(a.split.read_text())
    wrapper = YOLO(str(a.weights))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio == 1 and model.args.overlap_mask
    for p in model.parameters():
        p.requires_grad_(False)
    criteria = model.init_criterion()
    criterion = criteria.one2one
    head = model.model[-1]
    data, sources, identity = prepare_dataset(a, split, wrapper.names)
    captured = {}
    hooks = [branch[-1].register_forward_pre_hook(lambda _, x, level=l: captured.__setitem__(level, x[0].detach()))
             for l, branch in enumerate(head.one2one_cv4)]
    order = {}
    original_rasterizer = augment_module.polygons2masks_overlap
    def record_order(*args, **kwargs):
        masks, indices = original_rasterizer(*args, **kwargs)
        order["indices"] = indices.copy()
        return masks, indices
    augment_module.polygons2masks_overlap = record_order
    (a.out / "images").mkdir()
    index, counts, audits = {}, Counter(), []
    start = time.monotonic()
    for group in ("fit", "dev", "val"):
        domain = "val" if group == "val" else "train"
        source = sources[domain]
        ds = YOLODataset(img_path=str(data / f"{group}.txt"), imgsz=640, batch_size=1,
            augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
            data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
        assert {int(Path(p).stem) for p in ds.im_files} == set(split[group])
        by_id = {int(Path(p).stem): j for j, p in enumerate(ds.im_files)}
        index[group] = []
        for step, iid in enumerate(split[group], 1):
            j = by_id[iid]
            raw_label = ds.labels[j]
            pre_ids = []
            for cls, poly in zip(raw_label["cls"].flatten(), raw_label["segments"]):
                key = np.r_[cls, poly.flatten()].astype(np.float32)
                matches = [aid for arr, aid in identity[iid] if np.array_equal(key, arr)]
                assert len(matches) == 1, (iid, matches)
                pre_ids.append(matches[0])
            order.clear()
            sample = ds[j]
            annids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
            assert len(annids) == len(sample["cls"])
            batch = YOLODataset.collate_fn([sample])
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            x = batch["img"].float() / 255
            assert x.shape == (1, 3, 640, 640) and batch["masks"].shape[-2:] == (640, 640)
            with torch.no_grad():
                _, raw = model(x)
                assignment, _, _ = criterion.get_assigned_targets_and_loss(raw["one2one"], batch)
                many, _, _ = criteria.one2many.get_assigned_targets_and_loss(raw["one2many"], batch)
            fg, owner, target_boxes = (t[0] for t in assignment[:3])
            ids = torch.where(fg)[0]
            owners = owner[ids]
            assert len(set(owners.tolist())) == len(owners), "one2one has repeated GT"
            pred = raw["one2one"]
            proto = pred["proto"][0]
            coeff = pred["mask_coefficient"][0].T
            h = torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            recovered = torch.cat([head.one2one_cv4[l][-1](captured[l])[0].flatten(1).T for l in range(3)])
            torch.testing.assert_close(recovered, coeff, atol=1e-5, rtol=1e-5)
            boxes = head._get_decode_boxes(pred)[0].T
            levels = torch.cat([torch.full((captured[l].shape[-2] * captured[l].shape[-1],), l, dtype=torch.long) for l in range(3)])
            scores, classes, top_ids = head.get_topk_index(pred["scores"].permute(0, 2, 1).sigmoid(), 300)
            present = {(r, c) for r, c, s in zip(top_ids.flatten().tolist(), classes.flatten().tolist(), scores.flatten().tolist()) if s > .001}
            original_boxes = ops.scale_boxes((640, 640), boxes.clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"])
            classmap = {c: k for k, c in enumerate(sorted(source.cats))}
            rows = []
            for rid, g in zip(ids.tolist(), owners.tolist()):
                ann = source.anns[annids[g]]
                gb = torch.tensor(ann["bbox"], device="cuda").float()
                gb[2:] += gb[:2]
                rows.append(dict(image_id=iid, annotation_id=ann["id"], raw_id=rid, level=int(levels[rid]),
                    gt_index=g, area=ann["area"], box_iou=float(box_iou(gb[None], original_boxes[rid:rid+1])[0, 0]),
                    retained_with_gt_class=(rid, classmap[ann["category_id"]]) in present,
                    gt_class_score=float(pred["scores"][0, classmap[ann["category_id"]], rid].sigmoid())))
            image = dict(proto=proto.cpu(), coeff=coeff.cpu(), h=h.cpu(), boxes=boxes.cpu(), levels=levels,
                masks=sample["masks"][0].cpu(), owners=owners.cpu(), target_boxes=target_boxes[ids].cpu(),
                rows=rows, phi=F.adaptive_avg_pool2d(proto, (4, 4)).flatten().cpu(),
                top_ids=top_ids.cpu(), top_classes=classes.cpu(), top_scores=scores.cpu(), class_logits=pred["scores"][0].cpu(),
                original_shape=sample["ori_shape"], ratio_pad=sample["ratio_pad"],
                source_split=domain, all_annotation_ids=annids, segmentation_gain=float(model.args.box))
            if rows:
                if len(audits) < 6 or step == 1:
                    # Perturb coefficients to verify the actual function and gradient,
                    # not just equality at a potentially stationary original output.
                    c = (coeff[ids].detach() + torch.randn_like(coeff[ids]) * .03).requires_grad_(True)
                    modified = pred["mask_coefficient"].detach().clone()
                    modified[0, :, ids] = c.T
                    official = criterion.loss({**pred, "mask_coefficient": modified}, batch)[0][1]
                    expected_grad = torch.autograd.grad(official, c)[0]
                    rois = target_rois(image)
                    replay = roi_losses(c[None], rois)[0] / len(rows) * model.args.box
                    actual_grad = torch.autograd.grad(replay, c)[0]
                    torch.testing.assert_close(replay, official, atol=3e-5, rtol=3e-5)
                    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)
                    audits.append(dict(image_id=iid, group=group, loss=float(official.detach()),
                        value_error=float(abs(official.detach()-replay.detach())),
                        gradient_max_error=float((expected_grad-actual_grad).abs().max())))
                if group == "val":
                    masks = decode(image, coeff[ids], ids.cpu())
                    for row, m in zip(rows, masks):
                        gt = source.annToMask(source.anns[row["annotation_id"]]).astype(bool)
                        row["initial_iou"] = float((m & gt).sum() / max((m | gt).sum(), 1))
            torch.save(image, a.out / "images" / f"{iid:012d}.pt")
            index[group].append(dict(image_id=iid, n=len(rows)))
            counts[group + "_images"] += 1
            counts[group + "_gt"] += len(annids)
            counts[group + "_positive"] += len(rows)
            counts[group + "_one2many_positive"] += int(many[0].sum())
            if step % 25 == 0 or step == len(split[group]):
                state = dict(stage="official_cache", group=group, images=step, total=len(split[group]), elapsed=time.monotonic()-start)
                print(json.dumps(state), flush=True)
                write(a.out / "PROGRESS.json", state)
        write(a.out / "INDEX.json", index)
    augment_module.polygons2masks_overlap = original_rasterizer
    for hook in hooks:
        hook.remove()
    write(a.out / "LOSS_EQUIVALENCE.json", audits)
    write(a.out / "COMPLETE.json", dict(counts=counts, mask_ratio=model.args.mask_ratio,
        overlap_mask=model.args.overlap_mask, branch="one2one", source="official YOLODataset + official TAL + official segmentation objective",
        scope="Frozen checkpoint, no augmentations, no reconstruction of historical assignments; official merged polygons for supervision, original COCO masks for evaluation."))


def optimize(c0, roi, iterations):
    c = c0.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([c], lr=1, max_iter=iterations, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
    def closure():
        opt.zero_grad()
        v = roi_losses(c[None, None], [roi])[0]
        v.backward()
        return v
    opt.step(closure)
    loss = roi_losses(c[None, None], [roi])[0]
    grad = torch.autograd.grad(loss, c)[0]
    return c.detach(), dict(loss=float(loss.detach()), gradient_norm=float(grad.norm()), iterations=opt.state[c]["n_iter"])


def oracle(a):
    index = json.loads((a.bank / "INDEX.json").read_text())
    all_rows = []
    for item in index["val"]:
        image = load(a.bank / "images" / f'{item["image_id"]:012d}.pt')
        all_rows += [{**r, "row_index": k} for k, r in enumerate(image["rows"])]
    rng = random.Random(20260924)
    bad = [r for r in all_rows if r["initial_iou"] < .75]
    good = [r for r in all_rows if r["initial_iou"] >= .75]
    chosen = rng.sample(bad, min(a.each_group, len(bad))) + rng.sample(good, min(a.each_group, len(good)))
    grouped = defaultdict(list)
    for r in chosen:
        grouped[r["image_id"]].append(r)
    coco = COCO(str(a.annotations))
    rows, coefficients, controls = [], {}, []
    start = time.monotonic()
    for iid, group in grouped.items():
        image = load(a.bank / "images" / f"{iid:012d}.pt")
        selected = [r["row_index"] for r in group]
        rois = target_rois(image, selected)
        values = []
        for r, roi in zip(group, rois):
            initial = image["coeff"][r["raw_id"]].cuda()
            cp, sp = optimize(initial, roi, a.iterations)
            cz, sz = optimize(torch.zeros_like(initial), roi, a.iterations)
            best = cp if sp["loss"] <= sz["loss"] else cz
            values.append(best)
            coefficients[str(r["annotation_id"])] = best.cpu()
            r.update(initial_loss=float(roi_losses(initial[None, None], [roi])[0]),
                from_prediction=sp, from_zero=sz, oracle_loss=min(sp["loss"], sz["loss"]))
        masks = decode(image, torch.stack(values), torch.tensor([r["raw_id"] for r in group]))
        for r, m in zip(group, masks):
            gt = coco.annToMask(coco.anns[r["annotation_id"]]).astype(bool)
            r["final_iou_before"] = r["initial_iou"]
            r["final_iou_after"] = float((m & gt).sum() / max((m | gt).sum(), 1))
            rows.append(r)
        if len(group) >= 2 and len(controls) < 20:
            pair_rois = rois[:2]
            c = torch.stack([image["coeff"][r["raw_id"]] for r in group[:2]]).cuda().requires_grad_(True)
            joint_grad = torch.autograd.grad(roi_losses(c[None], pair_rois)[0], c)[0]
            separate = []
            for k, roi in enumerate(pair_rois):
                cc = c[k].detach().clone().requires_grad_(True)
                separate.append(torch.autograd.grad(roi_losses(cc[None, None], [roi])[0], cc)[0])
            err = float((joint_grad - torch.stack(separate)).abs().max())
            assert err < 1e-6
            opt = torch.optim.LBFGS([c], lr=1, max_iter=a.iterations, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
            def closure():
                opt.zero_grad()
                value = roi_losses(c[None], pair_rois)[0]
                value.backward()
                return value
            opt.step(closure)
            controls.append(dict(image_id=iid, gradient_max_error=err,
                independent_loss=sum(r["from_prediction"]["loss"] for r in group[:2]),
                joint_loss=float(roi_losses(c[None], pair_rois)[0].detach())))
        state = dict(stage="official_oracle", instances=len(rows), total=len(chosen), elapsed=time.monotonic()-start)
        print(json.dumps(state), flush=True)
        write(a.out / "PROGRESS.json", state)
        write(a.out / "ROWS.json", rows)
    summary = {"selection": {"failures": len(bad), "successes": len(good)}, "groups": {}, "joint_controls": controls}
    for name, pred in {"final_failure": lambda r:r["initial_iou"] < .75,
                       "final_success": lambda r:r["initial_iou"] >= .75,
                       "retained_failure": lambda r:r["initial_iou"] < .75 and r["retained_with_gt_class"]}.items():
        rr = [r for r in rows if pred(r)]
        summary["groups"][name] = dict(n=len(rr), before=float(np.mean([r["initial_iou"] for r in rr])) if rr else None,
            after=float(np.mean([r["final_iou_after"] for r in rr])) if rr else None,
            repairs=sum(r["initial_iou"] < .75 <= r["final_iou_after"] for r in rr),
            damages=sum(r["final_iou_after"] < .75 <= r["initial_iou"] for r in rr))
    write(a.out / "SUMMARY.json", summary)
    torch.save(coefficients, a.out / "oracle_coefficients.pt")
    write(a.out / "COMPLETE.json", dict(instances=len(rows), pairs=len(controls)))


def split_metadata(bank, index):
    meta = {}
    hs, ps, cs = [], [], []
    for group, items in index.items():
        vals = []
        for item in items:
            image = load(bank / "images" / f'{item["image_id"]:012d}.pt')
            if not image["rows"]:
                continue
            ids = [r["raw_id"] for r in image["rows"]]
            vals.append({**item, "phi": image["phi"]})
            if group == "fit":
                hs.append(image["h"][ids]); cs.append(image["coeff"][ids])
                ps.append(image["phi"].repeat(len(ids), 1))
        rng = random.Random(20260924)
        donors = list(range(len(vals)))
        rng.shuffle(donors)
        assert len(donors) > 1
        for k, j in enumerate(donors):
            vals[j]["donor_phi"] = vals[donors[(k+1) % len(donors)]]["phi"]
        meta[group] = vals
    h, p, c = torch.cat(hs), torch.cat(ps), torch.cat(cs)
    stats = (h.mean(0), h.std(0).clamp_min(.01), p.mean(0), p.std(0).clamp_min(.01), c.std(0).clamp_min(.1))
    return meta, stats


def predict(models, image, item, all_candidates=False):
    ids = torch.arange(len(image["coeff"])) if all_candidates else torch.tensor([r["raw_id"] for r in image["rows"]])
    h = image["h"][ids].cuda()
    c = image["coeff"][ids].cuda()
    levels = image["levels"][ids].cuda()
    values = []
    for name, net in models.items():
        phi = item["donor_phi"] if name == "condition_shuffle" else image["phi"]
        values.append(c + net(h, phi.cuda().expand(len(ids), -1), levels))
    return torch.stack(values), ids


def evaluate(a, meta, models):
    coco = COCO(str(a.annotations))
    rows = []
    for step, item in enumerate(meta["val"], 1):
        image = load(a.bank / "images" / f'{item["image_id"]:012d}.pt')
        with torch.no_grad():
            values, ids = predict(models, image, item)
            rois = target_rois(image)
            c0 = image["coeff"][ids].cuda()
            all_values = torch.cat([c0[None], values])
            losses = []
            for k, roi in enumerate(rois):
                losses.append(roi_losses(all_values[:, k:k+1], [roi]).cpu().tolist())
            ious = {}
            for name, c in zip(["original", *models], all_values):
                masks = decode(image, c, ids)
                ious[name] = []
                for r, m in zip(image["rows"], masks):
                    gt = coco.annToMask(coco.anns[r["annotation_id"]]).astype(bool)
                    ious[name].append(float((m & gt).sum() / max((m | gt).sum(), 1)))
        for k, r in enumerate(image["rows"]):
            assert abs(ious["original"][k] - r["initial_iou"]) < 1e-7
            rows.append({**r, "iou": {name: value[k] for name, value in ious.items()},
                         "loss": dict(zip(["original", *models], losses[k]))})
        if step % 25 == 0:
            print(json.dumps(dict(stage="official_probe_eval", images=step, total=len(meta["val"]))), flush=True)
    summaries = {}
    for group, predicate in {"all_official_positive": lambda r:True,
                             "retained_with_gt_class": lambda r:r["retained_with_gt_class"]}.items():
        rr = [r for r in rows if predicate(r)]
        summaries[group] = {}
        for name in ("original", *models):
            summaries[group][name] = dict(n=len(rr), mean_iou=float(np.mean([r["iou"][name] for r in rr])),
                mean_loss=float(np.mean([r["loss"][name] for r in rr])), mask75=sum(r["iou"][name] >= .75 for r in rr),
                repairs=sum(r["iou"]["original"] < .75 <= r["iou"][name] for r in rr),
                damages=sum(r["iou"][name] < .75 <= r["iou"]["original"] for r in rr))
    write(a.out / "VAL_ROWS.json", rows)
    write(a.out / "VAL_SUMMARY.json", summaries)


def probes(a):
    index = json.loads((a.bank / "INDEX.json").read_text())
    meta, stats = split_metadata(a.bank, index)
    models, optimizers, states = {}, {}, {}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        models[arm] = Probe(arm, *stats).cuda()
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=.0001)
        states[arm] = dict(best_dev=float("inf"), stale=0, history=[], parameters=sum(p.numel() for p in models[arm].parameters()))
    start = time.monotonic()
    active = dict(models)
    for epoch in range(1, a.epochs+1):
        totals = Counter()
        rng = random.Random(a.seed + epoch)
        items = list(meta["fit"])
        rng.shuffle(items)
        for offset in range(0, len(items), a.batch_images):
            minibatch = items[offset:offset+a.batch_images]
            n = sum(item["n"] for item in minibatch)
            for name in active:
                optimizers[name].zero_grad(set_to_none=True)
            for item in minibatch:
                image = load(a.bank / "images" / f'{item["image_id"]:012d}.pt')
                rois = target_rois(image)
                values, _ = predict(active, image, item)
                loss = roi_losses(values, rois) * image["segmentation_gain"]
                assert torch.isfinite(loss).all()
                (loss.sum() / n).backward()
                for name, value in zip(active, loss.detach().tolist()):
                    totals[name] += value
            for name, net in active.items():
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
                optimizers[name].step()
            if offset % (a.batch_images * 20) == 0:
                state = dict(stage="official_probe_fit", epoch=epoch, images=min(offset+a.batch_images, len(items)),
                    total=len(items), active=list(active), elapsed=time.monotonic()-start)
                print(json.dumps(state), flush=True)
                write(a.out / "PROGRESS.json", state)
        dev_sums = Counter()
        with torch.no_grad():
            for item in meta["dev"]:
                image = load(a.bank / "images" / f'{item["image_id"]:012d}.pt')
                rois = target_rois(image)
                values, _ = predict(active, image, item)
                loss = roi_losses(values, rois) * image["segmentation_gain"]
                for name, value in zip(active, loss.tolist()):
                    dev_sums[name] += value
        for name in list(active):
            dev_loss = dev_sums[name] / sum(r["n"] for r in meta["dev"])
            assert np.isfinite(dev_loss)
            row = dict(epoch=epoch, train_loss=totals[name]/sum(r["n"] for r in items), dev_loss=dev_loss)
            states[name]["history"].append(row)
            saved = dict(arm=name, state_dict=active[name].state_dict(), epoch=epoch, dev_loss=dev_loss, seed=a.seed)
            torch.save(saved, a.out / f"{name}_epoch{epoch}.pt")
            if dev_loss < states[name]["best_dev"]:
                states[name].update(best_dev=dev_loss, stale=0, chosen_epoch=epoch)
                torch.save(saved, a.out / f"{name}_best.pt")
            else:
                states[name]["stale"] += 1
            if states[name]["stale"] >= 5:
                del active[name]
        write(a.out / "TRAINING.json", states)
        print(json.dumps(dict(stage="official_probe_epoch", epoch=epoch, dev={n:s["best_dev"] for n,s in states.items()}, active=list(active))), flush=True)
        if not active:
            break
    for name, model in models.items():
        model.load_state_dict(load(a.out / f"{name}_best.pt")["state_dict"])
        model.eval()
    evaluate(a, meta, models)
    write(a.out / "COMPLETE.json", dict(arms=list(models), seed=a.seed, elapsed=time.monotonic()-start))


def analyze(a):
    rows = json.loads((a.probe / "VAL_ROWS.json").read_text())
    oracle_rows = json.loads((a.oracle / "ROWS.json").read_text())
    by_ann = {r["annotation_id"]: r for r in rows}
    matched = []
    for r in oracle_rows:
        pred = by_ann[r["annotation_id"]]
        assert pred["raw_id"] == r["raw_id"]
        assert abs(pred["loss"]["original"] - r["initial_loss"]) < 3e-4
        matched.append({**r, "probe_loss": pred["loss"], "probe_iou": pred["iou"]})
    comparisons = {}
    rng = np.random.default_rng(20260924)
    for group, keep in {"all":lambda r:True, "retained":lambda r:r["retained_with_gt_class"]}.items():
        rr = [r for r in rows if keep(r)]
        for comparator in ("linear", "mlp", "mlp_large", "condition_mean", "condition_shuffle"):
            if comparator not in rows[0]["iou"]:
                continue
            images = defaultdict(list)
            for r in rr:
                images[r["image_id"]].append([r["iou"]["condition_true"] - r["iou"][comparator],
                    int(r["iou"]["condition_true"]>=.75) - int(r["iou"][comparator]>=.75)])
            values = np.array([np.sum(v, axis=0) for v in images.values()])
            counts = np.array([len(v) for v in images.values()])
            draws = rng.integers(0, len(values), (2000, len(values)))
            boot = values[draws].sum(1) / counts[draws].sum(1)[:,None]
            comparisons[group+":true-vs-"+comparator] = dict(delta=(values.sum(0)/counts.sum()).tolist(),
                ci_low=np.quantile(boot,.025,axis=0).tolist(), ci_high=np.quantile(boot,.975,axis=0).tolist())
    gaps = {}
    initial = sum(r["initial_loss"] for r in matched)
    optimum = sum(r["oracle_loss"] for r in matched)
    for arm in rows[0]["loss"]:
        current = sum(r["probe_loss"][arm] for r in matched)
        gaps[arm] = dict(mean_loss=current/len(matched), fraction_of_finite_oracle_loss_gap_closed=(initial-current)/(initial-optimum))
    write(a.out / "SUMMARY.json", dict(paired= comparisons, oracle_subset_gap=gaps,
        scope="Single seed, frozen official one2one supervision, same-candidate IoU and BCE; not COCO AP; finite oracle, GT used only for training/oracle/evaluation."))
    write(a.out / "MATCHED_ORACLE_PROBE.json", matched)
    write(a.out / "COMPLETE.json", dict(val_instances=len(rows), oracle_instances=len(matched)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("cache", "oracle", "probes", "analyze"))
    for key in ("data", "weights", "split", "bank", "annotations", "probe", "oracle"):
        p.add_argument("--"+key, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-images", type=int, default=8)
    p.add_argument("--each-group", type=int, default=100)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--arms", default="linear,mlp,mlp_large,condition_mean,condition_true,condition_shuffle")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    globals()[args.stage](args)
