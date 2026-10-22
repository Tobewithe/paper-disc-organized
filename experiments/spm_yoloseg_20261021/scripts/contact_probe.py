"""Bounded frozen-feature contact probe. No detector/prototype training."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from pycocotools import mask as coco_mask


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def start(root, stage, contract):
    run = root / "runs" / ("RUN_SPM_" + stage.upper() + "_" + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    record = {"study_id": "STUDY_SPM_YOLOSEG_20261021", "run_id": run.name,
              "stage_id": stage, "execution_status": "running", "scientific_status": "not_adjudicated",
              "started_at": now(), "executor": os.uname().nodename, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
              "command": sys.argv, "protocol_sha256": sha(root / "EXECUTION_PROTOCOL.md"),
              "config_sha256": sha(root / "EXECUTION_CONFIG.json"), "source_sha256": sha(__file__),
              "optimizer_steps": 0, "detector_optimizer_steps": 0, "exit_code": None}
    save(run / "run.json", record)
    save(run / "EVAL_CONTRACT.json", contract)
    return run, record


def finish(run, record, status, summary, **extra):
    record.update(execution_status=status, ended_at=now(), **extra)
    save(run / "SUMMARY.json", summary)
    save(run / "run.json", record)
    manifest = {str(p.relative_to(run)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                for p in run.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    save(run / "ARTIFACT_MANIFEST.json", manifest)


def boundaries(ids, overlap, radius=4):
    fg = ids > 0
    c, b = np.zeros_like(fg), np.zeros_like(fg)
    for axis in (0, 1):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        z = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        x, y = ids[a], ids[z]
        cc = (x > 0) & (y > 0) & (x != y)
        bb = (x > 0) != (y > 0)
        c[a] |= cc; c[z] |= cc
        b[a] |= bb; b[z] |= bb
    dc = cv2.distanceTransform((~c).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    bgdist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    invalid = cv2.dilate(overlap.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    invalid[:12] = True; invalid[-12:] = True; invalid[:, :12] = True; invalid[:, -12:] = True
    frac = cv2.blur(fg.astype(np.float32), (17, 17))
    lab = np.full(ids.shape, -1, np.int8)
    lab[fg & (dc > 12) & (db > 12) & (frac >= .98) & ~invalid] = 2
    lab[fg & (db <= radius) & (dc > 12) & ~invalid] = 0
    lab[fg & (dc <= radius) & (bgdist > 8) & (frac >= .98) & ~invalid] = 1
    return lab, frac


def letterbox(image, ids, overlap):
    h, w = image.shape[:2]; ratio = 640 / max(h, w)
    nh, nw = round(h * ratio), round(w * ratio)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    rgb = np.full((640, 640, 3), 114, np.uint8)
    target = np.zeros((640, 640), np.int32)
    bad = np.zeros((640, 640), np.uint8)
    rgb[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    target[top:top+nh, left:left+nw] = cv2.resize(ids, (nw, nh), interpolation=cv2.INTER_NEAREST)
    bad[top:top+nh, left:left+nw] = cv2.resize(overlap.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    return rgb, target, bad.astype(bool), {"ratio": ratio, "left": left, "top": top, "new_hw": [nh, nw]}


def p0(root, run, cfg):
    inp = root / "inputs"; data = json.loads((inp / "annotations.json").read_text())
    assert sha(inp / "annotations.json") == cfg["annotation_sha256"]
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)
    (run / "cache").mkdir(); (run / "qa").mkdir()
    rows, totals, family_counts = [], Counter(), defaultdict(Counter)
    yy, xx = np.meshgrid(np.arange(4, 640, 8), np.arange(4, 640, 8), indexing="ij")
    for k, im in enumerate(data["images"]):
        path = inp / im["file_name"]
        assert sha(path) == im["source_sha256"], path
        image = cv2.imread(str(path)); assert image is not None
        assert list(image.shape[:2]) == [im["height"], im["width"]]
        ids = np.zeros(image.shape[:2], np.int32); overlap = np.zeros_like(ids, bool)
        for j, ann in enumerate(by_image[im["id"]], 1):
            seg = ann["segmentation"]
            rle = coco_mask.merge(coco_mask.frPyObjects(seg, im["height"], im["width"])) if isinstance(seg, list) else seg
            mask = coco_mask.decode(rle).astype(bool)
            assert mask.shape == ids.shape
            overlap |= (ids > 0) & mask
            ids[mask] = j
        image, ids, overlap, transform = letterbox(image, ids, overlap)
        lab, fraction = boundaries(ids, overlap)
        rng = np.random.default_rng(cfg["seed"] + im["id"])
        sel = []
        for label in (0, 1, 2):
            candidates = np.flatnonzero(lab[yy, xx].ravel() == label)
            if len(candidates): sel.extend(rng.choice(candidates, min(cfg["tokens_per_class_image"], len(candidates)), replace=False))
        sel = np.array(sorted(sel), dtype=np.int64)
        y, x = yy.ravel()[sel], xx.ravel()[sel]; labels = lab[y, x]
        counts = {str(t): int((labels == t).sum()) for t in (0, 1, 2)}
        totals.update(counts); family_counts[im["group_id"]].update(counts)
        np.savez_compressed(run / "cache" / f"{im['id']}.npz", image=image, x=x, y=y, labels=labels,
                            fg_fraction=fraction[y, x], grid_index=sel)
        row = dict(im); row.update(token_counts=counts, transform=transform,
                                    overlap_pixels=int(overlap.sum()), cache=f"cache/{im['id']}.npz")
        rows.append(row)
        if k % 19 == 0:
            view = image.copy()
            for px, py, label in zip(x, y, labels):
                cv2.circle(view, (int(px), int(py)), 3, {0: (255, 150, 0), 1: (0, 0, 255), 2: (0, 255, 0)}[int(label)], -1)
            cv2.putText(view, f"id={im['id']} C=red B=blue I=green", (8, 24), 0, .6, (255, 255, 255), 1)
            cv2.imwrite(str(run / "qa" / f"{im['id']}.jpg"), view)
        if k % 20 == 0: print("p0", k+1, len(data["images"]), flush=True)
    eligible = [g for g, cnt in family_counts.items() if min(cnt[str(t)] for t in (0, 1, 2)) >= 16]
    eligible.sort(key=lambda g: hashlib.sha256((str(cfg["seed"]) + g).encode()).hexdigest())
    gate = len(eligible) >= 8
    dev = eligible[:max(4, round(len(eligible) * .3))] if gate else []
    for row in rows: row["split"] = "dev" if row["group_id"] in dev else "train"
    split = {"rows": rows, "dev_groups": dev, "eligible_groups": eligible, "family_counts": dict(family_counts),
             "group_identity": "inferred filename family; physical pen/camera identities unknown",
             "external_test_used": False, "native_detection_failure_labels": "not_measured_in_this_feature_probe"}
    save(run / "SPLITS_AND_LEDGER.json", split)
    result = {"p0_gate": gate, "images": len(rows), "videos": len({r['video_id'] for r in rows}),
              "inferred_families": len(family_counts), "eligible_families": len(eligible),
              "dev_families": len(dev), "token_counts": dict(totals), "family_counts": dict(family_counts),
              "source_test_excluded": True, "scope": "exploratory grouped development, not verified cross-pen generalization"}
    if gate: save(root / "P0_POINTER.json", {"run_id": run.name, "ledger_sha256": sha(run / "SPLITS_AND_LEDGER.json")})
    return result


def edge_features(image):
    a = image.astype(np.float32) / 255
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mu = cv2.blur(gray, (9, 9)); std = np.sqrt(np.maximum(cv2.blur(gray*gray, (9, 9)) - mu*mu, 0))
    y, x = np.indices(gray.shape, dtype=np.float32)
    return np.dstack([a, gray, gx, gy, mag, cv2.blur(mag, (9, 9)), mu, std, x/640, y/640])


def extract(root, run, cfg, smoke):
    from ultralytics import YOLO
    import ultralytics
    pointer = json.loads((root / "P0_POINTER.json").read_text()); source = root / "runs" / pointer["run_id"]
    assert sha(source / "SPLITS_AND_LEDGER.json") == pointer["ledger_sha256"]
    ledger = json.loads((source / "SPLITS_AND_LEDGER.json").read_text())
    assert sha(cfg["weights"]) == cfg["weights_sha256"]
    assert ultralytics.__version__ == "8.4.100"
    head_path = Path(ultralytics.__file__).parent / "nn/modules/head.py"
    assert sha(head_path) == cfg["head_source_sha256"]
    torch.manual_seed(cfg["seed"]); torch.set_num_threads(4)
    model = YOLO(cfg["weights"]).model.to("cuda").eval().float()
    for p in model.parameters(): p.requires_grad_(False)
    captured = {}
    hook = model.model[-1].register_forward_pre_hook(lambda mod, args: captured.update(p3=args[0][0].detach().clone(), p4=args[0][1].detach().clone()))
    arrays = defaultdict(list); records = []; feature_shapes = None
    rows = ledger["rows"][:4] if smoke else ledger["rows"]
    initial = {name: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for name, t in model.state_dict().items()}
    for k, row in enumerate(rows):
        z = np.load(source / row["cache"]); image, x, y = z["image"], z["x"], z["y"]
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0).to("cuda") / 255
        with torch.no_grad(): model(tensor)
        p3, p4 = captured["p3"], captured["p4"]
        assert tuple(p3.shape[-2:]) == (80, 80)
        feature_shapes = {"p3": list(p3.shape), "p4": list(p4.shape)}
        indices = torch.as_tensor(z["grid_index"], device="cuda")
        arrays["E1"].append(p3.flatten(2)[0, :, indices].T.cpu())
        arrays["E2"].append(TF.unfold(p3, 3, padding=1)[0, :, indices].T.cpu())
        arrays["E3"].append(TF.interpolate(p4, p3.shape[-2:], mode="bilinear", align_corners=False).flatten(2)[0, :, indices].T.cpu())
        arrays["E0"].append(torch.from_numpy(edge_features(image)[y, x].copy()))
        for i, label in enumerate(z["labels"]):
            records.append({"image_id": row["id"], "group": row["group_id"], "split": row["split"],
                            "label": int(label), "fg_fraction": float(z["fg_fraction"][i]), "x": int(x[i]), "y": int(y[i])})
        if k % 20 == 0: print("extract", k+1, len(rows), flush=True)
    hook.remove()
    final = {name: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for name, t in model.state_dict().items()}
    assert initial == final
    arrays = {k: torch.cat(v).float() for k, v in arrays.items()}
    torch.save({"features": arrays, "records": records}, run / "FEATURES.pt")
    if not smoke: save(root / "FEATURE_POINTER.json", {"run_id": run.name, "sha256": sha(run / "FEATURES.pt")})
    return {"images": len(rows), "tokens": len(records), "feature_shapes": feature_shapes,
            "frozen_model_unchanged": True, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "feature_columns": {k: v.shape[1] for k, v in arrays.items()},
            "p0_run": source.name}


def average_precision(y, score):
    y = np.asarray(y, int); score = np.asarray(score)
    if y.sum() == 0: return None
    order = np.argsort(-score, kind="stable"); y, score = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(y)-1]
    tp = np.cumsum(y)[ends]; precision = tp / (ends+1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def probe(root, run, record, cfg, smoke):
    pointer = json.loads((root / "FEATURE_POINTER.json").read_text()); path = root / "runs" / pointer["run_id"] / "FEATURES.pt"
    assert sha(path) == pointer["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]; features = payload["features"]
    torch.set_num_threads(4); results = {}; step_count = 0
    for task, negative in (("C_vs_I", 2), ("C_vs_B", 0)):
        rng = np.random.default_rng(cfg["seed"])
        selected = []
        by_image = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(records): by_image[row["image_id"]][row["label"]].append(i)
        for rows in by_image.values():
            n = min(len(rows[1]), len(rows[negative]))
            if n: selected.extend(rng.choice(rows[1], n, replace=False)); selected.extend(rng.choice(rows[negative], n, replace=False))
        selected = np.asarray(sorted(selected), int)
        y = torch.tensor([float(records[i]["label"] == 1) for i in selected])
        train = np.array([records[i]["split"] == "train" for i in selected]); test = ~train
        assert train.any() and test.any() and y[train].sum() > 0 and y[test].sum() > 0
        groups = np.array([records[i]["group"] for i in selected])[test]
        task_result = {"train_tokens": int(train.sum()), "dev_tokens": int(test.sum()),
                       "dev_prevalence": float(y[test].mean()), "arms": {}}
        raw_scores = {}
        for arm in ("E0", "E1", "E2", "E3"):
            torch.manual_seed(cfg["seed"])
            a = features[arm][selected]
            mu = a[train].mean(0); std = a[train].std(0).clamp_min(.01)
            a = (a - mu) / std
            net = torch.nn.Linear(a.shape[1], 1) if arm in ("E1", "E3") else torch.nn.Sequential(torch.nn.Linear(a.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
            opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
            train_ids = torch.as_tensor(np.flatnonzero(train)); gen = torch.Generator().manual_seed(cfg["seed"])
            steps = 10 if smoke else cfg["steps_per_arm_task"]
            for _ in range(steps):
                idx = train_ids[torch.randint(len(train_ids), (256,), generator=gen)]
                loss = TF.binary_cross_entropy_with_logits(net(a[idx]).flatten(), y[idx])
                assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); opt.step(); step_count += 1
            with torch.no_grad(): score = net(a[test]).flatten().numpy()
            raw_scores[arm] = score
            per_group = {}
            target = y[test].numpy()
            for group in sorted(set(groups)):
                take = groups == group
                if min(int(target[take].sum()), int((1-target[take]).sum())) >= 8:
                    per_group[str(group)] = average_precision(target[take], score[take])
            task_result["arms"][arm] = {"pooled_AP": average_precision(target, score),
                    "group_macro_AP": float(np.mean(list(per_group.values()))) if per_group else None,
                    "per_group_AP": per_group, "parameters": sum(p.numel() for p in net.parameters()),
                    "last_train_loss": float(loss), "steps": steps}
            torch.save({"model": net.state_dict(), "mean": mu, "std": std, "source_features_sha256": pointer["sha256"]}, run / f"{task}_{arm}.pt")
            print(task, arm, task_result["arms"][arm], flush=True)
        common = sorted(set(task_result["arms"]["E2"]["per_group_AP"]) & set(task_result["arms"]["E0"]["per_group_AP"]))
        delta = np.array([task_result["arms"]["E2"]["per_group_AP"][g] - task_result["arms"]["E0"]["per_group_AP"][g] for g in common])
        if len(delta):
            rng = np.random.default_rng(1729); boots = rng.choice(delta, (2000, len(delta))).mean(1)
            task_result["E2_minus_E0"] = {"mean": float(delta.mean()), "ci95": np.quantile(boots, [.025, .975]).tolist(), "groups": len(delta)}
        else: task_result["E2_minus_E0"] = {"mean": None, "ci95": None, "groups": 0}
        np.savez_compressed(run / f"{task}_PREDICTIONS.npz", y=y[test].numpy(), groups=groups, **raw_scores)
        results[task] = task_result
    record["optimizer_steps"] = step_count
    primary = results["C_vs_I"]["E2_minus_E0"]
    if smoke: verdict = "smoke_passed_not_scientific"
    elif primary["groups"] < 4: verdict = "inconclusive_cluster_support"
    elif primary["mean"] >= .05 and primary["ci95"][0] > 0 and results["C_vs_I"]["arms"]["E2"]["group_macro_AP"] >= .65:
        verdict = "exploratory_contact_readability_supported"
    elif primary["ci95"][1] < .05: verdict = "minimum_contact_readability_increment_not_met"
    else: verdict = "inconclusive_effect_precision"
    save(run / "DECISION.json", {"status": verdict, "primary": "C_vs_I E2-minus-E0 inferred-family macro AP",
            "claim_scope": "exploratory pretrained frozen-feature readability, not AP utility or physical-pen generalization",
            "next_action": "train-only contact-token prototype coverage on existing features" if verdict == "exploratory_contact_readability_supported" else "prioritize the observed pixel/local-geometry representation evidence; do not expand appearance memory training"})
    return {"verdict": verdict, "tasks": results, "feature_run": pointer["run_id"], "source_test_excluded": True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["p0", "extract_smoke", "extract", "probe_smoke", "probe"])
    parser.add_argument("--root", required=True); a = parser.parse_args()
    root = Path(a.root); cfg = json.loads((root / "EXECUTION_CONFIG.json").read_text())
    lock = root / ("LOCK_" + a.stage)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    run, record = start(root, a.stage, cfg); print("RUN", run.name, flush=True)
    t = time.monotonic()
    try:
        if a.stage == "p0": result = p0(root, run, cfg)
        elif a.stage.startswith("extract"): result = extract(root, run, cfg, a.stage.endswith("smoke"))
        else: result = probe(root, run, record, cfg, a.stage.endswith("smoke"))
        result["elapsed_seconds"] = time.monotonic() - t
        finish(run, record, "completed", result, exit_code=0, scientific_status=result.get("verdict", "p0_or_features_only"))
        save(root / (a.stage.upper() + "_POINTER.json"), {"run_id": run.name}) if a.stage not in ("p0", "extract") else None
        print("DONE", json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException:
        (run / "ERROR.log").write_text(traceback.format_exc())
        finish(run, record, "failed", {"error": traceback.format_exc(), "elapsed_seconds": time.monotonic()-t}, exit_code=1)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__": main()
